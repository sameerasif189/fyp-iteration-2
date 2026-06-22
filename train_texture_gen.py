"""
Training script for the Texture Corruption U-Net.

Trains a conditional U-Net to generate horror texture corruptions
(blood, grime, cracks, decay, scorch) from clean textures.

Usage:
  python train_texture_gen.py --epochs 30 --batch-size 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.iteration2.texture_gen_model import (
    TextureCorruptionUNet, PatchDiscriminator, IMG_SIZE, count_params
)
from src.iteration2.texture_dataset import TextureCorruptionDataset


def perceptual_loss_simple(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple perceptual loss using multi-scale feature comparison.
    (Avoids VGG dependency -- uses downsampled L1 at multiple scales)
    """
    loss = torch.tensor(0.0, device=pred.device)
    p, t = pred, target

    for scale in range(4):
        loss = loss + nn.functional.l1_loss(p, t)
        if p.shape[-1] > 4:
            p = nn.functional.avg_pool2d(p, 2)
            t = nn.functional.avg_pool2d(t, 2)

    return loss / 4.0


def train(
    epochs: int = 30,
    batch_size: int = 8,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    l1_weight: float = 100.0,
    perceptual_weight: float = 10.0,
    adversarial_weight: float = 1.0,
    assets_dir: str = "assets",
    checkpoint_dir: str = "checkpoints/texture_gen",
    model_out: str = "models/texture_generator.pt",
    onnx_out: str = "models/texture_generator.onnx",
    device_str: str = "auto",
    samples_per_level: int = 200,
    min_delta: float = 1e-3,
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[texture-train] device={device}")

    assets_path = Path(assets_dir)
    has_real = (assets_path / "textures").exists() and any((assets_path / "textures").iterdir())

    dataset = TextureCorruptionDataset(
        assets_dir=assets_path if has_real else None,
        samples_per_level=samples_per_level,
    )
    print(f"[texture-train] dataset prepared: {len(dataset)} samples")

    if len(dataset) == 0:
        print("[texture-train] ERROR: no training data")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[texture-train] dataloader batches per epoch: {len(loader)}")

    generator = TextureCorruptionUNet().to(device)
    discriminator = PatchDiscriminator().to(device)

    print(f"[texture-train] generator params: {count_params(generator):,}")
    print(f"[texture-train] discriminator params: {count_params(discriminator):,}")

    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

    bce = nn.BCEWithLogitsLoss()

    history = []
    best_g_loss = float("inf")
    patience_counter = 0
    patience = 8

    print(f"[texture-train] starting: {epochs} epochs, {len(dataset)} samples, batch={batch_size}")
    print(f"[texture-train] early stopping: patience={patience}, min_delta={min_delta}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_g_total = 0.0
        epoch_d_total = 0.0
        epoch_l1 = 0.0
        step_count = 0

        progress = tqdm(loader, desc=f"texture epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (clean, corrupted_gt, levels) in enumerate(progress):
            clean = clean.to(device)
            corrupted_gt = corrupted_gt.to(device)
            levels = levels.to(device)
            bs = clean.size(0)

            # --- Generator forward ---
            fake_corrupted = generator(clean, levels)

            # --- Train Discriminator ---
            d_real = discriminator(corrupted_gt, levels)
            d_fake = discriminator(fake_corrupted.detach(), levels)

            real_label = torch.ones_like(d_real) * 0.9
            fake_label = torch.zeros_like(d_fake) + 0.1

            d_loss_real = bce(d_real, real_label)
            d_loss_fake = bce(d_fake, fake_label)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            d_fake_for_g = discriminator(fake_corrupted, levels)
            g_adv = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_l1 = nn.functional.l1_loss(fake_corrupted, corrupted_gt)
            g_perceptual = perceptual_loss_simple(fake_corrupted, corrupted_gt)

            g_loss = (adversarial_weight * g_adv +
                      l1_weight * g_l1 +
                      perceptual_weight * g_perceptual)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_g_total += g_loss.item()
            epoch_d_total += d_loss.item()
            epoch_l1 += g_l1.item()
            step_count += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                progress.set_postfix({
                    "G": f"{g_loss.item():.4f}",
                    "D": f"{d_loss.item():.4f}",
                    "L1": f"{g_l1.item():.4f}",
                })
            if (batch_idx + 1) % 25 == 0:
                print(
                    f"[step texture] epoch={epoch}/{epochs} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"G={g_loss.item():.4f} D={d_loss.item():.4f} L1={g_l1.item():.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()

        dt = time.perf_counter() - t0
        avg_g = epoch_g_total / max(step_count, 1)
        avg_d = epoch_d_total / max(step_count, 1)
        avg_l1 = epoch_l1 / max(step_count, 1)

        record = {
            "epoch": epoch,
            "g_loss": round(avg_g, 6),
            "d_loss": round(avg_d, 6),
            "l1_loss": round(avg_l1, 6),
            "time_s": round(dt, 2),
        }
        history.append(record)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"G={avg_g:.4f} D={avg_d:.4f} L1={avg_l1:.4f} "
            f"time={dt:.1f}s"
        )

        if epoch % 5 == 0:
            ckpt_path = ckpt_dir / f"texture_gen_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

        if avg_g < (best_g_loss - min_delta):
            best_g_loss = avg_g
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "g_loss": best_g_loss,
            }, model_path)
            print(f"  [best] saved {model_path}")
        else:
            patience_counter += 1
            print(f"  [plateau] no significant improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  [early-stop] converged/plateaued for {patience} epochs, stopping")
                break

    # ONNX export
    generator.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_level = torch.tensor([3], dtype=torch.long, device=device)
    onnx_path = Path(onnx_out)

    torch.onnx.export(
        generator, (dummy_input, dummy_level), str(onnx_path),
        input_names=["clean_texture", "stress_level"],
        output_names=["corrupted_texture"],
        dynamic_axes={
            "clean_texture": {0: "batch"},
            "stress_level": {0: "batch"},
            "corrupted_texture": {0: "batch"},
        },
        opset_version=18, dynamo=False,
    )
    print(f"[onnx] exported {onnx_path}")

    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] texture generator training complete. best_g_loss={best_g_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Texture Corruption U-Net")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/texture_gen")
    parser.add_argument("--model-out", type=str, default="models/texture_generator.pt")
    parser.add_argument("--onnx-out", type=str, default="models/texture_generator.onnx")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--samples-per-level", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        assets_dir=args.assets_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_out=args.model_out,
        onnx_out=args.onnx_out,
        device_str=args.device,
        samples_per_level=args.samples_per_level,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
