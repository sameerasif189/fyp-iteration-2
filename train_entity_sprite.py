"""
Training script for the Entity Sprite Generator.

Trains a conditional DCGAN to generate 128x128 RGBA entity sprites
from stress level + design class + noise.

Usage:
  python train_entity_sprite.py --epochs 60 --batch-size 64
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

from src.iteration2.entity_sprite_model import (
    EntitySpriteGenerator, EntitySpriteDiscriminator,
    NOISE_DIM, NUM_DESIGNS, count_params
)
from src.iteration2.entity_sprite_dataset import EntitySpriteDataset


def gradient_penalty(discriminator, real, fake, stress_levels, design_classes, device):
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out, _ = discriminator(interp, stress_levels, design_classes)
    grad = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grad.view(bs, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train(
    epochs: int = 120,
    batch_size: int = 48,
    lr_g: float = 1.5e-4,
    lr_d: float = 3e-4,
    n_critic: int = 4,
    gp_weight: float = 10.0,
    aux_weight: float = 1.5,
    samples_per_level: int = 900,
    assets_dir: str = "assets",
    checkpoint_dir: str = "checkpoints/entity_sprite",
    model_out: str = "models/entity_sprite_gen.pt",
    onnx_out: str = "models/entity_sprite_gen.onnx",
    device_str: str = "auto",
    min_delta: float = 3e-4,
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[entity-sprite] device={device}")

    dataset = EntitySpriteDataset(samples_per_level=samples_per_level,
                                  assets_dir=assets_dir)
    print(f"[entity-sprite] dataset: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[entity-sprite] batches/epoch: {len(loader)}")

    gen = EntitySpriteGenerator().to(device)
    disc = EntitySpriteDiscriminator().to(device)

    print(f"[entity-sprite] G params: {count_params(gen):,}")
    print(f"[entity-sprite] D params: {count_params(disc):,}")

    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.9))
    sch_g = optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=max(20, epochs), eta_min=lr_g * 0.2
    )
    sch_d = optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=max(20, epochs), eta_min=lr_d * 0.2
    )

    ce = nn.CrossEntropyLoss()

    history = []
    best_g_loss = float("inf")
    patience_counter = 0
    patience = 20

    print(f"[entity-sprite] starting: {epochs} epochs, batch={batch_size}")
    print(f"[entity-sprite] early stopping: patience={patience}, min_delta={min_delta}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_d, epoch_g = 0.0, 0.0
        d_steps, g_steps = 0, 0

        progress = tqdm(loader, desc=f"entity epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (real_imgs, stress_levels, design_classes) in enumerate(progress):
            real_imgs = real_imgs.to(device)
            stress_levels = stress_levels.to(device)
            design_classes = design_classes.to(device)
            bs = real_imgs.size(0)

            noise = torch.randn(bs, NOISE_DIM, device=device)
            fake_imgs = gen(noise, stress_levels, design_classes).detach()

            d_real, aux_real = disc(real_imgs, stress_levels, design_classes)
            d_fake, _ = disc(fake_imgs, stress_levels, design_classes)
            gp = gradient_penalty(disc, real_imgs, fake_imgs, stress_levels,
                                  design_classes, device)
            d_loss = (d_fake.mean() - d_real.mean() + gp_weight * gp
                      + aux_weight * ce(aux_real, design_classes))

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            epoch_d += d_loss.item()
            d_steps += 1

            if (batch_idx + 1) % n_critic == 0:
                noise = torch.randn(bs, NOISE_DIM, device=device)
                fake_imgs = gen(noise, stress_levels, design_classes)
                g_rf, g_aux = disc(fake_imgs, stress_levels, design_classes)
                g_loss = (-g_rf.mean()
                          + aux_weight * ce(g_aux, design_classes))

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                epoch_g += g_loss.item()
                g_steps += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                progress.set_postfix({
                    "D": f"{d_loss.item():.4f}",
                    "G": f"{(g_loss.item() if (batch_idx + 1) % n_critic == 0 else float('nan')):.4f}",
                })
            if (batch_idx + 1) % 25 == 0:
                g_display = g_loss.item() if (batch_idx + 1) % n_critic == 0 else float("nan")
                print(
                    f"[step entity] epoch={epoch}/{epochs} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"D={d_loss.item():.4f} G={g_display:.4f}"
                )

        dt = time.perf_counter() - t0
        avg_d = epoch_d / max(d_steps, 1)
        avg_g = epoch_g / max(g_steps, 1)

        history.append({"epoch": epoch, "d_loss": round(avg_d, 6),
                        "g_loss": round(avg_g, 6), "time_s": round(dt, 2)})

        sch_g.step()
        sch_d.step()
        print(
            f"[epoch {epoch}/{epochs}] D={avg_d:.4f} G={avg_g:.4f} "
            f"lrG={opt_g.param_groups[0]['lr']:.2e} "
            f"lrD={opt_d.param_groups[0]['lr']:.2e} time={dt:.1f}s"
        )

        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"entity_sprite_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "gen_state": gen.state_dict(),
                        "disc_state": disc.state_dict()}, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

        if g_steps > 0 and avg_g < (best_g_loss - min_delta):
            best_g_loss = avg_g
            patience_counter = 0
            torch.save({"epoch": epoch, "gen_state": gen.state_dict(),
                        "g_loss": best_g_loss}, model_path)
            print(f"  [best] saved {model_path}")
        else:
            patience_counter += 1
            print(f"  [plateau] no improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  [early-stop] converged after {epoch} epochs")
                break

    gen.eval()
    dummy_noise = torch.randn(1, NOISE_DIM, device=device)
    dummy_level = torch.tensor([4], dtype=torch.long, device=device)
    dummy_design = torch.tensor([0], dtype=torch.long, device=device)
    onnx_path = Path(onnx_out)
    try:
        torch.onnx.export(gen, (dummy_noise, dummy_level, dummy_design), str(onnx_path),
            input_names=["noise", "stress_level", "design_class"],
            output_names=["sprite"],
            dynamic_axes={"noise": {0: "batch"}, "stress_level": {0: "batch"},
                          "design_class": {0: "batch"}, "sprite": {0: "batch"}},
            opset_version=18, dynamo=False)
        print(f"[onnx] exported {onnx_path}")
    except Exception as e:
        print(f"[onnx] export failed: {e}")

    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] entity sprite training complete. best_g_loss={best_g_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--samples-per-level", type=int, default=900)
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--min-delta", type=float, default=3e-4)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          samples_per_level=args.samples_per_level,
          assets_dir=args.assets_dir, device_str=args.device,
          min_delta=args.min_delta)
