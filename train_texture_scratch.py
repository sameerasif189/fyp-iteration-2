"""
Training script for the Texture-From-Scratch Generator.

Trains a conditional DCGAN to generate 128x128 horror textures
from just stress level + noise, no input image needed.

Usage:
  python train_texture_scratch.py --epochs 40 --batch-size 32
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

from src.iteration2.texture_scratch_model import (
    TextureScratchGenerator, TextureScratchDiscriminator,
    NOISE_DIM, count_params
)
from src.iteration2.texture_scratch_dataset import TextureScratchDataset


def gradient_penalty(discriminator, real, fake, stress_levels, device):
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_out = discriminator(interp, stress_levels)
    grad = torch.autograd.grad(
        outputs=d_out, inputs=interp,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]
    grad_norm = grad.view(bs, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train(
    epochs: int = 40,
    batch_size: int = 32,
    lr_g: float = 2e-4,
    lr_d: float = 4e-4,
    n_critic: int = 3,
    gp_weight: float = 10.0,
    assets_dir: str = "assets",
    checkpoint_dir: str = "checkpoints/texture_scratch",
    model_out: str = "models/texture_scratch_gen.pt",
    onnx_out: str = "models/texture_scratch_gen.onnx",
    device_str: str = "auto",
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[tex-scratch] device={device}")

    dataset = TextureScratchDataset(assets_dir=assets_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    gen = TextureScratchGenerator().to(device)
    disc = TextureScratchDiscriminator().to(device)

    print(f"[tex-scratch] generator params: {count_params(gen):,}")
    print(f"[tex-scratch] discriminator params: {count_params(disc):,}")

    opt_g = optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.9))

    history = []
    best_g_loss = float("inf")
    patience_counter = 0
    patience = 10

    print(f"[tex-scratch] starting: {epochs} epochs, {len(dataset)} samples, batch={batch_size}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_d, epoch_g = 0.0, 0.0
        d_steps, g_steps = 0, 0

        for batch_idx, (real_imgs, stress_levels) in enumerate(loader):
            real_imgs = real_imgs.to(device)
            stress_levels = stress_levels.to(device)
            bs = real_imgs.size(0)

            noise = torch.randn(bs, NOISE_DIM, device=device)
            fake_imgs = gen(noise, stress_levels).detach()

            d_real = disc(real_imgs, stress_levels).mean()
            d_fake = disc(fake_imgs, stress_levels).mean()
            gp = gradient_penalty(disc, real_imgs, fake_imgs, stress_levels, device)
            d_loss = d_fake - d_real + gp_weight * gp

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            epoch_d += d_loss.item()
            d_steps += 1

            if (batch_idx + 1) % n_critic == 0:
                noise = torch.randn(bs, NOISE_DIM, device=device)
                fake_imgs = gen(noise, stress_levels)
                g_loss = -disc(fake_imgs, stress_levels).mean()

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()
                epoch_g += g_loss.item()
                g_steps += 1

        dt = time.perf_counter() - t0
        avg_d = epoch_d / max(d_steps, 1)
        avg_g = epoch_g / max(g_steps, 1)

        history.append({"epoch": epoch, "d_loss": round(avg_d, 6),
                        "g_loss": round(avg_g, 6), "time_s": round(dt, 2)})

        print(f"[epoch {epoch}/{epochs}] D={avg_d:.4f} G={avg_g:.4f} time={dt:.1f}s")

        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"tex_scratch_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "gen_state": gen.state_dict(),
                        "disc_state": disc.state_dict()}, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

        if g_steps > 0 and avg_g < best_g_loss:
            best_g_loss = avg_g
            patience_counter = 0
            torch.save({"epoch": epoch, "gen_state": gen.state_dict(),
                        "g_loss": best_g_loss}, model_path)
            print(f"  [best] saved {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [early-stop] no improvement for {patience} epochs")
                break

    gen.eval()
    dummy_noise = torch.randn(1, NOISE_DIM, device=device)
    dummy_level = torch.tensor([3], dtype=torch.long, device=device)
    onnx_path = Path(onnx_out)
    try:
        torch.onnx.export(gen, (dummy_noise, dummy_level), str(onnx_path),
            input_names=["noise", "stress_level"], output_names=["texture"],
            dynamic_axes={"noise": {0: "batch"}, "stress_level": {0: "batch"},
                          "texture": {0: "batch"}},
            opset_version=18, dynamo=False)
        print(f"[onnx] exported {onnx_path}")
    except Exception as e:
        print(f"[onnx] export failed: {e}")

    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] texture-from-scratch training complete. best_g_loss={best_g_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          assets_dir=args.assets_dir, device_str=args.device)
