"""
Training script for the Fusion Generator model.

Trains the conditional generator with:
  - Range-matching MSE loss per stress level
  - Diversity regularisation (outputs should vary with noise)
  - Smoothness regularisation (adjacent stress levels should be close)
  - Proper checkpointing, logging, and ONNX export
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader

from .fusion_model import FusionGenerator, NOISE_DIM, NUM_OUTPUTS, count_parameters
from .dataset import FusionDataset


def diversity_loss(model: FusionGenerator, stress_levels: torch.Tensor,
                   device: torch.device, n_samples: int = 4) -> torch.Tensor:
    """Penalise low variance across different noise inputs for same stress level."""
    unique_levels = stress_levels.unique()
    total = torch.tensor(0.0, device=device)
    count = 0
    for lv in unique_levels:
        noises = torch.randn(n_samples, model.noise_dim, device=device)
        lvs = lv.expand(n_samples)
        outputs = model(lvs, noises)
        variance = outputs.var(dim=0).mean()
        total = total + torch.exp(-variance * 10.0)
        count += 1
    return total / max(count, 1)


def smoothness_loss(model: FusionGenerator, device: torch.device,
                    batch_noise: torch.Tensor) -> torch.Tensor:
    """Adjacent stress levels should produce nearby (but distinct) outputs."""
    total = torch.tensor(0.0, device=device)
    noise = batch_noise[:1].expand(5, -1)
    for lv in range(5):
        lv_a = torch.tensor([lv], device=device).expand(1)
        lv_b = torch.tensor([lv + 1], device=device).expand(1)
        n = noise[:1]
        out_a = model(lv_a, n)
        out_b = model(lv_b, n)
        diff = (out_a - out_b).pow(2).mean()
        total = total + torch.relu(diff - 0.15)
    return total / 5.0


def train(
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 3e-4,
    samples_per_level: int = 5000,
    checkpoint_dir: str = "checkpoints",
    model_out: str = "models/fusion_generator.pt",
    onnx_out: str = "models/fusion_generator.onnx",
    device_str: str = "auto",
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(onnx_out)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[train] device={device}")
    print(f"[train] building dataset: {samples_per_level} samples/level, {samples_per_level * 6} total")

    dataset = FusionDataset(samples_per_level=samples_per_level)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = FusionGenerator().to(device)
    print(f"[train] model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    patience = 8
    history = []

    print(f"[train] starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"[train] early stopping: patience={patience} epochs")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_div = 0.0
        epoch_smooth = 0.0
        step_count = 0
        t0 = time.perf_counter()

        for batch_idx, (levels, noise, targets) in enumerate(loader):
            levels = levels.to(device)
            noise = noise.to(device)
            targets = targets.to(device)

            outputs = model(levels, noise)

            loss_mse = mse(outputs, targets)
            loss_div = diversity_loss(model, levels, device) * 0.1
            loss_smooth = smoothness_loss(model, device, noise) * 0.05
            loss = loss_mse + loss_div + loss_smooth

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_div += loss_div.item()
            epoch_smooth += loss_smooth.item()
            step_count += 1

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(
                    f"  [step] epoch={epoch}/{epochs} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"loss={loss.item():.6f} "
                    f"mse={loss_mse.item():.6f} "
                    f"div={loss_div.item():.6f} "
                    f"smooth={loss_smooth.item():.6f}"
                )

        scheduler.step()
        dt = time.perf_counter() - t0
        avg_loss = epoch_loss / max(step_count, 1)
        avg_mse = epoch_mse / max(step_count, 1)
        avg_div = epoch_div / max(step_count, 1)
        avg_smooth = epoch_smooth / max(step_count, 1)

        record = {
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "avg_mse": round(avg_mse, 6),
            "avg_div": round(avg_div, 6),
            "avg_smooth": round(avg_smooth, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "time_s": round(dt, 2),
        }
        history.append(record)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"loss={avg_loss:.6f} mse={avg_mse:.6f} div={avg_div:.6f} smooth={avg_smooth:.6f} "
            f"lr={record['lr']:.2e} time={dt:.1f}s"
        )

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = ckpt_dir / f"fusion_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, model_path)
            print(f"[best] saved {model_path} (loss={best_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early-stop] no improvement for {patience} epochs, stopping")
                break

    # Final ONNX export
    model.eval()
    dummy_level = torch.tensor([3], dtype=torch.long, device=device)
    dummy_noise = torch.randn(1, NOISE_DIM, device=device)
    torch.onnx.export(
        model, (dummy_level, dummy_noise), str(onnx_path),
        input_names=["stress_level", "noise"],
        output_names=["parameters"],
        dynamic_axes={"stress_level": {0: "batch"}, "noise": {0: "batch"}, "parameters": {0: "batch"}},
        opset_version=17,
    )
    print(f"[onnx] exported {onnx_path}")

    # Save history
    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[history] saved {history_path}")

    # Sample outputs for verification
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS (trained model)")
    print("=" * 80)
    model.eval()
    for lv in range(6):
        params = model.generate(lv, device=device)
        print(f"\nStress {lv}:")
        for k, v in params.items():
            print(f"  {k:25s} = {v:.4f}")

    print(f"\n[train] done. best_loss={best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Generator model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--samples-per-level", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--model-out", type=str, default="models/fusion_generator.pt")
    parser.add_argument("--onnx-out", type=str, default="models/fusion_generator.onnx")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_level=args.samples_per_level,
        checkpoint_dir=args.checkpoint_dir,
        model_out=args.model_out,
        onnx_out=args.onnx_out,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
