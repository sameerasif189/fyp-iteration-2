"""
Training script for the WaveGAN Audio Generator.

Trains a conditional GAN on wildlife vocalizations + horror audio
to generate unique stress-conditioned horror audio in real-time.

Usage:
  python train_audio_gen.py --epochs 50 --batch-size 32
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

from src.iteration2.audio_gen_model import (
    AudioGenerator, AudioDiscriminator, NOISE_DIM, AUDIO_LENGTH,
    SAMPLE_RATE, count_params
)
from src.iteration2.audio_dataset import AudioTrainingDataset, SyntheticAudioDataset


def gradient_penalty(discriminator, real_audio, fake_audio, stress_levels, device):
    """WGAN-GP gradient penalty."""
    batch_size = real_audio.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = (alpha * real_audio + (1 - alpha) * fake_audio).requires_grad_(True)

    d_out = discriminator(interpolated, stress_levels)
    grad = torch.autograd.grad(
        outputs=d_out, inputs=interpolated,
        grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True
    )[0]

    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


def spectral_loss(real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
    """Multi-scale spectral loss for audio quality."""
    loss = torch.tensor(0.0, device=real_audio.device)

    for n_fft in [512, 1024, 2048]:
        real_spec = torch.stft(real_audio.squeeze(1), n_fft=n_fft,
                               return_complex=True, hop_length=n_fft // 4,
                               window=torch.hann_window(n_fft, device=real_audio.device))
        fake_spec = torch.stft(fake_audio.squeeze(1), n_fft=n_fft,
                               return_complex=True, hop_length=n_fft // 4,
                               window=torch.hann_window(n_fft, device=fake_audio.device))

        real_mag = real_spec.abs()
        fake_mag = fake_spec.abs()

        loss = loss + (real_mag - fake_mag).abs().mean()
        loss = loss + (torch.log(real_mag + 1e-7) - torch.log(fake_mag + 1e-7)).abs().mean()

    return loss / 3.0


def temporal_loss(real_audio: torch.Tensor, fake_audio: torch.Tensor) -> torch.Tensor:
    """Time-domain consistency loss to reduce buzzy artifacts."""
    l1 = torch.nn.functional.l1_loss(fake_audio, real_audio)
    # Compare smoothed envelopes
    pool = nn.AvgPool1d(kernel_size=256, stride=64, padding=128)
    env_real = pool(real_audio.abs())
    env_fake = pool(fake_audio.abs())
    env_l1 = torch.nn.functional.l1_loss(env_fake, env_real)
    return l1 + env_l1


def train(
    epochs: int = 50,
    batch_size: int = 64,
    lr_g: float = 1e-4,
    lr_d: float = 4e-4,
    n_critic: int = 5,
    gp_weight: float = 10.0,
    spectral_weight: float = 8.0,
    temporal_weight: float = 4.0,
    assets_dir: str = "assets",
    checkpoint_dir: str = "checkpoints/audio_gen",
    model_out: str = "models/audio_generator.pt",
    onnx_out: str = "models/audio_generator.onnx",
    device_str: str = "auto",
    use_synthetic: bool = False,
    min_delta: float = 8e-4,
):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type == "cuda":
        # Saturate the 5060 (Blackwell) properly: TF32 matmul + tuned cuDNN kernels.
        # WaveGAN is tiny so this mostly burns through launch overhead on 1D convs.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            free_b, total_b = torch.cuda.mem_get_info(device)
            gpu_name = torch.cuda.get_device_name(device)
            print(
                f"[audio-train] device={device} ({gpu_name}) "
                f"vram_free={free_b/1024**3:.2f} GiB / {total_b/1024**3:.2f} GiB "
                f"tf32=on cudnn.benchmark=on"
            )
        except Exception:
            print(f"[audio-train] device={device} tf32=on cudnn.benchmark=on")
    else:
        print(f"[audio-train] device={device}")

    assets_path = Path(assets_dir)
    if use_synthetic or not (assets_path / "audio").exists():
        print("[audio-train] using synthetic dataset (no real audio found)")
        dataset = SyntheticAudioDataset(samples_per_level=500)
    else:
        print(f"[audio-train] loading real audio from {assets_path}")
        dataset = AudioTrainingDataset(assets_path)
    print(f"[audio-train] dataset prepared: {len(dataset)} samples")

    if len(dataset) == 0:
        print("[audio-train] ERROR: no training data available")
        return

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    print(f"[audio-train] dataloader batches per epoch: {len(loader)} batch_size={batch_size}")

    generator = AudioGenerator().to(device)
    discriminator = AudioDiscriminator().to(device)

    print(f"[audio-train] generator params: {count_params(generator):,}")
    print(f"[audio-train] discriminator params: {count_params(discriminator):,}")

    opt_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    history = []
    best_g_loss = float("inf")
    patience_counter = 0
    patience = 10

    print(f"[audio-train] starting: {epochs} epochs, {len(dataset)} samples, batch={batch_size}")
    print(f"[audio-train] early stopping: patience={patience}, min_delta={min_delta}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_spec_loss = 0.0
        d_steps = 0
        g_steps = 0

        progress = tqdm(loader, desc=f"audio epoch {epoch}/{epochs}", leave=False)
        for batch_idx, (real_audio, stress_levels) in enumerate(progress):
            real_audio = real_audio.to(device)
            stress_levels = stress_levels.to(device)
            bs = real_audio.size(0)

            # --- Train Discriminator ---
            noise = torch.randn(bs, NOISE_DIM, device=device)
            fake_audio = generator(noise, stress_levels).detach()

            d_real = discriminator(real_audio, stress_levels).mean()
            d_fake = discriminator(fake_audio, stress_levels).mean()
            gp = gradient_penalty(discriminator, real_audio, fake_audio, stress_levels, device)

            d_loss = d_fake - d_real + gp_weight * gp

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            epoch_d_loss += d_loss.item()
            d_steps += 1

            # --- Train Generator every n_critic steps ---
            if (batch_idx + 1) % n_critic == 0:
                noise = torch.randn(bs, NOISE_DIM, device=device)
                fake_audio = generator(noise, stress_levels)

                g_adv = -discriminator(fake_audio, stress_levels).mean()
                g_spec = spectral_loss(real_audio, fake_audio) * spectral_weight
                g_temp = temporal_loss(real_audio, fake_audio) * temporal_weight
                g_loss = g_adv + g_spec + g_temp

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                epoch_g_loss += g_loss.item()
                epoch_spec_loss += (g_spec.item() + g_temp.item())
                g_steps += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                progress.set_postfix({
                    "D": f"{d_loss.item():.4f}",
                    "G": f"{(g_loss.item() if (batch_idx + 1) % n_critic == 0 else float('nan')):.4f}",
                    "spec+tmp": f"{((g_spec.item() + g_temp.item()) if (batch_idx + 1) % n_critic == 0 else float('nan')):.3f}",
                })
            if (batch_idx + 1) % 25 == 0:
                g_display = g_loss.item() if (batch_idx + 1) % n_critic == 0 else float("nan")
                st_display = (g_spec.item() + g_temp.item()) if (batch_idx + 1) % n_critic == 0 else float("nan")
                print(
                    f"[step audio] epoch={epoch}/{epochs} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"D={d_loss.item():.4f} G={g_display:.4f} ST={st_display:.4f}"
                )

        dt = time.perf_counter() - t0
        avg_d = epoch_d_loss / max(d_steps, 1)
        avg_g = epoch_g_loss / max(g_steps, 1)
        avg_spec = epoch_spec_loss / max(g_steps, 1)

        record = {
            "epoch": epoch,
            "d_loss": round(avg_d, 6),
            "g_loss": round(avg_g, 6),
            "spectral_loss": round(avg_spec, 6),
            "time_s": round(dt, 2),
        }
        history.append(record)

        print(
            f"[epoch {epoch}/{epochs}] "
            f"D={avg_d:.4f} G={avg_g:.4f} spec={avg_spec:.4f} "
            f"time={dt:.1f}s"
        )

        if epoch % 10 == 0:
            ckpt_path = ckpt_dir / f"audio_gen_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "generator_state": generator.state_dict(),
                "discriminator_state": discriminator.state_dict(),
                "opt_g_state": opt_g.state_dict(),
                "opt_d_state": opt_d.state_dict(),
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

        if g_steps > 0 and avg_g < (best_g_loss - min_delta):
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
    dummy_noise = torch.randn(1, NOISE_DIM, device=device)
    dummy_level = torch.tensor([3], dtype=torch.long, device=device)
    onnx_path = Path(onnx_out)
    torch.onnx.export(
        generator, (dummy_noise, dummy_level), str(onnx_path),
        input_names=["noise", "stress_level"],
        output_names=["audio"],
        dynamic_axes={"noise": {0: "batch"}, "stress_level": {0: "batch"}, "audio": {0: "batch"}},
        opset_version=18, dynamo=False,
    )
    print(f"[onnx] exported {onnx_path}")

    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"[done] audio generator training complete. best_g_loss={best_g_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Audio Generator (WaveGAN)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Default 64 fits comfortably in 8 GB (RTX 5060) for the 2.5M-param WaveGAN; "
             "lower if you OOM, raise to 96/128 to keep the GPU pegged."
    )
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/audio_gen")
    parser.add_argument("--model-out", type=str, default="models/audio_generator.pt")
    parser.add_argument("--onnx-out", type=str, default="models/audio_generator.onnx")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    parser.add_argument("--spectral-weight", type=float, default=8.0)
    parser.add_argument("--temporal-weight", type=float, default=4.0)
    parser.add_argument("--min-delta", type=float, default=8e-4)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        assets_dir=args.assets_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_out=args.model_out,
        onnx_out=args.onnx_out,
        device_str=args.device,
        use_synthetic=args.synthetic,
        spectral_weight=args.spectral_weight,
        temporal_weight=args.temporal_weight,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
