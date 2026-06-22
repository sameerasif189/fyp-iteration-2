import argparse
import json
import random
import time
from pathlib import Path


def _write_checkpoint(checkpoint_dir: Path, epoch: int, step: int, metrics: dict) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ckpt_epoch{epoch:02d}_step{step:05d}.json"
    checkpoint_path.write_text(json.dumps({"epoch": epoch, "step": step, "metrics": metrics}, indent=2), encoding="utf-8")
    return checkpoint_path


def run_training(
    epochs: int,
    steps_per_epoch: int,
    checkpoint_every: int,
    checkpoint_dir: Path,
    model_config_out: Path,
) -> None:
    rng = random.Random(395)
    global_step = 0
    best_loss = 10_000.0
    model_cfg = {
        "audio_scale": 1.0,
        "visual_scale": 1.0,
        "camera_scale": 1.0,
        "entity_scale": 1.0,
    }

    print("[train] starting training loop")
    print(f"[train] epochs={epochs}, steps_per_epoch={steps_per_epoch}, checkpoint_every={checkpoint_every}")
    print(f"[train] checkpoints={checkpoint_dir}")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step in range(1, steps_per_epoch + 1):
            global_step += 1
            time.sleep(0.03)
            synthetic_loss = max(0.05, 2.0 / (1.0 + 0.03 * global_step) + rng.uniform(-0.03, 0.03))
            synthetic_quality = min(1.0, 0.2 + 0.8 * (1.0 - synthetic_loss / 2.0))
            running_loss += synthetic_loss

            if step % 10 == 0 or step == 1:
                print(
                    "[train] "
                    f"epoch={epoch}/{epochs} step={step}/{steps_per_epoch} "
                    f"global_step={global_step} loss={synthetic_loss:.4f} quality={synthetic_quality:.4f}"
                )

            if global_step % checkpoint_every == 0:
                metrics = {
                    "loss": round(synthetic_loss, 6),
                    "quality": round(synthetic_quality, 6),
                }
                ckpt = _write_checkpoint(checkpoint_dir, epoch, global_step, metrics)
                print(f"[ckpt] saved {ckpt}")

                # Simulated model scaling update, exported for demo inference.
                model_cfg["audio_scale"] = round(0.9 + 0.2 * synthetic_quality, 4)
                model_cfg["visual_scale"] = round(0.85 + 0.25 * synthetic_quality, 4)
                model_cfg["camera_scale"] = round(0.8 + 0.3 * synthetic_quality, 4)
                model_cfg["entity_scale"] = round(0.75 + 0.35 * synthetic_quality, 4)
                model_config_out.parent.mkdir(parents=True, exist_ok=True)
                model_config_out.write_text(json.dumps(model_cfg, indent=2), encoding="utf-8")
                print(f"[model] exported config {model_config_out}")

                if synthetic_loss < best_loss:
                    best_loss = synthetic_loss
                    best_path = checkpoint_dir / "best_checkpoint.json"
                    best_path.write_text(
                        json.dumps({"epoch": epoch, "step": global_step, "loss": best_loss}, indent=2),
                        encoding="utf-8",
                    )
                    print(f"[best] updated {best_path}")

        avg_loss = running_loss / max(1, steps_per_epoch)
        print(f"[epoch] epoch={epoch} avg_loss={avg_loss:.4f}")

    print("[train] completed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 2 model training simulator with checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--model-config-out", type=str, default="models/model_config.json")
    args = parser.parse_args()
    run_training(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_every=max(1, args.checkpoint_every),
        checkpoint_dir=Path(args.checkpoint_dir),
        model_config_out=Path(args.model_config_out),
    )


if __name__ == "__main__":
    main()

