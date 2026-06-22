import argparse
import json
import time
from pathlib import Path
from typing import List

from .fusion_orchestrator import FusionOrchestrator
from .performance import PerfMonitor


def scripted_level(elapsed_s: float) -> int:
    if elapsed_s < 30:
        return 1 if elapsed_s > 12 else 0
    if elapsed_s < 120:
        if elapsed_s < 60:
            return 2
        if elapsed_s < 95:
            return 3
        return 4
    if elapsed_s < 160:
        return 5
    return 2


def run_auto(
    duration_s: int,
    tick_hz: int,
    catalog_path: Path,
    model_config_path: Path | None = None,
    progress_interval_s: float = 1.0,
) -> List[dict]:
    orchestrator = FusionOrchestrator(catalog_path=catalog_path, model_config_path=model_config_path)
    perf = PerfMonitor()
    dt = 1.0 / max(1, tick_hz)
    frames = []
    start = time.perf_counter()
    last_progress_print_s = -progress_interval_s
    last_assets = None

    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= duration_s:
            break
        requested = scripted_level(elapsed)

        perf.begin()
        out = orchestrator.step(requested_level=requested, now_s=elapsed, mode="auto")
        perf_metrics = perf.end()

        payload = out.to_dict()
        payload["perf"] = perf_metrics
        payload["fallback_reason"] = perf.state.fallback_reason
        frames.append(payload)
        if elapsed - last_progress_print_s >= progress_interval_s:
            last_progress_print_s = elapsed
            current_assets = payload["generated_assets"]
            changed = current_assets != last_assets
            change_marker = "CHANGED" if changed else "steady "
            print(
                "[demo-live] "
                f"t={elapsed:6.2f}s "
                f"stress={payload['stress_event']['level']} "
                f"[{change_marker}] "
                f"audio={payload['generated_assets']['audio_asset']} "
                f"visual={payload['generated_assets']['visual_asset']} "
                f"camera={payload['generated_assets']['camera_asset']} "
                f"entity={payload['generated_assets']['entity_asset']} "
                f"adaptive_ms={payload['perf']['adaptive_ms']:.3f} "
                f"quality_tier={int(payload['perf']['quality_tier'])}"
            )
            last_assets = current_assets
        time.sleep(dt)
    return frames


def run_manual(catalog_path: Path, model_config_path: Path | None = None) -> None:
    orchestrator = FusionOrchestrator(catalog_path=catalog_path, model_config_path=model_config_path)
    perf = PerfMonitor()
    start = time.perf_counter()
    print("Manual mode started. Enter stress levels 0..5 (or 'q' to quit).")
    while True:
        raw = input("stress> ").strip().lower()
        if raw == "q":
            break
        if raw not in {"0", "1", "2", "3", "4", "5"}:
            print("Invalid input. Use 0..5.")
            continue
        level = int(raw)
        elapsed = time.perf_counter() - start
        perf.begin()
        out = orchestrator.step(requested_level=level, now_s=elapsed, mode="manual")
        perf_metrics = perf.end()
        payload = out.to_dict()
        payload["perf"] = perf_metrics
        payload["fallback_reason"] = perf.state.fallback_reason
        print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 2 controlled demo runner")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto")
    parser.add_argument("--duration", type=int, default=200)
    parser.add_argument("--tick-hz", type=int, default=5)
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "assets" / "catalog.json"),
    )
    parser.add_argument("--output", type=str, default="demo_output.json")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--progress-interval", type=float, default=1.0)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    model_config_path = Path(args.model_config) if args.model_config else None
    if args.mode == "manual":
        run_manual(catalog_path, model_config_path=model_config_path)
        return

    frames = run_auto(
        duration_s=args.duration,
        tick_hz=args.tick_hz,
        catalog_path=catalog_path,
        model_config_path=model_config_path,
        progress_interval_s=max(0.2, args.progress_interval),
    )
    out_path = Path(args.output)
    out_path.write_text(json.dumps(frames, indent=2), encoding="utf-8")
    print(f"Wrote {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()

