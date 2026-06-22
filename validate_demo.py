import argparse
import json
from pathlib import Path

from src.iteration2.demo_runner import run_auto
from src.iteration2.validation import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Iteration 2 demo outputs")
    parser.add_argument("--duration", type=int, default=200)
    parser.add_argument("--tick-hz", type=int, default=5)
    parser.add_argument("--input", type=str, default="demo_output.json")
    parser.add_argument("--output", type=str, default="validation_report.json")
    parser.add_argument(
        "--catalog",
        type=str,
        default=str(Path(__file__).resolve().parent / "assets" / "catalog.json"),
    )
    parser.add_argument("--model-config", type=str, default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        frames = run_auto(
            duration_s=args.duration,
            tick_hz=args.tick_hz,
            catalog_path=Path(args.catalog),
            model_config_path=Path(args.model_config) if args.model_config else None,
        )
        input_path.write_text(json.dumps(frames, indent=2), encoding="utf-8")

    report = write_report(input_json=input_path, output_json=Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

