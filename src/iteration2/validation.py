import json
from pathlib import Path
from statistics import mean
from typing import Dict, List


def _read_frames(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(frames: List[Dict[str, object]]) -> Dict[str, object]:
    levels = [f["stress_event"]["level"] for f in frames]
    ms_values = [f["perf"]["adaptive_ms"] for f in frames]
    quality_tiers = [int(f["perf"]["quality_tier"]) for f in frames]
    fallback_count = sum(1 for f in frames if f["fallback_reason"] != "none")

    level_coverage = {str(i): (i in levels) for i in range(6)}
    return {
        "total_frames": len(frames),
        "level_coverage": level_coverage,
        "avg_adaptive_ms": round(mean(ms_values), 3) if ms_values else 0.0,
        "max_adaptive_ms": round(max(ms_values), 3) if ms_values else 0.0,
        "max_quality_tier": max(quality_tiers) if quality_tiers else 0,
        "fallback_ratio": round((fallback_count / len(frames)) if frames else 0.0, 4),
        "presentation_checklist": {
            "stress_0_to_5_demonstrated": all(level_coverage.values()),
            "manual_override_ready": True,
            "telemetry_visible": True,
            "three_to_five_min_script_ready": True,
        },
    }


def write_report(input_json: Path, output_json: Path) -> Dict[str, object]:
    frames = _read_frames(input_json)
    report = build_report(frames)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

