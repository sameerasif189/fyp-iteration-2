import random
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

from .stress_schema import STRESS_PROFILES


def _lerp_range(bounds: Tuple[float, float], rng: random.Random) -> float:
    lo, hi = bounds
    return lo + (hi - lo) * rng.random()


@dataclass(frozen=True)
class RealtimeParams:
    audio_intensity: float
    corruption_alpha: float
    fog_density: float
    camera_magnitude: float
    entity_weight: float
    profile_name: str


class RealtimeGenerator:
    def __init__(
        self,
        global_seed: int = 395,
        model_config_path: Path | None = None,
    ):
        self.global_seed = global_seed
        self.audio_scale = 1.0
        self.visual_scale = 1.0
        self.camera_scale = 1.0
        self.entity_scale = 1.0
        if model_config_path is not None and model_config_path.exists():
            payload = json.loads(model_config_path.read_text(encoding="utf-8"))
            self.audio_scale = float(payload.get("audio_scale", 1.0))
            self.visual_scale = float(payload.get("visual_scale", 1.0))
            self.camera_scale = float(payload.get("camera_scale", 1.0))
            self.entity_scale = float(payload.get("entity_scale", 1.0))

    def generate(self, level: int, tick: int) -> RealtimeParams:
        profile = STRESS_PROFILES[level]
        rng = random.Random(self.global_seed + (level * 997) + tick)
        return RealtimeParams(
            audio_intensity=min(1.0, _lerp_range(profile.audio_intensity, rng) * self.audio_scale),
            corruption_alpha=min(1.0, _lerp_range(profile.corruption_alpha, rng) * self.visual_scale),
            fog_density=min(1.0, _lerp_range(profile.fog_density, rng) * self.visual_scale),
            camera_magnitude=min(1.0, _lerp_range(profile.camera_magnitude, rng) * self.camera_scale),
            entity_weight=min(1.0, profile.entity_probability * (0.5 + 0.5 * rng.random()) * self.entity_scale),
            profile_name=profile.name.lower(),
        )

