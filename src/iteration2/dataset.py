"""
Synthetic training dataset for the Fusion Generator.

Generates (stress_level, noise, target_params) triples based on the
stress profiles with controlled randomness for realistic variation.
"""

import random
import math
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .stress_schema import STRESS_PROFILES
from .fusion_model import OUTPUT_NAMES, NOISE_DIM, NUM_STRESS_LEVELS


STRESS_TARGET_RANGES = {
    0: {
        "audio_intensity": (0.05, 0.15), "audio_dissonance": (0.0, 0.10),
        "audio_pitch_drift": (-0.1, 0.1), "audio_reverb_depth": (0.6, 0.8),
        "audio_transient_rate": (0.0, 0.05),
        "dsp_pitch_shift": (0.45, 0.55), "dsp_distortion": (0.0, 0.05),
        "dsp_filter_freq": (0.8, 1.0), "dsp_reverb_mix": (0.6, 0.8),
        "dsp_layer_blend": (0.0, 0.1), "dsp_pan": (0.45, 0.55),
        "visual_corruption": (0.0, 0.05), "visual_fog_density": (0.03, 0.08),
        "visual_light_temp": (0.7, 0.9), "visual_flicker_rate": (0.0, 0.02),
        "visual_grime_overlay": (0.0, 0.03),
        "camera_magnitude": (0.0, 0.05), "camera_aberration": (0.0, 0.02),
        "camera_noise": (0.0, 0.03), "camera_vignette": (0.0, 0.08),
        "camera_warp": (0.0, 0.01),
        "camera_feed_face_distort": (0.3, 0.6), "camera_feed_shadow": (0.1, 0.3),
        "camera_feed_darken": (0.2, 0.5), "camera_feed_figure": (0.05, 0.15),
        "entity_probability": (0.0, 0.0), "entity_opacity": (0.0, 0.0),
        "entity_aggression": (0.0, 0.0), "entity_morphology": (0.0, 0.0),
        "entity_aura": (0.0, 0.0),
    },
    1: {
        "audio_intensity": (0.15, 0.30), "audio_dissonance": (0.05, 0.20),
        "audio_pitch_drift": (-0.15, 0.15), "audio_reverb_depth": (0.5, 0.75),
        "audio_transient_rate": (0.02, 0.10),
        "dsp_pitch_shift": (0.40, 0.55), "dsp_distortion": (0.03, 0.12),
        "dsp_filter_freq": (0.65, 0.85), "dsp_reverb_mix": (0.5, 0.7),
        "dsp_layer_blend": (0.05, 0.2), "dsp_pan": (0.4, 0.6),
        "visual_corruption": (0.04, 0.10), "visual_fog_density": (0.07, 0.14),
        "visual_light_temp": (0.55, 0.75), "visual_flicker_rate": (0.01, 0.06),
        "visual_grime_overlay": (0.03, 0.08),
        "camera_magnitude": (0.03, 0.08), "camera_aberration": (0.01, 0.05),
        "camera_noise": (0.02, 0.06), "camera_vignette": (0.05, 0.15),
        "camera_warp": (0.0, 0.03),
        "camera_feed_face_distort": (0.2, 0.5), "camera_feed_shadow": (0.08, 0.25),
        "camera_feed_darken": (0.15, 0.4), "camera_feed_figure": (0.03, 0.10),
        "entity_probability": (0.0, 0.0), "entity_opacity": (0.0, 0.0),
        "entity_aggression": (0.0, 0.0), "entity_morphology": (0.0, 0.0),
        "entity_aura": (0.0, 0.0),
    },
    2: {
        "audio_intensity": (0.30, 0.48), "audio_dissonance": (0.15, 0.35),
        "audio_pitch_drift": (-0.25, 0.25), "audio_reverb_depth": (0.4, 0.65),
        "audio_transient_rate": (0.08, 0.20),
        "dsp_pitch_shift": (0.35, 0.55), "dsp_distortion": (0.10, 0.25),
        "dsp_filter_freq": (0.50, 0.75), "dsp_reverb_mix": (0.4, 0.6),
        "dsp_layer_blend": (0.15, 0.35), "dsp_pan": (0.35, 0.65),
        "visual_corruption": (0.10, 0.22), "visual_fog_density": (0.12, 0.22),
        "visual_light_temp": (0.40, 0.60), "visual_flicker_rate": (0.05, 0.15),
        "visual_grime_overlay": (0.08, 0.18),
        "camera_magnitude": (0.08, 0.18), "camera_aberration": (0.04, 0.12),
        "camera_noise": (0.05, 0.14), "camera_vignette": (0.12, 0.25),
        "camera_warp": (0.02, 0.08),
        "camera_feed_face_distort": (0.1, 0.35), "camera_feed_shadow": (0.05, 0.15),
        "camera_feed_darken": (0.1, 0.25), "camera_feed_figure": (0.02, 0.08),
        "entity_probability": (0.0, 0.05), "entity_opacity": (0.0, 0.05),
        "entity_aggression": (0.0, 0.02), "entity_morphology": (0.0, 0.03),
        "entity_aura": (0.0, 0.02),
    },
    3: {
        "audio_intensity": (0.45, 0.65), "audio_dissonance": (0.30, 0.55),
        "audio_pitch_drift": (-0.4, 0.4), "audio_reverb_depth": (0.3, 0.55),
        "audio_transient_rate": (0.15, 0.35),
        "dsp_pitch_shift": (0.25, 0.50), "dsp_distortion": (0.20, 0.45),
        "dsp_filter_freq": (0.35, 0.60), "dsp_reverb_mix": (0.3, 0.55),
        "dsp_layer_blend": (0.30, 0.55), "dsp_pan": (0.25, 0.75),
        "visual_corruption": (0.20, 0.40), "visual_fog_density": (0.18, 0.34),
        "visual_light_temp": (0.25, 0.45), "visual_flicker_rate": (0.12, 0.28),
        "visual_grime_overlay": (0.18, 0.35),
        "camera_magnitude": (0.15, 0.28), "camera_aberration": (0.10, 0.22),
        "camera_noise": (0.12, 0.25), "camera_vignette": (0.20, 0.38),
        "camera_warp": (0.06, 0.16),
        "camera_feed_face_distort": (0.02, 0.15), "camera_feed_shadow": (0.02, 0.08),
        "camera_feed_darken": (0.03, 0.12), "camera_feed_figure": (0.01, 0.04),
        "entity_probability": (0.05, 0.12), "entity_opacity": (0.05, 0.15),
        "entity_aggression": (0.02, 0.10), "entity_morphology": (0.05, 0.15),
        "entity_aura": (0.02, 0.08),
    },
    4: {
        "audio_intensity": (0.62, 0.82), "audio_dissonance": (0.50, 0.78),
        "audio_pitch_drift": (-0.6, 0.6), "audio_reverb_depth": (0.2, 0.45),
        "audio_transient_rate": (0.30, 0.55),
        "dsp_pitch_shift": (0.15, 0.40), "dsp_distortion": (0.35, 0.65),
        "dsp_filter_freq": (0.20, 0.45), "dsp_reverb_mix": (0.2, 0.45),
        "dsp_layer_blend": (0.50, 0.75), "dsp_pan": (0.15, 0.85),
        "visual_corruption": (0.38, 0.66), "visual_fog_density": (0.30, 0.50),
        "visual_light_temp": (0.12, 0.30), "visual_flicker_rate": (0.25, 0.50),
        "visual_grime_overlay": (0.35, 0.60),
        "camera_magnitude": (0.25, 0.45), "camera_aberration": (0.20, 0.40),
        "camera_noise": (0.22, 0.42), "camera_vignette": (0.35, 0.55),
        "camera_warp": (0.14, 0.30),
        "camera_feed_face_distort": (0.0, 0.05), "camera_feed_shadow": (0.0, 0.03),
        "camera_feed_darken": (0.0, 0.05), "camera_feed_figure": (0.0, 0.02),
        "entity_probability": (0.20, 0.40), "entity_opacity": (0.30, 0.60),
        "entity_aggression": (0.15, 0.35), "entity_morphology": (0.20, 0.45),
        "entity_aura": (0.15, 0.35),
    },
    5: {
        "audio_intensity": (0.78, 1.0), "audio_dissonance": (0.72, 1.0),
        "audio_pitch_drift": (-0.8, 0.8), "audio_reverb_depth": (0.1, 0.35),
        "audio_transient_rate": (0.50, 0.85),
        "dsp_pitch_shift": (0.05, 0.30), "dsp_distortion": (0.55, 0.90),
        "dsp_filter_freq": (0.10, 0.30), "dsp_reverb_mix": (0.1, 0.35),
        "dsp_layer_blend": (0.70, 1.0), "dsp_pan": (0.05, 0.95),
        "visual_corruption": (0.62, 0.95), "visual_fog_density": (0.46, 0.75),
        "visual_light_temp": (0.0, 0.18), "visual_flicker_rate": (0.40, 0.80),
        "visual_grime_overlay": (0.55, 0.90),
        "camera_magnitude": (0.40, 0.72), "camera_aberration": (0.35, 0.65),
        "camera_noise": (0.38, 0.70), "camera_vignette": (0.50, 0.80),
        "camera_warp": (0.28, 0.55),
        "camera_feed_face_distort": (0.0, 0.02), "camera_feed_shadow": (0.0, 0.01),
        "camera_feed_darken": (0.0, 0.02), "camera_feed_figure": (0.0, 0.01),
        "entity_probability": (0.40, 0.75), "entity_opacity": (0.55, 0.90),
        "entity_aggression": (0.35, 0.70), "entity_morphology": (0.40, 0.70),
        "entity_aura": (0.35, 0.65),
    },
}


def _sample_target(level: int, rng: random.Random) -> List[float]:
    ranges = STRESS_TARGET_RANGES[level]
    target = []
    for name in OUTPUT_NAMES:
        lo, hi = ranges[name]
        if name == "audio_pitch_drift":
            val = lo + (hi - lo) * rng.random()
            target.append((val + 1.0) / 2.0)
        else:
            target.append(lo + (hi - lo) * rng.random())
    return target


class FusionDataset(Dataset):
    """Generates samples on-the-fly for memory efficiency."""

    def __init__(self, samples_per_level: int = 5000, seed: int = 42):
        super().__init__()
        self.samples_per_level = samples_per_level
        self.total = samples_per_level * NUM_STRESS_LEVELS
        self.seed = seed
        self._cache: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        self._build()

    def _build(self):
        rng = random.Random(self.seed)
        for level in range(NUM_STRESS_LEVELS):
            for _ in range(self.samples_per_level):
                noise = torch.randn(NOISE_DIM)
                target = _sample_target(level, rng)
                self._cache.append((level, noise, torch.tensor(target, dtype=torch.float32)))

        indices = list(range(len(self._cache)))
        rng.shuffle(indices)
        self._cache = [self._cache[i] for i in indices]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int):
        level, noise, target = self._cache[idx]
        return torch.tensor(level, dtype=torch.long), noise, target
