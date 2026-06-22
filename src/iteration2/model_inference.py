"""
Model inference wrapper for the trained Fusion Generator.

Loads a trained .pt checkpoint and runs real-time inference.
Falls back to random parameter generation if no model is available.
"""

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fusion_model import FusionGenerator, NOISE_DIM, OUTPUT_NAMES


@dataclass
class ModelParams:
    # Audio generation
    audio_intensity: float = 0.0
    audio_dissonance: float = 0.0
    audio_pitch_drift: float = 0.0
    audio_reverb_depth: float = 0.0
    audio_transient_rate: float = 0.0
    # Audio DSP
    dsp_pitch_shift: float = 0.0
    dsp_distortion: float = 0.0
    dsp_filter_freq: float = 0.0
    dsp_reverb_mix: float = 0.0
    dsp_layer_blend: float = 0.0
    dsp_pan: float = 0.0
    # Visual / texture
    visual_corruption: float = 0.0
    visual_fog_density: float = 0.0
    visual_light_temp: float = 0.0
    visual_flicker_rate: float = 0.0
    visual_grime_overlay: float = 0.0
    # Camera distortion
    camera_magnitude: float = 0.0
    camera_aberration: float = 0.0
    camera_noise: float = 0.0
    camera_vignette: float = 0.0
    camera_warp: float = 0.0
    # Camera feed manipulation
    camera_feed_face_distort: float = 0.0
    camera_feed_shadow: float = 0.0
    camera_feed_darken: float = 0.0
    camera_feed_figure: float = 0.0
    # Entity
    entity_probability: float = 0.0
    entity_opacity: float = 0.0
    entity_aggression: float = 0.0
    entity_morphology: float = 0.0
    entity_aura: float = 0.0

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in OUTPUT_NAMES}


class FusionModelInference:
    def __init__(self, model_path: Path, device_str: str = "auto"):
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        self.model = FusionGenerator().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.loaded = True
        print(f"[model] loaded fusion generator from {model_path} (device={self.device})")

    @torch.no_grad()
    def generate(self, stress_level: int, seed: Optional[int] = None) -> ModelParams:
        sl = torch.tensor([stress_level], dtype=torch.long, device=self.device)
        if seed is not None:
            gen = torch.Generator(device=self.device if self.device.type == "cuda" else "cpu")
            gen.manual_seed(seed)
            noise = torch.randn(1, NOISE_DIM, generator=gen, device=self.device)
        else:
            noise = torch.randn(1, NOISE_DIM, device=self.device)

        out = self.model(sl, noise).squeeze(0).cpu().tolist()
        params = dict(zip(OUTPUT_NAMES, out))
        return ModelParams(**params)

    @torch.no_grad()
    def inference_time_ms(self, stress_level: int = 3, n_runs: int = 100) -> float:
        import time
        sl = torch.tensor([stress_level], dtype=torch.long, device=self.device)
        noise = torch.randn(1, NOISE_DIM, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.model(sl, noise)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000 / n_runs
        return elapsed
