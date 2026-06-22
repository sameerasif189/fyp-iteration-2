"""
Conditional Fusion Generator Model

Takes stress level (0-5) + noise vector as input and generates all
output parameters across audio, visual, camera, camera-feed, and entity domains.

Architecture:
  - Stress embedding + noise -> encoder -> bottleneck -> decoder heads
  - Each domain has its own decoder head for specialised output
  - Trained with range-matching loss + diversity loss + smoothness reg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_SPEC = {
    # Audio generation params (5)
    "audio_intensity": (0.0, 1.0),
    "audio_dissonance": (0.0, 1.0),
    "audio_pitch_drift": (-1.0, 1.0),
    "audio_reverb_depth": (0.0, 1.0),
    "audio_transient_rate": (0.0, 1.0),
    # Audio DSP params (6)
    "dsp_pitch_shift": (0.0, 1.0),
    "dsp_distortion": (0.0, 1.0),
    "dsp_filter_freq": (0.0, 1.0),
    "dsp_reverb_mix": (0.0, 1.0),
    "dsp_layer_blend": (0.0, 1.0),
    "dsp_pan": (0.0, 1.0),
    # Visual / texture params (5)
    "visual_corruption": (0.0, 1.0),
    "visual_fog_density": (0.0, 1.0),
    "visual_light_temp": (0.0, 1.0),
    "visual_flicker_rate": (0.0, 1.0),
    "visual_grime_overlay": (0.0, 1.0),
    # Camera distortion params (5)
    "camera_magnitude": (0.0, 1.0),
    "camera_aberration": (0.0, 1.0),
    "camera_noise": (0.0, 1.0),
    "camera_vignette": (0.0, 1.0),
    "camera_warp": (0.0, 1.0),
    # Camera feed manipulation params (4)
    "camera_feed_face_distort": (0.0, 1.0),
    "camera_feed_shadow": (0.0, 1.0),
    "camera_feed_darken": (0.0, 1.0),
    "camera_feed_figure": (0.0, 1.0),
    # Entity params (5)
    "entity_probability": (0.0, 1.0),
    "entity_opacity": (0.0, 1.0),
    "entity_aggression": (0.0, 1.0),
    "entity_morphology": (0.0, 1.0),
    "entity_aura": (0.0, 1.0),
}

NUM_OUTPUTS = len(OUTPUT_SPEC)
OUTPUT_NAMES = list(OUTPUT_SPEC.keys())
NOISE_DIM = 32
NUM_STRESS_LEVELS = 6


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class DomainHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionGenerator(nn.Module):
    """
    Conditional generator: stress_level + noise -> horror asset parameters.
    Outputs 30 parameters across all domains. ~500K params, <1ms inference on GPU.
    """

    def __init__(self, noise_dim: int = NOISE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.noise_dim = noise_dim
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, 64)

        self.encoder = nn.Sequential(
            nn.Linear(64 + noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        self.audio_head = DomainHead(hidden_dim, 5)
        self.dsp_head = DomainHead(hidden_dim, 6)
        self.visual_head = DomainHead(hidden_dim, 5)
        self.camera_head = DomainHead(hidden_dim, 5)
        self.camera_feed_head = DomainHead(hidden_dim, 4)
        self.entity_head = DomainHead(hidden_dim, 5)

    def forward(self, stress_level: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        emb = self.stress_embed(stress_level)
        x = torch.cat([emb, noise], dim=-1)
        h = self.encoder(x)

        audio = self.audio_head(h)
        dsp = self.dsp_head(h)
        visual = self.visual_head(h)
        camera = self.camera_head(h)
        camera_feed = self.camera_feed_head(h)
        entity = self.entity_head(h)

        return torch.cat([audio, dsp, visual, camera, camera_feed, entity], dim=-1)

    def generate(self, stress_level: int, device: torch.device = None,
                 noise: torch.Tensor = None) -> dict:
        if device is None:
            device = next(self.parameters()).device
        sl = torch.tensor([stress_level], dtype=torch.long, device=device)
        if noise is None:
            noise = torch.randn(1, self.noise_dim, device=device)
        with torch.no_grad():
            out = self.forward(sl, noise).squeeze(0).cpu().tolist()
        return dict(zip(OUTPUT_NAMES, out))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
