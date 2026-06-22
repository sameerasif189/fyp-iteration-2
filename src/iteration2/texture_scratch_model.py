"""
Texture-From-Scratch Generator.

Conditional DCGAN that generates entire 128x128 textures from just
a stress level + noise vector. No input image needed -- the model
learns to create horror-appropriate textures from training data.

Architecture:
  - Generator: stress_embed + noise -> 4x4 -> upsample to 128x128
  - Discriminator: 128x128 + stress_embed -> real/fake
  - ~5M params total, <8ms inference on GPU
"""

import torch
import torch.nn as nn


NOISE_DIM = 128
IMG_SIZE = 128
IMG_CHANNELS = 3
NUM_STRESS_LEVELS = 6


class TextureScratchGenerator(nn.Module):
    """Generates 128x128 RGB textures from stress level + noise."""

    def __init__(self, noise_dim: int = NOISE_DIM, stress_embed_dim: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)

        input_dim = noise_dim + stress_embed_dim

        self.project = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.LeakyReLU(0.2),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, IMG_CHANNELS, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        emb = self.stress_embed(stress_level)
        x = torch.cat([noise, emb], dim=-1)
        x = self.project(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.upsample(x)

    @torch.no_grad()
    def generate(self, stress_level: int, device: torch.device = None,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if noise is None:
            noise = torch.randn(1, self.noise_dim, device=device)
        sl = torch.tensor([stress_level], dtype=torch.long, device=device)
        img = self.forward(noise, sl)
        return img.squeeze(0)


class TextureScratchDiscriminator(nn.Module):
    """PatchGAN discriminator for generated textures."""

    def __init__(self, stress_embed_dim: int = 32):
        super().__init__()
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)

        self.convs = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256 + stress_embed_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        x = self.convs(img)
        x = self.pool(x).view(x.size(0), -1)
        emb = self.stress_embed(stress_level)
        x = torch.cat([x, emb], dim=-1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
