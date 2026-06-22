"""
Conditional U-Net for Texture Corruption/Horror Overlay Generation.

Architecture:
  - Encoder: clean 256x256 RGB texture patch downsampled via strided convs
  - Conditioning: stress level embedding injected at bottleneck
  - Decoder: upsampled back to 256x256 with skip connections
  - Output: corrupted/horror variant (blood, grime, cracks, decay)
  - ~3M params, <5ms inference at 256x256 on GPU

Trained on paired data (clean -> corrupted via procedural generation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


IMG_SIZE = 256
NUM_STRESS_LEVELS = 6


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
        else:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TextureCorruptionUNet(nn.Module):
    """
    Conditional U-Net: clean_texture + stress_level -> corrupted_texture.
    Uses skip connections for preserving texture detail.
    """

    def __init__(self, stress_embed_dim: int = 64):
        super().__init__()
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)

        self.enc1 = ConvBlock(3, 64)               # 256 -> 128
        self.enc2 = ConvBlock(64, 128)             # 128 -> 64
        self.enc3 = ConvBlock(128, 256)            # 64 -> 32
        self.enc4 = ConvBlock(256, 512)            # 32 -> 16
        self.enc5 = ConvBlock(512, 512)            # 16 -> 8

        self.bottleneck_fc = nn.Linear(stress_embed_dim, 512 * 8 * 8)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
        )

        self.dec5 = UpBlock(512, 512, dropout=0.3)    # 8 -> 16
        self.dec4 = UpBlock(1024, 256, dropout=0.3)   # 16 -> 32
        self.dec3 = UpBlock(512, 128)                  # 32 -> 64
        self.dec2 = UpBlock(256, 64)                   # 64 -> 128
        self.dec1 = UpBlock(128, 32)                   # 128 -> 256

        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        emb = self.stress_embed(stress_level)
        cond = self.bottleneck_fc(emb)
        cond = cond.view(-1, 512, 8, 8)
        bottleneck = self.bottleneck_conv(torch.cat([e5, cond], dim=1))

        d5 = self.dec5(bottleneck)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.final(d1)

    @torch.no_grad()
    def corrupt(self, clean_texture: torch.Tensor, stress_level: int,
                device: torch.device = None) -> torch.Tensor:
        """Run corruption inference on a single texture."""
        if device is None:
            device = next(self.parameters()).device
        if clean_texture.dim() == 3:
            clean_texture = clean_texture.unsqueeze(0)
        clean_texture = clean_texture.to(device)
        sl = torch.tensor([stress_level], dtype=torch.long, device=device)
        corrupted = self.forward(clean_texture, sl)
        return corrupted.squeeze(0)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for texture realism."""

    def __init__(self, stress_embed_dim: int = 64):
        super().__init__()
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)
        self.stress_expand = nn.Linear(stress_embed_dim, IMG_SIZE * IMG_SIZE)

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, img: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        emb = self.stress_embed(stress_level)
        cond_map = self.stress_expand(emb).view(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, cond_map], dim=1)
        return self.model(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
