"""
Entity Sprite Generator -- 128x128 RGBA with 8 design classes.

Conditional DCGAN that generates 128x128 RGBA entity sprites from
stress level + design class + noise.  Outputs include alpha channel
for transparency.

Design classes (NUM_DESIGNS=8):
  0 wraith   -- tall, thin, flowing
  1 brute    -- massive, wide shoulders
  2 crawler  -- low, multi-limbed
  3 specter  -- floating, ethereal
  4 stalker  -- humanoid, wrong proportions
  5 abomination -- asymmetric, fused
  6 shade    -- featureless dark silhouette
  7 parasite -- small, clustered, tentacled

Architecture:
  Generator:  stress_embed + design_embed + noise -> 4x4 -> upsample to 128x128 RGBA
  Discriminator: 128x128 RGBA + stress + design -> real/fake + aux design class
  ~3.5M params, <6ms inference on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


NOISE_DIM = 64
IMG_SIZE = 128
IMG_CHANNELS = 4
NUM_STRESS_LEVELS = 6
NUM_DESIGNS = 8

DESIGN_NAMES = [
    "wraith", "brute", "crawler", "specter",
    "stalker", "abomination", "shade", "parasite",
]


class SelfAttention(nn.Module):
    """Channel self-attention for coherent spatial structure."""

    def __init__(self, ch: int):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class ResUpBlock(nn.Module):
    """Residual upsampling block for stronger structure generation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class EntitySpriteGenerator(nn.Module):
    """Generates 128x128 RGBA entity sprites from stress + design + noise."""

    def __init__(self, noise_dim: int = NOISE_DIM, stress_embed_dim: int = 32,
                 design_embed_dim: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)
        self.design_embed = nn.Embedding(NUM_DESIGNS, design_embed_dim)

        input_dim = noise_dim + stress_embed_dim + design_embed_dim

        self.project = nn.Sequential(
            nn.Linear(input_dim, 384 * 4 * 4),
            nn.LeakyReLU(0.2),
        )

        self.up1 = ResUpBlock(384, 256)  # 8x8
        self.up2 = ResUpBlock(256, 192)  # 16x16
        self.attn = SelfAttention(192)
        self.up3 = ResUpBlock(192, 128)  # 32x32
        self.up4 = ResUpBlock(128, 96)   # 64x64
        self.up5 = ResUpBlock(96, 64)    # 128x128
        self.to_rgba = nn.Conv2d(64, IMG_CHANNELS, 3, 1, 1)

        self.rgb_act = nn.Tanh()
        self.alpha_act = nn.Sigmoid()

    def forward(self, noise: torch.Tensor, stress_level: torch.Tensor,
                design_class: torch.Tensor) -> torch.Tensor:
        s_emb = self.stress_embed(stress_level)
        d_emb = self.design_embed(design_class)
        x = torch.cat([noise, s_emb, d_emb], dim=-1)
        x = self.project(x)
        x = x.view(x.size(0), 384, 4, 4)
        x = self.up1(x)   # 8x8
        x = self.up2(x)   # 16x16
        x = self.attn(x)  # self-attention at 16x16
        x = self.up3(x)   # 32x32
        x = self.up4(x)   # 64x64
        x = self.up5(x)   # 128x128
        x = self.to_rgba(x)
        rgb = self.rgb_act(x[:, :3])
        alpha = self.alpha_act(x[:, 3:4])
        return torch.cat([rgb, alpha], dim=1)

    @torch.no_grad()
    def generate(self, stress_level: int, design_class: int,
                 device: torch.device = None,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if noise is None:
            noise = torch.randn(1, self.noise_dim, device=device)
        sl = torch.tensor([stress_level], dtype=torch.long, device=device)
        dc = torch.tensor([design_class], dtype=torch.long, device=device)
        img = self.forward(noise, sl, dc)
        return img.squeeze(0)


class EntitySpriteDiscriminator(nn.Module):
    """Discriminator with auxiliary design classifier head."""

    def __init__(self, stress_embed_dim: int = 32, design_embed_dim: int = 32):
        super().__init__()
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)
        self.design_embed = nn.Embedding(NUM_DESIGNS, design_embed_dim)

        self.convs = nn.Sequential(
            spectral_norm(nn.Conv2d(IMG_CHANNELS, 48, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(48, 96, 4, 2, 1)),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(96, 160, 4, 2, 1)),
            nn.InstanceNorm2d(160),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(160, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(256, 320, 4, 2, 1)),
            nn.LeakyReLU(0.2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 321 + stress_embed_dim + design_embed_dim
        self.realfake = nn.Sequential(
            nn.Linear(feat_dim, 192),
            nn.LeakyReLU(0.2),
            nn.Linear(192, 1),
        )
        self.aux_design = nn.Sequential(
            nn.Linear(321, 96),
            nn.LeakyReLU(0.2),
            nn.Linear(96, NUM_DESIGNS),
        )

    def forward(self, img: torch.Tensor, stress_level: torch.Tensor,
                design_class: torch.Tensor):
        x = self.convs(img)
        x = self.pool(x).view(x.size(0), -1)
        # Minibatch standard deviation feature improves diversity pressure.
        mb_std = torch.std(x, dim=0, unbiased=False).mean().expand(x.size(0), 1)
        x = torch.cat([x, mb_std], dim=1)
        s_emb = self.stress_embed(stress_level)
        d_emb = self.design_embed(design_class)
        rf = self.realfake(torch.cat([x, s_emb, d_emb], dim=-1))
        aux = self.aux_design(x)
        return rf, aux


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
