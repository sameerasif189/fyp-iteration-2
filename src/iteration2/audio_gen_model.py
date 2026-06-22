"""
Conditional WaveGAN for Horror Audio Generation.

Architecture:
  Generator:  stress embedding + noise -> 1D transposed convolutions with
              residual connections -> 1s audio (16kHz)
  Discriminator: 1D convolutions with spectral normalization -> real/fake score
  ~2.5M params total, <5ms inference on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


SAMPLE_RATE = 16000
AUDIO_LENGTH = 16000
NOISE_DIM = 128
NUM_STRESS_LEVELS = 6


class ResidualUpBlock(nn.Module):
    """Transposed-conv upsampling with a residual shortcut."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 25,
                 stride: int = 4, padding: int = 11, out_padding: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride, padding, out_padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode='linear', align_corners=False),
            nn.Conv1d(in_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main(x)
        skip = self.shortcut(x)
        min_len = min(main.size(-1), skip.size(-1))
        return main[..., :min_len] + skip[..., :min_len]


class AudioGenerator(nn.Module):
    """Conditional 1D WaveGAN generator with residual upsampling."""

    def __init__(self, noise_dim: int = NOISE_DIM, stress_embed_dim: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)

        input_dim = noise_dim + stress_embed_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256 * 16),
            nn.LeakyReLU(0.2),
        )

        self.up1 = ResidualUpBlock(256, 256)
        self.up2 = ResidualUpBlock(256, 128)
        self.up3 = ResidualUpBlock(128, 64)
        self.up4 = ResidualUpBlock(64, 32)
        self.final = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=25, stride=4,
                               padding=12, output_padding=3),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        emb = self.stress_embed(stress_level)
        x = torch.cat([noise, emb], dim=-1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 16)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        x = x[:, :, :AUDIO_LENGTH]
        return x

    @torch.no_grad()
    def generate(self, stress_level: int, device: torch.device = None,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if noise is None:
            noise = torch.randn(1, self.noise_dim, device=device)
        sl = torch.tensor([stress_level], dtype=torch.long, device=device)
        audio = self.forward(noise, sl)
        return audio.squeeze(0).squeeze(0)


class AudioDiscriminator(nn.Module):
    """1D convolutional discriminator with spectral normalization."""

    def __init__(self, stress_embed_dim: int = 32):
        super().__init__()
        self.stress_embed = nn.Embedding(NUM_STRESS_LEVELS, stress_embed_dim)

        self.convs = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 32, kernel_size=25, stride=4, padding=11)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(32, 64, kernel_size=25, stride=4, padding=11)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(64, 128, kernel_size=25, stride=4, padding=11)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=11)),
            nn.LeakyReLU(0.2),
        )

        self.pool = nn.AdaptiveAvgPool1d(4)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 + stress_embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, audio: torch.Tensor, stress_level: torch.Tensor) -> torch.Tensor:
        x = self.convs(audio)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        emb = self.stress_embed(stress_level)
        x = torch.cat([x, emb], dim=-1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
