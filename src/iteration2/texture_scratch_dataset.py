"""
Dataset for training the texture-from-scratch generator.

Loads all real textures organized by stress level from the downloaded
assets. Resizes to 128x128 and normalizes to [-1, 1] for GAN training.
Falls back to synthetic procedural textures if no real data available.
"""

import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMG_SIZE = 128


class TextureScratchDataset(Dataset):
    """Loads real textures by stress level for unconditional generation training."""

    def __init__(self, assets_dir: str = "assets", img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.samples = []
        self._load_real(Path(assets_dir))
        if len(self.samples) == 0:
            self._generate_synthetic()
        print(f"[texture-scratch-dataset] loaded {len(self.samples)} samples")

    def _load_real(self, assets_path: Path):
        if not HAS_PIL:
            return
        tex_dir = assets_path / "textures"
        if not tex_dir.exists():
            return

        for level in range(6):
            level_dir = tex_dir / str(level)
            if not level_dir.exists():
                continue
            for asset_dir in level_dir.iterdir():
                if not asset_dir.is_dir():
                    continue
                for img_file in asset_dir.glob("*Color*"):
                    try:
                        img = Image.open(img_file).convert("RGB")
                        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                        tensor = torch.from_numpy(arr).permute(2, 0, 1)
                        self.samples.append((tensor, level))
                    except Exception:
                        continue

        entity_dir = tex_dir / "entity"
        if entity_dir.exists():
            for asset_dir in entity_dir.iterdir():
                if not asset_dir.is_dir():
                    continue
                for img_file in asset_dir.glob("*Color*"):
                    try:
                        img = Image.open(img_file).convert("RGB")
                        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
                        tensor = torch.from_numpy(arr).permute(2, 0, 1)
                        self.samples.append((tensor, random.choice([3, 4, 5])))
                    except Exception:
                        continue

    def _generate_synthetic(self):
        rng = random.Random(42)
        for level in range(6):
            for _ in range(200):
                img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
                darkness = level / 5.0
                base_r = 0.6 - darkness * 0.5 + rng.uniform(-0.1, 0.1)
                base_g = 0.55 - darkness * 0.45 + rng.uniform(-0.1, 0.1)
                base_b = 0.5 - darkness * 0.4 + rng.uniform(-0.1, 0.1)
                img[0] = base_r
                img[1] = base_g
                img[2] = base_b

                for _ in range(int(10 + level * 20)):
                    x = rng.randint(0, self.img_size - 1)
                    y = rng.randint(0, self.img_size - 1)
                    r = rng.randint(2, 6 + level * 3)
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            px, py = x + dx, y + dy
                            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                                dist = math.hypot(dx, dy) / r
                                if dist < 1.0:
                                    factor = (1 - dist) * darkness * 0.3
                                    img[0, py, px] -= factor
                                    img[1, py, px] -= factor * 0.8
                                    img[2, py, px] -= factor * 0.6

                noise_amp = 0.02 + level * 0.02
                img += np.random.randn(3, self.img_size, self.img_size).astype(np.float32) * noise_amp
                img = np.clip(img * 2 - 1, -1, 1)

                self.samples.append((torch.from_numpy(img), level))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img, level = self.samples[idx]
        return img, torch.tensor(level, dtype=torch.long)
