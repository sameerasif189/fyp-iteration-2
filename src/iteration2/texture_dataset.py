"""
Texture dataset for training the corruption U-Net.

Creates paired training data: clean textures from AmbientCG -> procedurally
corrupted variants at each stress level. Corruptions include blood stains,
grime, cracks, scorch marks, and organic decay.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .texture_gen_model import IMG_SIZE, NUM_STRESS_LEVELS

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_texture(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load a texture file and resize to target size."""
    if not HAS_PIL:
        raise ImportError("Pillow is required. Install with: pip install Pillow")
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def apply_blood_stains(img: np.ndarray, intensity: float,
                       rng: random.Random) -> np.ndarray:
    """Add procedural blood stain overlays."""
    h, w = img.shape[:2]
    result = img.copy()
    num_stains = int(intensity * 28) + 3

    for _ in range(num_stains):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        radius = rng.randint(5, int(20 + 30 * intensity))

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    dist = np.sqrt(dx * dx + dy * dy) / radius
                    if dist < 1.0:
                        alpha = (1.0 - dist ** 1.5) * intensity * 0.95
                        noise = rng.uniform(-0.05, 0.05)
                        blood_r = 0.55 + noise + rng.uniform(0.0, 0.28)
                        blood_g = 0.01 + noise * 0.4
                        blood_b = 0.01 + noise * 0.25
                        result[py, px, 0] = result[py, px, 0] * (1 - alpha) + blood_r * alpha
                        result[py, px, 1] = result[py, px, 1] * (1 - alpha) + blood_g * alpha
                        result[py, px, 2] = result[py, px, 2] * (1 - alpha) + blood_b * alpha
    return np.clip(result, 0, 1)


def apply_grime(img: np.ndarray, intensity: float, rng: random.Random) -> np.ndarray:
    """Add dark grime/dirt overlay."""
    h, w = img.shape[:2]
    result = img.copy()

    grime_map = np.zeros((h, w), dtype=np.float32)
    num_patches = int(intensity * 40) + 5

    for _ in range(num_patches):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        size = rng.randint(10, int(40 + 60 * intensity))
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    dist = max(abs(dx), abs(dy)) / size
                    if dist < 1.0 and rng.random() < 0.6:
                        grime_map[py, px] = max(grime_map[py, px],
                                                (1.0 - dist) * intensity)

    for c in range(3):
        darken = rng.uniform(0.1, 0.3)
        result[:, :, c] = result[:, :, c] * (1.0 - grime_map * darken)

    return np.clip(result, 0, 1)


def apply_cracks(img: np.ndarray, intensity: float, rng: random.Random) -> np.ndarray:
    """Add procedural crack patterns."""
    h, w = img.shape[:2]
    result = img.copy()
    num_cracks = int(intensity * 18) + 4

    for _ in range(num_cracks):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        length = rng.randint(26, int(90 + 180 * intensity))
        angle = rng.uniform(0, 2 * np.pi)
        crack_color = np.array([rng.uniform(0.05, 0.15)] * 3)

        for step in range(length):
            nx = int(x + np.cos(angle) * step)
            ny = int(y + np.sin(angle) * step)
            if 0 <= nx < w and 0 <= ny < h:
                width = max(1, int(2 * (1 - step / length)))
                for dw in range(-width, width + 1):
                    px = nx + dw if rng.random() < 0.5 else nx
                    py = ny + dw if rng.random() >= 0.5 else ny
                    if 0 <= px < w and 0 <= py < h:
                        alpha = intensity * 1.05 * (1 - step / length)
                        result[py, px] = result[py, px] * (1 - alpha) + crack_color * alpha

            if rng.random() < 0.1:
                angle += rng.uniform(-0.5, 0.5)

    return np.clip(result, 0, 1)


def apply_scorch(img: np.ndarray, intensity: float, rng: random.Random) -> np.ndarray:
    """Add scorch/burn marks."""
    h, w = img.shape[:2]
    result = img.copy()
    num_marks = int(intensity * 10) + 2

    for _ in range(num_marks):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        radius = rng.randint(10, int(30 + 50 * intensity))

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    dist = np.sqrt(dx * dx + dy * dy) / radius
                    if dist < 1.0:
                        alpha = (1.0 - dist) * intensity * 0.6
                        scorch_val = 0.05 + 0.1 * dist
                        result[py, px] = result[py, px] * (1 - alpha) + scorch_val * alpha

    return np.clip(result, 0, 1)


def generate_corruption(clean: np.ndarray, stress_level: int,
                        rng: random.Random) -> np.ndarray:
    """Apply stress-appropriate corruption to a clean texture."""
    intensity = stress_level / 5.0
    corrupted = clean.copy()

    if stress_level >= 1:
        corrupted = apply_grime(corrupted, intensity * 0.8, rng)

    if stress_level >= 2:
        corrupted = apply_cracks(corrupted, intensity * 0.9, rng)

    if stress_level >= 3:
        corrupted = apply_blood_stains(corrupted, intensity * 0.8, rng)

    if stress_level >= 4:
        corrupted = apply_scorch(corrupted, intensity * 0.9, rng)
        # Re-apply cracks+blood at high stress for severe visible damage.
        corrupted = apply_cracks(corrupted, min(1.0, intensity * 1.1), rng)
        corrupted = apply_blood_stains(corrupted, min(1.0, intensity * 1.0), rng)

    if stress_level >= 1:
        darken = intensity * 0.45
        corrupted = corrupted * (1.0 - darken)

    return np.clip(corrupted, 0, 1)


class TextureCorruptionDataset(Dataset):
    """
    Dataset of (clean, corrupted, stress_level) triples.

    If real textures are available (from AmbientCG download), uses those.
    Otherwise generates synthetic procedural textures for training.
    """

    def __init__(self, assets_dir: Path = None, samples_per_level: int = 200,
                 augment: bool = True, seed: int = 42):
        super().__init__()
        self.augment = augment
        self.rng = random.Random(seed)
        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

        if assets_dir and (assets_dir / "textures").exists():
            self._load_real_textures(assets_dir / "textures", samples_per_level)
        else:
            self._generate_synthetic(samples_per_level)

        print(f"[texture-dataset] loaded {len(self.samples)} pairs")

    def _load_real_textures(self, textures_dir: Path, samples_per_level: int):
        """Load real downloaded textures and generate corrupted pairs."""
        print(f"[texture-dataset] scanning real textures at {textures_dir}")
        all_color_maps = []
        manifest = textures_dir.parent / "ingested_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                by_stress = data.get("images_by_stress", {})
                for _, paths in by_stress.items():
                    for p in paths:
                        fp = Path(p)
                        if not fp.is_absolute():
                            fp = textures_dir.parent.parent / fp
                        if fp.exists():
                            all_color_maps.append(str(fp))
            except Exception as e:
                print(f"[texture-dataset] manifest read failed: {e}")
        for level_dir in textures_dir.iterdir():
            if level_dir.is_dir():
                for asset_dir in level_dir.iterdir():
                    if asset_dir.is_dir():
                        for f in asset_dir.glob("*Color*"):
                            all_color_maps.append(str(f))
        if not all_color_maps:
            # Fallback for mixed raw files not following *Color* naming.
            exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tga", "*.tif", "*.tiff"]
            for ext in exts:
                for f in textures_dir.parent.rglob(ext):
                    all_color_maps.append(str(f))
        # deduplicate paths while preserving order
        seen = set()
        deduped = []
        for p in all_color_maps:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        all_color_maps = deduped

        if not all_color_maps:
            self._generate_synthetic(samples_per_level)
            return

        target = samples_per_level * NUM_STRESS_LEVELS
        for i in range(target):
            tex_path = self.rng.choice(all_color_maps)
            try:
                clean = load_texture(tex_path)
            except Exception:
                continue

            level = self.rng.randint(0, NUM_STRESS_LEVELS - 1)
            corrupted = generate_corruption(clean, level, self.rng)

            if self.augment:
                clean, corrupted = self._augment_pair(clean, corrupted)

            self.samples.append((clean, corrupted, level))
            if (i + 1) % 200 == 0:
                print(f"[texture-dataset] generated {i + 1}/{target} training pairs")

    def _generate_synthetic(self, samples_per_level: int):
        """Generate synthetic texture pairs for training without real data."""
        np_rng = np.random.RandomState(self.rng.randint(0, 99999))

        for level in range(NUM_STRESS_LEVELS):
            for _ in range(samples_per_level):
                clean = self._make_synthetic_texture(np_rng)
                corrupted = generate_corruption(clean, level, self.rng)
                self.samples.append((clean, corrupted, level))

    def _make_synthetic_texture(self, np_rng: np.random.RandomState) -> np.ndarray:
        """Create a synthetic base texture (Perlin-noise-like)."""
        size = IMG_SIZE
        img = np.zeros((size, size, 3), dtype=np.float32)

        base_color = np_rng.uniform(0.3, 0.8, size=3)
        img[:] = base_color

        for scale in [4, 8, 16, 32]:
            noise = np_rng.randn(size // scale, size // scale, 3) * 0.05
            noise_up = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
            noise_up = noise_up[:size, :size, :]
            img += noise_up

        return np.clip(img, 0, 1)

    def _augment_pair(self, clean: np.ndarray,
                      corrupted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same random augmentation to both clean and corrupted."""
        if self.rng.random() < 0.5:
            clean = np.fliplr(clean).copy()
            corrupted = np.fliplr(corrupted).copy()
        if self.rng.random() < 0.5:
            clean = np.flipud(clean).copy()
            corrupted = np.flipud(corrupted).copy()
        if self.rng.random() < 0.25:
            k = self.rng.choice([1, 2, 3])
            clean = np.rot90(clean, k).copy()
            corrupted = np.rot90(corrupted, k).copy()
        return clean, corrupted

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean, corrupted, level = self.samples[idx]
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float()
        corrupted_t = torch.from_numpy(corrupted).permute(2, 0, 1).float()
        level_t = torch.tensor(level, dtype=torch.long)
        return clean_t, corrupted_t, level_t
