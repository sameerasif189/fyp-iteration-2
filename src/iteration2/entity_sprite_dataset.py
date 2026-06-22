"""
Dataset for training the entity sprite generator.

Generates procedural RGBA silhouettes of horror entities at 128x128 for all
8 design classes at various stress levels.  These serve as training targets
for the GAN to learn the distribution of entity shapes per design.

Design classes:
  0 wraith   -- tall thin flowing robes/tendrils
  1 brute    -- massive wide shoulders thick limbs
  2 crawler  -- low multi-limbed insectoid
  3 specter  -- floating ethereal fragmented
  4 stalker  -- humanoid wrong proportions
  5 abomination -- asymmetric fused organic horror
  6 shade    -- featureless dark silhouette glowing features
  7 parasite -- small clustered tentacled

Stress level affects opacity / distortion / detail intensity (0-5).
"""

import math
import random
import json
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .entity_sprite_model import IMG_SIZE, NUM_DESIGNS, NUM_STRESS_LEVELS

_S = IMG_SIZE  # 128


def _ellipse(img: np.ndarray, cx: int, cy: int, rx: int, ry: int,
             rgb: tuple, opacity: float, power: float = 1.5):
    """Draw a soft ellipse into RGBA img (C, H, W)."""
    y0, y1 = max(0, cy - ry), min(_S, cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(_S, cx + rx + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            dx = (x - cx) / max(rx, 1)
            dy = (y - cy) / max(ry, 1)
            d = math.sqrt(dx * dx + dy * dy)
            if d < 1.0:
                a = (1.0 - d ** power) * opacity
                img[0, y, x] = max(img[0, y, x], rgb[0]) if a > img[3, y, x] else img[0, y, x]
                img[1, y, x] = max(img[1, y, x], rgb[1]) if a > img[3, y, x] else img[1, y, x]
                img[2, y, x] = max(img[2, y, x], rgb[2]) if a > img[3, y, x] else img[2, y, x]
                img[3, y, x] = max(img[3, y, x], a)


def _trapezoid(img: np.ndarray, cx: int, top_y: int, bot_y: int,
               top_hw: int, bot_hw: int, rgb: tuple, opacity: float):
    """Draw a tapered body shape."""
    h = max(bot_y - top_y, 1)
    for y in range(max(0, top_y), min(_S, bot_y)):
        t = (y - top_y) / h
        hw = top_hw + (bot_hw - top_hw) * t
        for x in range(max(0, int(cx - hw)), min(_S, int(cx + hw))):
            edge = 1.0 - abs(x - cx) / max(hw, 1)
            a = edge * opacity * 0.9
            if a > 0 and a > img[3, y, x]:
                img[0, y, x] = rgb[0]
                img[1, y, x] = rgb[1]
                img[2, y, x] = rgb[2]
                img[3, y, x] = a


def _tendril(img: np.ndarray, sx: float, sy: float, angle: float,
             length: int, rgb: tuple, opacity: float, rng: random.Random):
    """Draw a wandering tendril."""
    tx, ty = float(sx), float(sy)
    for seg in range(length):
        angle += rng.uniform(-0.35, 0.35)
        tx += math.cos(angle) * 2.0
        ty += math.sin(angle) * 1.6
        ix, iy = int(tx), int(ty)
        fade = 1.0 - seg / max(length, 1)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                px, py = ix + dx, iy + dy
                if 0 <= px < _S and 0 <= py < _S:
                    a = opacity * fade * 0.8
                    if a > img[3, py, px]:
                        img[0, py, px] = rgb[0]
                        img[1, py, px] = rgb[1]
                        img[2, py, px] = rgb[2]
                        img[3, py, px] = a


def _eyes(img: np.ndarray, cx: int, cy: int, spread: int, r: int,
          glow_rgb: tuple, opacity: float):
    """Draw glowing eyes."""
    for ex_off in [-spread, spread]:
        ex, ey = cx + ex_off, cy
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = ex + dx, ey + dy
                if 0 <= px < _S and 0 <= py < _S:
                    d = math.hypot(dx, dy) / max(r, 1)
                    if d < 1.0:
                        a = (1.0 - d) * opacity
                        img[0, py, px] = glow_rgb[0]
                        img[1, py, px] = glow_rgb[1]
                        img[2, py, px] = glow_rgb[2]
                        img[3, py, px] = min(1.0, img[3, py, px] + a)


# ---------- per-design drawing functions ----------

def _draw_wraith(img, level, opacity, rng):
    """Tall, thin, flowing form with tendrils at high stress."""
    cx = _S // 2 + rng.randint(-6, 6)
    head_y = _S // 6 + rng.randint(-4, 4)
    hr = rng.randint(10, 16)
    bw_top = rng.randint(8, 14)
    bw_bot = rng.randint(18, 30)
    bh = rng.randint(50, 75)
    rgb = (-0.65 + rng.uniform(-0.15, 0.1),
           -0.85 + rng.uniform(-0.1, 0.05),
           -0.75 + rng.uniform(-0.1, 0.05))
    _ellipse(img, cx, head_y, hr, hr, rgb, opacity, 1.2)
    _trapezoid(img, cx, head_y + hr - 3, head_y + hr + bh,
               bw_top, bw_bot, rgb, opacity)
    if level >= 4:
        for _ in range(rng.randint(3, 7)):
            ang = rng.uniform(-0.4, 0.4) + math.pi / 2
            _tendril(img, cx + rng.randint(-bw_bot, bw_bot),
                     head_y + hr + bh * 0.6, ang,
                     rng.randint(12, 30), rgb, opacity * 0.7, rng)
    if level >= 5:
        _eyes(img, cx, head_y, int(hr * 0.4), 4, (0.8, -0.5, -0.6), 0.9)


def _draw_brute(img, level, opacity, rng):
    """Massive wide-shouldered form with horns."""
    cx = _S // 2 + rng.randint(-5, 5)
    head_y = _S // 5 + rng.randint(-3, 5)
    hr = rng.randint(12, 18)
    shoulder_w = rng.randint(30, 46)
    bh = rng.randint(40, 60)
    rgb = (-0.5 + rng.uniform(-0.2, 0.05),
           -0.8 + rng.uniform(-0.1, 0.05),
           -0.85 + rng.uniform(-0.1, 0.0))
    _ellipse(img, cx, head_y, hr, int(hr * 0.85), rgb, opacity, 1.0)
    _trapezoid(img, cx, head_y + hr - 4, head_y + hr + bh,
               shoulder_w // 2, int(shoulder_w * 0.35), rgb, opacity)
    for side in (-1, 1):
        arm_x = cx + side * int(shoulder_w * 0.48)
        for ay in range(head_y + hr, head_y + hr + int(bh * 0.7)):
            for dx in range(-4, 5):
                px = arm_x + dx
                if 0 <= px < _S and 0 <= ay < _S:
                    img[0, ay, px] = rgb[0]
                    img[1, ay, px] = rgb[1]
                    img[2, ay, px] = rgb[2]
                    img[3, ay, px] = max(img[3, ay, px], opacity * 0.75)
    if level >= 4:
        horn_rgb = (-0.3, -0.6, -0.7)
        for side in (-1, 1):
            hx = cx + side * int(hr * 0.55)
            for hy in range(head_y - hr, head_y - hr + rng.randint(8, 16)):
                for dx in range(-3, 4):
                    px = hx + side * abs(hy - (head_y - hr)) // 2 + dx
                    if 0 <= px < _S and 0 <= hy < _S:
                        img[3, hy, px] = max(img[3, hy, px], opacity * 0.6)
                        img[0, hy, px] = horn_rgb[0]
                        img[1, hy, px] = horn_rgb[1]
                        img[2, hy, px] = horn_rgb[2]
    if level >= 5:
        _eyes(img, cx, head_y, int(hr * 0.35), 5, (0.9, -0.2, -0.5), 0.9)


def _draw_crawler(img, level, opacity, rng):
    """Low insectoid with many legs."""
    cx = _S // 2 + rng.randint(-8, 8)
    cy = _S // 2 + rng.randint(0, 15)
    bw = rng.randint(30, 50)
    bh = rng.randint(18, 30)
    hr = rng.randint(8, 13)
    rgb = (-0.55 + rng.uniform(-0.15, 0.1),
           -0.75 + rng.uniform(-0.1, 0.05),
           -0.6 + rng.uniform(-0.15, 0.05))
    _ellipse(img, cx, cy - bh // 2 - hr + 3, hr, hr, rgb, opacity, 1.3)
    _ellipse(img, cx, cy, bw // 2, bh // 2, rgb, opacity * 0.9, 1.1)
    num_legs = rng.randint(3, 6)
    for side in (-1, 1):
        for i in range(num_legs):
            lx = cx + side * int(bw * 0.4)
            ly = cy - bh // 3 + int(i * bh * 0.6 / max(num_legs - 1, 1))
            ang = side * (0.8 + i * 0.15) + rng.uniform(-0.2, 0.2)
            _tendril(img, lx, ly, ang, rng.randint(10, 22), rgb, opacity * 0.8, rng)
    if level >= 5:
        _eyes(img, cx, cy - bh // 2 - hr + 3, int(hr * 0.45), 3,
              (0.7, 0.2, -0.6), 0.85)


def _draw_specter(img, level, opacity, rng):
    """Floating ethereal form with fragmented edges."""
    cx = _S // 2 + rng.randint(-10, 10)
    cy = _S // 3 + rng.randint(-5, 10)
    body_r = rng.randint(20, 35)
    rgb = (-0.4 + rng.uniform(-0.2, 0.1),
           -0.55 + rng.uniform(-0.15, 0.1),
           -0.45 + rng.uniform(-0.15, 0.1))
    _ellipse(img, cx, cy, body_r, int(body_r * 1.4), rgb, opacity * 0.65, 1.8)
    fade_h = rng.randint(25, 50)
    for y in range(cy + body_r, min(_S, cy + body_r + fade_h)):
        t = (y - cy - body_r) / max(fade_h, 1)
        hw = int(body_r * (1.0 - t * 0.6))
        a = opacity * 0.5 * (1.0 - t)
        for x in range(max(0, cx - hw), min(_S, cx + hw)):
            noise_a = a * rng.uniform(0.3, 1.0)
            if noise_a > img[3, y, x]:
                img[0, y, x] = rgb[0]
                img[1, y, x] = rgb[1]
                img[2, y, x] = rgb[2]
                img[3, y, x] = noise_a
    if level >= 4:
        for _ in range(rng.randint(4, 8)):
            fx = cx + rng.randint(-body_r - 10, body_r + 10)
            fy = cy + rng.randint(-body_r, body_r + fade_h)
            fr = rng.randint(3, 8)
            _ellipse(img, fx, fy, fr, fr, rgb, opacity * 0.4, 2.0)
    if level >= 5:
        _eyes(img, cx, cy - int(body_r * 0.3), int(body_r * 0.3), 4,
              (0.5, 0.5, 0.9), 0.8)


def _draw_stalker(img, level, opacity, rng):
    """Humanoid with wrong proportions -- long arms, hunched."""
    cx = _S // 2 + rng.randint(-6, 6)
    head_y = _S // 5 + rng.randint(-3, 6)
    hr = rng.randint(9, 14)
    torso_h = rng.randint(28, 42)
    torso_w = rng.randint(14, 22)
    arm_len = rng.randint(40, 60)
    rgb = (-0.7 + rng.uniform(-0.15, 0.1),
           -0.8 + rng.uniform(-0.1, 0.05),
           -0.75 + rng.uniform(-0.1, 0.05))
    _ellipse(img, cx, head_y, hr, int(hr * 0.9), rgb, opacity, 1.2)
    _trapezoid(img, cx, head_y + hr - 2, head_y + hr + torso_h,
               torso_w // 2, int(torso_w * 0.4), rgb, opacity)
    for side in (-1, 1):
        ax = cx + side * (torso_w // 2 + 2)
        ay = head_y + hr + 2
        ang = side * 0.3 + math.pi * 0.45
        for seg in range(arm_len):
            ang += rng.uniform(-0.06, 0.06)
            ax += math.cos(ang) * 1.4
            ay += math.sin(ang) * 1.1
            ix, iy = int(ax), int(ay)
            for d in range(-2, 3):
                px, py = ix + d, iy
                if 0 <= px < _S and 0 <= py < _S:
                    a = opacity * (1.0 - seg / arm_len) * 0.8
                    if a > img[3, py, px]:
                        img[0, py, px] = rgb[0]
                        img[1, py, px] = rgb[1]
                        img[2, py, px] = rgb[2]
                        img[3, py, px] = a
    leg_h = rng.randint(20, 35)
    _trapezoid(img, cx, head_y + hr + torso_h - 2,
               head_y + hr + torso_h + leg_h,
               int(torso_w * 0.3), int(torso_w * 0.15), rgb, opacity * 0.85)
    if level >= 5:
        _eyes(img, cx, head_y - 2, int(hr * 0.35), 4, (0.9, 0.1, -0.4), 0.9)


def _draw_abomination(img, level, opacity, rng):
    """Asymmetric fused mass of organic horror."""
    cx = _S // 2 + rng.randint(-8, 8)
    cy = _S // 2 + rng.randint(-8, 8)
    num_blobs = rng.randint(4, 8)
    rgb_base = (-0.6 + rng.uniform(-0.2, 0.1),
                -0.7 + rng.uniform(-0.15, 0.1),
                -0.55 + rng.uniform(-0.15, 0.05))
    for _ in range(num_blobs):
        bx = cx + rng.randint(-25, 25)
        by = cy + rng.randint(-25, 25)
        rx = rng.randint(8, 22)
        ry = rng.randint(8, 22)
        tint = tuple(c + rng.uniform(-0.1, 0.1) for c in rgb_base)
        _ellipse(img, bx, by, rx, ry, tint, opacity * rng.uniform(0.5, 0.95), 1.3)
    if level >= 4:
        for _ in range(rng.randint(4, 10)):
            sx = cx + rng.randint(-20, 20)
            sy = cy + rng.randint(-20, 20)
            ang = rng.uniform(0, 2 * math.pi)
            _tendril(img, sx, sy, ang, rng.randint(10, 25),
                     rgb_base, opacity * 0.6, rng)
    if level >= 5:
        num_eyes = rng.randint(2, 5)
        for _ in range(num_eyes):
            ex = cx + rng.randint(-18, 18)
            ey = cy + rng.randint(-18, 18)
            _eyes(img, ex, ey, rng.randint(2, 6), 3,
                  (0.8, -0.3, -0.5), 0.75)


def _draw_shade(img, level, opacity, rng):
    """Featureless dark silhouette with glowing eyes/mouth."""
    cx = _S // 2 + rng.randint(-6, 6)
    head_y = _S // 5 + rng.randint(-3, 5)
    hr = rng.randint(12, 18)
    bh = rng.randint(45, 70)
    bw = rng.randint(18, 30)
    rgb = (-0.95, -0.95, -0.95)
    _ellipse(img, cx, head_y, hr, hr, rgb, opacity, 0.8)
    _trapezoid(img, cx, head_y + hr - 3, head_y + hr + bh,
               bw // 2, bw, rgb, opacity * 0.95)
    if level >= 4:
        _eyes(img, cx, head_y, int(hr * 0.35), 5,
              (0.9, 0.8, 0.2), 0.85)
    if level >= 5:
        mouth_y = head_y + int(hr * 0.5)
        mouth_w = int(hr * 0.3)
        for dy in range(-2, 4):
            for dx in range(-mouth_w, mouth_w + 1):
                px, py = cx + dx, mouth_y + dy
                if 0 <= px < _S and 0 <= py < _S:
                    d = abs(dx) / max(mouth_w, 1)
                    a = (1.0 - d) * 0.7
                    img[0, py, px] = 0.9
                    img[1, py, px] = 0.3
                    img[2, py, px] = -0.2
                    img[3, py, px] = min(1.0, img[3, py, px] + a)


def _draw_parasite(img, level, opacity, rng):
    """Small clustered tentacled form."""
    cx = _S // 2 + rng.randint(-10, 10)
    cy = _S // 2 + rng.randint(-5, 10)
    core_r = rng.randint(10, 18)
    rgb = (-0.5 + rng.uniform(-0.2, 0.1),
           -0.65 + rng.uniform(-0.15, 0.1),
           -0.45 + rng.uniform(-0.2, 0.1))
    _ellipse(img, cx, cy, core_r, core_r, rgb, opacity * 0.85, 1.1)
    num_clusters = rng.randint(3, 6)
    for i in range(num_clusters):
        ang = (i / num_clusters) * 2 * math.pi + rng.uniform(-0.3, 0.3)
        dist = core_r + rng.randint(5, 15)
        bx = cx + int(math.cos(ang) * dist)
        by = cy + int(math.sin(ang) * dist)
        br = rng.randint(5, 10)
        _ellipse(img, bx, by, br, br, rgb, opacity * 0.7, 1.4)
    num_tentacles = rng.randint(5, 10)
    for i in range(num_tentacles):
        ang = (i / num_tentacles) * 2 * math.pi + rng.uniform(-0.4, 0.4)
        sx = cx + int(math.cos(ang) * core_r * 0.8)
        sy = cy + int(math.sin(ang) * core_r * 0.8)
        _tendril(img, sx, sy, ang, rng.randint(15, 35), rgb, opacity * 0.65, rng)
    if level >= 5:
        _eyes(img, cx, cy - int(core_r * 0.3), int(core_r * 0.3), 3,
              (0.6, 0.7, -0.3), 0.8)


_DESIGN_DRAW = [
    _draw_wraith, _draw_brute, _draw_crawler, _draw_specter,
    _draw_stalker, _draw_abomination, _draw_shade, _draw_parasite,
]


class EntitySpriteDataset(Dataset):
    """Procedurally generates 128x128 RGBA entity sprites for GAN training."""

    def __init__(self, samples_per_level: int = 300, img_size: int = IMG_SIZE,
                 assets_dir: str = "assets"):
        self.img_size = img_size
        self.samples = []
        self.assets_dir = Path(assets_dir)

        self._load_real_sprites(assets_dir)
        self._generate_procedural(samples_per_level)

        print(f"[entity-sprite-dataset] {len(self.samples)} samples "
              f"({NUM_DESIGNS} designs x {NUM_STRESS_LEVELS} levels)")

    def _load_real_sprites(self, assets_dir: str):
        if not HAS_PIL:
            return
        from .entity_sprite_model import DESIGN_NAMES

        sprite_dir = Path(assets_dir) / "entity_sprites"
        if not sprite_dir.exists():
            sprite_dir = None

        name_to_id = {n: i for i, n in enumerate(DESIGN_NAMES)}
        loaded = 0

        if sprite_dir is not None:
            for level_dir in sprite_dir.iterdir():
                if not level_dir.is_dir():
                    continue
                try:
                    level = int(level_dir.name)
                except ValueError:
                    continue
                for img_file in level_dir.glob("*.png"):
                    try:
                        pil = Image.open(img_file).convert("RGBA")
                        design = NUM_DESIGNS - 1
                        stem = img_file.stem.lower()
                        for dname, did in name_to_id.items():
                            if stem.startswith(dname):
                                design = did
                                break
                        self._append_rgba_sample(pil, level, design)
                        loaded += 1
                    except Exception:
                        continue

        # Extra robust data source: ingest monster-like images from manifest + extracted folders.
        extra_paths = self._collect_candidate_entity_images()
        rng = random.Random(1337)
        for img_path in extra_paths:
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            stem = img_path.stem.lower()
            design = NUM_DESIGNS - 1
            for dname, did in name_to_id.items():
                if dname in stem:
                    design = did
                    break
            # Bias heavy entity training toward stress 4-5 where entities are visible in demo.
            stress_choices = [4, 4, 5, 5, 3]
            level = stress_choices[rng.randrange(len(stress_choices))]
            rgba_pil = self._extract_foreground_rgba(pil)
            if rgba_pil is None:
                continue
            # Add multiple augmented variants per source image.
            aug_count = 4
            for k in range(aug_count):
                aug = self._augment_rgba(rgba_pil, rng_seed=abs(hash((str(img_path), k))) % (2 ** 31))
                self._append_rgba_sample(aug, level, design)
                loaded += 1

        if loaded:
            print(f"[entity-sprite-dataset] loaded {loaded} real textured sprites")

    def _collect_candidate_entity_images(self):
        """Collect monster-like images from ingest manifest and extracted folders."""
        paths = []
        manifest = self.assets_dir / "ingested_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                for p in data.get("entity_images", []):
                    fp = Path(p)
                    if not fp.is_absolute():
                        fp = (Path.cwd() / fp).resolve()
                    if fp.exists():
                        paths.append(fp)
            except Exception:
                pass

        keywords = (
            "monster", "creature", "demon", "zombie", "ghost", "horror",
            "wraith", "brute", "crawler", "specter", "stalker", "abomination",
            "shade", "parasite", "alien", "mutant", "slender", "ghoul",
        )
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".tif", ".tiff")
        for root_name in ("_extracted", "entity_sprites"):
            root = self.assets_dir / root_name
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    low = str(f).lower()
                    if any(k in low for k in keywords):
                        paths.append(f)

        # dedup and cap to avoid exploding RAM
        uniq = []
        seen = set()
        for p in paths:
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            uniq.append(p)
        if len(uniq) > 2000:
            rng = random.Random(2048)
            rng.shuffle(uniq)
            uniq = uniq[:2000]
        print(f"[entity-sprite-dataset] candidate entity images: {len(uniq)}")
        return uniq

    def _extract_foreground_rgba(self, pil_rgb: "Image.Image"):
        """Robust foreground extraction from arbitrary monster image."""
        w, h = pil_rgb.size
        if w < 24 or h < 24:
            return None
        arr = np.array(pil_rgb, dtype=np.float32)
        # Estimate background from borders.
        bw = max(4, min(min(w, h) // 10, 20))
        border = np.concatenate([
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)
        dist = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
        alpha = np.clip((dist - 22.0) * 6.0, 0, 255).astype(np.uint8)

        # Fallback edge-based alpha if bg-key result is poor.
        cov = float(np.mean(alpha > 18))
        if cov < 0.06 or cov > 0.92:
            lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
            gy, gx = np.gradient(lum)
            edge = np.sqrt(gx * gx + gy * gy)
            thr = np.percentile(edge, 70)
            alpha = np.clip((edge - thr) * 22.0, 0, 255).astype(np.uint8)
            cov = float(np.mean(alpha > 18))
            if cov < 0.04:
                return None

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr.astype(np.uint8)
        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba, mode="RGBA")

    def _augment_rgba(self, pil_rgba: "Image.Image", rng_seed: int):
        rng = random.Random(rng_seed)
        img = pil_rgba.copy()
        # Affine jitter
        angle = rng.uniform(-12.0, 12.0)
        scale = rng.uniform(0.88, 1.14)
        nw = max(32, int(img.width * scale))
        nh = max(32, int(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS).rotate(angle, resample=Image.BICUBIC, expand=True)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Contrast/brightness jitter in RGB only.
        arr = np.array(img, dtype=np.float32)
        gain = rng.uniform(0.8, 1.25)
        bias = rng.uniform(-16.0, 16.0)
        arr[:, :, :3] = np.clip(arr[:, :, :3] * gain + bias, 0, 255)
        # Slight alpha erosion/dilation effect.
        a_gain = rng.uniform(0.85, 1.2)
        arr[:, :, 3] = np.clip(arr[:, :, 3] * a_gain, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    def _append_rgba_sample(self, pil_rgba: "Image.Image", level: int, design: int):
        pil = pil_rgba.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(pil, dtype=np.float32)
        rgba = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)
        rgba[:3] = (arr[:, :, :3].transpose(2, 0, 1) / 127.5) - 1.0
        rgba[3] = arr[:, :, 3] / 255.0
        self.samples.append((torch.from_numpy(rgba), int(level), int(design)))

    def _generate_procedural(self, samples_per_level: int):
        rng = random.Random(42)
        samples_per_design = max(1, samples_per_level // NUM_DESIGNS)

        for level in range(NUM_STRESS_LEVELS):
            if level <= 1:
                opacity_base = 0.05 + level * 0.08
            elif level <= 3:
                opacity_base = 0.25 + (level - 2) * 0.2
            else:
                opacity_base = 0.6 + (level - 4) * 0.3

            for design_id in range(NUM_DESIGNS):
                for _ in range(samples_per_design):
                    img = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)
                    opacity = min(1.0, opacity_base + rng.uniform(-0.05, 0.05))

                    if level <= 1 and rng.random() < 0.5:
                        pass
                    else:
                        _DESIGN_DRAW[design_id](img, level, opacity, rng)

                    img = np.clip(img, -1.0, 1.0)
                    self.samples.append((torch.from_numpy(img), level, design_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, level, design = self.samples[idx]
        return img, torch.tensor(level, dtype=torch.long), torch.tensor(design, dtype=torch.long)
