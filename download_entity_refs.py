"""
Generate entity sprite training references from downloaded entity textures.

Uses the AmbientCG entity textures (Bark, Leather, Moss, etc.) to create
128x128 RGBA silhouettes with entity-like shapes applied as texture fills.
This gives the GAN realistic texture data to learn from.

Usage:
  python download_entity_refs.py
"""

import json
import math
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SPRITES_DIR = ASSETS_DIR / "entity_sprites"
ENTITY_TEX_DIR = ASSETS_DIR / "textures" / "entity"

TARGET = 128
NUM_DESIGNS = 8
DESIGN_NAMES = [
    "wraith", "brute", "crawler", "specter",
    "stalker", "abomination", "shade", "parasite",
]


def _make_silhouette_mask(design: int, size: int, rng: random.Random) -> "Image.Image":
    """Create a white-on-black silhouette mask for a given design archetype."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = size // 2, size // 2

    if design == 0:  # wraith
        head_r = rng.randint(12, 20)
        draw.ellipse([cx - head_r, cy // 3 - head_r, cx + head_r, cy // 3 + head_r], fill=220)
        body_top = cy // 3 + head_r - 4
        pts = [(cx - rng.randint(8, 16), body_top),
               (cx + rng.randint(8, 16), body_top),
               (cx + rng.randint(20, 35), size - 5),
               (cx - rng.randint(20, 35), size - 5)]
        draw.polygon(pts, fill=200)
    elif design == 1:  # brute
        head_r = rng.randint(14, 20)
        draw.ellipse([cx - head_r, size // 6 - head_r, cx + head_r, size // 6 + head_r], fill=230)
        sw = rng.randint(35, 50)
        bh = rng.randint(50, 70)
        draw.rectangle([cx - sw // 2, size // 6 + head_r - 5,
                        cx + sw // 2, size // 6 + head_r + bh], fill=210)
        for side in [-1, 1]:
            ax = cx + side * sw // 2
            draw.rectangle([ax - 5, size // 6 + head_r, ax + 5,
                            size // 6 + head_r + int(bh * 0.8)], fill=180)
    elif design == 2:  # crawler
        bw, bh = rng.randint(40, 55), rng.randint(20, 32)
        head_r = rng.randint(10, 16)
        draw.ellipse([cx - head_r, cy - bh // 2 - head_r,
                       cx + head_r, cy - bh // 2 + head_r], fill=210)
        draw.ellipse([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2], fill=200)
        for side in [-1, 1]:
            for i in range(rng.randint(3, 5)):
                ly = cy - bh // 3 + int(i * bh * 0.5 / 4)
                lx = cx + side * bw // 2
                draw.line([(lx, ly), (lx + side * rng.randint(12, 25),
                           ly + rng.randint(8, 20))], fill=170, width=3)
    elif design == 3:  # specter
        r = rng.randint(25, 40)
        draw.ellipse([cx - r, cy // 2 - r, cx + r, cy // 2 + int(r * 1.5)], fill=150)
        for _ in range(rng.randint(5, 10)):
            fx = cx + rng.randint(-r - 10, r + 10)
            fy = cy // 2 + rng.randint(-r, int(r * 1.5) + 20)
            fr = rng.randint(4, 10)
            draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=rng.randint(80, 140))
    elif design == 4:  # stalker
        head_r = rng.randint(10, 16)
        draw.ellipse([cx - head_r, size // 5 - head_r, cx + head_r, size // 5 + head_r], fill=220)
        tw = rng.randint(16, 24)
        th = rng.randint(30, 45)
        draw.rectangle([cx - tw // 2, size // 5 + head_r - 3,
                        cx + tw // 2, size // 5 + head_r + th], fill=200)
        for side in [-1, 1]:
            pts = [(cx + side * tw // 2, size // 5 + head_r),
                   (cx + side * rng.randint(30, 50), size - rng.randint(10, 25)),
                   (cx + side * tw // 2 + side * 3, size // 5 + head_r + th)]
            draw.polygon(pts, fill=170)
    elif design == 5:  # abomination
        for _ in range(rng.randint(4, 8)):
            bx = cx + rng.randint(-30, 30)
            by = cy + rng.randint(-30, 30)
            rx, ry = rng.randint(10, 25), rng.randint(10, 25)
            draw.ellipse([bx - rx, by - ry, bx + rx, by + ry], fill=rng.randint(160, 230))
    elif design == 6:  # shade
        head_r = rng.randint(14, 22)
        draw.ellipse([cx - head_r, size // 5 - head_r, cx + head_r, size // 5 + head_r], fill=250)
        bw_top = rng.randint(12, 20)
        bw_bot = rng.randint(24, 38)
        bh = rng.randint(50, 75)
        pts = [(cx - bw_top, size // 5 + head_r - 4),
               (cx + bw_top, size // 5 + head_r - 4),
               (cx + bw_bot, size // 5 + head_r + bh),
               (cx - bw_bot, size // 5 + head_r + bh)]
        draw.polygon(pts, fill=240)
    elif design == 7:  # parasite
        core_r = rng.randint(12, 20)
        draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=210)
        for i in range(rng.randint(4, 8)):
            ang = (i / 8) * 2 * math.pi + rng.uniform(-0.3, 0.3)
            dist = core_r + rng.randint(5, 18)
            bx = cx + int(math.cos(ang) * dist)
            by = cy + int(math.sin(ang) * dist)
            br = rng.randint(5, 12)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=180)
            draw.line([(cx + int(math.cos(ang) * core_r * 0.7),
                        cy + int(math.sin(ang) * core_r * 0.7)),
                       (bx, by)], fill=160, width=2)

    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
    return mask


def generate_textured_sprites():
    """Generate RGBA sprites by applying texture fills to silhouette masks."""
    if not HAS_DEPS:
        print("[entity-refs] Pillow/numpy required")
        return

    print("=" * 60)
    print("GENERATING ENTITY SPRITE TRAINING REFERENCES")
    print("=" * 60)

    tex_paths = []
    if ENTITY_TEX_DIR.exists():
        for d in ENTITY_TEX_DIR.iterdir():
            if d.is_dir():
                for f in d.glob("*Color*"):
                    tex_paths.append(f)
    all_tex_dirs = ASSETS_DIR / "textures"
    for level_dir in all_tex_dirs.iterdir():
        if level_dir.is_dir() and level_dir.name.isdigit():
            for asset_dir in level_dir.iterdir():
                if asset_dir.is_dir():
                    for f in asset_dir.glob("*Color*"):
                        tex_paths.append(f)
                        break

    if not tex_paths:
        print("  [warn] no texture color maps found, skipping")
        return

    print(f"  found {len(tex_paths)} texture color maps for fills")

    rng = random.Random(42)
    total = 0

    for stress in range(6):
        level_dir = SPRITES_DIR / str(stress)
        level_dir.mkdir(parents=True, exist_ok=True)

        if stress <= 1:
            sprites_per_design = 3
        elif stress <= 3:
            sprites_per_design = 6
        else:
            sprites_per_design = 10

        for design_id in range(NUM_DESIGNS):
            for i in range(sprites_per_design):
                tex_path = rng.choice(tex_paths)
                try:
                    tex = Image.open(tex_path).convert("RGB")
                    tex = tex.resize((TARGET, TARGET), Image.LANCZOS)
                except Exception:
                    continue

                mask = _make_silhouette_mask(design_id, TARGET, rng)
                tex_arr = np.array(tex, dtype=np.float32)
                darkness = 0.3 + 0.12 * stress
                tex_arr = tex_arr * darkness
                if stress >= 3:
                    tex_arr[:, :, 0] = np.clip(tex_arr[:, :, 0] * (1.0 + 0.15 * stress), 0, 255)
                    tex_arr[:, :, 1] = np.clip(tex_arr[:, :, 1] * (1.0 - 0.08 * stress), 0, 255)

                rgba = np.zeros((TARGET, TARGET, 4), dtype=np.uint8)
                rgba[:, :, :3] = np.clip(tex_arr, 0, 255).astype(np.uint8)
                rgba[:, :, 3] = np.array(mask)

                out = Image.fromarray(rgba, "RGBA")
                fname = f"{DESIGN_NAMES[design_id]}_{i:03d}.png"
                out.save(str(level_dir / fname))
                total += 1

    print(f"  [done] generated {total} textured entity sprites")

    catalog = {
        "source": "procedural silhouettes filled with AmbientCG/PolyHaven textures",
        "total": total,
        "designs": DESIGN_NAMES,
        "target_size": TARGET,
    }
    (SPRITES_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))


if __name__ == "__main__":
    generate_textured_sprites()
