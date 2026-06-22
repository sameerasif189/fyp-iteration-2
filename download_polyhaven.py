"""
Download textures from Poly Haven (polyhaven.com) -- all CC0.

Poly Haven has 800+ free PBR textures at up to 8K resolution.
This script downloads 1K JPGs to keep storage reasonable.

Usage:
  python download_polyhaven.py --max 100
"""

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import requests

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
TEXTURES_DIR = ASSETS_DIR / "textures"

POLYHAVEN_API = "https://api.polyhaven.com/assets"
POLYHAVEN_FILES = "https://api.polyhaven.com/files"

STRESS_KEYWORDS = {
    0: ["white", "clean", "marble", "plaster", "fabric", "wood", "tile"],
    1: ["concrete", "brick", "paint", "wallpaper", "ceramic", "plywood"],
    2: ["old", "worn", "stone", "gravel", "sand", "clay", "stucco"],
    3: ["rust", "moss", "crack", "weathered", "corroded", "lichen"],
    4: ["dark", "damaged", "burnt", "coal", "charcoal", "metal", "rough"],
    5: ["lava", "volcanic", "scorch", "ash", "obsidian", "embers", "rock"],
}


def classify_stress(name: str, tags: list) -> int:
    name_lower = name.lower()
    all_text = name_lower + " " + " ".join(t.lower() for t in tags)

    for level in range(5, -1, -1):
        for kw in STRESS_KEYWORDS[level]:
            if kw in all_text:
                return level
    return 2


def download_polyhaven_textures(max_total: int = 100):
    print("=" * 60)
    print("DOWNLOADING TEXTURES FROM POLY HAVEN (CC0)")
    print("=" * 60)

    print("  fetching asset list...")
    try:
        resp = requests.get(POLYHAVEN_API, params={"t": "textures"}, timeout=30)
        resp.raise_for_status()
        assets = resp.json()
    except Exception as e:
        print(f"  [error] failed to fetch asset list: {e}")
        return

    print(f"  found {len(assets)} textures on Poly Haven")

    downloaded = 0
    for asset_id, asset_info in assets.items():
        if downloaded >= max_total:
            break

        tags = asset_info.get("tags", [])
        stress_level = classify_stress(asset_id, tags)
        out_dir = TEXTURES_DIR / str(stress_level) / f"PH_{asset_id}"

        if out_dir.exists():
            downloaded += 1
            continue

        try:
            files_resp = requests.get(f"{POLYHAVEN_FILES}/{asset_id}", timeout=30)
            files_resp.raise_for_status()
            files_data = files_resp.json()
        except Exception as e:
            print(f"    [warn] failed to get files for {asset_id}: {e}")
            continue

        tex_maps = files_data.get("Diffuse", files_data.get("diffuse", {}))
        if not tex_maps:
            tex_maps = files_data.get("Color", files_data.get("color", {}))
        if not tex_maps:
            continue

        res_key = "1k" if "1k" in tex_maps else next(iter(tex_maps), None)
        if not res_key:
            continue

        fmt_data = tex_maps[res_key]
        jpg_url = None
        for fmt in ["jpg", "png"]:
            if fmt in fmt_data:
                jpg_url = fmt_data[fmt].get("url")
                if jpg_url:
                    break

        if not jpg_url:
            continue

        print(f"    downloading PH_{asset_id} -> stress {stress_level}...")
        try:
            dl = requests.get(jpg_url, timeout=60)
            dl.raise_for_status()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"PH_{asset_id}_Color.jpg"
            out_path.write_bytes(dl.content)
            downloaded += 1
            print(f"    [ok] PH_{asset_id} ({downloaded}/{max_total})")
            time.sleep(0.5)
        except Exception as e:
            print(f"    [fail] {asset_id}: {e}")

    print(f"\n  [done] downloaded {downloaded} textures from Poly Haven")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=100,
                        help="Maximum textures to download")
    args = parser.parse_args()
    download_polyhaven_textures(max_total=args.max)
