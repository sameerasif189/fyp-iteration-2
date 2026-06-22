"""
Asset downloader for Iteration 2.

Fetches:
  - PBR textures from AmbientCG API v3 (CC0, 2K JPG)
  - Horror/ambient audio from Freesound API v2
  - Wildlife vocalizations from Freesound for audio model training

Usage:
  python download_assets.py --textures --audio --wildlife --freesound-key YOUR_KEY
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import requests

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
TEXTURES_DIR = ASSETS_DIR / "textures"
AUDIO_DIR = ASSETS_DIR / "audio"
WILDLIFE_DIR = ASSETS_DIR / "wildlife"

AMBIENTCG_API = "https://ambientcg.com/api/v2/full_json"

TEXTURE_QUERIES_BY_STRESS = {
    0: {"tags": ["Plaster", "Wood", "Tiles", "Marble", "Fabric", "Paint",
                 "Carpet", "WoodFloor", "Terrazzo", "Chipboard", "Paper"],
        "keywords": ["clean", "smooth", "white", "neutral", "new", "polished", "fresh"]},
    1: {"tags": ["Plaster", "Concrete", "Wood", "Wallpaper", "Tiles", "Brick",
                 "Chipboard", "WoodSiding", "Planks", "PaintedWood"],
        "keywords": ["plain", "light", "indoor", "simple", "faded", "matte"]},
    2: {"tags": ["Concrete", "Metal", "Wood", "Brick", "Stone", "Plaster",
                 "Gravel", "Soil", "Clay", "CorrugatedSteel", "Paving"],
        "keywords": ["worn", "old", "aged", "dirty", "stained", "dusty", "grimy"]},
    3: {"tags": ["Concrete", "Metal", "Ground", "Rock", "Brick", "Leather",
                 "Rust", "Moss", "Lichen", "CrackedAsphalt", "WornMetal"],
        "keywords": ["cracked", "rust", "weathered", "decay", "moss", "corroded", "peeling"]},
    4: {"tags": ["Metal", "Ground", "Rock", "Concrete", "Asphalt", "Bark",
                 "Coal", "Charcoal", "CrackedEarth", "SlateRoof", "OxidizedMetal"],
        "keywords": ["damaged", "corroded", "dark", "burnt", "rough", "shattered", "mangled"]},
    5: {"tags": ["Ground", "Rock", "Metal", "Lava", "Ice", "Concrete",
                 "Embers", "Obsidian", "DriedMud", "MoltenRock", "BurntWood"],
        "keywords": ["scorched", "destroyed", "dark", "charred", "volcanic", "hellish", "ruined"]},
}

ENTITY_TEXTURE_QUERIES = [
    "Fabric", "Leather", "Bark", "Organic",
    "Moss", "Mud", "Bone", "Ice",
    "Rust", "Coal", "Charcoal", "Slate",
    "Obsidian", "DriedMud", "Fur", "Scales",
    "Coral", "Fungus", "Lichen", "Rock",
    "Metal", "WornMetal", "OxidizedMetal", "CrackedEarth",
]

AUDIO_QUERIES_BY_STRESS = {
    0: ["room tone ambient", "soft wind indoor", "quiet ambience", "calm background", "peaceful interior"],
    1: ["subtle tension", "quiet creak", "soft drip echo", "light suspense", "uneasy ambience"],
    2: ["tension drone", "eerie ambience", "creaking wood", "distant rumble", "suspense build"],
    3: ["dark ambient horror", "distant thunder", "unsettling drone", "ominous bass", "dread ambience"],
    4: ["horror stinger", "scream distant", "distorted bass", "monster growl", "creature roar"],
    5: ["horror scream", "dark horror intense", "terrifying ambient", "demonic voice", "nightmare sound"],
}

ENTITY_AUDIO_QUERIES = [
    "monster breathing",
    "creature growl",
    "ghost whisper",
    "demon voice",
    "footsteps slow creepy",
    "heartbeat fast",
    "bone crack",
    "shadow whoosh",
    "inhuman scream",
    "horror creature",
    "zombie groan",
    "werewolf snarl",
    "chains dragging",
    "creepy laughter",
    "flesh ripping",
    "wet footsteps",
    "heavy breathing horror",
    "scratching wall",
    "teeth chattering",
    "low rumble creature",
]

WILDLIFE_QUERIES = [
    "bat screech",
    "fox scream night",
    "owl call spooky",
    "raven croak",
    "wolf howl",
    "cat hiss",
    "crow caw",
    "insect buzz night",
    "snake hiss",
    "rat squeak",
    "hawk screech",
    "frog croak night",
]


def download_textures(max_per_level: int = 6):
    """Download PBR textures from AmbientCG organized by stress level."""
    print("\n" + "=" * 60)
    print("DOWNLOADING TEXTURES FROM AMBIENTCG")
    print("=" * 60)

    for level, config in TEXTURE_QUERIES_BY_STRESS.items():
        level_dir = TEXTURES_DIR / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)

        existing = list(level_dir.iterdir())
        if len(existing) >= max_per_level:
            print(f"  [stress {level}] already have {len(existing)} textures, skipping")
            continue

        print(f"\n  [stress {level}] searching for textures...")
        downloaded = 0

        for tag in config["tags"]:
            if downloaded >= max_per_level:
                break

            params = {
                "type": "Material",
                "q": tag,
                "limit": 10,
                "sort": "Popular",
                "include": "downloadData",
            }

            try:
                resp = requests.get(AMBIENTCG_API, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    [warn] API request failed for tag '{tag}': {e}")
                continue

            assets = data.get("foundAssets", [])
            for asset in assets:
                if downloaded >= max_per_level:
                    break

                asset_id = asset.get("assetId", "unknown")
                asset_dir = level_dir / asset_id

                if asset_dir.exists():
                    downloaded += 1
                    continue

                downloads = asset.get("downloadFolders", {})
                download_url = None

                for folder in downloads.get("default", {}).get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", []):
                    if "2K-JPG" in folder.get("attribute", ""):
                        download_url = folder.get("downloadLink")
                        break
                    elif "1K-JPG" in folder.get("attribute", ""):
                        download_url = folder.get("downloadLink")

                if not download_url:
                    dl_list = downloads.get("default", {}).get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", [])
                    if dl_list:
                        download_url = dl_list[0].get("downloadLink")

                if not download_url:
                    continue

                print(f"    downloading {asset_id}...")
                try:
                    dl_resp = requests.get(download_url, timeout=120)
                    dl_resp.raise_for_status()

                    asset_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
                        for member in zf.namelist():
                            if member.endswith((".jpg", ".png", ".jpeg")):
                                filename = Path(member).name
                                with open(asset_dir / filename, "wb") as f:
                                    f.write(zf.read(member))

                    downloaded += 1
                    print(f"    [ok] {asset_id} ({downloaded}/{max_per_level})")
                    time.sleep(1)
                except Exception as e:
                    print(f"    [fail] {asset_id}: {e}")

        print(f"  [stress {level}] downloaded {downloaded} texture sets")


def download_freesound(queries: List[str], output_dir: Path, api_key: str,
                       max_per_query: int = 5, label: str = "audio"):
    """Download audio clips from Freesound API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Token {api_key}"}
    base_url = "https://freesound.org/apiv2"

    downloaded_total = 0

    for query in queries:
        params = {
            "query": query,
            "filter": "duration:[0.5 TO 30]",
            "sort": "rating_desc",
            "fields": "id,name,previews,duration,tags",
            "page_size": max_per_query,
        }

        try:
            resp = requests.get(f"{base_url}/search/text/", params=params,
                                headers=headers, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except Exception as e:
            print(f"    [warn] search failed for '{query}': {e}")
            continue

        for sound in results:
            sound_id = sound["id"]
            name = sound["name"].replace("/", "_").replace("\\", "_")[:50]
            filename = f"{sound_id}_{name}.mp3"
            filepath = output_dir / filename

            if filepath.exists():
                downloaded_total += 1
                continue

            preview_url = sound.get("previews", {}).get("preview-hq-mp3")
            if not preview_url:
                preview_url = sound.get("previews", {}).get("preview-lq-mp3")
            if not preview_url:
                continue

            try:
                dl_resp = requests.get(preview_url, headers=headers, timeout=30)
                dl_resp.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(dl_resp.content)
                downloaded_total += 1
                print(f"    [ok] {filename[:60]}")
                time.sleep(0.5)
            except Exception as e:
                print(f"    [fail] {filename[:40]}: {e}")

    print(f"  [{label}] downloaded {downloaded_total} clips to {output_dir}")
    return downloaded_total


def download_audio(api_key: str, max_per_query: int = 5):
    """Download horror audio organized by stress level."""
    print("\n" + "=" * 60)
    print("DOWNLOADING HORROR AUDIO FROM FREESOUND")
    print("=" * 60)

    for level, queries in AUDIO_QUERIES_BY_STRESS.items():
        level_dir = AUDIO_DIR / str(level)
        print(f"\n  [stress {level}] searching: {queries}")
        download_freesound(queries, level_dir, api_key,
                           max_per_query=max_per_query, label=f"stress-{level}")


def download_wildlife(api_key: str, max_per_query: int = 8):
    """Download wildlife vocalizations for audio model training."""
    print("\n" + "=" * 60)
    print("DOWNLOADING WILDLIFE VOCALIZATIONS FROM FREESOUND")
    print("=" * 60)

    print(f"  queries: {WILDLIFE_QUERIES}")
    download_freesound(WILDLIFE_QUERIES, WILDLIFE_DIR, api_key,
                       max_per_query=max_per_query, label="wildlife")


def download_entity_textures(max_total: int = 30):
    """Download organic/dark textures for entity skin rendering."""
    print("\n" + "=" * 60)
    print("DOWNLOADING ENTITY TEXTURES FROM AMBIENTCG")
    print("=" * 60)

    entity_dir = TEXTURES_DIR / "entity"
    entity_dir.mkdir(parents=True, exist_ok=True)

    existing = [d for d in entity_dir.iterdir() if d.is_dir()] if entity_dir.exists() else []
    if len(existing) >= max_total:
        print(f"  already have {len(existing)} entity textures, skipping")
        return

    downloaded = 0
    for tag in ENTITY_TEXTURE_QUERIES:
        if downloaded >= max_total:
            break
        params = {
            "type": "Material",
            "q": tag,
            "limit": 8,
            "sort": "Popular",
            "include": "downloadData",
        }
        try:
            resp = requests.get(AMBIENTCG_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [warn] API failed for '{tag}': {e}")
            continue

        for asset in data.get("foundAssets", []):
            if downloaded >= max_total:
                break
            asset_id = asset.get("assetId", "unknown")
            asset_dir = entity_dir / asset_id
            if asset_dir.exists():
                downloaded += 1
                continue

            downloads = asset.get("downloadFolders", {})
            download_url = None
            for folder in downloads.get("default", {}).get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", []):
                if "2K-JPG" in folder.get("attribute", ""):
                    download_url = folder.get("downloadLink")
                    break
                elif "1K-JPG" in folder.get("attribute", ""):
                    download_url = folder.get("downloadLink")
            if not download_url:
                dl_list = downloads.get("default", {}).get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", [])
                if dl_list:
                    download_url = dl_list[0].get("downloadLink")
            if not download_url:
                continue

            print(f"    downloading {asset_id}...")
            try:
                dl_resp = requests.get(download_url, timeout=120)
                dl_resp.raise_for_status()
                asset_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
                    for member in zf.namelist():
                        if member.endswith((".jpg", ".png", ".jpeg")):
                            filename = Path(member).name
                            with open(asset_dir / filename, "wb") as f:
                                f.write(zf.read(member))
                downloaded += 1
                print(f"    [ok] {asset_id} ({downloaded}/{max_total})")
                time.sleep(1)
            except Exception as e:
                print(f"    [fail] {asset_id}: {e}")

    print(f"  [entity textures] downloaded {downloaded} sets")


def download_entity_audio(api_key: str, max_per_query: int = 8):
    """Download entity-specific audio (growls, whispers, footsteps)."""
    print("\n" + "=" * 60)
    print("DOWNLOADING ENTITY AUDIO FROM FREESOUND")
    print("=" * 60)

    entity_audio_dir = ASSETS_DIR / "entity_audio"
    print(f"  queries: {ENTITY_AUDIO_QUERIES}")
    download_freesound(ENTITY_AUDIO_QUERIES, entity_audio_dir, api_key,
                       max_per_query=max_per_query, label="entity-audio")


def generate_catalog():
    """Generate catalog_real.json from downloaded assets."""
    print("\n" + "=" * 60)
    print("GENERATING CATALOG")
    print("=" * 60)

    catalog: Dict[str, object] = {
        "textures": {},
        "audio": {},
        "wildlife": [],
        "entity_textures": [],
        "entity_audio": [],
    }

    for level in range(6):
        tex_dir = TEXTURES_DIR / str(level)
        if tex_dir.exists():
            texture_sets = []
            for asset_dir in sorted(tex_dir.iterdir()):
                if asset_dir.is_dir():
                    color_maps = [str(f.relative_to(BASE_DIR))
                                  for f in asset_dir.glob("*Color*")]
                    if color_maps:
                        texture_sets.append({
                            "id": asset_dir.name,
                            "path": str(asset_dir.relative_to(BASE_DIR)),
                            "color": color_maps[0] if color_maps else None,
                            "normal": next((str(f.relative_to(BASE_DIR))
                                            for f in asset_dir.glob("*Normal*")), None),
                            "roughness": next((str(f.relative_to(BASE_DIR))
                                               for f in asset_dir.glob("*Roughness*")), None),
                        })
            catalog["textures"][str(level)] = texture_sets

        audio_dir = AUDIO_DIR / str(level)
        if audio_dir.exists():
            clips = [str(f.relative_to(BASE_DIR))
                     for f in sorted(audio_dir.glob("*.mp3"))]
            catalog["audio"][str(level)] = clips

    if WILDLIFE_DIR.exists():
        catalog["wildlife"] = [str(f.relative_to(BASE_DIR))
                               for f in sorted(WILDLIFE_DIR.glob("*.mp3"))]

    entity_tex_dir = TEXTURES_DIR / "entity"
    if entity_tex_dir.exists():
        entity_textures = []
        for asset_dir in sorted(entity_tex_dir.iterdir()):
            if asset_dir.is_dir():
                color_maps = [str(f.relative_to(BASE_DIR)) for f in asset_dir.glob("*Color*")]
                if color_maps:
                    entity_textures.append({
                        "id": asset_dir.name,
                        "path": str(asset_dir.relative_to(BASE_DIR)),
                        "color": color_maps[0],
                        "normal": next((str(f.relative_to(BASE_DIR)) for f in asset_dir.glob("*Normal*")), None),
                        "roughness": next((str(f.relative_to(BASE_DIR)) for f in asset_dir.glob("*Roughness*")), None),
                    })
        catalog["entity_textures"] = entity_textures

    entity_audio_dir = ASSETS_DIR / "entity_audio"
    if entity_audio_dir.exists():
        catalog["entity_audio"] = [str(f.relative_to(BASE_DIR))
                                   for f in sorted(entity_audio_dir.glob("*.mp3"))]

    catalog_path = ASSETS_DIR / "catalog_real.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"  [ok] wrote {catalog_path}")
    print(f"       textures: {sum(len(v) for v in catalog['textures'].values())} sets")
    print(f"       audio: {sum(len(v) for v in catalog['audio'].values())} clips")
    print(f"       wildlife: {len(catalog.get('wildlife', []))} clips")
    print(f"       entity textures: {len(catalog.get('entity_textures', []))} sets")
    print(f"       entity audio: {len(catalog.get('entity_audio', []))} clips")


def main():
    parser = argparse.ArgumentParser(description="Download assets for Iteration 2")
    parser.add_argument("--textures", action="store_true", help="Download PBR textures from AmbientCG")
    parser.add_argument("--audio", action="store_true", help="Download horror audio from Freesound")
    parser.add_argument("--wildlife", action="store_true", help="Download wildlife vocalizations")
    parser.add_argument("--entities", action="store_true", help="Download entity textures + audio")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--freesound-key", type=str, default=os.environ.get("FREESOUND_API_KEY", ""),
                        help="Freesound API key (or set FREESOUND_API_KEY env var)")
    parser.add_argument("--max-textures-per-level", type=int, default=15)
    parser.add_argument("--max-audio-per-query", type=int, default=8)
    parser.add_argument("--max-entity-textures", type=int, default=30)
    parser.add_argument("--catalog-only", action="store_true", help="Only regenerate catalog")
    args = parser.parse_args()

    if args.catalog_only:
        generate_catalog()
        return

    do_all = args.all or (not args.textures and not args.audio and not args.wildlife and not args.entities)

    if do_all or args.textures:
        download_textures(max_per_level=args.max_textures_per_level)

    if do_all or args.entities:
        download_entity_textures(max_total=args.max_entity_textures)

    if do_all or args.audio:
        if not args.freesound_key:
            print("[error] Freesound API key required. Use --freesound-key or set FREESOUND_API_KEY")
            print("        Get a free key at: https://freesound.org/apiv2/apply/")
        else:
            download_audio(args.freesound_key, max_per_query=args.max_audio_per_query)

    if do_all or args.entities:
        if args.freesound_key:
            download_entity_audio(args.freesound_key)
        else:
            print("[warn] skipping entity audio (no Freesound key)")

    if do_all or args.wildlife:
        if not args.freesound_key:
            print("[error] Freesound API key required for wildlife downloads")
        else:
            download_wildlife(args.freesound_key)

    generate_catalog()
    print("\n[done] Asset download complete!")


if __name__ == "__main__":
    main()
