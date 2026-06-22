"""
Ingest mixed asset types into a unified texture manifest.

Handles:
- Image files across common extensions (png/jpg/jpeg/webp/bmp/tga/tif/tiff)
- ZIP archives (extracts and scans recursively)
- OBJ/MTL material references (collects map_Kd/map_* textures)

Usage:
  python ingest_assets.py
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".tif", ".tiff"}

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
EXTRACT_DIR = ASSETS_DIR / "_extracted"
MANIFEST_PATH = ASSETS_DIR / "ingested_manifest.json"


def _safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = out_dir / member.filename
            if not str(target.resolve()).startswith(str(out_dir.resolve())):
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _parse_mtl_images(mtl_path: Path) -> list[Path]:
    refs: list[Path] = []
    try:
        text = mtl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return refs
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("map_"):
            parts = re.split(r"\s+", s)
            if len(parts) >= 2:
                tex_name = parts[-1].strip().strip('"')
                p = (mtl_path.parent / tex_name).resolve()
                if p.exists() and p.suffix.lower() in IMG_EXTS:
                    refs.append(p)
    return refs


def _parse_obj_mtls(obj_path: Path) -> list[Path]:
    mtls: list[Path] = []
    try:
        text = obj_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return mtls
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("mtllib "):
            mtl_name = s[7:].strip().strip('"')
            p = (obj_path.parent / mtl_name).resolve()
            if p.exists() and p.suffix.lower() == ".mtl":
                mtls.append(p)
    return mtls


def _guess_stress(path: Path) -> int:
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if p in {"0", "1", "2", "3", "4", "5"}:
            return int(p)
    joined = " ".join(parts)
    if any(k in joined for k in ["blood", "crack", "horror", "decay", "rust", "burn"]):
        return 4
    if any(k in joined for k in ["lava", "scorch", "hell", "demon"]):
        return 5
    if any(k in joined for k in ["clean", "white", "marble", "tile"]):
        return 1
    return 3


def build_manifest() -> dict:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Extract ZIP files.
    zip_files = list(ASSETS_DIR.rglob("*.zip"))
    for z in zip_files:
        rel = z.relative_to(ASSETS_DIR)
        out = EXTRACT_DIR / rel.with_suffix("")
        if not out.exists():
            try:
                _safe_extract_zip(z, out)
                print(f"[ingest] extracted zip: {z}")
            except Exception as e:
                print(f"[ingest] failed zip {z}: {e}")

    # 2) Collect direct images.
    all_images: set[Path] = set()
    for p in ASSETS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            all_images.add(p.resolve())

    # 3) Collect OBJ->MTL->texture refs.
    for obj in ASSETS_DIR.rglob("*.obj"):
        for mtl in _parse_obj_mtls(obj):
            for img in _parse_mtl_images(mtl):
                all_images.add(img.resolve())

    by_stress: dict[int, list[str]] = {i: [] for i in range(6)}
    entity_images: list[str] = []
    for img in sorted(all_images):
        s = _guess_stress(img)
        rel = img.relative_to(BASE_DIR).as_posix() if str(img).startswith(str(BASE_DIR)) else str(img)
        by_stress[s].append(rel)
        low = rel.lower()
        if "entity" in low or "monster" in low or "creature" in low or "sprite" in low:
            entity_images.append(rel)

    manifest = {
        "images_by_stress": by_stress,
        "entity_images": entity_images,
        "totals": {
            "zip_files": len(zip_files),
            "images": sum(len(v) for v in by_stress.values()),
            "entity_images": len(entity_images),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build_manifest()
    print(
        f"[ingest] manifest written: {MANIFEST_PATH} | "
        f"images={m['totals']['images']} entity_images={m['totals']['entity_images']}"
    )
