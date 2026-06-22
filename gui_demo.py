"""
Iteration 2 Full GUI Demo -- Integrates all generation modules.

Panels:
  1. Environment: base tile (no fog) vs corrupted side-by-side; shared flicker only
  2. Camera Distortion: post-processing effects on environment view
  3. Audio: waveform + frequency visualization with DSP effects
  4. Camera Feed: live webcam with face distortion & shadow injection
  5. Entity: procedural entity rendering with model parameters
  6. HUD: all 30 parameters displayed with live bars

Controls:
  0-5: set stress level
  C: toggle camera feed
  Q/ESC: quit
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pygame
import pygame.gfxdraw
import torch

from src.iteration2.fusion_orchestrator import FusionOrchestrator
from src.iteration2.stress_schema import STRESS_PROFILES
from src.iteration2.performance import PerfMonitor
from src.iteration2.entity_gen import EntityParams, generate_entity_frame
from src.iteration2.camera_feed_fx import CameraFeedParams, CameraFeedProcessor
from src.iteration2.audio_gen_model import AudioGenerator, NOISE_DIM as AUDIO_NOISE_DIM
from src.iteration2.audio_horror_profile import (
    music_bed_level_gain,
    postprocess_wavegan_for_stress,
    wavegan_playback_volume,
    wavegan_quality_ok,
)
from src.iteration2.musicgen_audio_runtime import MusicGenRealtimeGen
from src.iteration2.procedural_audio_gen import ProceduralAudioGen
from src.iteration2.audio_content_filter import (
    filter_audio_paths,
    is_forbidden_audio,
    is_musicy_audio,
)
from src.iteration2.entity_sprite_model import (
    EntitySpriteGenerator, NUM_DESIGNS, DESIGN_NAMES,
)
from src.iteration2.texture_gen_model import TextureCorruptionUNet

try:
    import gltf as _panda_gltf  # panda3d-gltf
    import simplepbr as _panda_simplepbr
    from direct.showbase.ShowBase import ShowBase as _PandaShowBase
    from panda3d.core import (
        AmbientLight as _PAmbientLight,
        AntialiasAttrib as _PAntialiasAttrib,
        DirectionalLight as _PDirectionalLight,
        NodePath as _PNodePath,
        Vec3 as _PVec3,
        Vec4 as _PVec4,
        loadPrcFileData as _panda_load_prc,
    )
    _PANDA3D_AVAILABLE = True
    _PANDA3D_IMPORT_ERROR: Exception | None = None
except Exception as _e:
    _PANDA3D_AVAILABLE = False
    _PANDA3D_IMPORT_ERROR = _e

BASE_DIR = Path(__file__).resolve().parent
CATALOG = BASE_DIR / "assets" / "catalog.json"
CATALOG_REAL = BASE_DIR / "assets" / "catalog_real.json"
DEFAULT_MODEL = BASE_DIR / "models" / "fusion_generator.pt"
AUDIO_MODEL = BASE_DIR / "models" / "audio_generator.pt"
DEFAULT_MUSICGEN_LOCAL = BASE_DIR / "models" / "musicgen-small"
ENTITY_MODEL = BASE_DIR / "models" / "entity_sprite_gen.pt"
TEXTURE_MODEL = BASE_DIR / "models" / "texture_generator.pt"
MODEL_AUDIO_SR = 16000
MIXER_AUDIO_SR = 44100
ENTITY_MESH_CATALOG = BASE_DIR / "assets" / "entity_meshes" / "catalog.json"
NEW_ENTITIES_DIR = BASE_DIR / "new entities"
# Optional PNG/WebP blood & crack decals (dark-on-transparent preferred). Loaded lazily.
DAMAGE_OVERLAY_DIR = BASE_DIR / "assets" / "damage_overlays"

_cached_damage_assets: tuple[list[Path], list[Path]] | None = None


def damage_overlay_paths() -> tuple[list[Path], list[Path]]:
    """Return (crack_paths, blood_paths) under assets/damage_overlays/, if any."""
    global _cached_damage_assets
    if _cached_damage_assets is not None:
        return _cached_damage_assets
    crack: list[Path] = []
    blood: list[Path] = []
    if DAMAGE_OVERLAY_DIR.is_dir():
        for p in sorted(DAMAGE_OVERLAY_DIR.iterdir()):
            if not p.is_file():
                continue
            low = p.name.lower()
            if p.suffix.lower() not in (".png", ".webp", ".jpg", ".jpeg", ".bmp"):
                continue
            if any(k in low for k in ("crack", "fract", "split", "fissure")):
                crack.append(p)
            if any(k in low for k in ("blood", "splat", "stain", "drip")):
                blood.append(p)
    _cached_damage_assets = (crack, blood)
    return _cached_damage_assets


def _stamp_overlay_image(
    target: pygame.Surface,
    rng: random.Random,
    path: Path,
    *,
    span_frac: tuple[float, float],
    opacity: int,
    rotation_deg: tuple[float, float],
) -> bool:
    try:
        img = pygame.image.load(str(path)).convert_alpha()
    except Exception:
        return False
    w, h = target.get_width(), target.get_height()
    iw, ih = img.get_width(), img.get_height()
    if iw < 2 or ih < 2:
        return False
    span_lo, span_hi = span_frac
    target_span = int(min(w, h) * rng.uniform(span_lo, span_hi))
    scale = target_span / max(iw, ih)
    nw, nh = max(6, int(iw * scale)), max(6, int(ih * scale))
    img_s = pygame.transform.smoothscale(img, (nw, nh))
    img_s = pygame.transform.rotate(img_s, rng.uniform(*rotation_deg))
    tw, th = img_s.get_size()
    ox = rng.randint(max(-tw // 3, -(w // 12)), max(1, w - int(tw * 0.72)))
    oy = rng.randint(max(-th // 3, -(h // 12)), max(1, h - int(th * 0.72)))
    blank = pygame.Surface((w, h), pygame.SRCALPHA)
    blank.blit(img_s, (ox, oy))
    blank.set_alpha(max(0, min(255, int(opacity))))
    target.blit(blank, (0, 0))
    return True

W, H = 1760, 1024
DEFAULT_FPS = 30

STRESS_COLORS = {
    0: (60, 180, 80),
    1: (120, 180, 60),
    2: (210, 200, 40),
    3: (220, 140, 30),
    4: (200, 50, 30),
    5: (180, 20, 20),
}

STRESS_NAMES = {
    0: "Calm",
    1: "Wary",
    2: "Nervous",
    3: "Anxious",
    4: "Frightened",
    5: "Terrified",
}

ENTITY_MESH_MAP = {
    "wraith": "wraith_monster.glb",
    "brute": "brute_xbot.glb",
    "crawler": "crawler_brainstem.glb",
    "specter": "specter_ghost.glb",
    "stalker": "stalker_figure.glb",
    "abomination": "abomination_mass.glb",
    "shade": "shade_silhouette.glb",
    "parasite": "parasite_cluster.glb",
}

STRESS_4_DESIGNS = [0, 3, 6, 4]   # wraith, specter, shade, stalker
# Keep high-stress set focused on strong monster silhouettes (avoid blob-like parasite look).
STRESS_5_DESIGNS = [1, 2, 5, 4]   # brute, crawler, abomination, stalker


def _load_entity_mesh_catalog() -> dict:
    if not ENTITY_MESH_CATALOG.exists():
        return {}
    try:
        data = json.loads(ENTITY_MESH_CATALOG.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


ENTITY_MESH_CATALOG_DATA = _load_entity_mesh_catalog()


def _mesh_backed_design_ids() -> list[int]:
    """Design ids that have corresponding mesh assets in the catalog."""
    mesh_designs = {
        str(v.get("design", "")).strip().lower()
        for v in ENTITY_MESH_CATALOG_DATA.values()
        if isinstance(v, dict)
    }
    out = [i for i, name in enumerate(DESIGN_NAMES) if name.lower() in mesh_designs]
    return out


def _list_new_entities_meshes() -> list[Path]:
    """Return mesh files from external 'new entities' folder only."""
    if not NEW_ENTITIES_DIR.exists():
        return []
    out: list[Path] = []
    for ext in ("*.obj", "*.glb", "*.gltf", "*.fbx"):
        out.extend(sorted(NEW_ENTITIES_DIR.glob(ext)))
    return out


def _pick_weighted_new_entity_mesh(paths: list[Path], rng: random.Random) -> Path | None:
    """Pick one mesh per run uniformly at random from ``new entities/``.

    All four meshes (velociraptor / illiakan / cameraman / distortus_rex) are
    treated as equally likely so the run isn't biased toward the dinosaur
    silhouettes (velociraptor + distortus_rex would otherwise dominate ~2/3 of
    runs). Each mesh now has a flat ~25% chance.
    """
    if not paths:
        return None
    pick = rng.choice(paths)
    return pick


def _design_id_from_mesh_name(name: str) -> int:
    low = name.lower()
    for i, dname in enumerate(DESIGN_NAMES):
        if dname.lower() in low:
            return i
    # Meshes without explicit design keyword map to brute silhouette.
    return 1


def _texture_path_is_wood(path_str: str) -> bool:
    """True if filepath suggests lumber / planks / wood decking (AmbientCG naming, etc.)."""
    s = path_str.lower().replace("\\", "/")
    hints = (
        "wood", "lumber", "plank", "plywood", "veneer", "parquet",
        "birch", "oak", "mahogany", "teak", "walnut", "cedar",
        "pine", "fir", "timber", "bamboo", "hardwood", "roughwood",
        "deck", "flooring", "pallet",
        "_wood", "-wood", "/wood_", "/wood-", "wooddeck", "woodshelf",
        "fine woods", "antiqueoak", "woodfloors", "fine wood",
    )
    return any(h in s for h in hints)


class RealTextureBank:
    """Loads real PBR textures from catalog_real.json by stress level."""

    def __init__(self, target_w: int, target_h: int, substrate_filter: str = "wood"):
        self.target_w = target_w
        self.target_h = target_h
        self._substrate_filter = str(substrate_filter or "wood").lower()
        self.texture_paths = {}
        self._surf_cache = {}
        self._max_per_level = 180
        self._load()

    def _load(self):
        def _is_pbr_color_map(path_str: str) -> bool:
            s = path_str.lower().replace("\\", "/")
            name = Path(s).name
            if any(bad in s for bad in ["/entity_sprites/", "/_extracted/", "/wildlife/", "/audio/"]):
                return False
            if "/assets/textures/" not in s:
                return False
            good_tokens = ("color", "albedo", "basecolor", "diffuse")
            bad_tokens = ("normal", "roughness", "metal", "ao", "height", "displacement", "specular", "opacity")
            # Prefer explicit color maps, but allow plain texture images if they are in assets/textures.
            if not any(t in name for t in good_tokens):
                if not name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tga", ".tif", ".tiff")):
                    return False
            if any(t in name for t in bad_tokens):
                return False
            return True

        if not CATALOG_REAL.exists():
            print("[textures] no catalog_real.json found, using procedural")
            return

        catalog = json.loads(CATALOG_REAL.read_text(encoding="utf-8"))
        tex_data = catalog.get("textures", {})

        for level_str, tex_list in tex_data.items():
            level = int(level_str)
            self.texture_paths.setdefault(level, [])
            for tex_info in tex_list:
                color_path = tex_info.get("color")
                if color_path:
                    full_path = BASE_DIR / color_path
                    if not full_path.exists() or not _is_pbr_color_map(str(full_path)):
                        continue
                    fp_s = str(full_path)
                    if self._substrate_filter == "wood" and not _texture_path_is_wood(fp_s):
                        continue
                    self.texture_paths[level].append(fp_s)

        # Extra fallback: ingest mixed image files only when strict PBR maps are sparse.
        manifest_path = BASE_DIR / "assets" / "ingested_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                by_stress = data.get("images_by_stress", {})
                for lvl_str, paths in by_stress.items():
                    try:
                        level = int(lvl_str)
                    except Exception:
                        level = int(lvl_str) if isinstance(lvl_str, int) else 3
                    self.texture_paths.setdefault(level, [])
                    # Only augment levels that are currently low.
                    if len(self.texture_paths.get(level, [])) >= 24:
                        continue
                    for p in paths:
                        fp = Path(p)
                        if not fp.is_absolute():
                            fp = BASE_DIR / p
                        fs = str(fp)
                        if fp.exists() and _is_pbr_color_map(fs):
                            if self._substrate_filter == "wood" and not _texture_path_is_wood(fs):
                                continue
                            self.texture_paths[level].append(fs)
            except Exception as e:
                print(f"  [warn] failed to read ingested_manifest.json: {e}")

        # De-duplicate and cap per stress level to keep startup/snappiness stable.
        total = 0
        for lvl, paths in list(self.texture_paths.items()):
            dedup = list(dict.fromkeys(paths))
            if lvl >= 4 and self._substrate_filter != "wood":
                rock_kw = ("rock", "stone", "gravel", "slate", "asphalt", "concrete", "coal")
                rock = [p for p in dedup if any(k in p.lower() for k in rock_kw)]
                non_rock = [p for p in dedup if p not in rock]
                dedup = rock + non_rock
            if len(dedup) > self._max_per_level:
                rng = random.Random(1000 + lvl * 37)
                rng.shuffle(dedup)
                dedup = dedup[: self._max_per_level]
            self.texture_paths[lvl] = dedup
            total += len(dedup)

        if self._substrate_filter == "wood":
            merged: list[str] = []
            for lvl in range(6):
                merged.extend(self.texture_paths.get(lvl, []))
            merged = list(dict.fromkeys(merged))
            if merged:
                for lvl in range(6):
                    if not self.texture_paths.get(lvl):
                        self.texture_paths[lvl] = merged.copy()

        filter_note = f", filter={self._substrate_filter}" if self._substrate_filter != "all" else ""
        print(
            f"[textures] indexed {total} real textures "
            f"across {len(self.texture_paths)} stress levels (lazy-load{filter_note})"
        )

    def get(self, level: int, index: int = 0) -> pygame.Surface:
        """Get a real texture for a given stress level, or None."""
        paths = self.texture_paths.get(level, [])
        if paths:
            p = paths[index % len(paths)]
            return self.load_scaled_path(p, cache_tag=("lvl", level, p))
        return None

    def load_scaled_path(
        self, path_str: str, cache_tag: tuple | None = None
    ) -> pygame.Surface | None:
        """Decode + scale any stored asset path; cache key ignores stress bucket."""
        tag = cache_tag if cache_tag is not None else ("path", path_str)
        key = (self.target_w, self.target_h, tag, path_str)
        if key in self._surf_cache:
            return self._surf_cache[key]
        fp = Path(path_str)
        if not fp.exists():
            return None
        try:
            surf = pygame.image.load(str(fp))
            surf = pygame.transform.scale(surf, (self.target_w, self.target_h))
            if len(self._surf_cache) > 112:
                self._surf_cache.pop(next(iter(self._surf_cache)))
            self._surf_cache[key] = surf
            return surf
        except Exception:
            return None

    def any_level_paths(self, prefer: int | None = None) -> tuple[int, list[str]]:
        """Return (level_key, paths) for sampling when a tier is empty."""
        if prefer is not None:
            for d in range(6):
                lv = max(0, min(5, prefer + (d if d % 2 == 0 else -((d + 1) // 2))))
                if self.texture_paths.get(lv):
                    return lv, self.texture_paths[lv]
        for lv in range(6):
            if self.texture_paths.get(lv):
                return lv, self.texture_paths[lv]
        return prefer or 0, []

    @property
    def available(self) -> bool:
        return any(len(v) > 0 for v in self.texture_paths.values())


class EntityTextureBank:
    """Loads entity sprites only from dedicated entity sprite folders."""

    def __init__(self, target_w: int = 256, target_h: int = 256):
        self.target_w = target_w
        self.target_h = target_h
        self.texture_paths = {}
        self.design_paths = {}
        self._surf_cache = {}
        self._max_per_level = 120
        self._load()

    def _load(self):
        self.texture_paths = {i: [] for i in range(6)}
        self.design_paths = {i: [] for i in range(NUM_DESIGNS)}
        loaded = 0
        # Use only the new entity sprite folders (ignore generic manifest textures).
        candidate_roots = [NEW_ENTITIES_DIR]
        for sprite_root in candidate_roots:
            if not sprite_root.exists():
                continue
            for level_dir in sprite_root.iterdir():
                if not level_dir.is_dir():
                    continue
                try:
                    level = int(level_dir.name)
                except Exception:
                    continue
                if level < 0 or level > 5:
                    continue
                self.texture_paths.setdefault(level, [])
                for ext in ("*.png", "*.webp", "*.jpg", "*.jpeg"):
                    for f in level_dir.glob(ext):
                        self.texture_paths[level].append(str(f))
                        loaded += 1
            if loaded > 0:
                break

        for lvl, paths in list(self.texture_paths.items()):
            dedup = list(dict.fromkeys(paths))
            if len(dedup) > self._max_per_level:
                rng = random.Random(2000 + lvl * 53)
                rng.shuffle(dedup)
                dedup = dedup[: self._max_per_level]
            self.texture_paths[lvl] = dedup

        # Build design-specific path buckets from filename keywords.
        design_kw = {
            0: ("wraith", "ghost", "spectre", "phantom"),
            1: ("brute", "ogre", "tank", "xbot"),
            2: ("crawler", "spider", "insect", "skitter"),
            3: ("specter", "spirit", "ethereal"),
            4: ("stalker", "hunter", "lurker"),
            5: ("abomination", "mutant", "flesh", "horror"),
            6: ("shade", "shadow", "dark"),
            7: ("parasite", "worm", "slug", "tentacle"),
        }
        all_entity_paths = list(dict.fromkeys(self.texture_paths.get(4, []) + self.texture_paths.get(5, [])))
        for p in all_entity_paths:
            low = p.lower()
            assigned = False
            for did, kws in design_kw.items():
                if any(k in low for k in kws):
                    self.design_paths[did].append(p)
                    assigned = True
            if not assigned:
                # distribute unmatched assets round-robin for guaranteed coverage
                did = abs(hash(low)) % NUM_DESIGNS
                self.design_paths[did].append(p)

        print(f"[entity-textures] indexed {loaded} entity sprites from {NEW_ENTITIES_DIR}")

    def get(self, level: int, index: int = 0) -> pygame.Surface | None:
        paths = self.texture_paths.get(level, [])
        if paths:
            p = paths[index % len(paths)]
            key = (level, p)
            if key in self._surf_cache:
                return self._surf_cache[key]
            try:
                surf = pygame.image.load(p)
                surf = pygame.transform.scale(surf, (self.target_w, self.target_h))
                if len(self._surf_cache) > 64:
                    self._surf_cache.pop(next(iter(self._surf_cache)))
                self._surf_cache[key] = surf
                return surf
            except Exception:
                return None
        return None

    @property
    def available(self) -> bool:
        return any(len(v) > 0 for v in self.texture_paths.values())

    def get_design_sprite(
        self, level: int, design_id: int, seed: int, out_w: int, out_h: int
    ) -> pygame.Surface | None:
        """Create sprite-like RGBA entity from uploaded assets (auto background removal)."""
        paths = self.design_paths.get(design_id, [])
        if not paths:
            paths = self.texture_paths.get(level, [])
        if not paths:
            return None

        rng = random.Random(seed + design_id * 73 + level * 11)
        p = paths[rng.randrange(len(paths))]
        cache_key = ("design", p, out_w, out_h)
        if cache_key in self._surf_cache:
            return self._surf_cache[cache_key]
        try:
            src = pygame.image.load(p).convert()
            src = pygame.transform.smoothscale(src, (out_w, out_h)).convert()
        except Exception:
            return None
        arr = pygame.surfarray.array3d(src).astype(np.float32)  # (w, h, 3)
        w, h = arr.shape[0], arr.shape[1]

        # Estimate background color from corners, then key it out.
        cs = []
        n = max(4, min(16, min(w, h) // 12))
        cs.append(arr[:n, :n].reshape(-1, 3))
        cs.append(arr[w - n :, :n].reshape(-1, 3))
        cs.append(arr[:n, h - n :].reshape(-1, 3))
        cs.append(arr[w - n :, h - n :].reshape(-1, 3))
        bg = np.mean(np.concatenate(cs, axis=0), axis=0)
        dist = np.sqrt(np.sum((arr - bg) ** 2, axis=2))
        alpha = np.clip((dist - 18.0) * 8.0, 0, 255).astype(np.uint8)
        if float(alpha.mean()) < 12.0:
            lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            alpha = np.clip((lum - np.percentile(lum, 45)) * 7.0, 0, 255).astype(np.uint8)
        coverage = float(np.mean(alpha > 16))
        # Reject tiny-dot or broad rectangle masks from raw tile textures.
        if coverage < 0.08 or coverage > 0.58:
            return None

        rgba = np.zeros((w, h, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr.astype(np.uint8)
        rgba[:, :, 3] = alpha
        spr = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA").convert_alpha()

        # Mild per-design variation
        if rng.random() < 0.5:
            spr = pygame.transform.flip(spr, True, False)
        if rng.random() < 0.3:
            sw = max(72, int(out_w * rng.uniform(0.9, 1.1)))
            sh = max(96, int(out_h * rng.uniform(0.9, 1.1)))
            spr = pygame.transform.smoothscale(spr, (sw, sh))
        if len(self._surf_cache) > 96:
            self._surf_cache.pop(next(iter(self._surf_cache)))
        self._surf_cache[cache_key] = spr
        return spr


class RealAudioBank:
    """Loads real/generated audio clips from catalog for playback info."""

    def __init__(self):
        self.clips = {}
        self.clips_by_role = {}
        self._load()

    @staticmethod
    def _classify_clip(path: str) -> str:
        p = path.lower()
        impact_kw = ("hit", "sting", "stab", "impact", "shock", "scream", "roar", "laugh", "death")
        creature_kw = ("monster", "demon", "creature", "growl", "snarl", "ghost", "zombie", "alien")
        tension_kw = ("drone", "tension", "suspense", "rumble", "horror", "dark", "creak", "thunder", "ominous")
        if any(k in p for k in impact_kw):
            return "impact"
        if any(k in p for k in creature_kw):
            return "creature"
        if any(k in p for k in tension_kw):
            return "tension"
        return "ambient"

    def _load(self):
        if not CATALOG_REAL.exists():
            return
        catalog = json.loads(CATALOG_REAL.read_text(encoding="utf-8"))
        audio_data = catalog.get("audio", {})
        n_blocked = 0
        n_music_dropped = 0
        # First pass: load each *folder* (catalog key) as-is.
        per_folder_valid: dict[int, list[str]] = {}
        per_folder_by_role: dict[int, dict[str, list[str]]] = {}
        for level_str, clip_list in audio_data.items():
            folder = int(level_str)
            valid: list[str] = []
            by_role = {"ambient": [], "tension": [], "creature": [], "impact": []}
            for clip_path in clip_list:
                full = BASE_DIR / clip_path
                if not full.exists():
                    continue
                full_str = str(full)
                # Hard content filter: never expose forbidden files regardless
                # of stress level (sexual / gendered distress). The musical
                # loop filter only applies at L4/L5 — at lower stress, melodic
                # tension drones / piano stings are fine ambience.
                if is_forbidden_audio(full_str):
                    n_blocked += 1
                    continue
                if folder >= 4 and is_musicy_audio(full_str):
                    n_music_dropped += 1
                    continue
                valid.append(full_str)
                role = self._classify_clip(full_str)
                by_role[role].append(full_str)
            per_folder_valid[folder] = valid
            per_folder_by_role[folder] = by_role

        # Second pass: build the level -> clips views using a sliding 2-folder
        # window so each stress level draws from itself + the level below.
        #   L0 -> folder 0
        #   L1 -> folders 0 + 1
        #   L2 -> folders 1 + 2
        #   L3 -> folders 2 + 3
        #   L4 -> folders 3 + 4
        #   L5 -> folders 4 + 5
        for level in range(6):
            folders = [0] if level <= 0 else [level - 1, level]
            merged_valid: list[str] = []
            merged_role: dict[str, list[str]] = {
                "ambient": [], "tension": [], "creature": [], "impact": []
            }
            seen: set[str] = set()
            for f in folders:
                for p in per_folder_valid.get(f, []):
                    key = p.lower()
                    if key not in seen:
                        seen.add(key)
                        merged_valid.append(p)
                for role, paths in per_folder_by_role.get(f, {}).items():
                    for p in paths:
                        if p not in merged_role[role]:
                            merged_role[role].append(p)
            self.clips[level] = merged_valid
            self.clips_by_role[level] = merged_role

        total = sum(len(v) for v in self.clips.values())
        suffix = ""
        if n_blocked or n_music_dropped:
            suffix = f" (filtered forbidden={n_blocked} musical_L4L5={n_music_dropped})"
        print(
            f"[audio] cataloged {total} audio clip slots across {len(self.clips)} stress levels"
            f" (sliding 2-folder window){suffix}"
        )

    def get_clip_path(self, level: int, index: int = 0) -> str:
        """Get path to an audio clip for a stress level."""
        clips = self.clips.get(level, [])
        if clips:
            return clips[index % len(clips)]
        return None

    @property
    def available(self) -> bool:
        return any(len(v) > 0 for v in self.clips.values())

    def get_role_clip(self, level: int, role: str, index: int = 0) -> str:
        role_map = self.clips_by_role.get(level, {})
        clips = role_map.get(role, [])
        if not clips:
            clips = self.clips.get(level, [])
        if clips:
            return clips[index % len(clips)]
        return None


class EntityAudioBank:
    """Dedicated source of entity / creature sound effects.

    Loads every audio file in ``assets/entity_audio/`` (flat folder, no stress
    sub-folders), applies the content filter, and exposes a flat clip pool.
    Used as the *only* source for creature-role FX one-shots in the demo so
    monster sounds are routed independently from the per-stress fallback bed
    (which lives in ``assets/audio/{0..5}``).
    """

    def __init__(self, root: Path = BASE_DIR / "assets" / "entity_audio"):
        self.root = Path(root)
        self.clips: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.root.is_dir():
            print(f"[entity-audio] no folder at {self.root}; creature FX disabled")
            return
        candidates: list[Path] = []
        for ext in ("*.mp3", "*.wav", "*.ogg", "*.flac"):
            candidates.extend(sorted(self.root.glob(ext)))
        n_total = len(candidates)
        n_blocked = 0
        for p in candidates:
            if is_forbidden_audio(p):
                n_blocked += 1
                continue
            self.clips.append(str(p))
        print(
            f"[entity-audio] loaded {len(self.clips)}/{n_total} clips from {self.root}"
            + (f" (filtered forbidden={n_blocked})" if n_blocked else "")
        )

    @property
    def available(self) -> bool:
        return len(self.clips) > 0


class EntityAudioPicker:
    """Shuffle-once-per-pass picker over a flat EntityAudioBank clip pool."""

    __slots__ = ("_bank", "_rng", "_queue")

    def __init__(self, bank: EntityAudioBank, rng: random.Random):
        self._bank = bank
        self._rng = rng
        self._queue: list[str] = []
        self._refill()

    def _refill(self) -> None:
        paths = list(self._bank.clips)
        self._rng.shuffle(paths)
        self._queue = paths

    def next_path(self) -> str | None:
        if not self._bank.available:
            return None
        if not self._queue:
            self._refill()
        if not self._queue:
            return None
        return self._queue.pop(0)


class RotatingRoleClipPicker:
    """Shuffle-once-per-pass clip queues so the same file is not chosen every refresh at a tier."""

    __slots__ = ("_bank", "_role", "_rng", "_queues")

    def __init__(self, bank: RealAudioBank, role: str, rng: random.Random):
        self._bank = bank
        self._role = role
        self._rng = rng
        self._queues: dict[int, list[str]] = {}
        if bank.available:
            for lvl in sorted(self._bank.clips_by_role.keys()):
                self._queues[int(lvl)] = []
                self._refill(int(lvl))

    def _refill(self, level: int) -> None:
        role_map = self._bank.clips_by_role.get(level, {})
        paths = list(role_map.get(self._role, []))
        if not paths:
            paths = list(self._bank.clips.get(level, []))
        self._rng.shuffle(paths)
        self._queues[level] = paths

    def next_path(self, level: int) -> str | None:
        if not self._bank.available:
            return None
        level = max(0, min(5, int(level)))
        q = self._queues.get(level)
        if q is None:
            self._queues[level] = []
            self._refill(level)
            q = self._queues[level]
        if not q:
            self._refill(level)
            q = self._queues[level]
        if not q:
            return None
        return q.pop(0)


def gen_base_texture(w: int, h: int, seed: int) -> pygame.Surface:
    surf = pygame.Surface((w, h))
    rng = random.Random(seed)
    base_r, base_g, base_b = 90, 85, 75
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            v = rng.randint(-12, 12)
            c = (max(0, min(255, base_r + v)),
                 max(0, min(255, base_g + v)),
                 max(0, min(255, base_b + v)))
            pygame.draw.rect(surf, c, (x, y, 4, 4))
    return surf


def build_varied_base_texture(
    bank: RealTextureBank,
    level: int,
    w: int,
    h: int,
    compose_seed: int,
    *,
    texture_session_seed: int,
) -> pygame.Surface:
    """
    Compose the environment base from catalog assets (weighted around current stress),
    reshuffled each demo via ``texture_session_seed`` so layouts of tiles differ each run.
    """
    rng = random.Random(((compose_seed ^ texture_session_seed) & 0xFFFFFFFF) ^ (level << 17))
    canvas = pygame.Surface((w, h))
    canvas.fill((32, 30, 28))
    if not bank.available:
        return gen_base_texture(w, h, seed=compose_seed ^ texture_session_seed)

    def tier_offset() -> int:
        return rng.choices([-1, 0, 0, 0, 1], weights=[1, 8, 8, 6, 1])[0]

    tier_levels = [level]
    for _ in range(3):
        tier_levels.append(max(0, min(5, level + tier_offset())))

    chosen_paths: list[str] = []
    for tl in tier_levels:
        bucket = bank.texture_paths.get(tl, [])
        if not bucket:
            _, bucket = bank.any_level_paths(prefer=tl)
        if not bucket:
            return gen_base_texture(w, h, seed=compose_seed + 911)
        available = list(dict.fromkeys([p for p in bucket if p not in chosen_paths]))
        pool = available if available else bucket
        path_str = rng.choice(pool)
        chosen_paths.append(path_str)

    tiles: list[pygame.Surface | None] = []
    for p in chosen_paths:
        tiles.append(bank.load_scaled_path(p))

    tiles = [t for t in tiles if t is not None]
    if not tiles:
        return gen_base_texture(w, h, seed=compose_seed + 913)

    base = pygame.transform.smoothscale(tiles[0], (w, h))
    canvas.blit(base, (0, 0))
    if len(tiles) > 1:
        t2 = pygame.transform.smoothscale(tiles[1], (w, h))
        t2.set_alpha(rng.randint(68, 100))
        canvas.blit(t2, (0, 0))
    if len(tiles) > 2:
        t3 = pygame.transform.smoothscale(tiles[2], (w, h))
        t3.set_alpha(rng.randint(38, 62))
        canvas.blit(t3, (0, 0))
    if len(tiles) > 3:
        t4 = pygame.transform.smoothscale(tiles[3], (w, h))
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        for _ in range(18 + rng.randint(0, 10)):
            px = rng.randint(0, w - 1)
            py = rng.randint(0, h - 1)
            pr = rng.randint(28, min(140, max(w, h) // 2))
            pygame.draw.circle(mask, (255, 255, 255, rng.randint(20, 40)), (px, py), pr)
        t4.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        canvas.blit(t4, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return canvas


def _damage_strength(corruption_alpha: float, stress_level: int) -> float:
    st = stress_level / 5.0
    return float(min(1.0, corruption_alpha * (0.26 + 0.74 * st) + 0.065 * stress_level))


def _texture_damage_seed(session_seed: int, level: int, mixing_salt: int) -> int:
    """Mix per-demo session, stress level, and variant id for stable unique damage seeds."""
    s = int(session_seed) & 0xFFFFFFFF
    L = max(0, min(5, int(level)))
    m = int(mixing_salt) & 0xFFFFFFFF
    return int(s ^ ((L * 2654435761) & 0xFFFFFFFF) ^ ((m * 1597334677) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _blood_rect_overlay(
    rng: random.Random,
    overlay,
    w: int,
    h: int,
    stress_level: int,
    dmg: float,
) -> None:
    """Semi-transparent red rectangles; count and alpha range scale with stress (constant style)."""
    sl = max(0, min(5, int(stress_level)))
    st = float(sl) / 5.0
    n_blocks = max(5, min(170, int(12 + sl * 24 + dmg * 34)))
    alpha_lo = max(14, int(24 + 50 * st + 18 * dmg))
    alpha_hi = min(228, int(52 + 150 * st + 38 * dmg))
    alpha_lo = min(alpha_lo, alpha_hi)

    for __ in range(n_blocks):
        if rng.random() < 0.64:
            rw = rng.randint(max(12, max(10, w // 35)), max(22, min(w - 6, w // 4)))
            rh = rng.randint(2, max(5, h // 45 + int(5 * st + 4 * dmg)))
        else:
            rh = rng.randint(max(12, max(10, h // 32)), max(14, min(h - 8, h // 7)))
            rw = rng.randint(2, max(8, w // 52 + int(4 * st)))
        rw = max(2, min(rw, w))
        rh = max(2, min(rh, h))
        x = rng.randint(0, max(0, w - rw))
        y = rng.randint(0, max(0, h - rh))

        rr = rng.randint(118, 232)
        gg = rng.randint(0, 44)
        bb = rng.randint(4, 55)
        aa = rng.randint(alpha_lo, alpha_hi)
        if rng.random() < 0.22 + 0.1 * st:
            aa = int(aa * rng.uniform(0.35, 0.82))

        tile = pygame.Surface((rw, rh), pygame.SRCALPHA)
        tile.fill((rr, gg, bb, max(18, min(240, aa))))
        overlay.blit(tile, (x, y))


def _twig_crack_overlay(
    overlay,
    w: int,
    h: int,
    stress_level: int,
    dmg: float,
    damage_seed: int,
) -> None:
    """Sparse jagged fissures at mixed headings (mostly diagonal / oblique): variable width, rim + void."""

    sl = max(0, min(5, int(stress_level)))
    st = float(sl) / 5.0
    r = random.Random(damage_seed & 0xFFFFFFFF)

    # Scale stroke & steps to texture extent (fixes fixed-pixel cracks on arbitrary w×h surfaces).
    diag = math.hypot(float(max(1, w)), float(max(1, h)))
    ref_diag = math.hypot(480.0, 280.0)
    geom_scale = max(0.52, min(2.35, diag / ref_diag))

    xl = float(max(6.0, w * 0.032))
    xr = float(min(w - 7.0, w - max(6.0, w * 0.032)))
    yt = float(max(6.0, h * 0.054))
    yb = float(min(h - 7.0, h - max(6.0, h * 0.048)))

    n_cracks = max(3, min(14, int(4 + round(sl * 0.8) + int(dmg * 4.6))))
    w_tip = max(0.22, min(1.85, geom_scale * (0.28 + 0.14 * (1.0 - st) + 0.16 * dmg)))
    w_mid = max(0.5, min(5.2, geom_scale * max(0.62, min(3.25, 1.0 + 0.62 * dmg + 0.22 * (1.0 - st)))))
    alpha_core = max(176, min(249, int(232 - 16 * st + 12 * dmg)))
    alpha_shadow = max(74, min(148, int(104 - 9 * st + 20 * dmg)))
    alpha_rim = max(88, min(182, int(152 - 15 * st + 11 * dmg)))

    def width_along(t: float) -> float:
        """Narrow at tips, wider mid-length (hairline → gouge → hairline)."""
        d = abs(t - 0.5) * 2.0
        bulge = 1.0 - math.pow(d, 1.2)
        return w_tip + (w_mid - w_tip) * bulge

    def tangents_dense(strip_xyz: list[tuple[float, float, float]], i: int) -> tuple[float, float, float, float]:
        """Tangent + left normal along a densely sampled crack (same convention as tangents)."""
        n = len(strip_xyz)
        if n < 2:
            return 0.0, 1.0, -1.0, 0.0
        if i <= 0:
            vx = strip_xyz[1][0] - strip_xyz[0][0]
            vy = strip_xyz[1][1] - strip_xyz[0][1]
        elif i >= n - 1:
            vx = strip_xyz[-1][0] - strip_xyz[-2][0]
            vy = strip_xyz[-1][1] - strip_xyz[-2][1]
        else:
            vx = strip_xyz[i + 1][0] - strip_xyz[i - 1][0]
            vy = strip_xyz[i + 1][1] - strip_xyz[i - 1][1]
        leng = math.hypot(vx, vy)
        if leng < 1e-6:
            return 0.0, 1.0, -1.0, 0.0
        tx, ty = vx / leng, vy / leng
        nx, ny = -ty, tx
        return tx, ty, nx, ny

    def densify_with_t(
        orig: list[tuple[float, float]],
        max_step: float,
    ) -> list[tuple[float, float, float]]:
        """Insert points along each segment so stroke caps overlap—keeps linewidth taper via t∈[0,1]."""
        if len(orig) < 2:
            return [(orig[0][0], orig[0][1], 0.0)] if len(orig) == 1 else []
        denom = max(1, len(orig) - 1)
        out: list[tuple[float, float, float]] = []
        for i in range(len(orig) - 1):
            xa, ya = orig[i]
            xb, yb = orig[i + 1]
            t_a = i / denom
            t_b = (i + 1) / denom
            dx, dy = xb - xa, yb - ya
            seglen = math.hypot(dx, dy)
            if i == 0:
                out.append((xa, ya, t_a))
            elif seglen < 1e-6:
                out.append((xb, yb, t_b))
                continue
            n_sub = max(1, int(math.ceil(seglen / max_step)))
            for j in range(1, n_sub):
                u = j / float(n_sub)
                out.append((xa + dx * u, ya + dy * u, t_a * (1.0 - u) + t_b * u))
            out.append((xb, yb, t_b))
        return out

    def trace_fissure(
        x0: float,
        y0: float,
        phi_hint: float,
        phi_bounds: tuple[float, float],
    ) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
        """Random-walk cracks in direction space (phi = radians CCW from +x); sin(phi) feeds +y/down."""
        pts: list[tuple[float, float]] = [(x0, y0)]
        x, y = x0, y0
        pc_lo = max(-0.497 * math.pi, min(phi_bounds[0], phi_bounds[1]))
        pc_hi = min(1.497 * math.pi, max(phi_bounds[0], phi_bounds[1]))
        path_budget = diag * float(r.uniform(0.26, min(0.92, 0.55 + 0.32 * dmg + 0.08 * (1.0 - st))))
        travelled = 0.0
        gs = geom_scale
        max_steps = min(420, max(48, int(diag * 1.95)))
        phi_cur = float(phi_hint + r.gauss(0.0, 0.12 + 0.05 * dmg))
        phi_cur = max(pc_lo, min(pc_hi, phi_cur))

        branches: list[list[tuple[float, float]]] = []
        for _ in range(max_steps):
            if travelled >= path_budget:
                break
            phi_cur += float(r.gauss(0.0, 0.15 + 0.06 * (1.0 - st)))
            if r.random() < 0.11 + 0.05 * (1.0 - st):
                phi_cur += float(r.choice((-0.62, 0.62, -1.05, 1.05)))
            phi_cur = max(pc_lo, min(pc_hi, phi_cur))

            step = r.uniform(2.08, 7.82) * gs
            cand_x = x + math.cos(phi_cur) * step
            cand_y = y + math.sin(phi_cur) * step
            nx = max(xl, min(xr, cand_x))
            ny = max(yt, min(yb, cand_y))

            clipped = nx != cand_x or ny != cand_y
            if clipped:
                # Glance along bounding box edge instead of growing a vertical column stuck on margin.
                if nx == xr and math.cos(phi_cur) > 0.15:
                    phi_cur = math.pi + r.gauss(0.0, 0.09)
                elif nx == xl and math.cos(phi_cur) < -0.15:
                    phi_cur = r.gauss(0.0, 0.09)
                if ny == yb and math.sin(phi_cur) > 0.12:
                    phi_cur -= r.uniform(0.35, 0.92)
                elif ny == yt and math.sin(phi_cur) < -0.12:
                    phi_cur += r.uniform(0.35, 0.92)
                phi_cur = max(pc_lo, min(pc_hi, phi_cur))
                cand_x = x + math.cos(phi_cur) * step * 0.92
                cand_y = y + math.sin(phi_cur) * step * 0.92
                nx = max(xl, min(xr, cand_x))
                ny = max(yt, min(yb, cand_y))

            travelled += math.hypot(nx - x, ny - y)
            x, y = nx, ny
            pts.append((x, y))
            if len(pts) > 14 and r.random() < 0.026 + 0.012 * dmg:
                bx, by = x, y
                side = float(r.choice((-1.0, 1.0)))
                stub_phi = phi_cur + side * math.pi / 2.0 + r.gauss(0.0, 0.08)
                stub: list[tuple[float, float]] = [(bx, by)]
                vx, vy = bx, by
                for __ in range(r.randint(4, 11)):
                    step_s = r.uniform(1.6, 4.95) * gs
                    vx += math.cos(stub_phi) * step_s
                    vy += math.sin(stub_phi) * step_s
                    stub_phi += r.gauss(0.0, 0.2)
                    vx = max(xl, min(xr, vx))
                    vy = max(yt, min(yb, vy))
                    stub.append((vx, vy))
                if len(stub) >= 3:
                    branches.append(stub)
        return pts, branches

    def draw_polyline(pts: list[tuple[float, float]], width_scale: float) -> None:
        if len(pts) < 2:
            return
        max_step = max(1.15, geom_scale * 1.48)
        strip = densify_with_t(pts, max_step=max_step)
        nd = len(strip)
        if nd < 2:
            return
        # Fully joined silhouette: overlapping stroked segments (fixes gaps vs sparse circles-only).
        for i in range(nd - 1):
            x0, y0, t0 = strip[i]
            x1, y1, t1 = strip[i + 1]
            hw0 = width_along(t0) * width_scale
            hw1 = width_along(t1) * width_scale
            lw_core = max(1, int(round(max(hw0, hw1, 0.45) * 2.08)))
            lw_shadow = min(42, lw_core + 2 + int(round(geom_scale)))
            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))
            pygame.draw.line(overlay, (12, 11, 15, alpha_shadow), p0, p1, lw_shadow)
            pygame.draw.line(overlay, (0, 0, 0, alpha_core), p0, p1, lw_core)
        # Rounded joins + crevice rim (light asymmetric edge)—sample so corners stay welded.
        for i in range(nd):
            px, py, ti = strip[i]
            hw = width_along(ti) * width_scale
            _, _, nx, ny = tangents_dense(strip, i)
            ir = max(1, int(round(max(0.5, hw))))
            icx, icy = int(round(px)), int(round(py))
            pygame.draw.circle(overlay, (0, 0, 0, alpha_core), (icx, icy), ir)
            rim_r = max(1, int(round(max(0.45, hw * 0.42))))
            lrx = int(round(px + nx * (hw * 0.62 + 0.35)))
            lry = int(round(py + ny * (hw * 0.62 + 0.35)))
            pygame.draw.circle(overlay, (58, 54, 60, alpha_rim), (lrx, lry), rim_r)

    span_x = max(1.0, xr - xl)
    span_y_inner = max(1.0, yb - yt)
    for k in range(n_cracks):
        frac = (k + 0.52 + r.uniform(-0.17, 0.17)) / max(1.0, float(n_cracks))
        band_sh = span_y_inner * float(r.uniform(0.03, min(0.34, 0.14 + 0.065 * dmg)))
        edge_roll = r.random()
        if edge_roll < 0.41:
            # Top edge seed — headings fan down‑left through down‑right & shallow diagonals
            x0 = xl + span_x * frac
            x0 = max(xl + 3.5, min(xr - 3.5, x0))
            y0 = float(r.uniform(yt + 2.0, yt + band_sh))
            hint = float(r.gauss(math.pi * 0.52, math.pi * 0.29))
            bounds = (-0.12 * math.pi, 1.12 * math.pi)
        elif edge_roll < 0.58:
            # Bottom edge seed — headings mostly upward quadrant with spread
            x0 = xl + span_x * frac
            x0 = max(xl + 3.5, min(xr - 3.5, x0))
            y_lo = max(yt + 4.0, yb - band_sh)
            y_hi = yb - 3.5
            if y_lo > y_hi:
                y_lo, y_hi = yt + 10.0, yb - 6.5
            y0 = float(r.uniform(y_lo, y_hi))
            hint = float(r.gauss(-math.pi * 0.48, math.pi * 0.38))
            bounds = (-1.06 * math.pi, 0.18 * math.pi)
        elif edge_roll < 0.78:
            # Left edge — travel right-ish (diagonal + flat runs)
            yy_lo = yt + band_sh * 0.42
            yy_hi = max(yt + 10.0, yb - band_sh * 0.42)
            if yy_lo > yy_hi:
                yy_lo, yy_hi = yt + 8.0, yb - 10.0
            y0 = float(r.uniform(yy_lo, yy_hi))
            xh = min(xr - 6.5, max(xl + 8.0, xl + max(band_sh, 18.0 * geom_scale)))
            x0 = float(r.uniform(xl + 3.5, xh))
            hint = float(r.gauss(math.pi * 0.06, math.pi * 0.34))
            bounds = (-0.48 * math.pi, 0.48 * math.pi)
        else:
            # Right edge — travel left-ish
            yy_lo = yt + band_sh * 0.42
            yy_hi = max(yt + 10.0, yb - band_sh * 0.42)
            if yy_lo > yy_hi:
                yy_lo, yy_hi = yt + 8.0, yb - 10.0
            y0 = float(r.uniform(yy_lo, yy_hi))
            x_lo_seed = max(xl + span_x * 0.08, xr - max(band_sh, 18.0 * geom_scale))
            x0 = float(r.uniform(min(x_lo_seed, xr - 8.5), xr - 3.5))
            hint = float(r.gauss(math.pi * 0.94, math.pi * 0.34))
            bounds = (0.52 * math.pi, (2.0 - 0.46) * math.pi)
        main_pts, stubs = trace_fissure(x0, y0, hint, bounds)
        if len(main_pts) >= 5:
            draw_polyline(main_pts, 1.0)
        for stb in stubs:
            if len(stb) >= 3:
                draw_polyline(stb, 0.48 + 0.12 * (1.0 - st))


def apply_corruption(
    surf: pygame.Surface,
    corrosion_alpha: float,
    *,
    stress_level: int = 2,
    damage_seed: int = 7919,
) -> pygame.Surface:
    out = surf.copy()
    w, h = out.get_size()
    rng = random.Random(damage_seed & 0x7FFFFFFF)
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    dmg = _damage_strength(max(0.0, float(corrosion_alpha)), max(0, min(5, int(stress_level))))

    sl = max(0, min(5, int(stress_level)))
    _blood_rect_overlay(rng, overlay, w, h, sl, dmg)
    _twig_crack_overlay(overlay, w, h, sl, dmg, damage_seed)

    num_grime = max(0, min(8, int(2 + dmg * 8)))
    for __ in range(num_grime):
        gx = rng.randint(0, w - 1)
        gy = rng.randint(0, h - 1)
        sz = rng.randint(4, int(10 + 18 * dmg))
        grime_col = (25 + rng.randint(0, 20), 22 + rng.randint(0, 15), 15)
        grime_surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
        for gy2 in range(sz):
            for gx2 in range(sz):
                dist = math.hypot(gx2 - sz / 2, gy2 - sz / 2) / max(1e-6, sz / 2)
                if dist < 1.0 and rng.random() < 0.72:
                    ga = int(110 * dmg * (1.0 - dist))
                    grime_surf.set_at((gx2, gy2), (*grime_col, ga))
        overlay.blit(grime_surf, (gx, gy))

    out.blit(overlay, (0, 0))
    return out


def apply_texture_distortion_hq(
    surf: pygame.Surface,
    level: int,
    corruption: float,
    grime: float,
    tick: int,
    *,
    variation_seed: int = 0,
) -> pygame.Surface:
    """Neutral grime + faint dark scratches only (no hue / tint overlays)."""
    out = surf.copy()
    w, h = out.get_size()
    rng = random.Random(
        (((tick // 3) * 41 + level * 997) ^ (variation_seed * 83492791)) & 0xFFFFFFFF
    )

    if grime > 0.05:
        grain = pygame.Surface((w, h), pygame.SRCALPHA)
        cap = max(420, min(9000, int(w * h * (0.00045 + 0.002 * grime))))
        for _ in range(cap):
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)
            v = rng.randint(6, 24)
            a = rng.randint(3, 14)
            grain.set_at((x, y), (v, v, v, a))
        out.blit(grain, (0, 0))

    scratches = pygame.Surface((w, h), pygame.SRCALPHA)
    n_scr = max(1, min(22, int(2 + level * 1 + corruption * 8)))
    lvl = float(max(0, min(5, int(level))))
    for _ in range(n_scr):
        sx = rng.randint(0, w - 1)
        sy = rng.randint(0, h - 1)
        ln = rng.randint(16, int(36 + 70 * corruption))
        ang = rng.uniform(-0.9, 0.9)
        ex = max(0, min(w - 1, int(sx + math.cos(ang) * ln)))
        ey = max(0, min(h - 1, int(sy + math.sin(ang) * ln)))
        fade = max(14, min(140, int(16 + 70 * corruption * (lvl / 5.0 + 0.2))))
        pygame.draw.line(
            scratches,
            (12 + rng.randint(0, 8), 10, 10, fade),
            (sx, sy),
            (ex, ey),
            1,
        )
    out.blit(scratches, (0, 0))

    return out


def apply_fog(surf: pygame.Surface, density: float, tick: int) -> pygame.Surface:
    out = surf.copy()
    w, h = out.get_size()
    fog_layer = pygame.Surface((w, h), pygame.SRCALPHA)
    fog_a = int(density * 200)
    drift = math.sin(tick * 0.05) * 20
    for band in range(0, h, 8):
        local_a = max(0, min(255, int(fog_a * (0.5 + 0.5 * math.sin((band + drift) * 0.03)))))
        pygame.draw.rect(fog_layer, (160, 165, 175, local_a), (0, band, w, 8))
    out.blit(fog_layer, (0, 0))
    return out


def apply_light_flicker(surf: pygame.Surface, flicker_hz: float, tick: int) -> pygame.Surface:
    phase = math.sin(tick * flicker_hz * 0.15)
    if phase < -0.6:
        dark = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        dark.fill((0, 0, 0, int(40 + 60 * abs(phase))))
        out = surf.copy()
        out.blit(dark, (0, 0))
        return out
    return surf


def apply_camera_distortion(surf: pygame.Surface, magnitude: float, tick: int) -> pygame.Surface:
    mag = magnitude * 0.5
    out = surf.copy()
    w, h = out.get_size()

    if mag > 0.05:
        vignette = pygame.Surface((w, h), pygame.SRCALPHA)
        border = int(10 + mag * 40)
        alpha_val = int(40 + 80 * mag)
        for i in range(border):
            a = int(alpha_val * (1.0 - i / border))
            pygame.draw.rect(vignette, (0, 0, 0, a), (i, i, w - 2 * i, h - 2 * i), 1)
        out.blit(vignette, (0, 0))

    if mag > 0.15:
        rng = random.Random((tick // 4) * 17)
        num_noise = int(mag * 150)
        for _ in range(num_noise):
            nx = rng.randint(0, w - 1)
            ny = rng.randint(0, h - 1)
            v = rng.randint(100, 200)
            a = int(25 * mag)
            noise_surf = pygame.Surface((2, 2), pygame.SRCALPHA)
            noise_surf.fill((v, v, v, a))
            out.blit(noise_surf, (nx, ny))

    if mag > 0.35:
        rng2 = random.Random((tick // 6) * 23)
        num_lines = int(mag * 3)
        for _ in range(num_lines):
            ly = rng2.randint(0, h - 1)
            lx_shift = rng2.randint(-int(mag * 6), int(mag * 6))
            strip_h = rng2.randint(1, 2)
            strip = out.subsurface(pygame.Rect(0, max(0, ly), w, min(strip_h, h - ly))).copy()
            out.blit(strip, (lx_shift, ly))

    return out


def apply_camera_distortion_advanced(
    surf: pygame.Surface,
    magnitude: float,
    aberration: float,
    noise: float,
    vignette: float,
    warp: float,
    tick: int,
) -> pygame.Surface:
    """Apply camera-only lens artifacts so this panel differs from texture corruption."""
    out = apply_camera_distortion(surf, magnitude, tick)
    w, h = out.get_size()

    if aberration > 0.05:
        shift = max(1, int(2 + aberration * 6))
        arr = pygame.surfarray.array3d(out)
        rb = np.roll(arr[:, :, 0], shift, axis=0)
        bb = np.roll(arr[:, :, 2], -shift, axis=1)
        arr[:, :, 0] = rb
        arr[:, :, 2] = bb
        out = pygame.surfarray.make_surface(arr)

    if warp > 0.05:
        wave = max(1, int(2 + warp * 8))
        warped = pygame.Surface((w, h))
        for y in range(h):
            offset = int(math.sin((y * 0.04) + tick * 0.08) * wave)
            src_rect = pygame.Rect(0, y, w, 1)
            warped.blit(out, (offset, y), src_rect)
        out = warped

    if noise > 0.05:
        rng = random.Random((tick // 2) * 101)
        grain = pygame.Surface((w, h), pygame.SRCALPHA)
        px_count = int(w * h * min(0.04, noise * 0.06))
        for _ in range(px_count):
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)
            v = rng.randint(80, 210)
            a = int(20 + noise * 40)
            grain.set_at((x, y), (v, v, v, a))
        out.blit(grain, (0, 0))

    if vignette > 0.02:
        vig = pygame.Surface((w, h), pygame.SRCALPHA)
        border = int(15 + vignette * 80)
        max_a = int(30 + vignette * 130)
        for i in range(border):
            a = int(max_a * (1.0 - i / max(1, border)))
            pygame.draw.rect(vig, (0, 0, 0, a), (i, i, w - 2 * i, h - 2 * i), 1)
        out.blit(vig, (0, 0))

    return out


def generate_camera_proxy_scene(width: int, height: int, level: int, tick: int) -> pygame.Surface:
    """Render a non-texture camera view so camera panel is independent from environment textures."""
    surf = pygame.Surface((width, height))
    base = 18 + level * 6
    for y in range(height):
        g = max(0, min(255, base + int(28 * (y / max(1, height - 1)))))
        pygame.draw.line(surf, (g, g, g + 4), (0, y), (width, y))

    # Hallway rails and vanishing point.
    vp_x = width // 2 + int(math.sin(tick * 0.03) * (4 + level))
    vp_y = int(height * 0.34)
    rail_col = (70 + level * 10, 70 + level * 8, 76 + level * 6)
    pygame.draw.line(surf, rail_col, (0, height - 1), (vp_x, vp_y), 2)
    pygame.draw.line(surf, rail_col, (width - 1, height - 1), (vp_x, vp_y), 2)
    pygame.draw.line(surf, (55, 55, 62), (0, int(height * 0.63)), (width, int(height * 0.63)), 2)

    # Flickering practical lights.
    rng = random.Random((tick // 3) * 19 + level * 17)
    for i in range(4):
        lx = int(width * (0.2 + 0.2 * i))
        ly = int(height * 0.18 + math.sin(tick * 0.08 + i) * 3)
        glow = 90 + int(60 * abs(math.sin(tick * 0.11 + i + level * 0.2)))
        pygame.draw.circle(surf, (glow, glow, min(255, glow + 20)), (lx, ly), 6)
        if rng.random() < 0.08 + level * 0.02:
            pygame.draw.circle(surf, (255, 255, 255), (lx, ly), 4)

    # Passing shadow figure in the fake camera scene.
    if level >= 2:
        sx = int((tick * (1 + level * 0.2)) % (width + 60)) - 30
        sy = int(height * 0.66)
        sh = 55 + level * 8
        sw = 18 + level * 2
        shad = pygame.Surface((sw, sh), pygame.SRCALPHA)
        shad.fill((0, 0, 0, 70 + level * 20))
        surf.blit(shad, (sx, sy - sh))

    return surf


class EntityMeshRenderer3D:
    """Offscreen Panda3D PBR renderer for a single GLB / GLTF mesh.

    The demo picks exactly ONE mesh from ``new entities/`` per run, and this class
    renders it to a ``pygame.Surface`` each frame so the existing pygame UI can
    blit it inside the entity panel. The Panda3D ``ShowBase`` runs in
    ``window-type offscreen`` mode so there is no second OS window.
    """

    _instance: "EntityMeshRenderer3D | None" = None

    def __init__(
        self,
        mesh_path: Path,
        width: int = 320,
        height: int = 320,
        target_size: float = 2.5,
    ) -> None:
        if EntityMeshRenderer3D._instance is not None:
            raise RuntimeError("only one EntityMeshRenderer3D may exist per process")
        if not _PANDA3D_AVAILABLE:
            raise RuntimeError(f"panda3d unavailable: {_PANDA3D_IMPORT_ERROR}")

        _panda_load_prc("", "window-type offscreen")
        _panda_load_prc("", "audio-library-name null")
        _panda_load_prc("", "notify-level-display fatal")
        _panda_load_prc("", "notify-level-glgsg fatal")
        _panda_load_prc("", "notify-level-gobj fatal")
        _panda_load_prc("", f"win-size {width} {height}")
        _panda_load_prc("", "framebuffer-multisample 1")
        _panda_load_prc("", "multisamples 4")
        _panda_load_prc("", "sync-video false")

        self.width = int(width)
        self.height = int(height)
        self.mesh_path = Path(mesh_path)
        self.target_size = float(target_size)

        self.base = _PandaShowBase()
        self.base.disableMouse()

        self.pbr = None
        self._using_pbr = False
        try:
            self.pbr = _panda_simplepbr.init(
                window=self.base.win,
                use_normal_maps=True,
                enable_shadows=False,
                max_lights=4,
            )
            self._using_pbr = True
        except Exception as exc:
            print(f"[entity-3d] simplepbr init failed ({exc}); using auto-shader")
            try:
                self.base.render.setShaderAuto()
            except Exception as exc2:
                print(f"[entity-3d] shader-auto fallback failed: {exc2}")

        self.base.render.setAntialias(_PAntialiasAttrib.MMultisample)
        self.bg_rgb = (0.045, 0.035, 0.050)
        self.base.setBackgroundColor(*self.bg_rgb, 1.0)

        amb = _PAmbientLight("amb")
        amb.setColor(_PVec4(0.22, 0.20, 0.24, 1.0))
        self.amb_np = self.base.render.attachNewNode(amb)
        self.base.render.setLight(self.amb_np)

        self.key_np = self._add_dir_light("key", _PVec4(1.05, 0.92, 0.78, 1.0), (40, -45, 0))
        self.fill_np = self._add_dir_light("fill", _PVec4(0.35, 0.30, 0.55, 1.0), (-130, -20, 0))
        self.rim_np = self._add_dir_light("rim", _PVec4(0.75, 0.40, 0.55, 1.0), (170, -15, 0))

        settings = _panda_gltf.GltfSettings(skip_animations=True)
        root = _panda_gltf.load_model(str(self.mesh_path), gltf_settings=settings)
        self.model = _PNodePath(root)
        self.model.reparentTo(self.base.render)
        self._normalize_model()

        self.base.camLens.setFov(46)
        self.base.camLens.setAspectRatio(self.width / max(1.0, float(self.height)))
        self.base.camLens.setNearFar(0.05, 200.0)
        self.cam_target = _PVec3(0.0, 0.0, self._model_height * 0.50)
        self.cam_distance = max(self.target_size, self._model_height) * 1.55
        # Front-on camera. Keep pitch flat and yaw locked so the mesh reads like
        # a posed character in front of the viewer, not an orbiting showcase.
        self.yaw = 0.0
        self.pitch = 0.0
        # User-driven camera overrides. When the user drags the entity panel,
        # ``user_camera_active`` is True and these offsets are applied on top
        # of whatever yaw the demo passes to ``render_surface_with_mask``. A
        # right-click resets them to the auto pose.
        self._base_cam_distance = float(self.cam_distance)
        self.manual_yaw_offset = 0.0
        self.manual_pitch = 0.0
        self.manual_zoom = 1.0
        self.user_camera_active = False
        self._set_cam()

        for _ in range(3):
            self.base.taskMgr.step()

        EntityMeshRenderer3D._instance = self

    def _add_dir_light(self, name, color, hpr):
        light = _PDirectionalLight(name)
        light.setColor(color)
        np_ = self.base.render.attachNewNode(light)
        np_.setHpr(*hpr)
        self.base.render.setLight(np_)
        return np_

    def _normalize_model(self) -> None:
        """Scale, auto-orient, and floor the model.

        Do not treat the longest axis as "up": for dinosaurs / creatures the
        longest axis is usually head-to-tail, which made the model stand on its
        tail and face upward. Instead, score a few common GLB axis conversions
        and choose the one where Z is tall enough to stand, but not so tall that
        the whole body length has become vertical.
        """
        self._model_height = self.target_size
        bnd = self.model.getTightBounds()
        if not bnd:
            return
        bmin, bmax = bnd
        size = bmax - bmin
        diag = max(size.length(), 1e-6)
        self.model.setScale(self.target_size / diag)

        base_axis_candidates = [
            (0.0, 0.0),       # already Panda / Z-up
            (-90.0, 0.0),     # common glTF Y-up -> Z-up conversion
            (90.0, 0.0),
            (0.0, -90.0),
            (0.0, 90.0),
        ]
        # Test fixed headings too. Panda's camera sits at -Y looking toward +Y;
        # combined with the -90 pitch (Y-up -> Z-up), heading 0 puts the
        # authored GLB front face toward the camera. Heading 180 turns the
        # creature's back to us (which is what the previous bias produced).
        heading_candidates = [0.0, 180.0, 90.0, -90.0]
        candidates = [
            (heading, pitch, roll)
            for pitch, roll in base_axis_candidates
            for heading in heading_candidates
        ]

        # Pre-pass: measure every candidate so we can decide whether this mesh
        # is a tall humanoid (lich / cameraman) or a horizontal creature
        # (velociraptor / distortus_rex). The original scorer punished any
        # height_ratio > 1.15 to stop dinosaurs from standing on their tails,
        # which also crushed the natural upright pose for slim characters.
        candidate_metrics: list[tuple] = []
        for hpr in candidates:
            self.model.setHpr(*hpr)
            bnd_c = self.model.getTightBounds()
            if not bnd_c:
                continue
            bmin_c, bmax_c = bnd_c
            s = bmax_c - bmin_c
            sx, sy, sz = max(abs(s.x), 1e-6), max(abs(s.y), 1e-6), max(abs(s.z), 1e-6)
            horizontal_long = max(sx, sy)
            horizontal_short = min(sx, sy)
            height_ratio = sz / max(horizontal_long, 1e-6)
            width_ratio = horizontal_long / max(horizontal_short, 1e-6)
            candidate_metrics.append(
                (hpr, horizontal_long, horizontal_short, sz, height_ratio, width_ratio)
            )

        max_height_ratio = max((m[4] for m in candidate_metrics), default=1.0)
        # Three regimes based on the maximum achievable height/width ratio
        # across every rotation candidate:
        #   - >= 2.5  → slim humanoid (one dim dominates; rare for our meshes
        #               because skeletons inflate the bounds box).
        #   - <= 1.55 → near-cubic humanoid (lich / cameraman). All dims are
        #               within ~30% of each other; the model is a character
        #               whose arms+body+rig give it a roughly cubic bound. We
        #               want the longest authored axis vertical.
        #   -  in between → quadruped / horizontal creature (velociraptor,
        #               distortus_rex). Keep the original dinosaur-friendly
        #               scoring with tail-stand penalty.
        shape: str
        if max_height_ratio >= 2.5:
            shape = "slim_humanoid"
        elif max_height_ratio <= 1.55:
            shape = "cubic_humanoid"
        else:
            shape = "horizontal_creature"

        best_hpr = candidates[0]
        best_score = float("inf")
        for hpr, h_long, h_short, sz, height_ratio, width_ratio in candidate_metrics:
            if shape == "slim_humanoid":
                upright_bonus = -1.6 * (height_ratio / max_height_ratio)
                lying_penalty = max(0.0, 0.55 - height_ratio) * 6.0
                cramped_penalty = max(0.0, 1.05 - width_ratio) * 0.3
                score = upright_bonus + lying_penalty + cramped_penalty
            elif shape == "cubic_humanoid":
                # All dims similar. Reward the orientation where height
                # equals max attainable (the longest authored axis vertical).
                # No tail-stand worry because the dims are nearly the same;
                # the tallest pose IS the natural upright pose.
                upright_bonus = -1.4 * (height_ratio / max_height_ratio)
                flat_penalty = max(0.0, 0.55 - height_ratio) * 5.0
                score = upright_bonus + flat_penalty
            else:
                # Horizontal creature: prefer body length lying horizontally
                # with moderate height (not lying flat, not on its tail).
                flat_penalty = max(0.0, 0.36 - height_ratio) * 7.0
                tail_stand_penalty = max(0.0, height_ratio - 1.15) * 9.0
                cramped_penalty = max(0.0, 1.18 - width_ratio) * 0.7
                score = flat_penalty + tail_stand_penalty + cramped_penalty

            side_view_penalty = 0.12 if abs(abs(hpr[0]) - 90.0) < 1e-3 else 0.0
            back_penalty = 0.30 if abs(abs(hpr[0]) - 180.0) < 1e-3 else 0.0
            front_bonus = -0.10 if abs(hpr[0]) < 1e-3 else 0.0
            score += side_view_penalty + back_penalty + front_bonus

            # Only nudge toward authored axis for quadrupeds — for humanoids
            # the default axes are often the lying-down pose.
            if shape == "horizontal_creature" and (
                hpr == (0.0, -90.0, 0.0) or hpr == (0.0, 0.0, 0.0)
            ):
                score -= 0.08
            if score < best_score:
                best_score = score
                best_hpr = hpr

        self.model.setHpr(*best_hpr)

        # Re-measure post-rotation so the floor + center are correct.
        bnd2 = self.model.getTightBounds()
        if bnd2:
            bmin2, bmax2 = bnd2
            center = (bmin2 + bmax2) * 0.5
            self.model.setPos(-center.x, -center.y, -bmin2.z)
            self._model_height = float(bmax2.z - bmin2.z)
        print(
            f"[entity-3d] auto-oriented {self.mesh_path.name} hpr={self.model.getHpr()} "
            f"final_height={self._model_height:.2f} "
            f"shape={shape} (max_height_ratio={max_height_ratio:.2f})"
        )

    def _set_cam(self) -> None:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        r = self.cam_distance
        cp = math.cos(pitch)
        self.base.camera.setPos(
            self.cam_target.x - r * cp * math.sin(yaw),
            self.cam_target.y - r * cp * math.cos(yaw),
            self.cam_target.z + r * math.sin(pitch),
        )
        self.base.camera.lookAt(self.cam_target)

    def render_surface(self, yaw_deg: float, intensity: float = 1.0) -> "pygame.Surface | None":
        """Render one frame and return an RGBA pygame.Surface."""
        result = self.render_surface_with_mask(yaw_deg, intensity)
        return result[0] if result is not None else None

    def render_surface_with_mask(
        self,
        yaw_deg: float,
        intensity: float = 1.0,
    ) -> "tuple[pygame.Surface, np.ndarray] | None":
        """Render one frame and return (surface, model_mask) where mask is a uint8
        ``np.ndarray`` of shape (H, W) with 255 on the model and 0 on the background.
        """
        self.yaw = float(yaw_deg) + float(self.manual_yaw_offset)
        self.pitch = float(self.manual_pitch) if self.user_camera_active else 0.0
        self.cam_distance = self._base_cam_distance * float(self.manual_zoom)
        self._set_cam()
        k = 0.55 + 0.85 * max(0.0, min(1.0, intensity))
        self.key_np.node().setColor(_PVec4(k * 1.05, k * 0.92, k * 0.78, 1.0))
        rim = 0.45 + 0.55 * max(0.0, min(1.0, intensity))
        self.rim_np.node().setColor(_PVec4(rim * 1.25, rim * 0.55, rim * 0.85, 1.0))

        self.base.taskMgr.step()
        self.base.taskMgr.step()

        tex = self.base.win.getScreenshot()
        if tex is None:
            return None
        ram = tex.getRamImageAs("RGBA")
        if not ram:
            return None
        data = bytes(ram)
        w, h = tex.getXSize(), tex.getYSize()
        try:
            surf = pygame.image.frombuffer(data, (w, h), "RGBA")
        except Exception:
            return None
        surf = pygame.transform.flip(surf, False, True)

        arr = np.frombuffer(pygame.image.tobytes(surf, "RGB"), dtype=np.uint8)
        arr = arr.reshape(h, w, 3).astype(np.int16)

        cs = max(1, min(8, w // 16, h // 16))
        corner_samples = np.concatenate([
            arr[0:cs, 0:cs].reshape(-1, 3),
            arr[0:cs, -cs:].reshape(-1, 3),
            arr[-cs:, 0:cs].reshape(-1, 3),
            arr[-cs:, -cs:].reshape(-1, 3),
        ], axis=0)
        bg = np.median(corner_samples, axis=0).astype(np.int16)

        diff = np.abs(arr - bg).max(axis=2)
        mask = (diff > 22).astype(np.uint8)

        if mask.shape[0] > 4 and mask.shape[1] > 4:
            eroded = mask.copy()
            eroded[1:] &= mask[:-1]
            eroded[:-1] &= mask[1:]
            eroded[:, 1:] &= mask[:, :-1]
            eroded[:, :-1] &= mask[:, 1:]
            mask = eroded

        mask = (mask * 255).astype(np.uint8)
        return surf, mask

    def adjust_user_camera(
        self,
        *,
        dyaw: float = 0.0,
        dpitch: float = 0.0,
        dzoom: float = 0.0,
    ) -> None:
        """Mutate the user's yaw/pitch offsets and zoom factor.

        Called from the demo's mouse-drag handler. Angles are degrees, zoom
        is a multiplicative delta (positive zooms out, negative zooms in).
        Pitch is clamped to avoid flipping the camera through the poles.
        """
        if dyaw or dpitch or dzoom:
            self.user_camera_active = True
        if dyaw:
            self.manual_yaw_offset = (self.manual_yaw_offset + float(dyaw)) % 360.0
        if dpitch:
            self.manual_pitch = float(
                max(-80.0, min(80.0, self.manual_pitch + float(dpitch)))
            )
        if dzoom:
            self.manual_zoom = float(
                max(0.35, min(3.5, self.manual_zoom * (1.0 + float(dzoom))))
            )

    def reset_user_camera(self) -> None:
        """Drop user overrides and return to the auto-computed pose."""
        self.manual_yaw_offset = 0.0
        self.manual_pitch = 0.0
        self.manual_zoom = 1.0
        self.user_camera_active = False

    def shutdown(self) -> None:
        try:
            self.base.destroy()
        except Exception:
            pass
        EntityMeshRenderer3D._instance = None


class EntityDamageGenerator:
    """Blood + cracks that read as part of the surface, not pasted decals.

    Blood is a continuous float density field that **morphs every frame** via
    diffusion, masked gravity flow, and lateral smear while staying confined to
    the silhouette. It is composited with multiply-style darkening + tint into
    the rendered RGBA buffer. Cracks are drawn as vector strokes then softened.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed if seed is not None else random.getrandbits(31))
        self.crack_layer: pygame.Surface | None = None
        self._blood_field: np.ndarray | None = None
        self.layer_size: tuple[int, int] = (0, 0)
        self.last_level: int = -1
        self.last_seed: int = -1

    def regenerate(self, width: int, height: int, level: int, seed: int) -> None:
        """Rebuild crack decals and re-seed the morphing blood field."""
        self.layer_size = (width, height)
        self.last_level = int(level)
        self.last_seed = int(seed)
        self.rng = random.Random(seed)
        h, w = height, width
        if level >= 3:
            self._blood_field = self._seed_blood_field(w, h, level)
        else:
            self._blood_field = np.zeros((h, w), dtype=np.float32)
        self.crack_layer = self._make_crack_layer(w, h, level)

    def _ensure_size(self, width: int, height: int, level: int) -> None:
        if (
            self._blood_field is None
            or self.crack_layer is None
            or self.layer_size != (width, height)
            or self.last_level != level
        ):
            self.regenerate(width, height, level, self.rng.getrandbits(31))

    @staticmethod
    def _damage_amounts(level: int) -> tuple[int, int]:
        """Return (num_blood_seeds, num_cracks) for a stress level.

        Damage now begins at level 3 (faint scuffs + a few weeping spots) so the
        transition from "wary" to "hurt" reads visually on-screen rather than
        flipping on at level 4. Levels 4 and 5 are much more present than before.
        """
        if level <= 2:
            return 0, 0
        if level == 3:
            return 6, 3
        if level == 4:
            return 18, 7
        return 34, 14

    def _seed_blood_field(self, w: int, h: int, level: int) -> np.ndarray:
        """Sparse soft Gaussian seeds — the morph step turns these into streaks.

        A mix of small "weeping" seeds and a few large "deep wounds" reads
        much better than a uniform spray; the morph step then drips, smears,
        and re-pools them every frame.
        """
        try:
            from scipy import ndimage
        except Exception:
            ndimage = None
        field = np.zeros((h, w), dtype=np.float32)
        if level < 3:
            return field
        n_seeds, _ = self._damage_amounts(level)
        yy, xx = np.ogrid[:h, :w]
        # Mix of small + medium + large droplets. Roughly 60% small, 30% medium,
        # 10% deep wounds. Higher levels get bigger amplitudes and a wider area.
        for i in range(n_seeds):
            cx = self.rng.randint(int(w * 0.08), int(w * 0.92))
            cy = self.rng.randint(int(h * 0.10), int(h * 0.78))
            roll = self.rng.random()
            if roll < 0.60:
                sig = float(self.rng.uniform(min(w, h) * 0.012, min(w, h) * 0.030))
                amp = float(self.rng.uniform(0.45, 0.95)) * (0.7 + 0.10 * level)
            elif roll < 0.90:
                sig = float(self.rng.uniform(min(w, h) * 0.030, min(w, h) * 0.055))
                amp = float(self.rng.uniform(0.95, 1.45)) * (0.7 + 0.10 * level)
            else:
                sig = float(self.rng.uniform(min(w, h) * 0.055, min(w, h) * 0.085))
                amp = float(self.rng.uniform(1.55, 2.20)) * (0.7 + 0.10 * level)
            field += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig + 1e-6))
        if ndimage is not None:
            sig0 = max(0.8, min(w, h) * 0.0055)
            field = ndimage.gaussian_filter(field, sigma=sig0)
        return np.clip(field, 0.0, 16.0)

    def _make_crack_layer(self, w: int, h: int, level: int) -> pygame.Surface:
        layer = pygame.Surface((w, h), pygame.SRCALPHA)
        if level < 3:
            return layer
        _, n_cracks = self._damage_amounts(level)
        # Bigger panel + more cracks: spread across body with variable lengths.
        for _ in range(n_cracks):
            ox = self.rng.randint(int(w * 0.10), int(w * 0.90))
            oy = self.rng.randint(int(h * 0.10), int(h * 0.88))
            n_branches = self.rng.randint(3, 6)
            base_len = int(min(w, h) * (0.10 + 0.04 * max(0, level - 3)))
            for _ in range(n_branches):
                self._draw_crack_branch(
                    layer,
                    ox,
                    oy,
                    angle=self.rng.uniform(0, 2 * math.pi),
                    length=self.rng.randint(base_len, int(base_len * 2.4)),
                    depth=2 + max(0, level - 3),
                    level=level,
                )
        try:
            from scipy import ndimage

            buf = pygame.image.tobytes(layer, "RGBA")
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
            a = ndimage.gaussian_filter(arr[..., 3].astype(np.float32), sigma=1.15)
            arr[..., 3] = np.clip(a, 0, 255).astype(np.uint8)
            arr[..., :3] = np.clip(
                ndimage.gaussian_filter(arr[..., :3].astype(np.float32), sigma=0.55),
                0,
                255,
            ).astype(np.uint8)
            layer = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA")
        except Exception:
            pass
        return layer

    def _draw_crack_branch(
        self, surf: pygame.Surface, x: int, y: int, angle: float, length: int,
        depth: int, level: int = 4,
    ) -> None:
        if depth <= 0 or length <= 2:
            return
        steps = max(4, length // 3)
        seg_len = max(1.0, length / steps)
        pts = [(x, y)]
        cur_a = angle
        for _ in range(steps):
            cur_a += self.rng.uniform(-0.55, 0.55)
            x += math.cos(cur_a) * seg_len
            y += math.sin(cur_a) * seg_len
            pts.append((int(x), int(y)))
        # Heavier strokes + slightly bloody undertone read much better against
        # the entity's body than the old near-pure-black hairlines.
        outer_alpha = min(255, 195 + 12 * (level - 3))
        inner_alpha = min(255, 230 + 6 * (level - 3))
        outer_w = 4 if level >= 4 else 3
        inner_w = 2 if level >= 5 else 1
        outer = (4, 3, 5, outer_alpha)
        inner = (78, 22, 26, inner_alpha)
        try:
            pygame.draw.lines(surf, outer, False, pts, outer_w)
            pygame.draw.lines(surf, inner, False, pts, inner_w)
        except ValueError:
            return
        if depth > 1 and len(pts) > 3:
            bx, by = pts[len(pts) // 2]
            self._draw_crack_branch(
                surf,
                bx,
                by,
                angle=cur_a + self.rng.uniform(-1.0, 1.0),
                length=max(4, length // 2),
                depth=depth - 1,
                level=level,
            )

    def _morph_blood_field(
        self,
        bf: np.ndarray,
        mask_np: np.ndarray,
        tick: int,
        level: int,
        ndimage,
    ) -> np.ndarray:
        """One simulation step: diffuse, drip, smear, confine to body.

        Bigger gravity flow + slower decay than before so you can actually
        watch drops trail down the model, plus occasional micro-spawns to keep
        the field evolving instead of converging to a static stain.
        """
        m = (mask_np.astype(np.float32) / 255.0).clip(0.0, 1.0)
        m_soft = ndimage.gaussian_filter(m, sigma=1.4)
        bf = ndimage.gaussian_filter(bf, sigma=0.55)
        t = float(tick) * 0.027

        # Vertical gravity drip — fractional shift creates sub-pixel motion that
        # reads as continuous flow. Heavier at higher stress.
        tier = max(0.0, min(1.0, (level - 2) / 3.0))
        dy = 0.85 + 0.45 * tier + 0.20 * math.sin(t * 1.05 + float(level) * 0.31)
        dx = 0.22 * math.sin(t * 0.72) + 0.08 * math.sin(t * 0.35 + bf.shape[1] * 0.02)
        bf = ndimage.shift(bf, shift=(dy, dx), order=1, mode="constant", cval=0.0)
        bf *= np.clip(0.18 + 0.82 * m_soft, 0.0, 1.0)

        # Lateral smear — drops widen sideways while flowing, like real surface tension.
        lateral = ndimage.gaussian_filter(bf, sigma=(0.42, 2.4))
        bf = np.maximum(bf * 0.94, lateral * m_soft * 0.34)

        # Slow ambient decay so streaks linger but eventually fade.
        bf *= 0.985

        # Micro-spawn: every ~25 ticks at level 4 / ~14 at level 5, add a small
        # fresh weep so the morph keeps changing instead of flattening out.
        spawn_period = 25 if level == 4 else 14 if level >= 5 else 999
        if level >= 4 and (tick % spawn_period) == (level * 7) % spawn_period:
            h, w = bf.shape
            for _ in range(1 + (level - 3)):
                cx = self.rng.randint(int(w * 0.12), int(w * 0.88))
                cy = self.rng.randint(int(h * 0.10), int(h * 0.55))
                sig = float(self.rng.uniform(min(w, h) * 0.012, min(w, h) * 0.026))
                amp = float(self.rng.uniform(0.55, 1.20))
                yy, xx = np.ogrid[:h, :w]
                spawn = amp * np.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig + 1e-6)
                )
                spawn *= m
                bf = np.maximum(bf, spawn.astype(np.float32))

        bf *= np.clip(0.28 + 0.72 * m_soft, 0.0, 1.0)
        return np.clip(bf, 0.0, 10.0)

    def composite(
        self,
        target: pygame.Surface,
        mask_np: "np.ndarray",
        level: int,
        tick: int,
    ) -> None:
        """Morph blood into ``target`` RGB, then overlay softened cracks.

        Color model: darken the base by ``wet`` (so the wound looks bloody and
        damp, not just tinted), drive R up + G/B down for arterial red, and
        add a sub-pixel specular highlight at the gradient peak so droplets
        read as glossy liquid instead of a paint stain. Cracks are masked to
        the silhouette so they never bleed off the model.
        """
        if level < 3:
            return
        w, h = target.get_size()
        self._ensure_size(w, h, level)

        if mask_np.shape[1] != w or mask_np.shape[0] != h:
            try:
                from PIL import Image

                m_img = Image.fromarray(mask_np)
                m_img = m_img.resize((w, h), Image.BILINEAR)
                mask_np = np.array(m_img, dtype=np.uint8)
            except Exception:
                mask_np = np.ones((h, w), dtype=np.uint8) * 255

        if self._blood_field is None or self._blood_field.shape != (h, w):
            return

        try:
            from scipy import ndimage
        except Exception:
            ndimage = None

        m = (mask_np.astype(np.float32) / 255.0).clip(0.0, 1.0)
        bf = self._blood_field.astype(np.float32)

        # Level scaling for how "loud" the blood reads. Level 3 is subtle
        # weeping; level 4 is bloody; level 5 is graphic.
        level_gain = 0.55 if level == 3 else 0.95 if level == 4 else 1.35

        if ndimage is not None:
            bf = self._morph_blood_field(bf, mask_np, tick, level, ndimage)
            self._blood_field = bf
            m_soft = ndimage.gaussian_filter(m, sigma=1.1)
            st = np.clip(bf * m * 0.86 * level_gain, 0.0, 1.0)
            wet = np.clip(st * 1.45, 0.0, 1.0)
            # Gradient peaks = where the surface curves quickly — those are
            # where wet liquid catches the light. We brighten those tiny dots.
            lowfreq = ndimage.gaussian_filter(bf, sigma=4.5)
            peaks = np.clip((bf - lowfreq) * m_soft * 2.6, 0.0, 1.0)
        else:
            m_soft = m
            bf = np.roll(bf, 1, axis=0) * 0.94
            bf *= m
            self._blood_field = np.clip(bf, 0, 10)
            st = np.clip(bf * m * 0.86 * level_gain, 0.0, 1.0)
            wet = np.clip(st * 1.45, 0.0, 1.0)
            peaks = np.clip(st * 0.40, 0.0, 1.0)

        buf = pygame.image.tobytes(target, "RGBA")
        targ = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
        rgb = targ[..., :3].astype(np.float32)
        # 1) Darken under the wet area (blood is darker than flesh)
        rgb *= 1.0 - 0.72 * wet[..., None]
        # 2) Push toward arterial red
        rgb[..., 0] += 68.0 * wet
        rgb[..., 1] -= 8.0 * wet
        rgb[..., 2] -= 6.0 * wet
        # 3) Concentrated red at the drop peaks
        rgb[..., 0] += 36.0 * peaks
        # 4) Specular highlight — a small bright dot near the drip peak makes it
        #    read as a glossy wet surface rather than a paint smear.
        spec = (peaks ** 1.6) * 0.85
        rgb[..., 0] += 28.0 * spec
        rgb[..., 1] += 12.0 * spec
        rgb[..., 2] += 12.0 * spec
        targ[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out = pygame.image.frombuffer(targ.tobytes(), (w, h), "RGBA")
        target.blit(out, (0, 0))

        if self.crack_layer is not None:
            buf2 = pygame.image.tobytes(self.crack_layer, "RGBA")
            arr = np.frombuffer(buf2, dtype=np.uint8).reshape(h, w, 4).copy()
            arr[..., 3] = (arr[..., 3].astype(np.uint16) * mask_np // 255).astype(np.uint8)
            crack_surf = pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA")
            target.blit(crack_surf, (0, 0))


def draw_entity_panel_3d(
    surf: "pygame.Surface",
    rect: "pygame.Rect",
    renderer: EntityMeshRenderer3D,
    damage_gen: "EntityDamageGenerator | None",
    tick: int,
    level: int,
    mesh_label: str,
    intensity: float,
) -> bool:
    """Render the chosen GLB into ``rect`` with generative damage overlay.

    Returns True on success.
    """
    # Keep the creature facing forward. The previous orbiting camera made the
    # larger 3D panel look like the entity was tilted / looking upward. The
    # renderer adds any user-driven yaw/pitch/zoom on top of this base value
    # (left-drag inside the panel, right-click / R to reset, wheel to zoom).
    yaw = 0.0
    result = renderer.render_surface_with_mask(yaw, intensity=intensity)
    if result is None:
        return False
    e_surf, mask_np = result

    pygame.draw.rect(surf, (10, 8, 12), rect)

    sw, sh = e_surf.get_size()
    if sw <= 0 or sh <= 0:
        return False
    rw, rh = rect.width, rect.height

    target_surf = pygame.transform.smoothscale(e_surf, (rw, rh))
    if damage_gen is not None and level >= 3:
        try:
            from PIL import Image
            mask_img = Image.fromarray(mask_np).resize((rw, rh), Image.BILINEAR)
            scaled_mask = np.array(mask_img, dtype=np.uint8)
        except Exception:
            scaled_mask = np.kron(
                mask_np,
                np.ones(
                    (max(1, rh // sh), max(1, rw // sw)),
                    dtype=np.uint8,
                ),
            )[:rh, :rw]
        damage_gen.composite(target_surf, scaled_mask, level=level, tick=tick)

    surf.blit(target_surf, rect.topleft)

    border_color = (60, 25, 40) if level <= 3 else (180, 60, 60) if level == 4 else (220, 80, 80)
    pygame.draw.rect(surf, border_color, rect, 2)

    bar_h = 22
    bar = pygame.Surface((rect.width, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 175))
    surf.blit(bar, (rect.left, rect.top))

    font_label_local = pygame.font.SysFont("consolas", 12, bold=True)
    dmg_summary = "clean"
    if damage_gen is not None and level >= 3:
        n_b, n_c = EntityDamageGenerator._damage_amounts(level)
        dmg_summary = f"blood={n_b} cracks={n_c}"
    label_text = (
        f"ENTITY 3D  mesh={Path(mesh_label).stem}  lv={level}  "
        f"yaw={renderer.yaw:5.1f}\u00b0  pitch={renderer.pitch:4.1f}\u00b0  "
        f"zoom={renderer.manual_zoom:4.2f}x  dmg={dmg_summary}"
    )
    surf.blit(
        font_label_local.render(label_text, True, (245, 200, 200)),
        (rect.left + 6, rect.top + 5),
    )

    # Hint along the bottom of the panel so the user knows the controls exist
    # and can tell at a glance whether they have overridden the auto pose.
    hint_h = 18
    hint = pygame.Surface((rect.width, hint_h), pygame.SRCALPHA)
    hint.fill((0, 0, 0, 150))
    surf.blit(hint, (rect.left, rect.bottom - hint_h))
    hint_color = (160, 220, 200) if renderer.user_camera_active else (140, 160, 170)
    hint_text = (
        "drag=orbit  wheel=zoom  right-click / R=reset"
        + ("  [USER]" if renderer.user_camera_active else "")
    )
    surf.blit(
        font_label_local.render(hint_text, True, hint_color),
        (rect.left + 6, rect.bottom - hint_h + 3),
    )
    return True


def _build_entity_texture(width: int, height: int, seed: int, level: int, aggression: float) -> pygame.Surface:
    """Generate a noisy organic texture for entity body fill."""
    tex = pygame.Surface((width, height), pygame.SRCALPHA)
    rng = random.Random(seed)
    base_dark = max(12, 40 - level * 4)
    rust = min(90, int(18 + aggression * 55 + level * 6))
    for y in range(height):
        for x in range(width):
            n = rng.randint(-18, 18)
            veins = int(18 * (0.5 + 0.5 * math.sin((x + y) * 0.08 + seed * 0.03)))
            r = max(0, min(255, base_dark + rust + n + veins))
            g = max(0, min(255, base_dark // 2 + n // 2))
            b = max(0, min(255, base_dark // 2 + n // 3))
            a = 210
            tex.set_at((x, y), (r, g, b, a))
    return tex


def _make_design_fallback_sprite(
    design_id: int,
    level: int,
    out_w: int,
    out_h: int,
    entity_texture_bank: EntityTextureBank | None = None,
) -> pygame.Surface:
    """Create a distinct per-design fallback sprite so pool diversity never collapses."""
    surf = pygame.Surface((out_w, out_h), pygame.SRCALPHA)
    cx, cy = out_w // 2, out_h // 2
    color = (120 + level * 12, 45 + level * 5, 38 + level * 4, 225)

    if design_id == 0:  # wraith (tall ghostly cloak)
        pygame.draw.ellipse(surf, color, (cx - 24, cy - 96, 48, 56))
        cloak = [
            (cx - 24, cy - 50), (cx + 24, cy - 50),
            (cx + 46, cy + 78), (cx + 10, cy + 116),
            (cx - 8, cy + 104), (cx - 42, cy + 116),
            (cx - 50, cy + 72),
        ]
        pygame.draw.polygon(surf, color, cloak)
    elif design_id == 1:  # brute (large torso, thick limbs)
        pygame.draw.ellipse(surf, color, (cx - 26, cy - 102, 52, 46))
        pygame.draw.rect(surf, color, (cx - 40, cy - 58, 80, 116), border_radius=14)
        pygame.draw.rect(surf, color, (cx - 58, cy - 36, 18, 90), border_radius=8)
        pygame.draw.rect(surf, color, (cx + 40, cy - 28, 18, 82), border_radius=8)
        pygame.draw.rect(surf, color, (cx - 26, cy + 58, 18, 54), border_radius=8)
        pygame.draw.rect(surf, color, (cx + 8, cy + 58, 18, 54), border_radius=8)
    elif design_id == 2:  # crawler (body + many legs)
        pygame.draw.ellipse(surf, color, (cx - 50, cy - 24, 100, 58))
        pygame.draw.ellipse(surf, color, (cx - 20, cy - 48, 40, 34))
        for i, off in enumerate((-46, -30, -14, 14, 30, 46)):
            pygame.draw.line(surf, color, (cx + off, cy + 6), (cx + off + (10 if off > 0 else -10), cy + 54), 5)
            if i % 2 == 0:
                pygame.draw.line(surf, color, (cx + off, cy - 2), (cx + off + (14 if off > 0 else -14), cy + 36), 3)
    elif design_id == 3:  # specter (floating torn form)
        pygame.draw.ellipse(surf, color, (cx - 34, cy - 96, 68, 90))
        pygame.draw.polygon(surf, color, [
            (cx - 34, cy - 30), (cx + 34, cy - 30),
            (cx + 20, cy + 92), (cx, cy + 74), (cx - 18, cy + 98),
        ])
    elif design_id == 4:  # stalker (thin humanoid with long arms)
        pygame.draw.ellipse(surf, color, (cx - 18, cy - 102, 36, 40))
        pygame.draw.rect(surf, color, (cx - 18, cy - 64, 36, 118), border_radius=10)
        pygame.draw.line(surf, color, (cx - 16, cy - 30), (cx - 54, cy + 62), 6)
        pygame.draw.line(surf, color, (cx + 16, cy - 26), (cx + 56, cy + 58), 6)
        pygame.draw.line(surf, color, (cx - 10, cy + 50), (cx - 24, cy + 114), 6)
        pygame.draw.line(surf, color, (cx + 10, cy + 50), (cx + 22, cy + 114), 6)
    elif design_id == 5:  # abomination (asymmetric fused masses)
        pygame.draw.ellipse(surf, color, (cx - 48, cy - 42, 86, 92))
        pygame.draw.ellipse(surf, (150, 66, 60, 215), (cx - 4, cy - 18, 56, 68))
        pygame.draw.ellipse(surf, (135, 58, 52, 205), (cx - 62, cy + 4, 42, 52))
        pygame.draw.line(surf, color, (cx + 16, cy + 26), (cx + 62, cy + 68), 6)
        pygame.draw.line(surf, color, (cx - 20, cy + 40), (cx - 52, cy + 106), 6)
    elif design_id == 6:  # shade (dark cloaked silhouette)
        shadow_col = (60, 45, 52, 235)
        pygame.draw.ellipse(surf, shadow_col, (cx - 24, cy - 96, 48, 50))
        pygame.draw.polygon(surf, shadow_col, [
            (cx - 30, cy - 48), (cx + 30, cy - 48),
            (cx + 42, cy + 106), (cx - 42, cy + 106),
        ])
    else:  # parasite (non-blob: segmented body + limbs)
        pygame.draw.ellipse(surf, color, (cx - 30, cy - 20, 60, 44))
        pygame.draw.ellipse(surf, (145, 62, 56, 220), (cx - 16, cy - 54, 32, 28))
        for off in (-34, -18, 18, 34):
            pygame.draw.line(surf, color, (cx + off, cy + 8), (cx + off + (12 if off > 0 else -12), cy + 52), 4)
        pygame.draw.line(surf, color, (cx - 8, cy + 20), (cx - 22, cy + 76), 4)
        pygame.draw.line(surf, color, (cx + 8, cy + 20), (cx + 20, cy + 76), 4)

    if entity_texture_bank and entity_texture_bank.available:
        tex = entity_texture_bank.get(level, design_id)
        if tex:
            tex = pygame.transform.smoothscale(tex, (out_w, out_h)).convert_alpha()
            mask = pygame.mask.from_surface(surf).to_surface(
                setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0)
            )
            tex.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            tex.blit(surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            return tex
    return surf


def draw_audio_waveform(surf: pygame.Surface, rect: pygame.Rect, intensity: float,
                        dissonance: float, tick: int):
    pygame.draw.rect(surf, (15, 15, 20), rect)
    pygame.draw.rect(surf, (50, 50, 60), rect, 1)

    mid_y = rect.centery
    num_points = rect.width
    rng = random.Random((tick // 3) * 11)

    for layer in range(3):
        points = []
        freq = 0.03 + layer * 0.02 + dissonance * 0.02
        amp = (rect.height * 0.35) * intensity * (1.0 - layer * 0.25)
        phase = tick * (0.06 + layer * 0.03)

        for i in range(num_points):
            x = rect.left + i
            base = math.sin(i * freq + phase) * amp
            noise = rng.uniform(-1, 1) * dissonance * amp * 0.5
            y = mid_y + base + noise
            y = max(rect.top + 2, min(rect.bottom - 2, y))
            points.append((x, int(y)))

        if len(points) > 1:
            if intensity < 0.25:
                color = (60, 200, 100)
            elif intensity < 0.5:
                color = (220, 210, 50)
            elif intensity < 0.75:
                color = (230, 140, 40)
            else:
                color = (220, 50, 40)
            pygame.draw.lines(surf, color, False, points, 2)

    freq_bars = 32
    bar_w = max(1, rect.width // freq_bars - 1)
    for i in range(freq_bars):
        bar_val = abs(math.sin(i * 0.5 + tick * 0.2)) * intensity
        bar_val += rng.uniform(0, 0.3) * dissonance
        bar_val = min(1.0, bar_val)
        bar_h = int(bar_val * rect.height * 0.4)
        bx = rect.left + i * (bar_w + 1)
        by = rect.bottom - bar_h
        if bar_val < 0.3:
            bar_col = (50, 160, 80, 120)
        elif bar_val < 0.6:
            bar_col = (200, 190, 40, 140)
        else:
            bar_col = (200, 50, 30, 160)
        bar_surf = pygame.Surface((bar_w, max(1, bar_h)), pygame.SRCALPHA)
        bar_surf.fill(bar_col)
        surf.blit(bar_surf, (bx, by))


def _pick_entity_design(level: int, tick: int) -> int:
    """Select a design class once per stress level."""
    if level <= 3:
        return STRESS_4_DESIGNS[0]
    designs = STRESS_4_DESIGNS if level == 4 else STRESS_5_DESIGNS
    rng = random.Random(level * 1009 + 17)
    return designs[rng.randrange(len(designs))]


def _build_fixed_entity_bundle(
    level: int,
    design_id: int,
    sprite_runtime,
    entity_textures,
    mesh_label: str | None = None,
) -> tuple[pygame.Surface, str, int] | None:
    """Build one entity sprite once per run and reuse it."""
    if level < 4:
        return None
    d_name = str(mesh_label or DESIGN_NAMES[design_id % NUM_DESIGNS])
    spr = None
    if entity_textures is not None:
        spr = entity_textures.get_design_sprite(
            level=level,
            design_id=design_id,
            seed=7000 + level * 173 + design_id * 29,
            out_w=180,
            out_h=240,
        )
    if spr is None:
        # Keep a deterministic fallback silhouette if no matching sprite exists.
        spr = _make_design_fallback_sprite(
            design_id, level, 180, 240, entity_texture_bank=None
        )
    if spr is None:
        return None
    return (spr, d_name, design_id)


def draw_entity_panel(surf: pygame.Surface, rect: pygame.Rect, params: EntityParams,
                      tick: int, level: int, sprite_runtime=None, model_only: bool = False,
                      entity_design_id: int = -1, entity_texture_bank=None,
                      entity_sprite_bundle=None, entity_motion_phase: float = 0.0):
    def _clamp_to_panel(x: int, y: int, w: int, h: int) -> tuple[int, int]:
        # Keep sprite fully inside entity panel bounds.
        min_x = rect.left
        min_y = rect.top
        max_x = rect.right - w
        max_y = rect.bottom - h
        return max(min_x, min(x, max_x)), max(min_y, min(y, max_y))

    bg_surf = pygame.Surface((rect.width, rect.height))
    grad_r = max(5, 8 + level * 3)
    grad_g = max(3, 4 + level)
    grad_b = max(5, 8 + level * 2)
    for y in range(rect.height):
        t = y / max(1, rect.height - 1)
        c = (int(grad_r * (1 - t * 0.5)), int(grad_g * (1 - t * 0.3)), int(grad_b * (1 - t * 0.4)))
        pygame.draw.line(bg_surf, c, (0, y), (rect.width, y))
    surf.blit(bg_surf, rect.topleft)
    pygame.draw.rect(surf, (50, 20, 20), rect, 1)

    if level < 4:
        font = pygame.font.SysFont("consolas", 14)
        label = font.render("no entity (stress < 4)", True, (60, 60, 60))
        surf.blit(label, (rect.centerx - label.get_width() // 2,
                          rect.centery - label.get_height() // 2))
        return

    # Keep entity always present on stress 4/5.
    forced_params = EntityParams(
        probability=1.0,
        opacity=params.opacity,
        aggression=params.aggression,
        morphology_distort=params.morphology_distort,
        aura_intensity=params.aura_intensity,
        height_scale=params.height_scale,
        width_scale=params.width_scale,
        movement_speed=params.movement_speed,
        eye_glow=params.eye_glow,
        flicker_rate=params.flicker_rate,
    )
    # Keep entities stable on stress 4/5 (no drifting animation in panel).
    entity_tick = level * 97 if level >= 4 else tick
    entity_data = generate_entity_frame(forced_params, entity_tick, rect.width, rect.height, seed=42)
    if not entity_data.visible:
        font = pygame.font.SysFont("consolas", 12)
        label = font.render("entity lurking...", True, (40, 10, 10))
        surf.blit(label, (rect.centerx - label.get_width() // 2,
                          rect.centery - label.get_height() // 2))
        return

    design_id = entity_design_id if entity_design_id >= 0 else _pick_entity_design(level, tick)
    design_name = DESIGN_NAMES[design_id % NUM_DESIGNS]
    mesh_name = ENTITY_MESH_MAP.get(design_name, "unknown.glb")

    if model_only:
        font = pygame.font.SysFont("consolas", 11)
        sprite_w = max(72, int(entity_data.width * 1.4))
        sprite_h = max(120, int(entity_data.height * 1.3))
        d_name = design_name
        spr = None
        from_bundle = entity_sprite_bundle is not None
        if from_bundle:
            base_spr, d_name, bundle_design_id = entity_sprite_bundle
            design_id = int(bundle_design_id)
            design_name = DESIGN_NAMES[design_id % NUM_DESIGNS]
            # Bundle name comes from selected mesh identity; keep mesh+design aligned.
            mesh_name = str(d_name)
            spr = base_spr.copy()
            sprite_w, sprite_h = spr.get_size()
        elif sprite_runtime and sprite_runtime.available:
            spr, d_name = sprite_runtime.generate_surface(
                stress_level=level,
                design_class=design_id,
                seed=1000 + level * 131 + design_id * 17,
                out_w=sprite_w,
                out_h=sprite_h,
            )
        else:
            d_name = design_name
            spr = _make_design_fallback_sprite(
                design_id, level, sprite_w, sprite_h, entity_texture_bank=None
            )
        if spr:
            # Fixed placement for stress 4/5 using prebuilt bundle.
            sx = rect.left + entity_data.x - sprite_w // 2
            sy = rect.top + entity_data.y - sprite_h // 2
            sx, sy = _clamp_to_panel(sx, sy, sprite_w, sprite_h)
            surf.blit(spr, (sx, sy))
            lbl = font.render(
                f"ENTITY: design={d_name} mesh={Path(mesh_name).name}",
                True, (200, 130, 100))
            surf.blit(lbl, (rect.left + 5, rect.top + 5))
            return
        lbl = font.render(f"ENTITY: generating {design_name}...", True, (200, 120, 80))
        surf.blit(lbl, (rect.left + 5, rect.top + 5))

    entity_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    alpha = int(min(255, entity_data.opacity * 255))
    alpha = max(alpha, 110 if level >= 4 else 80, 160 if level >= 5 else 0)
    skin_palettes = [
        (52, 42, 35),
        (61, 50, 41),
        (74, 58, 48),
        (80, 62, 50),
        (92, 66, 54),
        (105, 70, 58),
    ]
    skin = skin_palettes[max(0, min(5, level))]

    if entity_data.aura_radius > 0:
        for r_off in range(3):
            ar = entity_data.aura_radius + r_off * 8
            ac = (*entity_data.aura_color[:3], max(0, entity_data.aura_color[3] - r_off * 10))
            pygame.draw.circle(entity_surf, ac, (entity_data.x, entity_data.y), ar)

    design = "brute"
    mesh_name = ENTITY_MESH_MAP[design]

    body_color = (*skin, alpha)
    head_r = entity_data.width // 2
    if design == "brute":
        head_r = int(head_r * 1.15)
    elif design == "crawler":
        head_r = max(8, int(head_r * 0.72))
    head_y = entity_data.y - entity_data.height // 3
    pygame.draw.circle(entity_surf, body_color, (entity_data.x, head_y), head_r)

    neck_y = head_y + head_r - 3
    shoulder_w = int(entity_data.width * 0.55)
    body_bottom = entity_data.y + entity_data.height // 2
    if design == "wraith":
        body_points = [
            (entity_data.x - shoulder_w, body_bottom),
            (entity_data.x - int(shoulder_w * 0.92), neck_y + 12),
            (entity_data.x - entity_data.width // 6, neck_y - 2),
            (entity_data.x + entity_data.width // 6, neck_y - 2),
            (entity_data.x + int(shoulder_w * 0.92), neck_y + 12),
            (entity_data.x + shoulder_w, body_bottom),
        ]
    elif design == "brute":
        body_points = [
            (entity_data.x - int(shoulder_w * 1.25), body_bottom),
            (entity_data.x - int(shoulder_w * 1.05), neck_y + 22),
            (entity_data.x - entity_data.width // 3, neck_y + 4),
            (entity_data.x + entity_data.width // 3, neck_y + 4),
            (entity_data.x + int(shoulder_w * 1.05), neck_y + 22),
            (entity_data.x + int(shoulder_w * 1.25), body_bottom),
        ]
    else:  # crawler
        body_points = [
            (entity_data.x - int(shoulder_w * 1.2), body_bottom),
            (entity_data.x - int(shoulder_w * 0.7), neck_y + 26),
            (entity_data.x - entity_data.width // 4, neck_y + 18),
            (entity_data.x + entity_data.width // 4, neck_y + 18),
            (entity_data.x + int(shoulder_w * 0.7), neck_y + 26),
            (entity_data.x + int(shoulder_w * 1.2), body_bottom),
        ]
    pygame.draw.polygon(entity_surf, body_color, body_points)

    # Replace flat silhouette with textured body fill mask.
    texture = _build_entity_texture(
        rect.width,
        rect.height,
        seed=(tick // 4) * 31 + 7,
        level=level,
        aggression=params.aggression,
    )
    mask = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, alpha), (entity_data.x, head_y), head_r)
    pygame.draw.polygon(mask, (255, 255, 255, alpha), body_points)
    texture.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    entity_surf.blit(texture, (0, 0))

    # If trained sprite model is available, use it as primary visual source.
    if entity_sprite_bundle is not None:
        bundle_spr, bundle_design_name, _ = entity_sprite_bundle
        if bundle_spr:
            sprite_w, sprite_h = bundle_spr.get_size()
            design = bundle_design_name
            mesh_name = ENTITY_MESH_MAP.get(design, ENTITY_MESH_MAP["brute"])
            entity_surf.blit(
                bundle_spr,
                (entity_data.x - sprite_w // 2, entity_data.y - sprite_h // 2),
            )
    elif sprite_runtime and sprite_runtime.available:
        sprite_w = max(48, int(entity_data.width * (1.3 if level >= 4 else 1.0)))
        sprite_h = max(84, int(entity_data.height * (1.15 if level >= 4 else 1.0)))
        overlay_design_id = _pick_entity_design(level, tick)
        spr, inferred_design = sprite_runtime.generate_surface(
            stress_level=level,
            design_class=overlay_design_id,
            seed=(tick // 3) * 97 + int(params.aggression * 1000) + level * 43,
            out_w=sprite_w,
            out_h=sprite_h,
        )
        if spr:
            design = inferred_design
            mesh_name = ENTITY_MESH_MAP.get(design, ENTITY_MESH_MAP["brute"])
            entity_surf.blit(spr, (entity_data.x - sprite_w // 2, entity_data.y - sprite_h // 2))

    if design == "brute":
        horn_col = (min(255, skin[0] + 40), min(255, skin[1] + 30), min(255, skin[2] + 25), alpha)
        pygame.draw.polygon(entity_surf, horn_col, [
            (entity_data.x - head_r // 2, head_y - head_r + 4),
            (entity_data.x - head_r // 6, head_y - head_r - 14),
            (entity_data.x - head_r // 8, head_y - head_r + 8),
        ])
        pygame.draw.polygon(entity_surf, horn_col, [
            (entity_data.x + head_r // 2, head_y - head_r + 4),
            (entity_data.x + head_r // 6, head_y - head_r - 14),
            (entity_data.x + head_r // 8, head_y - head_r + 8),
        ])
    elif design == "crawler":
        for off in (-entity_data.width // 2, entity_data.width // 2):
            pygame.draw.line(entity_surf, (skin[0] + 20, skin[1] + 8, skin[2] + 8, alpha),
                             (entity_data.x + off, neck_y + 24),
                             (entity_data.x + off + int(off * 0.6), body_bottom + 14), 3)

    for rib_start, rib_end in entity_data.rib_lines:
        rib_col = (min(255, skin[0] + 35), min(255, skin[1] + 20), min(255, skin[2] + 20), min(255, alpha + 30))
        pygame.draw.line(entity_surf, rib_col, rib_start, rib_end, 1)

    for tendril in entity_data.tendrils:
        if len(tendril) > 1:
            for i in range(len(tendril) - 1):
                t = i / max(1, len(tendril) - 1)
                ta = int(alpha * (1.0 - t * 0.6))
                tcol = (min(255, skin[0] + 20), min(255, skin[1] + 12), min(255, skin[2] + 12), ta)
                thick = max(1, int(3 * (1.0 - t)))
                pygame.draw.line(entity_surf, tcol, tendril[i], tendril[i + 1], thick)

    for lx, ly in entity_data.limb_points:
        limb_col = (min(255, skin[0] + 20), min(255, skin[1] + 8), min(255, skin[2] + 8), min(255, alpha))
        pygame.draw.line(entity_surf, limb_col,
                         (entity_data.x, neck_y + 20), (lx, ly), 3)
        pygame.draw.circle(entity_surf, limb_col, (lx, ly), 3)

    if entity_data.has_eyes:
        for ex, ey in entity_data.eye_positions:
            glow_r = 7
            glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            for gr in range(glow_r, 0, -1):
                ga = int(entity_data.eye_color[3] * (gr / glow_r) * 0.4)
                gc = (*entity_data.eye_color[:3], ga)
                pygame.draw.circle(glow_surf, gc, (glow_r, glow_r), gr)
            entity_surf.blit(glow_surf, (ex - glow_r, ey - glow_r))
            pygame.draw.circle(entity_surf, entity_data.eye_color, (ex, ey), 3)
            pygame.draw.circle(entity_surf, (255, 255, 255, min(255, alpha)), (ex, ey), 1)

    if len(entity_data.mouth_points) >= 3:
        mouth_col = (30, 0, 0, min(255, alpha))
        pygame.draw.polygon(entity_surf, mouth_col, entity_data.mouth_points)
        pygame.draw.lines(entity_surf, (min(255, skin[0] + 25), min(255, skin[1] + 15), min(255, skin[2] + 15), alpha),
                          False, entity_data.mouth_points, 1)

    prng = random.Random((tick // 4) * 99)
    for px, py in entity_data.particle_positions:
        pa = prng.randint(10, max(11, min(255, alpha // 3)))
        ps = prng.randint(1, 3)
        pygame.draw.circle(entity_surf, (8, 4, 8, pa), (px, py), ps)

    surf.blit(entity_surf, rect.topleft)

    font = pygame.font.SysFont("consolas", 11)
    lbl = font.render(
        f"ENTITY: {entity_data.entity_type}  design={design}  mesh={Path(mesh_name).name}",
        True, (200, 120, 100)
    )
    surf.blit(lbl, (rect.left + 5, rect.top + 5))


def draw_camera_feed_panel(surf: pygame.Surface, rect: pygame.Rect,
                           feed_processor: CameraFeedProcessor,
                           params: CameraFeedParams, tick: int):
    pygame.draw.rect(surf, (5, 5, 8), rect)
    pygame.draw.rect(surf, (40, 80, 40), rect, 1)

    frame_rgb = feed_processor.get_frame_rgb(params)

    if frame_rgb is not None and frame_rgb.size > 0:
        h, w = frame_rgb.shape[:2]
        frame_surf = pygame.surfarray.make_surface(
            np.transpose(frame_rgb, (1, 0, 2)))
        scaled = pygame.transform.scale(frame_surf, (rect.width, rect.height))
        surf.blit(scaled, rect.topleft)
    else:
        # No synthetic camera fallback any more -- if the webcam fails,
        # render an explicit "NO WEBCAM" message instead of fake static.
        font_msg = pygame.font.SysFont("consolas", 14)
        msg = font_msg.render(
            "NO WEBCAM (synthetic camera removed)", True, (90, 140, 90)
        )
        surf.blit(
            msg,
            (rect.centerx - msg.get_width() // 2,
             rect.centery - msg.get_height() // 2),
        )

    font = pygame.font.SysFont("consolas", 11)
    cam_label = "WEBCAM" if feed_processor.has_webcam else "OFFLINE"
    lbl = font.render(f"CAMERA FEED [{cam_label}]", True, (80, 200, 80))
    surf.blit(lbl, (rect.left + 4, rect.top + 3))

    info = font.render(
        f"face_distort={params.face_distort_amount:.2f} "
        f"shadow={params.shadow_inject_prob:.2f}",
        True, (60, 150, 60))
    surf.blit(info, (rect.left + 4, rect.bottom - 16))


def draw_hud(surf: pygame.Surface, level: int, frame_data: dict, perf_data: dict,
             tick: int, elapsed: float, using_model: bool):
    hud_w = 300
    hud_rect = pygame.Rect(W - hud_w, 0, hud_w, H)
    pygame.draw.rect(surf, (12, 12, 18), hud_rect)
    pygame.draw.line(surf, (40, 40, 50), (hud_rect.left, 0), (hud_rect.left, H), 2)

    font_title = pygame.font.SysFont("consolas", 13, bold=True)
    font_sm = pygame.font.SysFont("consolas", 10)

    x0 = hud_rect.left + 8
    y = 6

    mode_label = "NEURAL" if using_model else "FALLBACK"
    mode_color = (80, 255, 120) if using_model else (255, 200, 60)
    title = font_title.render(f"LIVE [{mode_label}]", True, mode_color)
    surf.blit(title, (x0, y)); y += 18

    color = STRESS_COLORS[level]
    bar_w_total = hud_w - 20
    bar_h = 14
    pygame.draw.rect(surf, (30, 30, 35), (x0, y, bar_w_total, bar_h))
    fill_w = int(bar_w_total * (level / 5.0))
    pygame.draw.rect(surf, color, (x0, y, fill_w, bar_h))
    lbl = font_sm.render(f"Stress: {level} ({STRESS_NAMES[level]})", True, (255, 255, 255))
    surf.blit(lbl, (x0 + 4, y + 1))
    y += 18

    mp = frame_data.get("model_params", {})

    def _section(title_text, params_list, title_col):
        nonlocal y
        surf.blit(font_sm.render(title_text, True, title_col), (x0, y))
        y += 12
        for name, val in params_list:
            val_f = val if isinstance(val, float) else 0.0
            lbl_s = font_sm.render(f" {name}", True, (100, 100, 110))
            val_s = font_sm.render(f"{val_f:.3f}", True, title_col)
            surf.blit(lbl_s, (x0, y))
            surf.blit(val_s, (x0 + 140, y))

            bar_x = x0 + 195
            bar_max = hud_w - 20 - 195
            bw = max(0, int(min(1.0, abs(val_f)) * bar_max))
            pygame.draw.rect(surf, (25, 25, 35), (bar_x, y + 1, bar_max, 7))
            pygame.draw.rect(surf, title_col, (bar_x, y + 1, bw, 7))
            y += 11

    _section("AUDIO", [
        ("intensity", mp.get("audio_intensity", 0)),
        ("dissonance", mp.get("audio_dissonance", 0)),
        ("reverb", mp.get("audio_reverb_depth", 0)),
        ("transient", mp.get("audio_transient_rate", 0)),
    ], (220, 210, 60))

    _section("DSP", [
        ("pitch_shift", mp.get("dsp_pitch_shift", 0)),
        ("distortion", mp.get("dsp_distortion", 0)),
        ("filter", mp.get("dsp_filter_freq", 0)),
        ("reverb_mix", mp.get("dsp_reverb_mix", 0)),
        ("blend", mp.get("dsp_layer_blend", 0)),
    ], (180, 220, 80))

    _section("VISUAL", [
        ("corruption", mp.get("visual_corruption", 0)),
        ("fog", mp.get("visual_fog_density", 0)),
        ("light", mp.get("visual_light_temp", 0)),
        ("flicker", mp.get("visual_flicker_rate", 0)),
        ("grime", mp.get("visual_grime_overlay", 0)),
    ], (230, 150, 50))

    _section("CAMERA", [
        ("magnitude", mp.get("camera_magnitude", 0)),
        ("aberration", mp.get("camera_aberration", 0)),
        ("noise", mp.get("camera_noise", 0)),
        ("vignette", mp.get("camera_vignette", 0)),
        ("warp", mp.get("camera_warp", 0)),
    ], (200, 100, 220))

    _section("CAM FEED", [
        ("face_dist", mp.get("camera_feed_face_distort", 0)),
        ("shadow", mp.get("camera_feed_shadow", 0)),
        ("darken", mp.get("camera_feed_darken", 0)),
        ("figure", mp.get("camera_feed_figure", 0)),
    ], (80, 180, 80))

    _section("ENTITY", [
        ("probability", mp.get("entity_probability", 0)),
        ("opacity", mp.get("entity_opacity", 0)),
        ("aggression", mp.get("entity_aggression", 0)),
        ("morphology", mp.get("entity_morphology", 0)),
        ("aura", mp.get("entity_aura", 0)),
    ], (220, 60, 60))

    y += 4
    pygame.draw.line(surf, (40, 40, 50), (x0, y), (x0 + bar_w_total, y), 1)
    y += 4

    perf_entries = [
        ("Adaptive ms", f"{perf_data['adaptive_ms']:.2f}"),
        ("Quality", f"{int(perf_data['quality_tier'])}"),
        ("Frame", f"{tick}"),
        ("Time", f"{elapsed:.1f}s"),
    ]
    for label, value in perf_entries:
        lbl_surf = font_sm.render(f"{label}: {value}", True, (150, 150, 160))
        surf.blit(lbl_surf, (x0, y))
        y += 12

    y += 6
    help_lines = ["0-5:stress  C:cam  Q:quit"]
    for line in help_lines:
        surf.blit(font_sm.render(line, True, (70, 70, 80)), (x0, y))
        y += 12


class RealtimeAudioGen:
    """Generates unique audio per stress level using the trained WaveGAN."""

    def __init__(self, *, use_fp16: bool = True):
        self.generator = None
        self.device = None
        self._dtype = torch.float32
        self._use_fp16 = bool(use_fp16)
        self._load()

    def _load(self):
        if not AUDIO_MODEL.exists():
            print("[audio-gen] no trained audio model found, skipping real-time audio")
            return
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # GPU fast-path: TF32 + cuDNN tuning. WaveGAN is tiny (~2.5M params,
                # ~150 KB activations); this mostly cuts launch overhead so realtime
                # clip generation stays sub-millisecond on a 5060.
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            else:
                self.device = torch.device("cpu")

            self.generator = AudioGenerator().to(self.device)
            ckpt = torch.load(str(AUDIO_MODEL), map_location=self.device, weights_only=True)
            self.generator.load_state_dict(ckpt["generator_state"])
            self.generator.eval()
            if self.device.type == "cuda" and self._use_fp16:
                self.generator = self.generator.half()
                self._dtype = torch.float16
            print(f"[audio-gen] loaded WaveGAN on {self.device} (dtype={self._dtype})")
        except Exception as e:
            print(f"[audio-gen] failed to load: {e}")
            self.generator = None

    @property
    def available(self) -> bool:
        return self.generator is not None

    def generate_clip(self, stress_level: int, seed: int | None = None) -> np.ndarray:
        if not self.available:
            return None
        with torch.no_grad():
            if seed is not None:
                gen = torch.Generator(device=self.device if self.device.type == "cuda" else "cpu")
                gen.manual_seed(int(seed) & 0x7FFFFFFF)
                noise = torch.randn(
                    1, AUDIO_NOISE_DIM, generator=gen,
                    device=self.device, dtype=self._dtype,
                )
            else:
                noise = torch.randn(1, AUDIO_NOISE_DIM,
                                    device=self.device, dtype=self._dtype)
            sl = torch.tensor([int(stress_level)], dtype=torch.long, device=self.device)
            audio = self.generator(noise, sl).squeeze(0).squeeze(0)
        return audio.float().cpu().numpy()


def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32)
    src_x = np.arange(audio.shape[0], dtype=np.float32)
    dst_len = max(1, int(round(audio.shape[0] * float(dst_sr) / float(src_sr))))
    dst_x = np.linspace(0, audio.shape[0] - 1, dst_len, dtype=np.float32)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _to_stereo_sound(audio_mono: np.ndarray, volume: float) -> pygame.mixer.Sound:
    y = np.clip(audio_mono, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    stereo = np.stack([pcm, pcm], axis=1)
    snd = pygame.sndarray.make_sound(stereo.copy(order="C"))
    snd.set_volume(max(0.0, min(1.0, volume)))
    return snd


def _synthetic_ambient_mono(
    sr: int,
    *,
    seconds: float,
    seed: int,
    level: int,
) -> np.ndarray:
    """Cheap looping rumble for MusicGen bootstrap when no WAV / WaveGAN snapshot exists."""
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    n = max(512, int(sr * seconds))
    gain = float(0.052 + 0.011 * level + 0.018 * rng.random())
    x = rng.standard_normal(n).astype(np.float32) * gain
    decay = 0.91 + 0.07 * rng.random()
    acc = 0.0
    y = np.empty_like(x)
    for i in range(n):
        acc = decay * acc + (1.0 - decay) * x[i]
        y[i] = acc
    decay2 = 0.86 + 0.1 * rng.random()
    acc = 0.0
    z = np.empty_like(y)
    for i in range(n):
        acc = decay2 * acc + y[i]
        z[i] = acc
    z -= float(np.mean(z))
    peak = float(np.max(np.abs(z))) + 1e-9
    return np.clip(z / peak * 0.9, -1.0, 1.0).astype(np.float32)


def _make_musicgen_bootstrap_sound(
    *,
    stress_level: int,
    clip_index: int,
    real_audio: RealAudioBank,
    audio_gen: RealtimeAudioGen,
    audio_int: float,
    dissonance: float,
    sole_source_generated: bool,
    catalog_only_gap: bool = False,
    ambient_picker: RotatingRoleClipPicker | None = None,
    procedural_gen: ProceduralAudioGen | None = None,
) -> pygame.mixer.Sound:
    """Bootstrap ambience: catalog → procedural granular → trained WaveGAN → noise.

    ``catalog_only_gap`` skips generators (used when MusicGen is mid-render so we
    don't spike the GPU). The procedural generator is preferred over WaveGAN
    because it always produces a coherent clip while still being CPU-cheap.
    """
    lvl = max(0, min(5, int(stress_level)))
    ix = max(0, int(clip_index))
    sr = MIXER_AUDIO_SR
    seed = (((ix + 17) * 1009 + lvl * 7919) ^ 0x13579BDF) & 0xFFFFFFFF
    try:
        if real_audio.available:
            pth = ambient_picker.next_path(lvl) if ambient_picker is not None else None
            if pth is None:
                pth = real_audio.get_role_clip(lvl, "ambient", ix)
            if pth:
                s = pygame.mixer.Sound(pth)
                vm = float(0.12 + 0.035 * lvl + 0.1 * audio_int + 0.04 * dissonance)
                vm *= music_bed_level_gain(lvl)
                if sole_source_generated:
                    vm *= 1.06
                if catalog_only_gap:
                    vm *= 0.9
                s.set_volume(max(0.05, min(0.74, vm)))
                return s
    except Exception:
        pass
    if not catalog_only_gap and procedural_gen is not None and procedural_gen.available:
        try:
            bs = (((seed ^ 0x7F4A7C15) << 5) ^ (lvl * 0x9E3779B1)) & 0xFFFFFFFF
            raw_try = procedural_gen.generate_clip(lvl, seed=bs)
            if raw_try is not None:
                raw_try = postprocess_wavegan_for_stress(raw_try, lvl, audio_int, dissonance)
                raw_try = _resample_audio(raw_try, MODEL_AUDIO_SR, sr)
                vm = wavegan_playback_volume(
                    lvl,
                    audio_int,
                    dissonance,
                    sole_source=sole_source_generated,
                )
                return _to_stereo_sound(raw_try, volume=max(0.08, vm * 0.88))
        except Exception:
            pass
    if not catalog_only_gap:
        try:
            if audio_gen.available:
                bs = (((seed ^ 913) << 13) ^ (lvl * 0x517CC1B7)) & 0xFFFFFFFF
                picked = None
                for attempt in range(5):
                    raw_try = audio_gen.generate_clip(lvl, seed=(bs + attempt * 7919 + attempt * attempt) ^ 0x243F6A88)
                    if raw_try is None:
                        continue
                    raw_try = postprocess_wavegan_for_stress(raw_try, lvl, audio_int, dissonance)
                    if attempt < 4 and not wavegan_quality_ok(raw_try, lvl, audio_int, dissonance):
                        continue
                    picked = raw_try
                    break
                if picked is not None:
                    picked = _resample_audio(picked, MODEL_AUDIO_SR, sr)
                    vm = wavegan_playback_volume(
                        lvl,
                        audio_int,
                        dissonance,
                        sole_source=sole_source_generated,
                    )
                    return _to_stereo_sound(picked, volume=max(0.08, vm * 0.88))
        except Exception:
            pass
    mono = _synthetic_ambient_mono(sr, seconds=2.82, seed=seed, level=lvl)
    vm = float(0.12 + 0.034 * lvl + 0.08 * audio_int) * music_bed_level_gain(lvl)
    return _to_stereo_sound(mono, volume=max(0.07, min(0.52, vm)))


class EntitySpriteRuntime:
    """Runtime inference wrapper for trained entity sprite generator (128x128, 8 designs)."""

    def __init__(self):
        self.generator = None
        self.device = None
        self._cache = {}
        self._load()

    def _load(self):
        if not ENTITY_MODEL.exists():
            print("[entity-gen] no trained entity sprite model found")
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.generator = EntitySpriteGenerator().to(self.device)
            ckpt = torch.load(str(ENTITY_MODEL), map_location=self.device, weights_only=True)
            self.generator.load_state_dict(ckpt["gen_state"])
            self.generator.eval()
            print(f"[entity-gen] loaded sprite generator on {self.device} "
                  f"({NUM_DESIGNS} designs, 128x128)")
        except Exception as e:
            print(f"[entity-gen] failed to load: {e}")
            self.generator = None

    @property
    def available(self) -> bool:
        return self.generator is not None

    def clear_cache(self):
        self._cache.clear()

    @torch.no_grad()
    def generate_surface(self, stress_level: int, design_class: int,
                         seed: int, out_w: int, out_h: int
                         ) -> tuple[pygame.Surface | None, str]:
        if not self.available:
            return None, "none"

        cache_key = (stress_level, design_class, seed)
        if cache_key in self._cache:
            return self._cache[cache_key]

        design_name = DESIGN_NAMES[design_class % NUM_DESIGNS]
        best = None
        best_score = -1e9

        for k in range(4):
            gen = torch.Generator(
                device=self.device if self.device.type == "cuda" else "cpu")
            gen.manual_seed(seed + k * 131)
            noise = torch.randn(1, self.generator.noise_dim,
                                generator=gen, device=self.device)
            sl = torch.tensor([stress_level], dtype=torch.long, device=self.device)
            dc = torch.tensor([design_class], dtype=torch.long, device=self.device)
            sprite = self.generator(noise, sl, dc)[0]
            rgb = ((sprite[:3].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
                   ).clip(0, 255).astype(np.uint8)
            alpha = (sprite[3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            mask = alpha > 38
            coverage = float(mask.mean())
            if coverage < 0.01 or coverage > 0.70:
                continue
            ys, xs = np.where(mask)
            if ys.size < 8:
                continue
            bw = float(xs.max() - xs.min() + 1)
            bh = float(ys.max() - ys.min() + 1)
            aspect = bh / max(1.0, bw)
            if aspect > 4.0 or aspect < 0.3:
                continue
            rgb_var = float(np.var(rgb.astype(np.float32) * mask[..., None]))
            edge = float(np.abs(np.diff(alpha.astype(np.float32), axis=0)).mean())
            score = (coverage * 3.0 + rgb_var / 1200.0 + edge / 30.0
                     - abs(aspect - 1.6) * 0.5)
            if score > best_score:
                best_score = score
                best = (rgb, alpha)

        if best is None:
            return None, design_name

        rgb, alpha = best
        rgba = np.dstack([rgb, alpha])
        surf = pygame.image.frombuffer(
            rgba.tobytes(), (rgba.shape[1], rgba.shape[0]), "RGBA"
        ).convert_alpha()
        surf = pygame.transform.smoothscale(surf, (out_w, out_h))
        result = (surf, design_name)
        self._cache[cache_key] = result
        return result


class TextureRuntime:
    """Runtime inference wrapper for trained texture corruption model."""

    def __init__(self):
        self.model = None
        self.device = None
        self._load()

    def _load(self):
        if not TEXTURE_MODEL.exists():
            print("[texture-gen] no trained texture model found")
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TextureCorruptionUNet().to(self.device)
            ckpt = torch.load(str(TEXTURE_MODEL), map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["generator_state"])
            self.model.eval()
            print(f"[texture-gen] loaded corruption U-Net on {self.device}")
        except Exception as e:
            print(f"[texture-gen] failed to load: {e}")
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def corrupt_surface(self, surf: pygame.Surface, stress_level: int, tick: int = 0, strength: float = 0.8) -> pygame.Surface:
        if not self.available:
            return surf
        orig_w, orig_h = surf.get_size()
        small = pygame.transform.smoothscale(surf, (256, 256))
        src_uint8 = pygame.surfarray.array3d(small).astype(np.uint8)
        arr = src_uint8.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(self.device)
        out = self.model.corrupt(tensor[0], stress_level=stress_level, device=self.device)
        pred = ((out.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.float32)
        src = src_uint8.astype(np.float32)
        delta = pred - src
        gain = float(
            0.12
            + 0.28 * float(max(0.0, min(1.0, strength)))
            + 0.065 * float(stress_level) / 5.0
        )
        gain = min(0.38, gain)
        mixed = np.clip(src + delta * gain, 0, 255)

        diff_mag = np.mean(np.abs(delta), axis=2)
        crack_thresh = np.percentile(diff_mag, 86.0 - stress_level * 1.55)
        crack_mask = diff_mag > crack_thresh
        mixed[crack_mask] *= np.array([0.86, 0.84, 0.82], dtype=np.float32)

        mixed = np.clip(0.94 * mixed + 0.06 * src, 0, 255)
        out_np = mixed.astype(np.uint8)
        out_surf = pygame.surfarray.make_surface(np.transpose(out_np, (1, 0, 2)))
        return pygame.transform.smoothscale(out_surf, (orig_w, orig_h))


def main():
    parser = argparse.ArgumentParser(description="Iteration 2 Full GUI Demo")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to trained fusion_generator.pt")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable webcam capture")
    parser.add_argument(
        "--catalog-audio",
        action="store_true",
        help="Include catalog WAV/MP3 ambience, tension loops, and one-shots "
        "(default: synthesized bed only on the ambience channel)",
    )
    parser.add_argument(
        "--audio-backend",
        choices=["auto", "procedural", "wavegan", "musicgen"],
        default="musicgen",
        help="musicgen (default): local MusicGen first, then procedural/WaveGAN fallback if allowed; "
        "procedural: CPU granular+drone synth from real grains (always works, very fast); "
        "wavegan: models/audio_generator.pt trained checkpoint only; "
        "auto: try MusicGen then procedural.",
    )
    parser.add_argument(
        "--musicgen-model",
        type=str,
        default=str(DEFAULT_MUSICGEN_LOCAL),
        help=(
            "Local snapshot folder (recommended: models/musicgen-small) "
            "or HF hub id only if offline mode is OFF. Populate with:\n"
            '  "...\\miniconda3\\Scripts\\hf.exe" download facebook/musicgen-small '
            "--local-dir models/musicgen-small [--token HF_TOKEN or set HF_TOKEN]"
        ),
    )
    parser.add_argument(
        "--audio-remote-ok",
        action="store_true",
        help="Allow Hugging Face hub/cache when loading MusicGen. "
        "Default is local snapshot only (--musicgen-model folder, offline-friendly).",
    )
    parser.add_argument(
        "--musicgen-sync-load",
        action="store_true",
        help="Load MusicGen on the main thread (startup freeze). Default loads in the background "
        "while a stress-mapped WAV / WaveGAN / procedural bed plays on the ambience channel.",
    )
    parser.add_argument(
        "--musicgen-tokens",
        type=int,
        default=192,
        help="MusicGen max_new_tokens: each step is sequential decoding, so high values "
        "mean much longer GPU time (~192 default for snappier loops; ~512 for longer beds)",
    )
    parser.add_argument("--model-only", action="store_true", default=True,
                        help="Require trained model outputs only (no procedural/catalog fallback)")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Allow non-model fallbacks when a trained model is missing")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Target demo pacing (default: {DEFAULT_FPS}; capped 4–60)",
    )
    parser.add_argument("--time-scale", type=float, default=0.75,
                        help="Simulation clock scale multiplier (0.5 = slower)")
    parser.add_argument(
        "--env-texture-filter",
        choices=["wood", "all"],
        default="wood",
        help="Environment panel base tiles: wood (default, catalog paths matching lumber/planks/etc.) "
        "or all stress-catalog textures.",
    )
    args = parser.parse_args()
    generated_audio_only = not bool(getattr(args, "catalog_audio", False))

    mg_path = Path(args.musicgen_model).expanduser().resolve()
    mg_snapshot_ok = mg_path.is_dir()
    audio_force_local = not bool(getattr(args, "audio_remote_ok", False))

    if audio_force_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model_path = Path(args.model) if args.model else None

    run_entity_rng = random.Random(int(time.time() * 1000) ^ random.getrandbits(31))
    new_entities_meshes = _list_new_entities_meshes()
    if new_entities_meshes:
        print(
            f"[entity-3d] {len(new_entities_meshes)} mesh(es) available in "
            f"`new entities/` (uniform random pick): "
            + ", ".join(p.name for p in new_entities_meshes)
        )
    run_fixed_mesh_path = _pick_weighted_new_entity_mesh(new_entities_meshes, run_entity_rng)
    run_fixed_design = _design_id_from_mesh_name(run_fixed_mesh_path.name) if run_fixed_mesh_path else 1
    run_fixed_mesh_label = run_fixed_mesh_path.name if run_fixed_mesh_path else "new-entities-mesh-missing"
    if run_fixed_mesh_path is not None:
        print(f"[entity-3d] this run selected: {run_fixed_mesh_path.name}")

    content_w_preview = W - 300
    _feed_panel_w_preview = content_w_preview // 2 - 10
    entity_panel_w = max(320, _feed_panel_w_preview)
    # Big upright 3D entity viewport — was 240 tall, now 540 (2.25x area).
    # On a 5060 Ti laptop a single PBR mesh at this size is well under 1 ms.
    entity_panel_h = 540
    # Internal panda3d render target at ~1.5x the panel; supersampling is dirt
    # cheap here (one mesh, no shadows) and noticeably crisper than 1:1.
    render_w = int(entity_panel_w * 1.5)
    render_h = int(entity_panel_h * 1.5)

    entity_3d_renderer: EntityMeshRenderer3D | None = None
    entity_damage_gen: EntityDamageGenerator | None = None
    if run_fixed_mesh_path is not None and _PANDA3D_AVAILABLE:
        try:
            entity_3d_renderer = EntityMeshRenderer3D(
                run_fixed_mesh_path,
                width=render_w,
                height=render_h,
                target_size=2.5,
            )
            entity_damage_gen = EntityDamageGenerator(
                seed=int(run_entity_rng.getrandbits(31))
            )
            print(
                f"[entity-3d] live GLB renderer ready: {run_fixed_mesh_path.name} "
                f"({render_w}x{render_h} -> panel {entity_panel_w}x{entity_panel_h})"
            )
        except Exception as exc:
            print(f"[entity-3d] failed to init renderer ({exc}); falling back to 2D entity panel")
            entity_3d_renderer = None
            entity_damage_gen = None
    elif run_fixed_mesh_path is None:
        print("[entity-3d] no meshes in `new entities/`, using 2D entity panel")
    elif not _PANDA3D_AVAILABLE:
        print(f"[entity-3d] panda3d unavailable ({_PANDA3D_IMPORT_ERROR}); using 2D entity panel")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Iteration 2: Full Horror Generation Demo")
    clock = pygame.time.Clock()

    orchestrator = FusionOrchestrator(
        catalog_path=CATALOG,
        trained_model_path=model_path,
    )
    using_model = orchestrator.using_model
    perf = PerfMonitor()

    content_w = W - 300
    half_w = content_w // 2 - 10
    tex_tile_w = half_w // 2
    tex_tile_h = 140
    # Static layout: the entity panel never moves during a run, so compute its
    # rect once for both event-handling (mouse drag/zoom/reset) and rendering.
    static_feed_panel_w = content_w // 2 - 10
    entity_rect_static = pygame.Rect(static_feed_panel_w + 18, 424, static_feed_panel_w, 540)
    proc_fallback_seed = random.getrandbits(30)
    base_tex = gen_base_texture(half_w, 280, seed=proc_fallback_seed)

    real_textures = RealTextureBank(tex_tile_w, tex_tile_h, substrate_filter=args.env_texture_filter)
    entity_textures = EntityTextureBank(256, 256)
    real_audio = RealAudioBank()
    _cp_rng = random.Random((time.time_ns() & 0xFFFFFFFF) ^ 0x6A09E667)
    ambient_clip_picker = RotatingRoleClipPicker(real_audio, "ambient", _cp_rng)
    tension_clip_picker = RotatingRoleClipPicker(real_audio, "tension", random.Random(_cp_rng.randint(1, 10**9)))
    impact_clip_picker = RotatingRoleClipPicker(real_audio, "impact", random.Random(_cp_rng.randint(1, 10**9)))
    # Entity / creature one-shots come from assets/entity_audio/ exclusively,
    # routed independently from the stress-tier fallback bed.
    entity_audio_bank = EntityAudioBank()
    entity_clip_picker = EntityAudioPicker(entity_audio_bank, random.Random(_cp_rng.randint(1, 10**9)))
    texture_runtime = TextureRuntime()
    sprite_runtime = EntitySpriteRuntime()
    model_only = args.model_only and (not args.allow_fallback)

    if model_only:
        missing = []
        if not using_model:
            missing.append(f"fusion model ({model_path})")
        if not texture_runtime.available:
            missing.append(f"texture model ({TEXTURE_MODEL})")
        if not sprite_runtime.available:
            missing.append(f"entity sprite model ({ENTITY_MODEL})")

    try:
        pygame.mixer.init(frequency=MIXER_AUDIO_SR, size=-16, channels=2, buffer=1024)
        audio_initialized = True
        print("[audio] pygame.mixer initialized")
    except Exception as e:
        audio_initialized = False
        print(f"[audio] mixer init failed: {e}")

    audio_gen = RealtimeAudioGen()
    procedural_gen = ProceduralAudioGen(BASE_DIR / "assets")
    musicgen_gen: MusicGenRealtimeGen | None = None

    dont_load_musicgen = False
    musicgen_requested = args.audio_backend in ("auto", "musicgen")
    fallback_audio_allowed = bool(getattr(args, "allow_fallback", False)) or args.audio_backend == "auto"
    if args.audio_backend in ("auto", "musicgen") and audio_force_local:
        if not mg_snapshot_ok:
            msg = (
                f"MusicGen requires a local snapshot (default offline mode): missing folder\n  {mg_path}\n"
                "  Example (PATH may omit Scripts):\n"
                '  "...\\miniconda3\\Scripts\\hf.exe" download facebook/musicgen-small '
                "--local-dir models/musicgen-small [--token YOUR_HF_TOKEN]\n"
                "  Or use --audio-remote-ok so Hugging Face can download/cache weights."
            )
            if args.audio_backend == "musicgen" and not fallback_audio_allowed:
                raise SystemExit(msg)
            print(f"[audio] {msg}\n[audio] Falling back to procedural/WaveGAN.")
            dont_load_musicgen = True

    if args.audio_backend in ("auto", "musicgen") and not dont_load_musicgen:
        mg_id = str(mg_path) if mg_snapshot_ok else args.musicgen_model
        lf = audio_force_local
        # When the user explicitly selects musicgen, force a synchronous load
        # so we can pre-render one bed per stress level before pygame starts.
        # Auto-mode keeps the old defer-and-background behavior.
        force_sync = (args.audio_backend == "musicgen")
        mg_defer = (not force_sync) and (not bool(getattr(args, "musicgen_sync_load", False)))
        mg_try = MusicGenRealtimeGen(
            model_id=mg_id,
            max_new_tokens=args.musicgen_tokens,
            local_files_only=lf,
            defer_init=mg_defer,
        )
        if mg_defer:
            musicgen_gen = mg_try
            print(
                "[musicgen] loading in background — stress-mapped WAV bed on ambience until "
                "the first GPU clip (and between clips while the channel is idle); "
                "use --musicgen-sync-load for blocking init."
            )
            if mg_try.available:
                src = mg_id if mg_snapshot_ok else args.musicgen_model
                mo = "(local snapshot)" if mg_snapshot_ok else "(transformers Hub / cache)"
                print(f"[audio] MusicGen ready immediately {mo}: {src}")
        elif mg_try.available:
            musicgen_gen = mg_try
            src = mg_id if mg_snapshot_ok else args.musicgen_model
            mo = "(local snapshot)" if mg_snapshot_ok else "(transformers Hub / cache)"
            print(f"[audio] MusicGen load source {mo}: {src}")
        elif args.audio_backend == "musicgen":
            err = mg_try.last_load_error or "unknown error"
            if not fallback_audio_allowed:
                raise SystemExit(f"MusicGen backend required but failed: {err}")
            print(f"[audio] MusicGen unavailable ({err}); falling back to procedural/WaveGAN.")
        else:
            print(f"[audio] MusicGen unavailable ({mg_try.last_load_error}); using procedural/WaveGAN.")

    use_musicgen_bed = args.audio_backend in ("auto", "musicgen") and musicgen_gen is not None
    use_procedural_bed = (
        args.audio_backend == "procedural"
        or (musicgen_requested and fallback_audio_allowed)
        or (args.audio_backend == "auto" and musicgen_gen is None)
    ) and procedural_gen.available

    # Per-run MusicGen memory. Every generated clip is kept by stress level so
    # revisiting a level can replay an already rendered clip instantly instead
    # of waiting on the transformer again. This is intentionally in-memory only:
    # it resets each demo run and never writes generated audio to disk.
    musicgen_clip_cache: dict[int, list[np.ndarray]] = {lvl: [] for lvl in range(6)}
    musicgen_clip_cursor: dict[int, int] = {lvl: 0 for lvl in range(6)}
    musicgen_clip_last_played: dict[int, int | None] = {lvl: None for lvl in range(6)}
    musicgen_strict_only = (
        args.audio_backend == "musicgen"
        and generated_audio_only
        and not fallback_audio_allowed
    )

    def _remember_musicgen_clip(level: int, clip: np.ndarray) -> None:
        lvl = max(0, min(5, int(level)))
        if clip is None or clip.size <= 0:
            return
        # Keep a small rolling pool per stress level. Six clips is enough to
        # avoid repeats during normal fallback/catalog use without growing VRAM
        # or RAM over a long demo session.
        pool = musicgen_clip_cache[lvl]
        pool.append(np.asarray(clip, dtype=np.float32).copy())
        if len(pool) > 6:
            del pool[0]
            cur = musicgen_clip_cursor[lvl]
            last = musicgen_clip_last_played[lvl]
            musicgen_clip_cursor[lvl] = max(0, cur - 1)
            musicgen_clip_last_played[lvl] = None if last is None else max(0, last - 1)

    def _next_cached_musicgen_clip(level: int) -> np.ndarray | None:
        lvl = max(0, min(5, int(level)))
        pool = musicgen_clip_cache.get(lvl, [])
        if not pool:
            return None
        idx = musicgen_clip_cursor[lvl] % len(pool)
        if (
            not musicgen_strict_only
            and musicgen_clip_last_played[lvl] == idx
        ):
            if len(pool) <= 1:
                return None
            idx = (idx + 1) % len(pool)
        musicgen_clip_cursor[lvl] = (idx + 1) % len(pool)
        musicgen_clip_last_played[lvl] = idx
        return pool[idx]

    def _play_musicgen_array(
        raw: np.ndarray,
        level: int,
        audio_intensity: float,
        dissonance_value: float,
        out_channel,
        *,
        fade_ms: int,
    ) -> bool:
        if raw is None or raw.size <= 0 or out_channel is None or musicgen_gen is None:
            return False
        try:
            resampled = _resample_audio(raw, musicgen_gen.sample_rate, MIXER_AUDIO_SR)
            vol = wavegan_playback_volume(
                level,
                audio_intensity,
                dissonance_value,
                sole_source=generated_audio_only,
            )
            if not generated_audio_only:
                vol *= 0.28
            snd = _to_stereo_sound(resampled, volume=vol)
            if out_channel.get_busy():
                try:
                    out_channel.fadeout(80 if fade_ms <= 100 else 120)
                except Exception:
                    pass
            out_channel.play(snd, loops=0, fade_ms=fade_ms)
            return True
        except Exception:
            return False

    # Pre-render one MusicGen clip per stress level so the demo always has a
    # ready-to-play bed the moment a level is entered. Synchronous because we
    # want it done before the pygame loop starts.
    if use_musicgen_bed and musicgen_gen is not None and musicgen_gen.available:
        print("[musicgen] pre-rendering 1 bed per stress level (L0..L5)...")
        prewarm_t0 = time.perf_counter()
        prewarm_profiles = [
            (0, 0.10, 0.06),
            (1, 0.22, 0.14),
            (2, 0.38, 0.26),
            (3, 0.55, 0.42),
            (4, 0.72, 0.58),
            (5, 0.92, 0.78),
        ]
        for lvl, ai_seed, dis_seed in prewarm_profiles:
            lt0 = time.perf_counter()
            clip = musicgen_gen.generate_now(
                lvl, ai_seed, dis_seed, seed=lvl * 9001 + 7919
            )
            if clip is not None and clip.size > 0:
                _remember_musicgen_clip(lvl, clip)
                print(
                    f"  [musicgen] L{lvl} ready ({clip.size / musicgen_gen.sample_rate:.2f}s, "
                    f"{time.perf_counter() - lt0:.1f}s wall)"
                )
            else:
                print(f"  [musicgen] L{lvl} FAILED to prewarm")
        print(
            f"[musicgen] prewarm complete: {sum(1 for v in musicgen_clip_cache.values() if v)}/6 levels "
            f"in {time.perf_counter() - prewarm_t0:.1f}s"
        )

    if model_only and not audio_gen.available and not procedural_gen.available:
        if musicgen_gen is None:
            missing.append(f"audio (WaveGAN at {AUDIO_MODEL} and/or MusicGen snapshot)")
        elif not musicgen_gen.available:
            if musicgen_gen.is_loading:
                pass
            else:
                err = musicgen_gen.last_load_error or "MusicGen failed to load"
                missing.append(f"audio MusicGen: {err[:200]}")

    if model_only and missing:
        raise SystemExit("Model-only mode missing required models: " + ", ".join(missing))
    if generated_audio_only:
        if use_musicgen_bed:
            print(
                "[audio] MusicGen bed (default when available); stress steers prompts. "
                "--audio-remote-ok allows HF Hub/cache; "
                "--audio-backend procedural for CPU granular synth; "
                "--audio-backend wavegan for WaveGAN.pt; "
                "--catalog-audio also mixes WAVs to fill load-time gaps."
            )
        elif use_procedural_bed:
            print(
                "[audio] Procedural granular bed (CPU, real horror grains + drone + transients). "
                "Use --audio-backend musicgen for the neural model (+ GPU if CUDA)."
            )
        else:
            print(
                "[audio] WaveGAN bed; use --audio-backend musicgen for local MusicGen (+ GPU if CUDA), "
                "or --audio-backend procedural for the granular synth. "
                "Pass --catalog-audio to mix catalog clips with the gen bed."
            )
        if not use_musicgen_bed and not use_procedural_bed and not audio_gen.available:
            print(
                "[audio] warning: no WaveGAN checkpoint and procedural unavailable; "
                "install models/audio_generator.pt, use --audio-backend procedural, "
                "use MusicGen (default backend), or use --catalog-audio"
            )
    gen_audio_channel = None
    ambient_channel = None
    tension_channel = None
    fx_channel = None
    if audio_initialized:
        try:
            ambient_channel = pygame.mixer.Channel(0)
            tension_channel = pygame.mixer.Channel(1)
            fx_channel = pygame.mixer.Channel(2)
            gen_audio_channel = pygame.mixer.Channel(3)
        except Exception:
            pass

    musicgen_warm_started = bool(musicgen_gen and use_musicgen_bed and musicgen_gen.available)
    musicgen_first_chunk_played = False
    musicgen_boot_level: int | None = None
    musicgen_placeholder_failed_faded = False
    if musicgen_gen and use_musicgen_bed and musicgen_gen.available:
        musicgen_gen.enqueue_replace(0, 0.1, 0.06, seed=7919)

    camera_feed = None
    show_camera = not args.no_camera
    if show_camera:
        camera_feed = CameraFeedProcessor(camera_id=0)

    current_level = 0
    prev_level = -1
    tick = 0
    texture_session_seed = random.getrandbits(32)
    last_audio_refresh = -999
    target_fps = max(4, min(60, int(args.fps)))
    if generated_audio_only and use_musicgen_bed:
        audio_refresh_interval = max(int(target_fps * 3.5), target_fps * 2)
    elif generated_audio_only and use_procedural_bed:
        audio_refresh_interval = max(12, int(target_fps * 0.92))
    elif generated_audio_only:
        audio_refresh_interval = max(12, int(target_fps * 0.92))
    else:
        audio_refresh_interval = 200
    last_fx_tick = -999
    audio_clip_index = 0
    prev_gen_bed_busy = False
    start = time.perf_counter()
    time_scale = max(0.2, min(2.0, float(args.time_scale)))
    running = True

    # Keep a small per-level pool so visuals stay stable-ish but not hard-stuck.
    texture_cache: dict[int, list[tuple[pygame.Surface, pygame.Surface]]] = {}
    # Keep one monster identity for both stress 4 and 5 in this run (from new entities folder).
    fixed_entity_designs = {4: run_fixed_design, 5: run_fixed_design}
    entity_design_id = fixed_entity_designs[4]
    entity_pool: list[tuple[pygame.Surface, str, int]] = []
    fixed_entity_bundles: dict[int, tuple[pygame.Surface, str, int] | None] = {4: None, 5: None}
    entity_pool_idx = 0
    entity_last_switch_tick = 0
    entity_motion_phase = 0.0
    # Mouse-driven entity camera state. Left-drag inside the entity panel
    # orbits the camera (yaw + pitch), right-click resets, wheel zooms.
    entity_cam_dragging = False
    entity_cam_drag_anchor: tuple[int, int] | None = None
    # Sensitivity tuned so a 360 spin takes about a screen-width of drag.
    entity_cam_deg_per_px_yaw = 0.55
    entity_cam_deg_per_px_pitch = 0.42

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_c:
                    show_camera = not show_camera
                elif event.key == pygame.K_r and entity_3d_renderer is not None:
                    entity_3d_renderer.reset_user_camera()
                elif event.key == pygame.K_0 or event.key == pygame.K_KP0:
                    current_level = 0
                elif event.key == pygame.K_1 or event.key == pygame.K_KP1:
                    current_level = 1
                elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                    current_level = 2
                elif event.key == pygame.K_3 or event.key == pygame.K_KP3:
                    current_level = 3
                elif event.key == pygame.K_4 or event.key == pygame.K_KP4:
                    current_level = 4
                elif event.key == pygame.K_5 or event.key == pygame.K_KP5:
                    current_level = 5
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if entity_3d_renderer is not None and entity_rect_static.collidepoint(event.pos):
                    if event.button == 1:
                        entity_cam_dragging = True
                        entity_cam_drag_anchor = event.pos
                    elif event.button == 3:
                        entity_3d_renderer.reset_user_camera()
                    elif event.button == 4:
                        entity_3d_renderer.adjust_user_camera(dzoom=-0.10)
                    elif event.button == 5:
                        entity_3d_renderer.adjust_user_camera(dzoom=0.10)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    entity_cam_dragging = False
                    entity_cam_drag_anchor = None
            elif event.type == pygame.MOUSEMOTION:
                if (
                    entity_cam_dragging
                    and entity_3d_renderer is not None
                    and entity_cam_drag_anchor is not None
                ):
                    dx = event.pos[0] - entity_cam_drag_anchor[0]
                    dy = event.pos[1] - entity_cam_drag_anchor[1]
                    entity_3d_renderer.adjust_user_camera(
                        dyaw=dx * entity_cam_deg_per_px_yaw,
                        dpitch=-dy * entity_cam_deg_per_px_pitch,
                    )
                    entity_cam_drag_anchor = event.pos
            elif event.type == pygame.MOUSEWHEEL and entity_3d_renderer is not None:
                mx, my = pygame.mouse.get_pos()
                if entity_rect_static.collidepoint(mx, my):
                    entity_3d_renderer.adjust_user_camera(dzoom=-0.10 * float(event.y))

        elapsed = (time.perf_counter() - start) * time_scale

        perf.begin()
        t_orch0 = time.perf_counter()
        out = orchestrator.step(requested_level=current_level, now_s=elapsed, mode="manual")
        t_orch_ms = (time.perf_counter() - t_orch0) * 1000.0
        perf_metrics = perf.end()
        payload = out.to_dict()
        level = payload["stress_event"]["level"]
        mp = payload.get("model_params", {})

        corruption = mp.get("visual_corruption", payload["visual_command"]["corruption_alpha"])
        fog = mp.get("visual_fog_density", payload["visual_command"]["fog_density"])
        flicker = mp.get("visual_flicker_rate", 0.1) * 3.0
        cam_mag = mp.get("camera_magnitude", payload["camera_command"]["magnitude"])
        cam_aberration = mp.get("camera_aberration", 0.0)
        cam_noise = mp.get("camera_noise", 0.0)
        cam_vignette = mp.get("camera_vignette", 0.0)
        cam_warp = mp.get("camera_warp", 0.0)
        audio_int = mp.get("audio_intensity", payload["audio_command"]["intensity"])
        dissonance = mp.get("audio_dissonance", 0.1)

        musicgen_placeholder_ch = None
        bed_channel_for_gen = None
        if audio_initialized:
            musicgen_placeholder_ch = ambient_channel if generated_audio_only else gen_audio_channel
            bed_channel_for_gen = musicgen_placeholder_ch

        busy_before_ph = bool(bed_channel_for_gen and bed_channel_for_gen.get_busy())
        oneshot_gen_armed = False
        if bed_channel_for_gen and audio_initialized:
            if (
                use_musicgen_bed
                and musicgen_gen
                and musicgen_gen.available
                and musicgen_first_chunk_played
            ):
                oneshot_gen_armed = True
            elif use_procedural_bed and procedural_gen.available:
                oneshot_gen_armed = True
            elif (not use_musicgen_bed) and (not use_procedural_bed) and audio_gen.available:
                oneshot_gen_armed = True
        immediate_gen_bed_refresh = bool(
            oneshot_gen_armed and prev_gen_bed_busy and not busy_before_ph
        )

        if audio_initialized and use_musicgen_bed and musicgen_gen and musicgen_placeholder_ch:
            mg = musicgen_gen
            settled_fail = (
                not mg.available
                and not mg.is_loading
                and (mg.last_load_error is not None)
            )
            if settled_fail and not musicgen_placeholder_failed_faded:
                try:
                    musicgen_placeholder_ch.fadeout(280)
                except Exception:
                    pass
                musicgen_placeholder_failed_faded = True
                if use_procedural_bed or audio_gen.available:
                    use_musicgen_bed = False
                    last_audio_refresh = -999
                    print("[audio] MusicGen failed after startup; switched to procedural/WaveGAN fallback.")

            needs_boot_ph = (
                not musicgen_first_chunk_played
                and not settled_fail
                and (mg.is_loading or mg.available)
            )
            gap_between_mg_clips = (
                musicgen_first_chunk_played
                and mg.available
                and not settled_fail
                and not musicgen_placeholder_ch.get_busy()
            )
            if needs_boot_ph or gap_between_mg_clips:
                should_start_ph = (
                    musicgen_boot_level is None
                    or musicgen_boot_level != level
                    or (
                        needs_boot_ph
                        and mg.is_loading
                        and not musicgen_placeholder_ch.get_busy()
                    )
                    or gap_between_mg_clips
                )
                catalog_gap = gap_between_mg_clips
                if should_start_ph:
                    if musicgen_boot_level is not None and musicgen_boot_level != level:
                        try:
                            musicgen_placeholder_ch.fadeout(85)
                        except Exception:
                            pass
                    # Prefer a cached MusicGen clip from THIS run before falling
                    # back to catalog/procedural/WaveGAN. In strict MusicGen-only
                    # mode the prewarm guarantees the cache always has something
                    # for the current level, so the gap filler never plays
                    # pre-stored audio. Bootstrap path only runs when the cache
                    # is genuinely empty (e.g. non-strict mode while MusicGen is
                    # still loading asynchronously).
                    played_cached_filler = False
                    inline_level_changed = (level != prev_level)
                    # On a fresh level change the dedicated cache-replay block
                    # below handles the cached MusicGen clip. Skip the cached
                    # play here so we don't fade across two clips in one frame.
                    cache_eligible = (
                        bool(musicgen_clip_cache.get(level))
                        and not inline_level_changed
                    )
                    if cache_eligible:
                        cached_raw = _next_cached_musicgen_clip(level)
                        if cached_raw is not None and _play_musicgen_array(
                            cached_raw,
                            level,
                            audio_int,
                            dissonance,
                            musicgen_placeholder_ch,
                            fade_ms=180 if musicgen_boot_level is None else 120,
                        ):
                            played_cached_filler = True
                            musicgen_boot_level = level
                            musicgen_first_chunk_played = True
                            print(
                                f"[musicgen-cache] gap-fill L{level} "
                                f"({len(musicgen_clip_cache[level])} cached)"
                            )
                    elif (
                        musicgen_clip_cache.get(level)
                        and inline_level_changed
                    ):
                        # Suppress catalog/procedural bootstrap on level change
                        # when we have a cached clip — the dedicated block
                        # below will play it cleanly with the right fade.
                        played_cached_filler = True
                        musicgen_boot_level = level

                    if not played_cached_filler:
                        clip_ix_gap = audio_clip_index + (
                            (tick >> 8) + level * 3 if catalog_gap else 0
                        )
                        bsnd = _make_musicgen_bootstrap_sound(
                            stress_level=level,
                            clip_index=clip_ix_gap,
                            real_audio=real_audio,
                            audio_gen=audio_gen,
                            audio_int=audio_int,
                            dissonance=dissonance,
                            sole_source_generated=generated_audio_only,
                            catalog_only_gap=catalog_gap,
                            ambient_picker=ambient_clip_picker,
                            procedural_gen=procedural_gen,
                        )
                        try:
                            musicgen_placeholder_ch.play(
                                bsnd,
                                loops=-1,
                                fade_ms=260 if musicgen_boot_level is None else 180,
                            )
                            musicgen_boot_level = level
                        except Exception:
                            pass

            if mg.available and not musicgen_warm_started:
                ws = (((tick ^ (level << 11)) ^ 7919) * 9973) & 0x7FFFFFFF
                mg.enqueue_replace(level, audio_int, dissonance, ws)
                musicgen_warm_started = True

        level_changed = (level != prev_level)
        timer_audio_refresh = tick - last_audio_refresh >= audio_refresh_interval
        need_catalog_refresh = level_changed or timer_audio_refresh
        need_gen_refresh = level_changed or timer_audio_refresh or immediate_gen_bed_refresh

        if level_changed:
            texture_cache.clear()
            if sprite_runtime:
                sprite_runtime.clear_cache()
            if level >= 4:
                entity_design_id = fixed_entity_designs[level]
            else:
                entity_design_id = _pick_entity_design(level, tick)
            entity_pool = []
            entity_pool_idx = 0
            entity_last_switch_tick = tick
            entity_motion_phase = random.Random(level * 19 + tick).uniform(0.0, math.pi * 2.0)
            if level >= 4:
                # Build one fixed bundle per run and reuse it (no live re-generation).
                if fixed_entity_bundles[level] is None:
                    fixed_entity_bundles[level] = _build_fixed_entity_bundle(
                        level=level,
                        design_id=fixed_entity_designs[level],
                        sprite_runtime=sprite_runtime,
                        entity_textures=entity_textures,
                        mesh_label=run_fixed_mesh_label,
                    )
                if fixed_entity_bundles[level] is not None:
                    entity_pool = [fixed_entity_bundles[level]]
            audio_clip_index = random.randint(0, 999)

        if need_catalog_refresh or need_gen_refresh:
            if not level_changed and (timer_audio_refresh or immediate_gen_bed_refresh):
                audio_clip_index += 1

            if need_catalog_refresh:
                if audio_initialized and real_audio.available and not generated_audio_only:
                    try:
                        if level_changed:
                            if ambient_channel:
                                ambient_channel.fadeout(500)
                            if tension_channel:
                                tension_channel.fadeout(400)
                        base_path = ambient_clip_picker.next_path(level) or real_audio.get_role_clip(
                            level, "ambient", audio_clip_index
                        )
                        tense_path = tension_clip_picker.next_path(level) or real_audio.get_role_clip(
                            level, "tension", audio_clip_index + 3
                        )
                        if ambient_channel and base_path:
                            amb_snd = pygame.mixer.Sound(base_path)
                            g = music_bed_level_gain(level)
                            amb_snd.set_volume(
                                (0.12 + 0.18 * audio_int + 0.03 * level) * g
                            )
                            ambient_channel.play(amb_snd, loops=-1, fade_ms=350)
                        if tension_channel and (audio_int > 0.2 or dissonance > 0.15) and tense_path:
                            ten_snd = pygame.mixer.Sound(tense_path)
                            g = music_bed_level_gain(level)
                            ten_snd.set_volume(
                                (0.05 + 0.15 * audio_int + 0.08 * dissonance) * g
                            )
                            tension_channel.play(ten_snd, loops=-1, fade_ms=250)
                    except Exception:
                        pass

            if need_gen_refresh:
                if generated_audio_only and audio_initialized and level_changed and ambient_channel:
                    if not (use_musicgen_bed and musicgen_gen and not musicgen_first_chunk_played):
                        try:
                            ambient_channel.fadeout(220)
                        except Exception:
                            pass

                bed_channel = (
                    ambient_channel if generated_audio_only else gen_audio_channel
                )
                if use_musicgen_bed and musicgen_gen:
                    gen_seed = (
                        (tick // 3) * 101
                        + level * 17
                        + audio_clip_index * 31
                        + (0x4B1D if immediate_gen_bed_refresh else 0)
                    ) & 0x7FFFFFFF
                    musicgen_gen.enqueue_replace(level, audio_int, dissonance, gen_seed)

                elif use_procedural_bed and procedural_gen.available and audio_initialized and bed_channel:
                    try:
                        proc_seed = (
                            ((tick & 0xFFFFF) * 1009)
                            ^ (level * 0x9E37)
                            ^ (audio_clip_index * 0xC2B2AE35)
                            ^ (0x27D4EB2F if immediate_gen_bed_refresh else 0)
                        ) & 0xFFFFFFFF
                        raw = procedural_gen.generate_clip(level, seed=proc_seed)
                        if raw is not None:
                            raw = postprocess_wavegan_for_stress(
                                raw, level, audio_int, dissonance
                            )
                            raw = _resample_audio(raw, MODEL_AUDIO_SR, MIXER_AUDIO_SR)
                            vol = wavegan_playback_volume(
                                level,
                                audio_int,
                                dissonance,
                                sole_source=generated_audio_only,
                            )
                            if not generated_audio_only:
                                vol *= 0.28
                            snd = _to_stereo_sound(raw, volume=vol)
                            bed_channel.play(snd, loops=0, fade_ms=90 if generated_audio_only else 0)
                    except Exception:
                        pass

                elif audio_gen.available and audio_initialized and bed_channel:
                    try:
                        base_seed = (
                            ((tick & 0xFFFFF) * 1109)
                            ^ (level * 224737)
                            ^ (audio_clip_index * 12582917)
                            ^ 0x9E3779B1
                            ^ (0x85EBCA6B if immediate_gen_bed_refresh else 0)
                        ) & 0xFFFFFFFF
                        for attempt in range(5):
                            raw = audio_gen.generate_clip(
                                level, seed=base_seed + attempt * 7919 + attempt * attempt
                            )
                            if raw is None:
                                continue
                            raw = postprocess_wavegan_for_stress(
                                raw, level, audio_int, dissonance
                            )
                            if attempt < 4 and not wavegan_quality_ok(
                                raw, level, audio_int, dissonance
                            ):
                                continue
                            raw = _resample_audio(raw, MODEL_AUDIO_SR, MIXER_AUDIO_SR)
                            vol = wavegan_playback_volume(
                                level,
                                audio_int,
                                dissonance,
                                sole_source=generated_audio_only,
                            )
                            if not generated_audio_only:
                                vol *= 0.28
                            snd = _to_stereo_sound(raw, volume=vol)
                            bed_channel.play(snd, loops=0, fade_ms=90 if generated_audio_only else 0)
                            break
                    except Exception:
                        pass

            last_audio_refresh = tick
            prev_level = level

        # Play a cached MusicGen clip immediately when re-entering a level.
        # In strict MusicGen-only mode repeats are allowed because there is no
        # other bed source. In mixed/fallback mode a single already-heard cache
        # entry is skipped so catalog/procedural fallback can cover the gap
        # until a fresh MusicGen render arrives.
        if (
            use_musicgen_bed
            and musicgen_gen
            and audio_initialized
            and level_changed
        ):
            out_ch_pre = ambient_channel if generated_audio_only else gen_audio_channel
            if out_ch_pre is not None:
                cached_raw = _next_cached_musicgen_clip(level)
                if cached_raw is not None:
                    if _play_musicgen_array(
                        cached_raw,
                        level,
                        audio_int,
                        dissonance,
                        out_ch_pre,
                        fade_ms=80,
                    ):
                        cache_count = len(musicgen_clip_cache.get(level, []))
                        print(f"[musicgen-cache] replay L{level} ({cache_count} cached)")
                    musicgen_first_chunk_played = True

        if use_musicgen_bed and musicgen_gen and audio_initialized:
            out_ch = ambient_channel if generated_audio_only else gen_audio_channel
            if out_ch:
                try:
                    mg_result = musicgen_gen.poll_result()
                    if mg_result is not None:
                        result_level = int(mg_result.get("level", level))
                        m_raw = mg_result.get("wave")
                        if isinstance(m_raw, np.ndarray) and m_raw.size > 0:
                            _remember_musicgen_clip(result_level, m_raw)
                        if (
                            isinstance(m_raw, np.ndarray)
                            and m_raw.size > 0
                            and result_level == level
                        ):
                            if not musicgen_first_chunk_played:
                                musicgen_first_chunk_played = True
                            _play_musicgen_array(
                                m_raw,
                                level,
                                audio_int,
                                dissonance,
                                out_ch,
                                fade_ms=140 if generated_audio_only else 0,
                            )
                            print(
                                f"[musicgen-cache] stored+played L{result_level} "
                                f"({len(musicgen_clip_cache[result_level])} cached)"
                            )
                        elif isinstance(m_raw, np.ndarray) and m_raw.size > 0:
                            print(
                                f"[musicgen-cache] stored L{result_level} for later "
                                f"({len(musicgen_clip_cache[result_level])} cached)"
                            )
                except Exception:
                    pass

        if (
            audio_initialized
            and fx_channel
            and not generated_audio_only
            and (real_audio.available or entity_audio_bank.available)
        ):
            fx_interval = max(30, int(200 - 100 * audio_int - 40 * dissonance))
            if tick - last_fx_tick >= fx_interval:
                if audio_int > 0.75 or dissonance > 0.65:
                    role = "impact"
                elif mp.get("entity_aggression", 0.0) > 0.45:
                    role = "creature"
                else:
                    role = "tension"
                # Creature FX are routed through the dedicated entity audio
                # bank (assets/entity_audio/) so monster sounds are kept fully
                # separate from the per-stress catalog bed.
                if role == "creature":
                    fx_path = (
                        entity_clip_picker.next_path()
                        if entity_audio_bank.available
                        else None
                    )
                else:
                    fx_p = {
                        "impact": impact_clip_picker,
                        "tension": tension_clip_picker,
                    }.get(role, tension_clip_picker)
                    fx_path = (
                        fx_p.next_path(level)
                        if fx_p is not None
                        else None
                    ) or real_audio.get_role_clip(level, role,
                                                  audio_clip_index + tick // max(1, fx_interval))
                if fx_path:
                    try:
                        fx_snd = pygame.mixer.Sound(fx_path)
                        fx_snd.set_volume(0.06 + 0.05 * level + 0.06 * audio_int)
                        fx_channel.play(fx_snd)
                        last_fx_tick = tick
                    except Exception:
                        pass

        screen.fill((10, 10, 14))

        env_rect = pygame.Rect(8, 8, half_w, 280)

        t_env0 = time.perf_counter()
        if level in texture_cache and texture_cache[level]:
            variants = texture_cache[level]
            # Slow deterministic cycling between prebuilt variants.
            v_idx = (tick // 180) % len(variants)
            env_base_raw, env_damaged_raw = variants[v_idx]
            tex_src = f"CACHED[{v_idx + 1}/{len(variants)}]"
        else:
            tex_src = "PROC"
            generated_variants: list[tuple[pygame.Surface, pygame.Surface]] = []
            num_variants = 3 if level >= 4 else 2
            for k in range(num_variants):
                env_surf = pygame.Surface((half_w, 280))
                env_surf.fill((30, 28, 25))
                if real_textures.available:
                    tex_src = "REAL"
                    compose = (100 + level * 101 + k * 37) ^ texture_session_seed
                    varied = build_varied_base_texture(
                        real_textures,
                        level,
                        half_w,
                        280,
                        compose,
                        texture_session_seed=texture_session_seed,
                    )
                    env_surf.blit(varied, (0, 0))
                else:
                    env_surf.blit(base_tex, (0, 0))

                env_plain_before_damage = env_surf.copy()

                tex_variant_salt = _texture_damage_seed(texture_session_seed, level, k * 101 + 4043)
                if texture_runtime.available:
                    corrupt_tick = int(tex_variant_salt % 99_983) ^ (k * 7937)
                    env_tex_k = texture_runtime.corrupt_surface(
                        env_surf, level, tick=corrupt_tick, strength=corruption
                    )
                    tex_src = "REAL+MODEL" if tex_src == "REAL" else "MODEL"
                    detail_alpha = min(0.75, 0.16 + 0.30 * corruption + 0.08 * level)
                    dmg_seed = _texture_damage_seed(texture_session_seed, level, k * 211 + 9091)
                    env_tex_k = apply_corruption(
                        env_tex_k,
                        detail_alpha,
                        stress_level=level,
                        damage_seed=dmg_seed,
                    )
                    if not model_only:
                        env_tex_k = apply_texture_distortion_hq(
                            env_tex_k,
                            level,
                            corruption * 0.35,
                            mp.get("visual_grime_overlay", 0.0) * 0.45,
                            tick + k * 9,
                            variation_seed=_texture_damage_seed(
                                texture_session_seed, level, k * 503 + 8801
                            ),
                        )
                else:
                    dmg_seed = _texture_damage_seed(texture_session_seed, level, k * 307 + 7177)
                    env_tex_k = apply_corruption(
                        env_surf,
                        corruption,
                        stress_level=level,
                        damage_seed=dmg_seed,
                    )
                    env_tex_k = apply_texture_distortion_hq(
                        env_tex_k,
                        level,
                        corruption,
                        mp.get("visual_grime_overlay", 0.0),
                        tick + k * 7,
                        variation_seed=_texture_damage_seed(
                            texture_session_seed, level, k * 409 + 661
                        ),
                    )
                generated_variants.append((env_plain_before_damage, env_tex_k.copy()))

            texture_cache[level] = generated_variants
            env_base_raw, env_damaged_raw = generated_variants[0]

        env_base_vis = env_base_raw.copy()
        env_final = env_damaged_raw.copy()

        # BASE pane is the raw aged catalog texture: no fog, no flicker.
        # Only the CORRUPTED pane gets fog + flicker, so the contrast between
        # "what the wall actually looks like" and "what stress paints onto it" is sharp.
        env_final = apply_fog(env_final, fog, tick)
        env_final = apply_light_flicker(env_final, flicker, tick)
        t_env_ms = (time.perf_counter() - t_env0) * 1000.0

        gap = 6
        hdr_h = 32
        row_top = env_rect.top + hdr_h
        pane_h = env_rect.bottom - row_top - 8
        pane_h = max(40, pane_h)
        pane_w = max(24, (env_rect.width - 12 - gap) // 2)
        left_r = pygame.Rect(env_rect.left + 6, row_top, pane_w, pane_h)
        right_r = pygame.Rect(env_rect.left + 6 + pane_w + gap, row_top, pane_w, pane_h)
        scaled_base = pygame.transform.smoothscale(env_base_vis, (left_r.width, left_r.height))
        scaled_damage = pygame.transform.smoothscale(env_final, (right_r.width, right_r.height))
        screen.blit(scaled_base, left_r.topleft)
        screen.blit(scaled_damage, right_r.topleft)
        pygame.draw.rect(screen, STRESS_COLORS[level], env_rect, 2)
        pygame.draw.line(
            screen, (42, 44, 58), (left_r.right + 2, row_top), (left_r.right + 2, row_top + pane_h), 1
        )
        font_label = pygame.font.SysFont("consolas", 11, bold=True)
        font_tiny = pygame.font.SysFont("consolas", 10, bold=True)
        lbl = font_label.render(
            f"ENVIRONMENT [{tex_src}]  corr={corruption:.2f}  fog={fog:.2f}",
            True, (200, 200, 210))
        screen.blit(lbl, (env_rect.left + 4, env_rect.top + 2))
        screen.blit(font_tiny.render("BASE (undamaged)", True, (150, 200, 160)), (left_r.left + 2, env_rect.top + 18))
        screen.blit(font_tiny.render("CORRUPTED", True, (220, 150, 130)), (right_r.left + 2, env_rect.top + 18))

        # --- Camera Distortion Panel ---
        # Uses the LIVE webcam frame (no synthetic proxy scene). Effects from
        # apply_camera_distortion_advanced sit on top of the real camera.
        t_cam0 = time.perf_counter()
        cam_rect = pygame.Rect(half_w + 18, 8, half_w, 280)
        live_rgb = None
        if show_camera and camera_feed is not None:
            live_rgb = camera_feed.get_raw_frame_rgb()
        if live_rgb is not None and live_rgb.size > 0:
            cam_source = pygame.surfarray.make_surface(
                np.transpose(live_rgb, (1, 0, 2))
            )
            cam_source = pygame.transform.smoothscale(cam_source, (half_w, 280))
            cam_source = apply_light_flicker(cam_source, flicker * 0.25, tick)
            cam_surf = apply_camera_distortion_advanced(
                cam_source, cam_mag, cam_aberration, cam_noise, cam_vignette, cam_warp, tick
            )
            scaled = pygame.transform.scale(cam_surf, (cam_rect.width, cam_rect.height))
            screen.blit(scaled, cam_rect.topleft)
            cam_src_label = "scene=live_webcam"
        else:
            pygame.draw.rect(screen, (8, 8, 12), cam_rect)
            font_off = pygame.font.SysFont("consolas", 14)
            off_lbl = font_off.render(
                "NO WEBCAM (synthetic camera removed)", True, (140, 70, 90)
            )
            screen.blit(
                off_lbl,
                (cam_rect.centerx - off_lbl.get_width() // 2,
                 cam_rect.centery - off_lbl.get_height() // 2),
            )
            cam_src_label = "scene=webcam_unavailable"
        t_cam_ms = (time.perf_counter() - t_cam0) * 1000.0
        pygame.draw.rect(screen, (180, 80, 200), cam_rect, 2)
        lbl2 = font_label.render(
            f"CAMERA DISTORTION  mag={cam_mag:.2f} ab={cam_aberration:.2f} n={cam_noise:.2f} v={cam_vignette:.2f} w={cam_warp:.2f}",
            True, (200, 120, 220))
        screen.blit(lbl2, (cam_rect.left + 4, cam_rect.top + 2))
        screen.blit(font_label.render(cam_src_label, True, (170, 110, 200)),
                    (cam_rect.left + 4, cam_rect.top + 16))

        # --- Audio Waveform Panel ---
        audio_rect = pygame.Rect(8, 296, content_w - 8, 120)
        t_audio_vis0 = time.perf_counter()
        draw_audio_waveform(screen, audio_rect, audio_int, dissonance, tick)
        t_audio_vis_ms = (time.perf_counter() - t_audio_vis0) * 1000.0
        audio_src_parts = []
        if generated_audio_only:
            if use_musicgen_bed and musicgen_gen:
                audio_src_parts.append("MUSICGEN")
            elif use_procedural_bed and procedural_gen.available:
                audio_src_parts.append("PROCEDURAL")
            elif audio_gen.available:
                audio_src_parts.append("WAVEGAN")
            else:
                audio_src_parts.append("GENERATED_UNAVAILABLE")
        else:
            if real_audio.available:
                audio_src_parts.append("REAL_BED")
            if use_musicgen_bed and musicgen_gen:
                audio_src_parts.append("MUSICGEN")
            elif use_procedural_bed and procedural_gen.available:
                audio_src_parts.append("PROCEDURAL")
            elif audio_gen.available:
                audio_src_parts.append("WAVEGAN")
        audio_src = "+".join(audio_src_parts) if audio_src_parts else "SILENT"
        lbl3 = font_label.render(
            f"AUDIO [{audio_src}]  intensity={audio_int:.2f}  dissonance={dissonance:.2f}",
            True, (220, 210, 60))
        screen.blit(lbl3, (audio_rect.left + 4, audio_rect.top + 4))

        # --- Camera Feed Panel ---
        feed_panel_w = content_w // 2 - 10
        feed_rect = pygame.Rect(8, 424, feed_panel_w, 540)
        if show_camera and camera_feed:
            feed_params = CameraFeedParams.from_stress(level, mp)
            draw_camera_feed_panel(screen, feed_rect, camera_feed, feed_params, tick)
        else:
            pygame.draw.rect(screen, (8, 8, 12), feed_rect)
            pygame.draw.rect(screen, (30, 50, 30), feed_rect, 1)
            font_off = pygame.font.SysFont("consolas", 12)
            off_lbl = font_off.render("CAMERA FEED [OFF] (press C)", True, (50, 80, 50))
            screen.blit(off_lbl, (feed_rect.centerx - off_lbl.get_width() // 2,
                                  feed_rect.centery))

        # --- Entity Panel ---
        # Same rect that mouse-drag uses for click detection. Defined once
        # before the loop so the hit-test and the draw site never disagree.
        entity_rect = entity_rect_static
        entity_params = EntityParams.from_model_output(mp)
        t_entity0 = time.perf_counter()
        drew_3d_entity = False
        if entity_3d_renderer is not None and level >= 4:
            entity_intensity = (
                0.55 + 0.45 * max(0.0, min(1.0, (level - 3) / 2.0))
            ) * (0.7 + 0.6 * float(mp.get("entity_presence", 1.0)))
            if entity_damage_gen is not None and entity_damage_gen.last_level != level:
                entity_damage_gen.regenerate(
                    entity_rect.width,
                    entity_rect.height,
                    level,
                    seed=int(time.time() * 17) ^ (level * 7919),
                )
            try:
                drew_3d_entity = draw_entity_panel_3d(
                    screen,
                    entity_rect,
                    entity_3d_renderer,
                    entity_damage_gen,
                    tick=tick,
                    level=level,
                    mesh_label=run_fixed_mesh_label,
                    intensity=entity_intensity,
                )
            except Exception as exc:
                print(f"[entity-3d] render error this frame: {exc}")
                drew_3d_entity = False

        if not drew_3d_entity:
            draw_entity_panel(
                screen, entity_rect, entity_params, tick, level,
                sprite_runtime=sprite_runtime, model_only=model_only,
                entity_design_id=entity_design_id,
                entity_texture_bank=entity_textures,
                entity_sprite_bundle=(entity_pool[entity_pool_idx] if entity_pool else None),
                entity_motion_phase=entity_motion_phase,
            )
        t_entity_ms = (time.perf_counter() - t_entity0) * 1000.0

        # --- DSP Info Bar ---
        dsp_rect = pygame.Rect(8, 968, content_w - 8, 48)
        pygame.draw.rect(screen, (12, 15, 12), dsp_rect)
        pygame.draw.rect(screen, (40, 60, 40), dsp_rect, 1)
        font_dsp = pygame.font.SysFont("consolas", 11)
        dsp_text = (
            f"DSP: pitch={mp.get('dsp_pitch_shift', 0):.2f} "
            f"dist={mp.get('dsp_distortion', 0):.2f} "
            f"filter={mp.get('dsp_filter_freq', 0):.2f} "
            f"reverb={mp.get('dsp_reverb_mix', 0):.2f} "
            f"blend={mp.get('dsp_layer_blend', 0):.2f} "
            f"pan={mp.get('dsp_pan', 0):.2f}"
        )
        screen.blit(font_dsp.render(dsp_text, True, (120, 200, 120)),
                    (dsp_rect.left + 6, dsp_rect.top + 6))
        dsp_text2 = f"Audio Gen: transient_rate={mp.get('audio_transient_rate', 0):.2f}  reverb_depth={mp.get('audio_reverb_depth', 0):.2f}"
        screen.blit(font_dsp.render(dsp_text2, True, (100, 180, 100)),
                    (dsp_rect.left + 6, dsp_rect.top + 22))

        cur_design_name = DESIGN_NAMES[entity_design_id % NUM_DESIGNS] if level >= 4 else "n/a"
        perf_text = (
            f"Inference(ms): orch={t_orch_ms:.1f} env={t_env_ms:.1f} "
            f"cam={t_cam_ms:.1f} entity={t_entity_ms:.1f} "
            f"| tex={tex_src} audio={audio_src} "
            f"design={cur_design_name}"
        )
        screen.blit(font_dsp.render(perf_text, True, (170, 170, 200)),
                    (dsp_rect.left + 6, dsp_rect.top + 36))

        # --- HUD ---
        draw_hud(screen, level, payload, perf_metrics, tick, elapsed, using_model)

        pygame.display.flip()
        if audio_initialized and bed_channel_for_gen is not None:
            prev_gen_bed_busy = bool(bed_channel_for_gen.get_busy())
        clock.tick(target_fps)
        tick += 1

    if musicgen_gen:
        musicgen_gen.shutdown()

    if camera_feed:
        camera_feed.release()
    if entity_3d_renderer is not None:
        try:
            entity_3d_renderer.shutdown()
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    main()
