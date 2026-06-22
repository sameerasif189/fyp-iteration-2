"""
Audio asset content filter.

Single source of truth for which audio files are excluded from the demo's
playback, generation, and training pipelines. Two reasons we filter:

1. **Appropriateness** - audio packs sometimes include sexualized loops
   (``Dom-Moan``, ``Sub-Blowjob``, etc.) and gendered human distress
   (``Demonic Woman Scream``). Those are not appropriate for the FYP demo
   regardless of horror context, so they are removed at every ingestion site.
2. **Tone consistency** - the demo is "abandoned-place horror sound design",
   not "human suffering". Pure-music loops (boom-bap kicks, hard-style kicks,
   melodic bass loops with BPM tags) muddy the bed and are also filtered.

The filter is intentionally conservative: only the filename basename is
inspected, so it cannot have a false positive on file *contents*. Add new
keywords below — the spelling check is a case-insensitive substring match.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

__all__ = [
    "FORBIDDEN_KEYWORDS",
    "MUSIC_KEYWORDS",
    "is_forbidden_audio",
    "filter_audio_paths",
    "filter_forbidden_only",
]

# Hard block — gendered human distress + sexual loops + anything that reads as
# a human (or human-like) scream/wail/cry. Fox vocalisations are notorious for
# sounding like a woman screaming, so we explicitly block fox-vocal files too.
FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    # Gendered human references
    "woman", "women", "female", "girl", "girls", "lady", "ladies",
    "male",  # catches "male" and "female" (substring); blocks "young male" too
    # Sexual / inappropriate content
    "moan", "moaning", "groan", "groaning",
    "blowjob", "blow_job", "blow-job", "blowing",  # "Male creepy blowing"
    "porn", "erotic", "nsfw",
    "dom-",  "-dom_", "_dom-", " dom ",
    "sub-blowjob", "sub_blow",
    "intense-bpm", "intensebpm",
    "boyfriend", "girlfriend", "kiss-",
    # Human-distress vocalisations (block broadly; the demo still has plenty of
    # pure mechanical / drone / ambient horror grains left over).
    "scream", "screams", "screaming",
    "wail", "wails", "wailing",
    "shriek", "shrieks", "shrieking",
    "screech", "screeches", "screeching",
    "cry,", "_cry", "-cry", " cry",  # specific to avoid matching "crystal" etc
    "crying", "weep", "weeping", "sob ", "sobbing",
    "laugh", "laughter", "laughing", "giggle", "giggling",
    "chant", "chanting", "singing", "vocal",
    "breath", "breathing", "panting", "gasp", "gasping",
    # Fox vocalisations -- acoustically indistinguishable from a woman screaming.
    "fox_scream", "fox-scream", "fox scream", "fox,",
    "fox vocal", "fox_vocal", "fox-vocal",
    # Generic "demonic"/"human" voice tags often carry gendered distress.
    "demonic woman", "demonic-woman",
    "young male", "young female", "young woman", "young man",
)

# Soft block — overtly musical loops. The demo is sound design, not score, so
# tagged BPM loops, drum kits, kicks, basslines, EDM stylings dilute the horror.
# These are filtered at L4/L5 only (where MusicGen / procedural drive the bed)
# but allowed at lower stress where catalog ambient texture is welcome.
MUSIC_KEYWORDS: tuple[str, ...] = (
    "hardstyle", "kick", "kik ", "kik_", "808",
    "synthbass loop", "bassline loop", "drum loop", "drums loop",
    "ripped apart beat", "ripped-apart-beat",
    "epic logo", "request the honour",
    "piano synth drums loop",
)


def _basename_lower(p: str | os.PathLike[str]) -> str:
    """Lowercase basename — only the filename, not the whole path."""
    return Path(str(p)).name.lower()


def is_forbidden_audio(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` should be excluded everywhere (sexual / gendered distress)."""
    name = _basename_lower(path)
    return any(kw in name for kw in FORBIDDEN_KEYWORDS)


def is_musicy_audio(path: str | os.PathLike[str]) -> bool:
    """True if ``path`` is an overtly musical loop (BPM-tagged, drum kit, etc.)."""
    name = _basename_lower(path)
    return any(kw in name for kw in MUSIC_KEYWORDS)


def filter_forbidden_only(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    """Drop only the hard-blocked files; keep musical loops."""
    return [str(p) for p in paths if not is_forbidden_audio(p)]


def filter_audio_paths(
    paths: Iterable[str | os.PathLike[str]],
    *,
    drop_music: bool = False,
) -> list[str]:
    """Drop hard-blocked files; optionally also drop musical loops."""
    out: list[str] = []
    for p in paths:
        if is_forbidden_audio(p):
            continue
        if drop_music and is_musicy_audio(p):
            continue
        out.append(str(p))
    return out
