"""
Drive WaveGAN playback so stress level 0–5 matches the fusion schema:
ambient / wary unease at low levels → full horror density at 5.

Uses discrete level (WaveGAN conditioning) plus continuous fusion heads
(audio_intensity, audio_dissonance) so runtime sound follows the trained model,
not a fixed post chain.
"""

from __future__ import annotations

import numpy as np

NUM_STRESS_LEVELS = 6

_KERNEL_NARROW = np.array([0.08, 0.12, 0.16, 0.28, 0.16, 0.12, 0.08], dtype=np.float32)
_KERNEL_WIDE = np.array(
    [0.02, 0.04, 0.07, 0.10, 0.14, 0.18, 0.22, 0.18, 0.14, 0.10, 0.07, 0.04, 0.02],
    dtype=np.float32,
)


def _clip_level(level: int) -> int:
    return max(0, min(NUM_STRESS_LEVELS - 1, int(level)))


# Per-discrete-stress gain for generated bed + optional catalog scaling (0 = calm, 5 = peak).
# L4/L5 are deliberately pushed past 1.0 so volume saturates at the mixer hard
# ceiling -- combined with the procedural generator's higher event density this
# gives terrified tiers the perceived loudness/energy jump the calm tiers lack.
_MUSIC_BED_LEVEL_GAIN = np.array(
    [0.68, 0.74, 0.84, 0.96, 1.55, 1.95],
    dtype=np.float32,
)


def music_bed_level_gain(level: int) -> float:
    """Softer 0--3, clearly louder 4--5; multiplies MusicGen / WaveGAN / optional catalog bed."""
    i = _clip_level(level)
    return float(_MUSIC_BED_LEVEL_GAIN[i])


def ambient_smoothing_mix(level: int, audio_intensity: float, dissonance: float) -> float:
    """
    How much smoothing to apply: ~1 at level 0 (soft bed), ~0 at level 5 (raw horror).
    Fusion heads reduce smoothing when the model asks for aggression even mid-level.
    """
    lvl = _clip_level(level)
    tier = lvl / max(1, NUM_STRESS_LEVELS - 1)
    # Steeper than linear: low tiers stay mellow longer; high tiers open up fast.
    base = float(np.clip(1.0 - (tier ** 1.65), 0.0, 1.0))
    aggression = float(np.clip(audio_intensity, 0.0, 1.0)) * 0.45 + float(np.clip(dissonance, 0.0, 1.0)) * 0.32
    floor = 0.032 + 0.10 * float(1.0 - tier)
    return float(np.clip(base - aggression, floor, 0.93))


def postprocess_wavegan_for_stress(
    audio: np.ndarray,
    level: int,
    audio_intensity: float,
    dissonance: float,
) -> np.ndarray:
    """Level-aware smoothing: calm levels stay mellow; terrified levels keep grit and hits."""
    x = np.asarray(audio, dtype=np.float32).copy()
    if x.size == 0:
        return x
    x = np.clip(x, -1.0, 1.0)
    x = x - float(np.mean(x))

    lvl = _clip_level(level)
    tier = lvl / max(1, NUM_STRESS_LEVELS - 1)
    ai = float(np.clip(audio_intensity, 0.0, 1.0))
    dis = float(np.clip(dissonance, 0.0, 1.0))

    mix = ambient_smoothing_mix(level, audio_intensity, dissonance)
    kn = _KERNEL_NARROW / np.sum(_KERNEL_NARROW)
    kw = _KERNEL_WIDE / np.sum(_KERNEL_WIDE)
    narrow = np.convolve(x, kn, mode="same")
    wide = np.convolve(x, kw, mode="same")
    # blend wide->narrow as horror rises (mix drops)
    band = (1.0 - mix) * narrow + mix * wide
    grit = narrow - wide
    grit_w = float((0.05 + 0.20 * tier + 0.12 * dis) * (1.0 - 0.62 * mix))
    band = band + grit_w * grit
    x = band

    if tier > 0.22:
        win = max(24, min(x.size // 40, 600))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        low = np.convolve(x, kernel, mode="same")
        rumble_amp = float(0.065 * tier**1.2 + 0.075 * ai + 0.045 * dis)
        x = x + rumble_amp * low

    drive = float(0.82 + 0.95 * tier + 0.52 * ai + 0.38 * dis)
    x = np.tanh(np.clip(x, -8.0, 8.0) * drive)
    peak_drive = float(np.max(np.abs(x))) + 1e-8
    if peak_drive > 1e-6:
        x = x / peak_drive

    fade_n = min(512, max(64, x.size // 30))
    env = np.ones_like(x)
    env[:fade_n] = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    env[-fade_n:] = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    x *= env
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = x / peak
    return x


def wavegan_quality_ok(
    audio: np.ndarray,
    level: int,
    audio_intensity: float,
    dissonance: float = 0.0,
) -> bool:
    """
    Reject empty / dead / pure-hiss takes at low levels; allow harsh transients at 4–5.
    """
    if audio.size < 256:
        return False
    x = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-8))
    if rms < 0.018:
        return False
    zcr = float(((x[:-1] * x[1:]) < 0).mean())
    spec = np.fft.rfft(x)
    mag = np.abs(spec) + 1e-8
    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))

    lvl = _clip_level(level)
    tier = lvl / max(1, NUM_STRESS_LEVELS - 1)
    amp = float(np.clip(audio_intensity, 0.0, 1.0))
    di = float(np.clip(dissonance, 0.0, 1.0))
    chaos = float(np.clip(amp * 0.55 + di * 0.42, 0.0, 1.0))
    tier_edge = tier * tier

    zcr_max = 0.30 + 0.28 * tier + 0.12 * amp + 0.18 * chaos + 0.22 * chaos * tier_edge
    flat_max = 0.70 + 0.15 * tier + 0.06 * amp + 0.12 * chaos + 0.28 * chaos * tier_edge
    if zcr > zcr_max:
        return False
    if flatness > flat_max:
        return False
    return True


def wavegan_playback_volume(
    level: int,
    audio_intensity: float,
    dissonance: float,
    *,
    sole_source: bool,
) -> float:
    """
    Level 0--2 stay relatively quiet (ambient dread); 4--5 punch harder while still
    following fusion intensity/dissonance scalars from the checkpoint.
    """
    lvl = _clip_level(level)
    tier = lvl / max(1, NUM_STRESS_LEVELS - 1)
    ai = float(np.clip(audio_intensity, 0.0, 1.0))
    dis = float(np.clip(dissonance, 0.0, 1.0))
    # Discrete level steps 0→5 (+ continuous heads); final curve also uses music_bed_level_gain.
    fusion = 0.21 * ai + 0.052 * float(lvl) + 0.098 * dis
    ambient_headroom = (1.0 - tier) * (-0.065)
    panic_lift = (tier**2) * 0.062
    if lvl >= 4:
        # Stronger lift at L4/L5 so they punch above L3 even when fusion heads
        # are mid-range. Previous values gave only ~0.05/0.08 extra; doubled.
        panic_lift += 0.095 * float(lvl - 3)
    if lvl >= 5:
        panic_lift += 0.075
    vol = 0.11 + fusion + ambient_headroom + panic_lift
    if sole_source:
        vol += 0.038
    vol *= music_bed_level_gain(lvl)
    return float(np.clip(vol, 0.03, 1.0))
