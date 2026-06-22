"""
Procedural granular horror audio generator.

Drop-in replacement for ``RealtimeAudioGen`` (the WaveGAN wrapper in ``gui_demo``):
synthesises 1s @ 16kHz horror beds on the CPU in well under 20 ms per clip.

A 1s clip is mixed from four layers, weighted by stress level 0--5:

  1. ``drone``     - sub-bass sine + slow FM + inharmonic beating partial
  2. ``noise_bed`` - pink/brown noise -> bandpass with LFO sweep
  3. ``grains``    - 4--22 short grains (~80-220ms) sampled from
                     ``assets/wildlife`` and ``assets/audio/{level}``, pitch-shifted
                     and Hann-enveloped
  4. ``transients``- ring-modulated impulses with exp decay (active stress >= 3)

The output is normalized to roughly +/- 0.9 peak and then handed off to the
existing ``postprocess_wavegan_for_stress`` chain in ``audio_horror_profile`` so
the rest of the pipeline (smoothing, grit, rumble, level-aware volume) stays
identical to the WaveGAN path.

Public surface mirrors ``RealtimeAudioGen``:

    gen = ProceduralAudioGen(assets_dir)
    if gen.available:
        clip = gen.generate_clip(stress_level=3, seed=42)   # np.float32, shape (16000,)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from .audio_content_filter import (
    filter_audio_paths,
    is_forbidden_audio,
    is_musicy_audio,
)


SAMPLE_RATE = 16000
CLIP_LEN = 16000
NUM_STRESS_LEVELS = 6


_LEVEL_LAYER_WEIGHTS: Dict[int, Dict[str, float]] = {
    0: {"drone": 0.46, "noise": 0.30, "grains": 0.22, "transients": 0.02},
    1: {"drone": 0.40, "noise": 0.30, "grains": 0.27, "transients": 0.03},
    2: {"drone": 0.34, "noise": 0.28, "grains": 0.32, "transients": 0.06},
    3: {"drone": 0.28, "noise": 0.24, "grains": 0.36, "transients": 0.12},
    # L4/L5 lean hard on grains + transients so the field feels chaotic, not
    # just "louder ambient". Drone/noise are deliberately pulled back to make
    # room for impact density.
    4: {"drone": 0.18, "noise": 0.16, "grains": 0.46, "transients": 0.30},
    5: {"drone": 0.13, "noise": 0.12, "grains": 0.48, "transients": 0.42},
}

_LEVEL_GRAIN_COUNT: Dict[int, tuple[int, int]] = {
    0: (2, 4),
    1: (3, 6),
    2: (4, 9),
    3: (6, 12),
    # Much thicker overlapping-grain texture at high stress: 1 sec of audio
    # carries 15-30+ pitched/reversed sample-fragments stacked on top of each
    # other for a properly panicked sound bed.
    4: (15, 30),
    # L5 is the user is *at end*: relentless, no breathing room, never ambient.
    5: (32, 56),
}

_LEVEL_TRANSIENT_COUNT: Dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (0, 0),
    2: (0, 1),
    3: (1, 3),
    # L4/L5 ramp transient *density* dramatically: many short impact/screech
    # bursts per second. This is what carries the "I am in danger right now"
    # feeling that the previous tuning was missing.
    4: (5, 12),
    # L5 transient floor is loud and constant -- silence between hits is not
    # allowed at peak stress.
    5: (16, 28),
}

_LEVEL_DRIVE: Dict[int, float] = {
    0: 0.85, 1: 0.95, 2: 1.05, 3: 1.20,
    # Harder waveshaping at peak tiers -> more harmonic harshness without
    # blowing past the sample headroom. L5 is intentionally overdriven so the
    # waveshaper smashes the mix flat against the rails for a wall-of-sound feel.
    4: 1.85, 5: 3.20,
}


# ---------------------------------------------------------------------------
# Style variants: each generated clip rolls one of these "moods" so the
# fallback bed feels different every refresh even within the same stress
# level. Multipliers stack on top of _LEVEL_LAYER_WEIGHTS / _LEVEL_DRIVE; the
# parameter tweaks reshape drone, noise filter character, and grain/transient
# pitch ranges so consecutive clips don't sound like the same patch.
# ---------------------------------------------------------------------------
_STYLE_VARIANTS: List[Dict[str, float]] = [
    # tunnel: heavy drone, sparse grains, long-tail noise
    {"name_id": 0, "drone": 1.45, "noise": 1.20, "grains": 0.55, "transients": 0.65,
     "drone_freq_mul": 0.65, "noise_q": 0.7, "pitch_mul": 0.7, "drive_mul": 0.85},
    # swarm: thick overlapping grains, busy mid-band
    {"name_id": 1, "drone": 0.55, "noise": 0.75, "grains": 1.55, "transients": 1.10,
     "drone_freq_mul": 1.05, "noise_q": 1.2, "pitch_mul": 1.35, "drive_mul": 1.05},
    # ironworks: metallic ring-mod transients dominate
    {"name_id": 2, "drone": 0.70, "noise": 0.65, "grains": 0.85, "transients": 1.85,
     "drone_freq_mul": 1.40, "noise_q": 1.4, "pitch_mul": 1.15, "drive_mul": 1.20},
    # wet: low slow noise bed, low-pitched grains, drips
    {"name_id": 3, "drone": 1.10, "noise": 1.35, "grains": 0.95, "transients": 0.50,
     "drone_freq_mul": 0.80, "noise_q": 0.6, "pitch_mul": 0.55, "drive_mul": 0.90},
    # near: dry close-mic chaos, punchy transients
    {"name_id": 4, "drone": 0.45, "noise": 0.55, "grains": 1.10, "transients": 1.60,
     "drone_freq_mul": 1.15, "noise_q": 1.1, "pitch_mul": 1.05, "drive_mul": 1.30},
    # far: distant rumble, long tails
    {"name_id": 5, "drone": 1.35, "noise": 1.15, "grains": 0.75, "transients": 0.45,
     "drone_freq_mul": 0.55, "noise_q": 0.5, "pitch_mul": 0.85, "drive_mul": 0.80},
    # void: sub-only drone, hollow noise, sparse impacts
    {"name_id": 6, "drone": 1.70, "noise": 0.65, "grains": 0.55, "transients": 0.70,
     "drone_freq_mul": 0.45, "noise_q": 0.8, "pitch_mul": 0.65, "drive_mul": 0.85},
    # shriek: high pitch grain emphasis, screech transients
    {"name_id": 7, "drone": 0.55, "noise": 0.85, "grains": 1.40, "transients": 1.55,
     "drone_freq_mul": 1.25, "noise_q": 1.5, "pitch_mul": 1.75, "drive_mul": 1.25},
    # cavern: huge low-frequency space, slow movement, sparse echo-hits
    {"name_id": 8, "drone": 1.60, "noise": 1.05, "grains": 0.65, "transients": 0.55,
     "drone_freq_mul": 0.50, "noise_q": 0.6, "pitch_mul": 0.75, "drive_mul": 0.78},
    # static: broadband hiss-dominant, broken-radio character
    {"name_id": 9, "drone": 0.40, "noise": 1.75, "grains": 0.85, "transients": 0.95,
     "drone_freq_mul": 0.95, "noise_q": 1.8, "pitch_mul": 1.10, "drive_mul": 1.15},
    # decay: detuned drone with falling grains, "going wrong" feeling
    {"name_id": 10, "drone": 1.25, "noise": 0.80, "grains": 1.20, "transients": 0.85,
     "drone_freq_mul": 0.75, "noise_q": 0.9, "pitch_mul": 0.65, "drive_mul": 0.95},
    # frenzy: maxed transients + dense grains, rolling chaos
    {"name_id": 11, "drone": 0.45, "noise": 0.60, "grains": 1.60, "transients": 2.10,
     "drone_freq_mul": 1.20, "noise_q": 1.3, "pitch_mul": 1.45, "drive_mul": 1.35},
]


def _pick_style(
    rng: np.random.Generator,
    level: int,
    *,
    recent_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Pick a style for this clip with anti-repeat. Probability is level-aware
    (calm styles dominate low stress, aggressive styles dominate high stress),
    and any style used in the last few clips gets heavily down-weighted so the
    fallback bed stops feeling like a metronome."""
    # 12-style weight tables (last 4 are: cavern, static, decay, frenzy)
    if level <= 1:
        weights = np.array(
            [2.5, 0.4, 0.3, 1.5, 0.4, 2.0, 2.2, 0.3,  2.0, 1.8, 1.4, 0.3],
            dtype=np.float32,
        )
    elif level == 2:
        weights = np.array(
            [1.8, 1.0, 0.8, 1.5, 1.0, 1.6, 1.4, 0.8,  1.6, 1.6, 1.5, 0.7],
            dtype=np.float32,
        )
    elif level == 3:
        weights = np.array(
            [1.3, 1.6, 1.2, 1.3, 1.4, 1.2, 0.9, 1.4,  1.2, 1.5, 1.5, 1.3],
            dtype=np.float32,
        )
    elif level == 4:
        weights = np.array(
            [0.8, 2.0, 1.8, 0.9, 1.7, 0.7, 0.5, 1.9,  0.7, 1.4, 1.5, 2.1],
            dtype=np.float32,
        )
    else:
        # L5 = "the user is at end". Zero out every style with even a hint of
        # ambient breathing room (tunnel/wet/far/void/cavern/decay) so only the
        # relentless ones (swarm/ironworks/near/shriek/static/frenzy) can play.
        # Style ids in order: tunnel(0) swarm(1) ironworks(2) wet(3) near(4)
        # far(5) void(6) shriek(7) cavern(8) static(9) decay(10) frenzy(11)
        weights = np.array(
            [0.0, 2.6, 2.4, 0.0, 2.3, 0.0, 0.0, 2.6,  0.0, 1.4, 0.0, 3.0],
            dtype=np.float32,
        )

    if recent_ids:
        # Strong damping for the most recent style, lighter for older ones.
        damp = [0.15, 0.35, 0.55, 0.75]
        for slot, sid in enumerate(reversed(recent_ids[-4:])):
            if 0 <= sid < weights.size:
                weights[sid] *= damp[min(slot, len(damp) - 1)]
        if float(weights.sum()) <= 1e-6:
            weights = weights + 0.01

    weights = weights / weights.sum()
    idx = int(rng.choice(len(_STYLE_VARIANTS), p=weights))
    return _STYLE_VARIANTS[idx]


def _clip_level(level: int) -> int:
    return max(0, min(NUM_STRESS_LEVELS - 1, int(level)))


def _hann_envelope(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float32)
    t = np.linspace(0.0, np.pi, n, dtype=np.float32)
    return np.sin(t).astype(np.float32)


def _pitch_shift(samples: np.ndarray, semitones: float) -> np.ndarray:
    if abs(semitones) < 0.01 or samples.size < 4:
        return samples.astype(np.float32, copy=False)
    ratio = 2.0 ** (semitones / 12.0)
    n_out = max(2, int(round(samples.size / ratio)))
    src_x = np.arange(samples.size, dtype=np.float32)
    dst_x = np.linspace(0.0, samples.size - 1.0, n_out, dtype=np.float32)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Voss-McCartney-style approximation - cheap pink-ish 1/f noise."""
    white = rng.standard_normal(n).astype(np.float32)
    out = np.empty_like(white)
    acc = 0.0
    a = 0.985
    for i in range(n):
        acc = a * acc + (1.0 - a) * white[i]
        out[i] = acc + 0.45 * white[i]
    peak = float(np.max(np.abs(out))) + 1e-9
    return (out / peak).astype(np.float32)


def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32) * 0.06
    out = np.cumsum(white, dtype=np.float32)
    out = out - float(np.mean(out))
    peak = float(np.max(np.abs(out))) + 1e-9
    return (out / peak).astype(np.float32)


def _biquad_bandpass(audio: np.ndarray, center: np.ndarray, q: float,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """Time-varying single-pole bandpass approximation (cheap + stable)."""
    n = audio.size
    if center.size != n:
        center = np.interp(np.linspace(0, center.size - 1, n),
                           np.arange(center.size), center).astype(np.float32)
    center = np.clip(center, 30.0, sr * 0.45)
    rc = 1.0 / (2.0 * np.pi * np.maximum(center, 1.0))
    dt = 1.0 / sr
    alpha_lp = (dt / (rc + dt)).astype(np.float32)
    alpha_hp = (rc / (rc + dt)).astype(np.float32)

    lp = np.zeros(n, dtype=np.float32)
    lp[0] = alpha_lp[0] * audio[0]
    for i in range(1, n):
        lp[i] = lp[i - 1] + alpha_lp[i] * (audio[i] - lp[i - 1])

    hp = np.zeros(n, dtype=np.float32)
    hp[0] = lp[0]
    for i in range(1, n):
        hp[i] = alpha_hp[i] * (hp[i - 1] + lp[i] - lp[i - 1])

    bp = hp - hp.mean()
    peak = float(np.max(np.abs(bp))) + 1e-9
    return (bp / peak * q).astype(np.float32)


class ProceduralAudioGen:
    """Granular + procedural horror audio synth on CPU (no checkpoint, no GPU)."""

    def __init__(
        self,
        assets_dir: Optional[Path] = None,
        *,
        max_files_per_level: int = 22,
        max_grains_per_level: int = 192,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.sample_rate = SAMPLE_RATE
        self.clip_len = CLIP_LEN
        self._rng_seed = int(seed)
        self._grain_bank: Dict[int, List[np.ndarray]] = {
            i: [] for i in range(NUM_STRESS_LEVELS)
        }
        self._has_grains = False
        self._load_error: str | None = None
        # Per-level recent style history used to suppress back-to-back repeats
        # so consecutive procedural clips at the same stress level don't sound
        # like the same patch.
        self._recent_styles: Dict[int, List[int]] = {i: [] for i in range(NUM_STRESS_LEVELS)}

        if assets_dir is not None:
            try:
                self._load_grains(Path(assets_dir),
                                  max_files_per_level=max_files_per_level,
                                  max_grains_per_level=max_grains_per_level,
                                  verbose=verbose)
            except Exception as e:  # pragma: no cover - asset loading is best-effort
                self._load_error = str(e)
                if verbose:
                    print(f"[procedural-audio] grain load failed: {e}")

        self._has_grains = any(len(v) > 0 for v in self._grain_bank.values())
        if verbose:
            counts = ", ".join(
                f"L{lvl}={len(v)}" for lvl, v in self._grain_bank.items()
            )
            print(f"[procedural-audio] grain bank ready ({counts})")

    @property
    def available(self) -> bool:
        return True

    @property
    def last_load_error(self) -> str | None:
        return self._load_error

    def _load_grains(self, assets_dir: Path, *, max_files_per_level: int,
                     max_grains_per_level: int, verbose: bool) -> None:
        if not HAS_LIBROSA:
            self._load_error = "librosa not installed; procedural will run drone+noise only"
            if verbose:
                print(f"[procedural-audio] {self._load_error}")
            return

        audio_root = assets_dir / "audio"
        rng = random.Random(self._rng_seed)
        n_blocked = 0
        n_music_skipped = 0

        # Sliding 2-folder window per stress level:
        #   L0 -> folder 0
        #   L1 -> folders 0 + 1
        #   L2 -> folders 1 + 2
        #   L3 -> folders 2 + 3
        #   L4 -> folders 3 + 4
        #   L5 -> folders 4 + 5
        # This blends adjacent stress folders so each level gets more variety
        # and consecutive levels share some material for smoother transitions.
        def _folder_window_for_level(lvl: int) -> List[int]:
            if lvl <= 0:
                return [0]
            return [lvl - 1, lvl]

        for level in range(NUM_STRESS_LEVELS):
            files: List[Path] = []
            for folder_idx in _folder_window_for_level(level):
                folder_path = audio_root / str(folder_idx)
                if folder_path.is_dir():
                    files.extend(sorted(folder_path.glob("*.mp3")))
                    files.extend(sorted(folder_path.glob("*.wav")))
            # De-duplicate after combining two folders (rarely identical, but
            # protects against any catalog overlap).
            seen: set[str] = set()
            uniq: List[Path] = []
            for f in files:
                key = str(f).lower()
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(f)
            files = uniq
            # Hard content filter (sexual / gendered distress) always applies;
            # musical-loop filter only at L4/L5 where we want pure horror SFX.
            raw_before = len(files)
            files = [
                Path(p) for p in filter_audio_paths(
                    [str(f) for f in files],
                    drop_music=(level >= 4),
                )
            ]
            for folder_idx in _folder_window_for_level(level):
                n_blocked += sum(
                    1 for f in (audio_root / str(folder_idx)).glob("*")
                    if is_forbidden_audio(f)
                )
            if level >= 4:
                n_music_skipped += raw_before - len(files)
            rng.shuffle(files)
            for fp in files[:max_files_per_level]:
                self._extract_grains_into(level, fp, max_grains_per_level)
                if len(self._grain_bank[level]) >= max_grains_per_level:
                    break

        # NOTE: wildlife/entity_audio are NOT mixed into procedural grains any
        # more. Procedural strictly follows the sliding 2-folder window over
        # assets/audio/{level-1, level}. Entity sounds are handled by
        # EntityAudioBank in gui_demo.py (separate FX one-shots).

        if verbose and (n_blocked or n_music_skipped):
            print(
                f"[procedural-audio] content filter: forbidden={n_blocked} "
                f"musical_loops_dropped_L4_L5={n_music_skipped}"
            )

    def _extract_grains_into(self, level: int, fp: Path, cap: int) -> None:
        try:
            audio, _ = librosa.load(str(fp), sr=SAMPLE_RATE, mono=True)
        except Exception:
            return
        if audio.size < int(0.1 * SAMPLE_RATE):
            return

        audio = audio.astype(np.float32)
        rms_full = float(np.sqrt(np.mean(audio * audio) + 1e-9))
        if rms_full < 0.005:
            return
        audio = audio / (rms_full + 1e-9) * 0.25

        rng = np.random.default_rng(abs(hash((str(fp), level))) & 0xFFFFFFFF)
        min_g = int(0.06 * SAMPLE_RATE)
        max_g = int(0.22 * SAMPLE_RATE)
        n_picks = min(8, (audio.size // max_g) + 1)
        for _ in range(n_picks):
            if len(self._grain_bank[level]) >= cap:
                return
            length = int(rng.integers(min_g, max_g + 1))
            if length >= audio.size:
                continue
            start = int(rng.integers(0, audio.size - length))
            grain = audio[start:start + length].copy()
            peak = float(np.max(np.abs(grain)))
            if peak < 0.015:
                continue
            grain *= (0.6 / max(peak, 1e-6))
            grain = grain * _hann_envelope(grain.size)
            self._grain_bank[level].append(grain.astype(np.float32))

    def _drone(self, level: int, rng: np.random.Generator,
               freq_mul: float = 1.0) -> np.ndarray:
        t = np.arange(self.clip_len, dtype=np.float32) / SAMPLE_RATE
        tier = level / 5.0
        # Per-clip glissando: starts at fc, drifts +/- semitones over the
        # second so consecutive clips never sound static at the same level.
        glide_oct = float(rng.uniform(-0.35, 0.35) + (rng.uniform(-0.25, 0.25) * tier))
        gliss = (2.0 ** (glide_oct * t)).astype(np.float32)
        fc = (38.0 + 10.0 * tier + float(rng.uniform(-4.0, 5.0))) * float(freq_mul)
        fm_rate = float(rng.uniform(0.30, 1.40))
        fm_depth = 2.0 + 4.0 * tier + float(rng.uniform(-1.0, 1.8))
        mod = fm_depth * np.sin(2.0 * np.pi * fm_rate * t)
        phase = 2.0 * np.pi * (fc * t * gliss + np.cumsum(mod) / SAMPLE_RATE)
        drone = np.sin(phase).astype(np.float32)
        partial_ratio = 1.5 + 0.10 * float(rng.uniform(-1.0, 1.0))
        partial_phase = 2.0 * np.pi * fc * partial_ratio * t * gliss
        beat = np.sin(2.0 * np.pi * float(rng.uniform(0.15, 0.55)) * t)
        drone = drone + 0.34 * np.sin(partial_phase) * (0.6 + 0.4 * beat)
        sub_phase = 2.0 * np.pi * (fc * 0.5) * t * gliss
        drone = drone + 0.32 * np.sin(sub_phase)
        # Occasionally stack a dissonant fifth-ish partial for tonal chaos.
        if rng.random() < (0.20 + 0.35 * tier):
            diss_ratio = float(rng.uniform(1.32, 1.48))
            drone = drone + 0.22 * np.sin(2.0 * np.pi * fc * diss_ratio * t * gliss)
        peak = float(np.max(np.abs(drone))) + 1e-9
        return (drone / peak).astype(np.float32)

    def _noise_bed(self, level: int, rng: np.random.Generator,
                   q_mul: float = 1.0) -> np.ndarray:
        n = self.clip_len
        tier = level / 5.0
        pink = _pink_noise(n, rng)
        brown = _brown_noise(n, rng)
        # Per-clip pink/brown ratio jitter so the bed texture changes flavor.
        pink_w = float(rng.uniform(0.30, 0.65)) + 0.25 * tier
        brown_w = max(0.15, 1.0 - pink_w)
        mix = pink_w * pink + brown_w * brown

        n_pts = 6
        # Per-clip filter sweep range: sometimes narrow (single droning band),
        # sometimes wide (sweeping radio-like artefact).
        sweep_a = float(rng.uniform(120.0, 600.0))
        sweep_b = float(rng.uniform(2000.0, 5500.0 + 1500.0 * tier))
        if rng.random() < 0.3:
            sweep_a, sweep_b = sweep_b * 0.75, sweep_b
        sweep_lo, sweep_hi = (sweep_a, sweep_b) if sweep_a < sweep_b else (sweep_b, sweep_a)
        if sweep_hi - sweep_lo < 1.0:
            sweep_hi = sweep_lo + 1.0
        ctrl = rng.uniform(sweep_lo, sweep_hi, size=n_pts).astype(np.float32)
        center = np.interp(
            np.arange(n, dtype=np.float32),
            np.linspace(0, n - 1, n_pts, dtype=np.float32),
            ctrl,
        ).astype(np.float32)
        q = (0.85 + 0.35 * tier) * float(q_mul)
        bp = _biquad_bandpass(mix, center, q)

        # LFO envelope rate also jittered for variety.
        lfo_rate = float(rng.uniform(0.15, 1.20))
        envelope = 0.4 + 0.6 * np.abs(np.sin(
            2.0 * np.pi * lfo_rate * np.arange(n) / SAMPLE_RATE
        )).astype(np.float32)
        bed = bp * envelope
        peak = float(np.max(np.abs(bed))) + 1e-9
        return (bed / peak).astype(np.float32)

    def _grains(self, level: int, rng: np.random.Generator,
                pitch_mul: float = 1.0,
                cross_level_chance: float = 0.0) -> np.ndarray:
        out = np.zeros(self.clip_len, dtype=np.float32)
        tier = level / 5.0
        bank = list(self._grain_bank.get(level, []))
        # Cross-level grain pulls: occasionally inject grains from neighbouring
        # levels for unexpected colour ("a distant calm sound during chaos" or
        # "a sharp shriek during low ambience"). Capped per call so the active
        # level still dominates. At L5 we *only* borrow from L4 -- pulling in
        # L2/L3 calm grains during peak terror dilutes the relentlessness.
        if cross_level_chance > 0.0 and rng.random() < cross_level_chance:
            if level >= 5:
                allowed = (level - 1,)
            else:
                allowed = (level - 2, level - 1, level + 1, level + 2)
            other_levels = [
                lvl for lvl in allowed
                if 0 <= lvl < NUM_STRESS_LEVELS and self._grain_bank.get(lvl)
            ]
            if other_levels:
                donor_lvl = int(rng.choice(other_levels))
                donor = self._grain_bank[donor_lvl]
                k = min(len(donor), int(rng.integers(2, 6)))
                if k > 0:
                    idx = rng.choice(len(donor), size=k, replace=False)
                    bank = bank + [donor[int(i)] for i in idx]
        if not bank:
            for fallback_level in (level - 1, level + 1, level - 2, level + 2):
                if 0 <= fallback_level < NUM_STRESS_LEVELS and self._grain_bank.get(fallback_level):
                    bank = self._grain_bank[fallback_level]
                    break
        if not bank:
            return out

        lo, hi = _LEVEL_GRAIN_COUNT[level]
        n_grains = int(rng.integers(lo, hi + 1))
        # Wider pitch range at high stress -> grains scream higher / growl lower
        # for true atonal chaos instead of pitch-matched ambient.
        pitch_max = (4.0 + 14.0 * tier) * float(pitch_mul)
        # Per-clip pitch bias so some clips lean low (growl) and others high
        # (screech) instead of always being symmetric around zero.
        pitch_bias = float(rng.uniform(-pitch_max * 0.4, pitch_max * 0.4))

        for _ in range(n_grains):
            grain = bank[int(rng.integers(0, len(bank)))]
            semis = float(rng.uniform(-pitch_max, pitch_max)) + pitch_bias
            shifted = _pitch_shift(grain, semis)
            if shifted.size >= self.clip_len:
                shifted = shifted[: self.clip_len]
            start_max = max(1, self.clip_len - shifted.size)
            start = int(rng.integers(0, start_max))
            gain = float(rng.uniform(0.4, 0.95)) * (0.7 + 0.55 * tier)
            # Reverse rate climbs with stress so high tiers feel inside-out.
            if rng.random() < (0.18 + 0.22 * tier):
                shifted = shifted[::-1].copy()
            out[start:start + shifted.size] += shifted * gain

        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 1e-6:
            out = out / peak
        return out

    def _transients(self, level: int, rng: np.random.Generator,
                    pitch_mul: float = 1.0) -> np.ndarray:
        out = np.zeros(self.clip_len, dtype=np.float32)
        lo, hi = _LEVEL_TRANSIENT_COUNT[level]
        if hi == 0:
            return out
        tier = level / 5.0
        n = int(rng.integers(lo, hi + 1))
        # At L4/L5 push transient amplitudes harder and widen the spectral range
        # so screeches feel cutting, not polite.
        amp_min = 0.45 + 0.20 * tier
        amp_max = 0.95 + 0.30 * tier
        carrier_hi = (2400.0 + 2200.0 * tier) * float(pitch_mul)  # up to ~4.6 kHz at L5
        carrier_lo = max(80.0, 220.0 * float(pitch_mul))
        for _ in range(n):
            dur = int(rng.integers(160, 900))
            if dur >= self.clip_len:
                continue
            pos = int(rng.integers(0, self.clip_len - dur))
            t = np.arange(dur, dtype=np.float32) / SAMPLE_RATE
            carrier = float(rng.uniform(carrier_lo, carrier_hi))
            mod_rate = float(rng.uniform(35.0, 280.0 + 220.0 * tier))
            ring = (1.0 + (0.55 + 0.25 * tier) * np.sin(2.0 * np.pi * mod_rate * t))
            tone = np.sin(2.0 * np.pi * carrier * t) * ring
            noise_burst = rng.standard_normal(dur).astype(np.float32) * (0.45 + 0.30 * tier)
            decay_rate = float(rng.uniform(18.0, 80.0))
            env = np.exp(-t * decay_rate).astype(np.float32)
            amp = float(rng.uniform(amp_min, amp_max))
            out[pos:pos + dur] += (tone * 0.60 + noise_burst * 0.40) * env * amp

        # L5: lay down a second pass of very short, very dense impact crackles
        # so the second never goes quiet between the main transients.
        if level >= 5:
            extra = int(rng.integers(8, 16))
            for _ in range(extra):
                dur = int(rng.integers(80, 260))
                pos = int(rng.integers(0, self.clip_len - dur))
                t = np.arange(dur, dtype=np.float32) / SAMPLE_RATE
                burst = rng.standard_normal(dur).astype(np.float32)
                env = np.exp(-t * float(rng.uniform(60.0, 180.0))).astype(np.float32)
                out[pos:pos + dur] += burst * env * float(rng.uniform(0.45, 0.85))

        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 1e-6:
            out = out / peak
        return out

    def generate_clip(self, stress_level: int, seed: Optional[int] = None) -> np.ndarray:
        """Build one 1-second 16 kHz mono float32 clip for ``stress_level`` (0-5).

        Each call rolls a "style" (tunnel / swarm / ironworks / wet / near /
        far / void / shriek) that reshapes layer weights, drone frequency,
        noise filter Q, grain pitch range, and drive so consecutive clips at
        the same stress level sound noticeably different.
        """
        lvl = _clip_level(stress_level)
        rng = np.random.default_rng(
            ((int(seed) if seed is not None else self._rng_seed) ^ (lvl * 0x9E3779B1)) & 0xFFFFFFFF
        )
        base_w = _LEVEL_LAYER_WEIGHTS[lvl]
        recent = self._recent_styles.get(lvl, [])
        style = _pick_style(rng, lvl, recent_ids=recent)
        # Track style id so the next call avoids back-to-back repeats.
        sid = int(style.get("name_id", -1))
        if sid >= 0:
            recent.append(sid)
            if len(recent) > 6:
                del recent[0]
            self._recent_styles[lvl] = recent

        # Per-clip layer jitter: ~15% chance any one layer is dropped entirely
        # or doubled, giving rare "all noise no drone" / "huge sub drone only"
        # moments. Disabled at L5 -- peak terror needs every layer present at
        # full strength, no breathing room.
        is_peak = (lvl >= 5)

        def _jitter_layer(name: str, base_mul: float) -> float:
            if is_peak:
                return base_mul * float(rng.uniform(1.05, 1.30))
            r = rng.random()
            if r < 0.07:
                return 0.0
            if r < 0.18:
                return base_mul * float(rng.uniform(1.5, 2.1))
            return base_mul * float(rng.uniform(0.85, 1.15))

        w = {
            "drone": base_w["drone"] * _jitter_layer("drone", style["drone"]),
            "noise": base_w["noise"] * _jitter_layer("noise", style["noise"]),
            "grains": base_w["grains"] * _jitter_layer("grains", style["grains"]),
            "transients": base_w["transients"] * _jitter_layer("transients", style["transients"]),
        }
        drone = self._drone(lvl, rng, freq_mul=style["drone_freq_mul"]) * w["drone"]
        noise = self._noise_bed(lvl, rng, q_mul=style["noise_q"]) * w["noise"]
        # Higher cross-level pull chance at mid+ stress; rare at L0/L1 so calm
        # stays calm but never feels static.
        cross_chance = 0.05 + 0.10 * (lvl / max(1, NUM_STRESS_LEVELS - 1))
        if w["grains"] > 0.0 and self._has_grains:
            grains = self._grains(
                lvl, rng,
                pitch_mul=style["pitch_mul"],
                cross_level_chance=cross_chance,
            ) * w["grains"]
            # L5 "wall" pass: a second independent dense grain field is mixed
            # on top so the layered chaos becomes a solid wall, not a stream
            # of distinct events. The two grain layers stay decorrelated by
            # forking a fresh RNG branch.
            if is_peak:
                rng_wall = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                wall = self._grains(
                    lvl, rng_wall,
                    pitch_mul=max(0.85, style["pitch_mul"] * 0.9),
                    cross_level_chance=0.0,
                ) * w["grains"] * 0.85
                grains = grains + wall
        else:
            grains = np.zeros(self.clip_len, dtype=np.float32)
            noise = noise * (1.0 + w["grains"])
        transients = self._transients(lvl, rng, pitch_mul=style["pitch_mul"]) * w["transients"]
        # L5 transient reinforcement: a second decorrelated burst layer keeps
        # the field absolutely packed even when style's transient multiplier
        # is lower.
        if is_peak:
            rng_tx = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            transients = transients + self._transients(
                lvl, rng_tx, pitch_mul=style["pitch_mul"]
            ) * w["transients"] * 0.75

        mix = drone + noise + grains + transients
        drive = _LEVEL_DRIVE[lvl] * style["drive_mul"]
        mix = np.tanh(mix * drive).astype(np.float32)

        # Per-clip macro-envelope shape: occasional swell-in or duck-out so
        # the perceived loudness doesn't sit on the same RMS across clips.
        # AT L5 the macro envelope is *always* a forward push so there is no
        # quiet moment -- the bed only swells louder or holds at the rails.
        macro = np.ones(self.clip_len, dtype=np.float32)
        t_norm = np.linspace(0.0, 1.0, self.clip_len, dtype=np.float32)
        if is_peak:
            # Slight rising tilt always -- pushes harder into the clip end
            # so back-to-back clips feel like a continuous escalation.
            macro = 0.92 + 0.18 * t_norm
        else:
            env_shape = float(rng.random())
            if env_shape < 0.18:
                # slow swell
                macro = 0.55 + 0.45 * t_norm
            elif env_shape < 0.36:
                # ducking tail
                macro = 1.0 - 0.4 * (t_norm ** 2)
            elif env_shape < 0.50:
                # mid-clip dip ("breath caught")
                dip_pos = float(rng.uniform(0.30, 0.65))
                macro = 1.0 - 0.5 * np.exp(-((t_norm - dip_pos) ** 2) / 0.012)
        mix = mix * macro

        # Shorter fade at high stress so transients aren't shaved at clip edges.
        fade_n = min(384, self.clip_len // 40) if lvl < 4 else min(96, self.clip_len // 160)
        env = np.ones(self.clip_len, dtype=np.float32)
        env[:fade_n] = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        env[-fade_n:] = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        mix *= env
        # Level-aware target peak: low tiers keep headroom (gentle), high tiers
        # push right up to the rails so chunks stitch together without dropping
        # in perceived loudness between clips.
        target_peak = 0.78 + 0.05 * lvl  # 0.78..1.03
        target_peak = min(target_peak, 0.99)
        peak = float(np.max(np.abs(mix))) + 1e-9
        if peak > 1e-6:
            mix = mix * (target_peak / peak)
        return mix.astype(np.float32)
