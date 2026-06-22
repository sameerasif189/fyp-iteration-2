"""
Generate high-quality synthetic horror audio assets for training.

Creates realistic WAV files organized by stress level using:
  - FM synthesis for drones and tonal elements
  - Filtered noise for ambient textures
  - Transient synthesis for stingers and impacts
  - Envelope shaping for natural dynamics
  - Wildlife-like vocalizations using chirp + FM synthesis

Output: assets/audio/{0-5}/*.wav and assets/wildlife/*.wav
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import math
import random
import struct
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "assets" / "audio"
WILDLIFE_DIR = BASE_DIR / "assets" / "wildlife"


def write_wav(filepath: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Write a mono float32 array as 16-bit WAV."""
    audio = np.clip(audio, -1.0, 1.0)
    int16_audio = (audio * 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())


def fade_in_out(audio: np.ndarray, fade_samples: int = 800) -> np.ndarray:
    """Apply smooth fade in/out to avoid clicks."""
    n = len(audio)
    fade_in = np.linspace(0, 1, min(fade_samples, n // 4)) ** 2
    fade_out = np.linspace(1, 0, min(fade_samples, n // 4)) ** 2
    audio[:len(fade_in)] *= fade_in
    audio[-len(fade_out):] *= fade_out
    return audio


def generate_drone(duration_s: float, base_freq: float, harmonics: int = 5,
                   detune: float = 0.02, lfo_rate: float = 0.5,
                   noise_amount: float = 0.05, rng: np.random.RandomState = None) -> np.ndarray:
    """Generate a dark ambient drone with harmonics and LFO modulation."""
    if rng is None:
        rng = np.random.RandomState()
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    audio = np.zeros(n, dtype=np.float32)

    lfo = np.sin(2 * np.pi * lfo_rate * t)

    for h in range(1, harmonics + 1):
        freq = base_freq * h * (1.0 + rng.uniform(-detune, detune))
        amp = 0.3 / h
        phase = rng.uniform(0, 2 * np.pi)
        mod = 1.0 + lfo * (0.1 / h)
        audio += np.sin(2 * np.pi * freq * t * mod + phase) * amp

    if noise_amount > 0:
        noise = rng.randn(n) * noise_amount
        audio += noise

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.85


def generate_room_tone(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Very quiet room ambience with subtle air movement."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    noise = rng.randn(n).astype(np.float32) * 0.05
    from_scipy = False
    try:
        from scipy.signal import butter, lfilter
        b, a = butter(4, 800 / (SAMPLE_RATE / 2), btype='low')
        noise = lfilter(b, a, noise).astype(np.float32)
        from_scipy = True
    except ImportError:
        kernel_size = SAMPLE_RATE // 800
        kernel = np.ones(kernel_size) / kernel_size
        noise = np.convolve(noise, kernel, mode='same').astype(np.float32)

    slow_mod = np.sin(2 * np.pi * 0.1 * t) * 0.3 + 0.7
    audio = noise * slow_mod

    subtle_tone = np.sin(2 * np.pi * 50 * t) * 0.02
    audio += subtle_tone

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.3


def generate_tension_drone(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Dissonant tension drone with beating frequencies."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    f1 = rng.uniform(80, 120)
    f2 = f1 * 1.01 + rng.uniform(0.5, 2.0)
    f3 = f1 * 1.5 + rng.uniform(-3, 3)

    audio = (np.sin(2 * np.pi * f1 * t) * 0.4 +
             np.sin(2 * np.pi * f2 * t) * 0.35 +
             np.sin(2 * np.pi * f3 * t) * 0.25)

    lfo = np.sin(2 * np.pi * 0.3 * t)
    audio *= (0.7 + 0.3 * lfo)

    noise = rng.randn(n).astype(np.float32) * 0.08
    audio += noise

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.7


def generate_horror_stinger(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Sharp horror stinger / scare chord."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    attack = np.exp(-t * 3.0)
    freqs = [rng.uniform(200, 800) for _ in range(5)]
    audio = np.zeros(n, dtype=np.float32)

    for freq in freqs:
        detune = rng.uniform(-5, 5)
        audio += np.sin(2 * np.pi * (freq + detune) * t) * (0.3 / len(freqs))

    audio *= attack
    noise_burst = rng.randn(n).astype(np.float32) * 0.15 * np.exp(-t * 5.0)
    audio += noise_burst

    audio = fade_in_out(audio, fade_samples=200)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.9


def generate_creaking(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Creaking / groaning wood sound."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    base_freq = rng.uniform(150, 400)
    mod_freq = rng.uniform(20, 80)
    mod_depth = rng.uniform(50, 200)

    freq_mod = base_freq + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    phase = np.cumsum(freq_mod / SAMPLE_RATE) * 2 * np.pi
    audio = np.sin(phase) * 0.5

    env_freq = rng.uniform(2, 8)
    envelope = np.abs(np.sin(2 * np.pi * env_freq * t)) ** 0.5
    audio *= envelope

    noise = rng.randn(n).astype(np.float32) * 0.1
    audio += noise * envelope

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.6


def generate_dark_ambient(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Deep dark ambient texture with sub-bass and filtered noise."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    sub = np.sin(2 * np.pi * rng.uniform(30, 60) * t) * 0.3
    mid = np.sin(2 * np.pi * rng.uniform(100, 200) * t) * 0.2

    lfo1 = np.sin(2 * np.pi * 0.15 * t)
    lfo2 = np.sin(2 * np.pi * 0.07 * t)

    noise = rng.randn(n).astype(np.float32) * 0.15
    try:
        from scipy.signal import butter, lfilter
        cutoff = 300 + 200 * lfo1
        b, a = butter(2, 400 / (SAMPLE_RATE / 2), btype='low')
        noise = lfilter(b, a, noise).astype(np.float32)
    except ImportError:
        kernel = np.ones(40) / 40
        noise = np.convolve(noise, kernel, mode='same').astype(np.float32)

    audio = sub + mid * (0.5 + 0.5 * lfo2) + noise
    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.75


def generate_distorted_bass(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Distorted bass rumble for high stress."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    freq = rng.uniform(40, 80)
    audio = np.sin(2 * np.pi * freq * t)

    gain = rng.uniform(3.0, 8.0)
    audio = np.tanh(audio * gain)

    sub_freq = rng.uniform(20, 40)
    audio += np.sin(2 * np.pi * sub_freq * t) * 0.3

    lfo = np.sin(2 * np.pi * rng.uniform(1, 4) * t)
    audio *= (0.6 + 0.4 * lfo)

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.8


def generate_bat_screech(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize bat-like screech using FM synthesis."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    carrier = rng.uniform(2000, 5000)
    mod_freq = rng.uniform(100, 400)
    mod_depth = rng.uniform(500, 2000)

    chirp = np.linspace(1.0, rng.uniform(0.3, 1.5), n)
    freq_mod = carrier * chirp + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    phase = np.cumsum(freq_mod / SAMPLE_RATE) * 2 * np.pi
    audio = np.sin(phase)

    num_bursts = rng.randint(3, 8)
    envelope = np.zeros(n, dtype=np.float32)
    for _ in range(num_bursts):
        center = rng.uniform(0.1, 0.9) * duration_s
        width = rng.uniform(0.02, 0.1)
        burst = np.exp(-((t - center) / width) ** 2)
        envelope += burst
    envelope = np.clip(envelope, 0, 1)

    audio *= envelope
    audio = fade_in_out(audio, fade_samples=400)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.7


def generate_fox_scream(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize fox-like scream using formant synthesis."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    f0 = rng.uniform(400, 800)
    vibrato = np.sin(2 * np.pi * rng.uniform(4, 8) * t) * 30
    pitch_contour = f0 * np.linspace(0.8, 1.3, n) + vibrato

    phase = np.cumsum(pitch_contour / SAMPLE_RATE) * 2 * np.pi
    source = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)

    envelope = np.ones(n, dtype=np.float32)
    attack = int(0.05 * SAMPLE_RATE)
    decay = int(0.3 * SAMPLE_RATE)
    envelope[:attack] = np.linspace(0, 1, attack)
    if decay < n:
        envelope[-decay:] = np.linspace(1, 0, decay)

    audio = source * envelope
    noise = rng.randn(n).astype(np.float32) * 0.1 * envelope
    audio += noise

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.75


def generate_owl_call(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize owl-like hooting."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    audio = np.zeros(n, dtype=np.float32)

    num_hoots = rng.randint(2, 5)
    hoot_spacing = duration_s / (num_hoots + 1)

    for i in range(num_hoots):
        center = (i + 1) * hoot_spacing
        freq = rng.uniform(250, 450)
        width = rng.uniform(0.08, 0.15)

        hoot_env = np.exp(-((t - center) / width) ** 2)
        pitch_drop = np.where(t > center, 1.0 - (t - center) * 0.5, 1.0)
        pitch_drop = np.clip(pitch_drop, 0.7, 1.0)

        hoot = np.sin(2 * np.pi * freq * pitch_drop * t) * hoot_env
        audio += hoot

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.6


def generate_wolf_howl(duration_s: float, rng: np.random.RandomState) -> np.ndarray:
    """Synthesize wolf-like howl with pitch glide."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    start_freq = rng.uniform(200, 350)
    peak_freq = rng.uniform(500, 800)
    end_freq = rng.uniform(250, 400)

    pitch_peak = 0.3 + rng.uniform(-0.1, 0.1)
    pitch = np.where(t < pitch_peak * duration_s,
                     start_freq + (peak_freq - start_freq) * (t / (pitch_peak * duration_s)),
                     peak_freq + (end_freq - peak_freq) * ((t - pitch_peak * duration_s) / ((1 - pitch_peak) * duration_s)))

    vibrato = np.sin(2 * np.pi * rng.uniform(4, 7) * t) * 15
    pitch += vibrato

    phase = np.cumsum(pitch / SAMPLE_RATE) * 2 * np.pi
    audio = np.sin(phase) * 0.6 + np.sin(2 * phase) * 0.25 + np.sin(3 * phase) * 0.1

    attack = int(0.15 * SAMPLE_RATE)
    sustain_end = int(0.7 * n)
    release = n - sustain_end
    envelope = np.ones(n, dtype=np.float32)
    envelope[:attack] = np.linspace(0, 1, attack) ** 2
    envelope[sustain_end:] = np.linspace(1, 0, release) ** 1.5

    audio *= envelope

    breath = rng.randn(n).astype(np.float32) * 0.08 * envelope
    audio += breath

    audio = fade_in_out(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio * 0.8


STRESS_GENERATORS = {
    0: [
        ("room_tone", generate_room_tone, {"duration_s": 3.0}),
        ("soft_air", generate_room_tone, {"duration_s": 4.0}),
        ("quiet_ambience", generate_room_tone, {"duration_s": 5.0}),
    ],
    1: [
        ("subtle_drone", lambda d, r: generate_drone(d, 60, harmonics=2, noise_amount=0.02, rng=r), {"duration_s": 4.0}),
        ("light_tension", generate_tension_drone, {"duration_s": 3.0}),
        ("distant_creak", generate_creaking, {"duration_s": 2.0}),
    ],
    2: [
        ("tension_drone_a", generate_tension_drone, {"duration_s": 4.0}),
        ("creak_slow", generate_creaking, {"duration_s": 3.0}),
        ("eerie_ambient", generate_dark_ambient, {"duration_s": 5.0}),
    ],
    3: [
        ("dark_drone", lambda d, r: generate_drone(d, 80, harmonics=6, detune=0.04, noise_amount=0.1, rng=r), {"duration_s": 4.0}),
        ("horror_ambient", generate_dark_ambient, {"duration_s": 5.0}),
        ("tension_building", generate_tension_drone, {"duration_s": 4.0}),
    ],
    4: [
        ("horror_stinger_a", generate_horror_stinger, {"duration_s": 2.0}),
        ("distorted_bass", generate_distorted_bass, {"duration_s": 3.0}),
        ("intense_drone", lambda d, r: generate_drone(d, 50, harmonics=8, detune=0.06, noise_amount=0.15, rng=r), {"duration_s": 4.0}),
        ("harsh_creak", generate_creaking, {"duration_s": 2.0}),
    ],
    5: [
        ("terror_stinger", generate_horror_stinger, {"duration_s": 1.5}),
        ("extreme_bass", generate_distorted_bass, {"duration_s": 3.0}),
        ("chaos_drone", lambda d, r: generate_drone(d, 40, harmonics=10, detune=0.08, noise_amount=0.2, rng=r), {"duration_s": 4.0}),
        ("impact_stinger", generate_horror_stinger, {"duration_s": 1.0}),
        ("dark_intense", generate_dark_ambient, {"duration_s": 3.0}),
    ],
}

WILDLIFE_GENERATORS = [
    ("bat_screech", generate_bat_screech, {"duration_s": 1.5}),
    ("fox_scream", generate_fox_scream, {"duration_s": 2.5}),
    ("owl_call", generate_owl_call, {"duration_s": 3.0}),
    ("wolf_howl", generate_wolf_howl, {"duration_s": 4.0}),
]


def main():
    rng = np.random.RandomState(42)
    clips_per_variant = 8

    print("=" * 60)
    print("GENERATING HORROR AUDIO ASSETS")
    print("=" * 60)

    total_files = 0

    for level, generators in STRESS_GENERATORS.items():
        level_dir = AUDIO_DIR / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [stress {level}] generating {len(generators)} sound types x {clips_per_variant} variants...")

        for name, gen_func, kwargs in generators:
            for i in range(clips_per_variant):
                filename = f"{name}_{i:02d}.wav"
                filepath = level_dir / filename

                if filepath.exists():
                    total_files += 1
                    continue

                duration = kwargs.get("duration_s", 3.0) + rng.uniform(-0.5, 0.5)
                duration = max(1.0, duration)

                try:
                    audio = gen_func(duration, rng)
                    write_wav(str(filepath), audio)
                    total_files += 1
                except Exception as e:
                    print(f"    [warn] failed to generate {filename}: {e}")

        count = len(list(level_dir.glob("*.wav")))
        print(f"    -> {count} WAV files in stress {level}")

    print(f"\n  [stress audio] total: {total_files} clips")

    print(f"\n{'=' * 60}")
    print("GENERATING WILDLIFE VOCALIZATIONS")
    print("=" * 60)

    WILDLIFE_DIR.mkdir(parents=True, exist_ok=True)
    wildlife_count = 0

    for name, gen_func, kwargs in WILDLIFE_GENERATORS:
        print(f"\n  generating {name} x {clips_per_variant * 2} variants...")
        for i in range(clips_per_variant * 2):
            filename = f"{name}_{i:02d}.wav"
            filepath = WILDLIFE_DIR / filename

            if filepath.exists():
                wildlife_count += 1
                continue

            duration = kwargs.get("duration_s", 2.0) + rng.uniform(-0.3, 0.5)
            duration = max(0.5, duration)

            try:
                audio = gen_func(duration, rng)
                write_wav(str(filepath), audio)
                wildlife_count += 1
            except Exception as e:
                print(f"    [warn] failed: {e}")

    print(f"\n  [wildlife] total: {wildlife_count} clips")

    print(f"\n{'=' * 60}")
    print("UPDATING CATALOG")
    print("=" * 60)

    import json
    catalog_path = BASE_DIR / "assets" / "catalog_real.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        catalog = {"textures": {}, "audio": {}, "wildlife": []}

    for level in range(6):
        audio_dir = AUDIO_DIR / str(level)
        if audio_dir.exists():
            clips = [str(f.relative_to(BASE_DIR)) for f in sorted(audio_dir.glob("*.wav"))]
            catalog["audio"][str(level)] = catalog.get("audio", {}).get(str(level), []) + clips
            seen = set()
            catalog["audio"][str(level)] = [x for x in catalog["audio"][str(level)]
                                            if not (x in seen or seen.add(x))]

    if WILDLIFE_DIR.exists():
        wildlife_clips = [str(f.relative_to(BASE_DIR)) for f in sorted(WILDLIFE_DIR.glob("*.wav"))]
        existing = catalog.get("wildlife", [])
        all_wildlife = list(set(existing + wildlife_clips))
        all_wildlife.sort()
        catalog["wildlife"] = all_wildlife

    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"  [ok] updated {catalog_path}")

    audio_total = sum(len(v) for v in catalog.get("audio", {}).values())
    print(f"       audio: {audio_total} clips")
    print(f"       wildlife: {len(catalog.get('wildlife', []))} clips")

    print(f"\n[done] Audio generation complete!")


if __name__ == "__main__":
    main()
