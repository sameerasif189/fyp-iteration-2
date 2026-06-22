"""
Audio dataset for training the WaveGAN audio generator.

Loads wildlife vocalizations and horror audio clips, resamples to 16kHz mono,
normalizes, and slices into 1-second segments with stress-level labels.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_gen_model import SAMPLE_RATE, AUDIO_LENGTH, NUM_STRESS_LEVELS
from .audio_content_filter import is_forbidden_audio, is_musicy_audio

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def load_audio_file(filepath: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    if HAS_LIBROSA:
        audio, _ = librosa.load(filepath, sr=target_sr, mono=True)
        return audio
    else:
        raise ImportError("librosa is required for audio loading. Install with: pip install librosa")


def slice_audio(audio: np.ndarray, segment_length: int = AUDIO_LENGTH,
                hop_length: int = None) -> List[np.ndarray]:
    """Slice audio into fixed-length segments."""
    if hop_length is None:
        hop_length = segment_length // 2

    segments = []
    for start in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start:start + segment_length]
        if np.max(np.abs(segment)) > 0.01 and is_valid_segment(segment):
            segments.append(segment)
    return segments


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """DC-remove + RMS/peak normalize to [-1, 1]."""
    x = audio.astype(np.float32)
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x) + 1e-8))
    if rms > 1e-6:
        x = x * (0.22 / rms)
    peak = float(np.max(np.abs(x)))
    if peak > 0.98:
        x = x * (0.98 / peak)
    return np.clip(x, -1.0, 1.0)


def is_valid_segment(segment: np.ndarray) -> bool:
    """Reject silent/clipped/buzzy segments for cleaner training targets."""
    if segment.size == 0:
        return False
    peak = float(np.max(np.abs(segment)))
    rms = float(np.sqrt(np.mean(segment * segment) + 1e-8))
    if peak < 0.03 or rms < 0.008:
        return False
    zcr = float(((segment[:-1] * segment[1:]) < 0).mean())
    if zcr > 0.28:
        return False
    crest = peak / max(rms, 1e-6)
    if crest < 1.2:
        return False
    return True


class AudioTrainingDataset(Dataset):
    """
    Dataset for WaveGAN training.

    Loads audio from:
      - assets/wildlife/ -> high stress (4-5), used for transient generation
      - assets/audio/{0-5}/ -> stress-labeled ambient/horror clips
    """

    def __init__(self, assets_dir: Path, augment: bool = True, seed: int = 42):
        super().__init__()
        self.augment = augment
        self.rng = random.Random(seed)
        self.samples: List[Tuple[np.ndarray, int]] = []

        self._load_all(assets_dir)
        print(f"[audio-dataset] loaded {len(self.samples)} segments")

    def _load_all(self, assets_dir: Path):
        audio_dir = assets_dir / "audio"
        wildlife_dir = assets_dir / "wildlife"

        n_blocked = 0
        n_music_dropped = 0
        for level in range(NUM_STRESS_LEVELS):
            level_dir = audio_dir / str(level)
            if not level_dir.exists():
                continue
            for fp in list(level_dir.glob("*.mp3")) + list(level_dir.glob("*.wav")):
                if is_forbidden_audio(fp):
                    n_blocked += 1
                    continue
                # Drop musical loops at L4/L5 so the model learns pure horror
                # SFX rather than fragments of BPM-tagged synth basses.
                if level >= 4 and is_musicy_audio(fp):
                    n_music_dropped += 1
                    continue
                self._load_file(str(fp), level)

        if wildlife_dir.exists():
            for fp in list(wildlife_dir.glob("*.mp3")) + list(wildlife_dir.glob("*.wav")):
                if is_forbidden_audio(fp):
                    n_blocked += 1
                    continue
                wildlife_level = self.rng.choice([4, 5])
                self._load_file(str(fp), wildlife_level)

        if n_blocked or n_music_dropped:
            print(
                f"[audio-dataset] content filter dropped forbidden={n_blocked} "
                f"musical_L4_L5={n_music_dropped}"
            )

    def _load_file(self, filepath: str, level: int):
        try:
            audio = load_audio_file(filepath)
            segments = slice_audio(audio)
            for seg in segments:
                seg = normalize_audio(seg)
                self.samples.append((seg, level))
        except Exception as e:
            print(f"  [warn] failed to load {filepath}: {e}")

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Random augmentation for training diversity."""
        if self.rng.random() < 0.3:
            gain = self.rng.uniform(0.5, 1.5)
            audio = np.clip(audio * gain, -1.0, 1.0)

        if self.rng.random() < 0.2:
            shift = self.rng.randint(-SAMPLE_RATE // 4, SAMPLE_RATE // 4)
            audio = np.roll(audio, shift)

        if self.rng.random() < 0.15:
            noise_level = self.rng.uniform(0.001, 0.02)
            audio = audio + np.random.randn(len(audio)) * noise_level
            audio = np.clip(audio, -1.0, 1.0)

        return audio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, level = self.samples[idx]

        if self.augment:
            audio = self._augment(audio.copy())

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        level_tensor = torch.tensor(level, dtype=torch.long)
        return audio_tensor, level_tensor


class SyntheticAudioDataset(Dataset):
    """
    Fallback dataset using synthetic audio generation for when no real assets
    are available. Generates sine-based horror audio patterns.
    """

    def __init__(self, samples_per_level: int = 500, seed: int = 42):
        super().__init__()
        self.samples: List[Tuple[np.ndarray, int]] = []
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        for level in range(NUM_STRESS_LEVELS):
            for _ in range(samples_per_level):
                audio = self._generate_synthetic(level, rng, np_rng)
                self.samples.append((audio, level))

        rng.shuffle(self.samples)

    def _generate_synthetic(self, level: int, rng: random.Random,
                            np_rng: np.random.RandomState) -> np.ndarray:
        t = np.linspace(0, 1.0, AUDIO_LENGTH, dtype=np.float32)

        base_freq = 80 + level * 60 + rng.uniform(-20, 20)
        intensity = 0.1 + level * 0.15

        audio = np.sin(2 * np.pi * base_freq * t) * intensity

        num_harmonics = 2 + level
        for h in range(1, num_harmonics + 1):
            freq = base_freq * (h + 1) + rng.uniform(-10, 10)
            amp = intensity * (0.5 ** h) * (1 + level * 0.1)
            audio += np.sin(2 * np.pi * freq * t) * amp

        if level >= 3:
            lfo_freq = rng.uniform(2, 8)
            lfo = np.sin(2 * np.pi * lfo_freq * t)
            audio *= (1.0 + lfo * 0.3 * (level / 5.0))

        if level >= 4:
            num_transients = rng.randint(2, 5 + level)
            for _ in range(num_transients):
                pos = rng.randint(0, AUDIO_LENGTH - 400)
                duration = rng.randint(100, 400)
                amp_t = rng.uniform(0.3, 0.8) * (level / 5.0)
                freq_t = rng.uniform(200, 2000)
                transient = np.sin(2 * np.pi * freq_t * t[pos:pos + duration]) * amp_t
                envelope = np.exp(-np.linspace(0, 5, duration))
                audio[pos:pos + duration] += transient * envelope

        noise_level = 0.01 + level * 0.02
        audio += np_rng.randn(AUDIO_LENGTH) * noise_level

        audio = normalize_audio(audio)
        return audio

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio, level = self.samples[idx]
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        level_tensor = torch.tensor(level, dtype=torch.long)
        return audio_tensor, level_tensor
