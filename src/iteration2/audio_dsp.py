"""
Real-time DSP chain for audio manipulation.

Applies stress-driven audio effects to preloaded clips:
  - Pitch shifting
  - Distortion / saturation
  - Reverb (convolution-based)
  - Low-pass / high-pass filtering
  - Layering / crossfade between clips
  - Spatial panning

Controlled by fusion model parameters in real-time.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


SAMPLE_RATE = 16000


@dataclass
class DSPParams:
    pitch_shift_semitones: float = 0.0
    distortion: float = 0.0
    reverb_mix: float = 0.0
    reverb_decay: float = 0.3
    lowpass_freq: float = 8000.0
    highpass_freq: float = 20.0
    layer_blend: float = 0.0
    pan: float = 0.0
    volume: float = 1.0


def apply_pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Simple pitch shift via resampling (no time-stretch)."""
    if abs(semitones) < 0.01:
        return audio
    ratio = 2.0 ** (semitones / 12.0)
    n_out = int(len(audio) / ratio)
    indices = np.linspace(0, len(audio) - 1, n_out)
    shifted = np.interp(indices, np.arange(len(audio)), audio)
    if len(shifted) < len(audio):
        shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
    else:
        shifted = shifted[:len(audio)]
    return shifted


def apply_distortion(audio: np.ndarray, amount: float) -> np.ndarray:
    """Soft-clipping distortion."""
    if amount < 0.01:
        return audio
    gain = 1.0 + amount * 10.0
    x = audio * gain
    return np.tanh(x) * (1.0 / np.tanh(gain))


def apply_reverb(audio: np.ndarray, mix: float, decay: float) -> np.ndarray:
    """Simple algorithmic reverb using comb filters."""
    if mix < 0.01:
        return audio

    reverb_length = int(SAMPLE_RATE * decay)
    impulse = np.zeros(reverb_length)
    impulse[0] = 1.0

    delays = [int(SAMPLE_RATE * d) for d in [0.02, 0.035, 0.05, 0.07]]
    gains = [0.6, 0.45, 0.35, 0.25]

    for delay, gain in zip(delays, gains):
        if delay < reverb_length:
            impulse[delay] += gain * (decay ** 0.5)

    for i in range(1, len(impulse)):
        impulse[i] += impulse[i - 1] * 0.3 * decay

    impulse = impulse / (np.max(np.abs(impulse)) + 1e-8)

    wet = np.convolve(audio, impulse, mode='full')[:len(audio)]
    return audio * (1.0 - mix) + wet * mix


def apply_lowpass(audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
    """Simple first-order IIR low-pass filter."""
    if cutoff_freq >= SAMPLE_RATE / 2:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_freq)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (rc + dt)

    output = np.zeros_like(audio)
    output[0] = alpha * audio[0]
    for i in range(1, len(audio)):
        output[i] = output[i - 1] + alpha * (audio[i] - output[i - 1])
    return output


def apply_highpass(audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
    """Simple first-order IIR high-pass filter."""
    if cutoff_freq <= 1.0:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_freq)
    dt = 1.0 / SAMPLE_RATE
    alpha = rc / (rc + dt)

    output = np.zeros_like(audio)
    output[0] = audio[0]
    for i in range(1, len(audio)):
        output[i] = alpha * (output[i - 1] + audio[i] - audio[i - 1])
    return output


def apply_pan(audio_mono: np.ndarray, pan: float) -> np.ndarray:
    """Convert mono to stereo with panning (-1=left, 0=center, 1=right)."""
    pan_norm = (pan + 1.0) / 2.0
    left = audio_mono * np.sqrt(1.0 - pan_norm)
    right = audio_mono * np.sqrt(pan_norm)
    return np.stack([left, right], axis=-1)


def layer_audio(clip_a: np.ndarray, clip_b: np.ndarray, blend: float) -> np.ndarray:
    """Crossfade between two audio clips."""
    min_len = min(len(clip_a), len(clip_b))
    a = clip_a[:min_len]
    b = clip_b[:min_len]
    return a * (1.0 - blend) + b * blend


class AudioDSPChain:
    """Complete DSP processing chain driven by model parameters."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, params: DSPParams,
                secondary_audio: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply full DSP chain to audio clip."""
        if audio is None or len(audio) == 0:
            return np.zeros(self.sample_rate, dtype=np.float32)

        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        if secondary_audio is not None and params.layer_blend > 0.01:
            audio = layer_audio(audio, secondary_audio, params.layer_blend)

        audio = apply_pitch_shift(audio, params.pitch_shift_semitones)
        audio = apply_distortion(audio, params.distortion)
        audio = apply_lowpass(audio, params.lowpass_freq)
        audio = apply_highpass(audio, params.highpass_freq)
        audio = apply_reverb(audio, params.reverb_mix, params.reverb_decay)

        audio = audio * params.volume
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    @staticmethod
    def params_from_model_output(model_params: dict) -> DSPParams:
        """Convert fusion model outputs to DSP parameters."""
        intensity = model_params.get("audio_intensity", 0.5)
        dissonance = model_params.get("audio_dissonance", 0.0)
        reverb = model_params.get("audio_reverb_depth", 0.3)
        transient = model_params.get("audio_transient_rate", 0.0)

        return DSPParams(
            pitch_shift_semitones=(dissonance - 0.5) * 12.0,
            distortion=dissonance * 0.8,
            reverb_mix=reverb,
            reverb_decay=0.2 + reverb * 0.6,
            lowpass_freq=8000.0 - dissonance * 5000.0,
            highpass_freq=20.0 + transient * 200.0,
            layer_blend=min(1.0, intensity * 1.5),
            pan=(dissonance - 0.5) * 0.6,
            volume=0.3 + intensity * 0.7,
        )
