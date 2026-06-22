from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StressProfile:
    name: str
    audio_intensity: Tuple[float, float]
    dissonance: Tuple[float, float]
    corruption_alpha: Tuple[float, float]
    fog_density: Tuple[float, float]
    camera_magnitude: Tuple[float, float]
    entity_probability: float


STRESS_PROFILES: Dict[int, StressProfile] = {
    0: StressProfile("Calm", (0.05, 0.15), (0.00, 0.10), (0.00, 0.05), (0.03, 0.08), (0.00, 0.05), 0.00),
    1: StressProfile("Wary", (0.15, 0.30), (0.05, 0.20), (0.04, 0.10), (0.07, 0.14), (0.03, 0.08), 0.00),
    2: StressProfile("Nervous", (0.30, 0.48), (0.15, 0.35), (0.10, 0.22), (0.12, 0.22), (0.08, 0.18), 0.05),
    3: StressProfile("Anxious", (0.45, 0.65), (0.30, 0.55), (0.20, 0.40), (0.18, 0.34), (0.15, 0.28), 0.12),
    4: StressProfile("Frightened", (0.62, 0.82), (0.50, 0.78), (0.38, 0.66), (0.30, 0.50), (0.25, 0.45), 0.24),
    5: StressProfile("Terrified", (0.78, 1.00), (0.72, 1.00), (0.62, 0.95), (0.46, 0.75), (0.40, 0.72), 0.40),
}


class StressTransitionController:
    def __init__(self, hysteresis: int = 1, min_hold_s: float = 8.0, cooldown_after_peak_s: float = 14.0):
        self.hysteresis = hysteresis
        self.min_hold_s = min_hold_s
        self.cooldown_after_peak_s = cooldown_after_peak_s
        self._last_level = 0
        self._last_change_t = 0.0
        self._peak_t = -10_000.0

    def resolve_level(self, requested_level: int, now_s: float) -> int:
        requested_level = max(0, min(5, requested_level))
        level_delta = abs(requested_level - self._last_level)
        hold_elapsed = now_s - self._last_change_t

        if self._last_level == 5:
            peak_elapsed = now_s - self._peak_t
            if peak_elapsed < self.cooldown_after_peak_s and requested_level < 2:
                requested_level = 2

        if level_delta < self.hysteresis:
            return self._last_level
        if hold_elapsed < self.min_hold_s:
            return self._last_level

        self._last_level = requested_level
        self._last_change_t = now_s
        if requested_level == 5:
            self._peak_t = now_s
        return self._last_level

