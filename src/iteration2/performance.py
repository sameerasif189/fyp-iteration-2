from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional


@dataclass
class PerformanceConfig:
    target_fps: int = 30
    adaptive_budget_ms_avg: float = 7.0
    adaptive_budget_ms_peak: float = 10.0


@dataclass
class PerformanceState:
    quality_tier: int = 0
    fallback_reason: str = "none"


class PerfMonitor:
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self._start_t = 0.0
        self.state = PerformanceState()

    def begin(self) -> None:
        self._start_t = perf_counter()

    def end(self) -> Dict[str, float]:
        elapsed_ms = (perf_counter() - self._start_t) * 1000.0
        frame_budget_ms = 1000.0 / self.config.target_fps
        over_peak = elapsed_ms > self.config.adaptive_budget_ms_peak
        over_avg = elapsed_ms > self.config.adaptive_budget_ms_avg

        if over_peak:
            self.state.quality_tier = min(2, self.state.quality_tier + 1)
            self.state.fallback_reason = "peak_budget_exceeded"
        elif over_avg and self.state.quality_tier < 1:
            self.state.quality_tier = 1
            self.state.fallback_reason = "avg_budget_exceeded"
        elif not over_avg:
            self.state.quality_tier = max(0, self.state.quality_tier - 1)
            if self.state.quality_tier == 0:
                self.state.fallback_reason = "none"

        return {
            "adaptive_ms": elapsed_ms,
            "frame_budget_ms": frame_budget_ms,
            "quality_tier": float(self.state.quality_tier),
        }

