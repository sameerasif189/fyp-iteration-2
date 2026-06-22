from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class StressEvent:
    level: int
    timestamp_s: float
    mode: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AudioCommand:
    layer: str
    intensity: float
    seed: int
    profile: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VisualCommand:
    profile: str
    corruption_alpha: float
    fog_density: float
    light_flicker_hz: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CameraFxCommand:
    profile: str
    magnitude: float
    duration_s: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EntityCommand:
    entity_type: str
    weight: float
    duration_s: float
    enabled: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

