from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .asset_bank import PreloadedAssetBank
from .commands import AudioCommand, CameraFxCommand, EntityCommand, StressEvent, VisualCommand
from .realtime_modifiers import RealtimeGenerator
from .stress_schema import StressTransitionController, STRESS_PROFILES


@dataclass(frozen=True)
class FrameOutput:
    stress_event: StressEvent
    audio_command: AudioCommand
    visual_command: VisualCommand
    camera_command: CameraFxCommand
    entity_command: EntityCommand
    cache_stats: Dict[str, int]
    generated_assets: Dict[str, str]
    model_params: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "stress_event": self.stress_event.to_dict(),
            "audio_command": self.audio_command.to_dict(),
            "visual_command": self.visual_command.to_dict(),
            "camera_command": self.camera_command.to_dict(),
            "entity_command": self.entity_command.to_dict(),
            "cache_stats": self.cache_stats,
            "generated_assets": self.generated_assets,
            "model_params": self.model_params,
        }


class FusionOrchestrator:
    def __init__(
        self,
        catalog_path: Path,
        seed: int = 395,
        hysteresis: int = 1,
        min_hold_s: float = 8.0,
        cooldown_after_peak_s: float = 14.0,
        model_config_path: Path | None = None,
        trained_model_path: Path | None = None,
    ):
        self.asset_bank = PreloadedAssetBank(catalog_path)
        self.fallback_generator = RealtimeGenerator(global_seed=seed, model_config_path=model_config_path)
        self.transitions = StressTransitionController(
            hysteresis=hysteresis,
            min_hold_s=min_hold_s,
            cooldown_after_peak_s=cooldown_after_peak_s,
        )
        self.tick = 0
        self.seed = seed
        self.model_inference = None
        self.using_model = False

        if trained_model_path and trained_model_path.exists():
            try:
                from .model_inference import FusionModelInference
                self.model_inference = FusionModelInference(trained_model_path)
                self.using_model = True
                inf_ms = self.model_inference.inference_time_ms()
                print(f"[orchestrator] using trained model, avg inference={inf_ms:.3f}ms")
            except Exception as e:
                print(f"[orchestrator] model load failed ({e}), using fallback generator")

    def step(self, requested_level: int, now_s: float, mode: str = "auto") -> FrameOutput:
        level = self.transitions.resolve_level(requested_level=requested_level, now_s=now_s)
        self.asset_bank.warm_levels(level)

        model_params_dict: Dict[str, float] = {}

        if self.using_model and self.model_inference is not None:
            mp = self.model_inference.generate(level, seed=self.seed + self.tick)
            model_params_dict = mp.to_dict()
            audio_intensity = mp.audio_intensity
            corruption_alpha = mp.visual_corruption
            fog_density = mp.visual_fog_density
            camera_magnitude = mp.camera_magnitude
            entity_weight = mp.entity_probability
            light_flicker_hz = mp.visual_flicker_rate * 3.0
            profile_name = STRESS_PROFILES[level].name.lower()
        else:
            params = self.fallback_generator.generate(level=level, tick=self.tick)
            audio_intensity = params.audio_intensity
            corruption_alpha = params.corruption_alpha
            fog_density = params.fog_density
            camera_magnitude = params.camera_magnitude
            entity_weight = params.entity_weight
            light_flicker_hz = 0.4 + (2.0 * corruption_alpha)
            profile_name = params.profile_name

        audio_assets = self.asset_bank.get_assets("audio", level)
        visual_assets = self.asset_bank.get_assets("visual", level)
        camera_assets = self.asset_bank.get_assets("camera", level)
        entity_assets = self.asset_bank.get_assets("entity", level)

        audio_layer = audio_assets[self.tick % len(audio_assets)] if audio_assets else "amb_fallback"
        visual_profile = visual_assets[self.tick % len(visual_assets)] if visual_assets else "visual_fallback"
        camera_profile = camera_assets[self.tick % len(camera_assets)] if camera_assets else "cam_fallback"
        entity_type = entity_assets[self.tick % len(entity_assets)] if entity_assets else "none"

        event = StressEvent(level=level, timestamp_s=now_s, mode=mode)
        audio_cmd = AudioCommand(
            layer=audio_layer,
            intensity=audio_intensity,
            seed=self.seed + self.tick,
            profile=profile_name,
        )
        visual_cmd = VisualCommand(
            profile=visual_profile,
            corruption_alpha=corruption_alpha,
            fog_density=fog_density,
            light_flicker_hz=light_flicker_hz,
        )
        camera_cmd = CameraFxCommand(
            profile=camera_profile,
            magnitude=camera_magnitude,
            duration_s=0.8,
        )
        entity_cmd = EntityCommand(
            entity_type=entity_type,
            weight=entity_weight,
            duration_s=1.2,
            enabled=(level >= 4 and entity_type != "none"),
        )
        self.tick += 1
        return FrameOutput(
            stress_event=event,
            audio_command=audio_cmd,
            visual_command=visual_cmd,
            camera_command=camera_cmd,
            entity_command=entity_cmd,
            cache_stats={
                "active": self.asset_bank.active_cache_size,
                "warm": self.asset_bank.warm_cache_size,
            },
            generated_assets={
                "audio_asset": audio_layer,
                "visual_asset": visual_profile,
                "camera_asset": camera_profile,
                "entity_asset": entity_type,
            },
            model_params=model_params_dict,
        )
