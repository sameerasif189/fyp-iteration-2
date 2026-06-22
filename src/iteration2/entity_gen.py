"""
Entity Appearance Generator.

Procedural entity rendering system with model-driven parameters:
  - Body morphology (height, width, limb distortion)
  - Opacity and flicker pattern
  - Movement speed and direction
  - Glow/aura intensity
  - Eye presence, color, and count

Entity types:
  - shadow_pass: fleeting shadow at stress 4 (low opacity, slow drift)
  - silhouette_stare: standing figure at stress 5 (high opacity, near-still, eyes)
  - hallway_flash: momentary full-body flash
  - corner_lurk: partially visible figure peeking from edge

All rendering is procedural (no external sprites needed) so it works
as a standalone generation system exportable to Unity.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EntityParams:
    """Parameters driving entity appearance -- output by fusion model."""
    probability: float = 0.0
    opacity: float = 0.0
    aggression: float = 0.0
    morphology_distort: float = 0.0
    aura_intensity: float = 0.0
    height_scale: float = 1.0
    width_scale: float = 1.0
    movement_speed: float = 0.0
    eye_glow: float = 0.0
    flicker_rate: float = 0.0

    @classmethod
    def from_model_output(cls, model_params: dict) -> "EntityParams":
        """Construct from fusion model output dictionary."""
        prob = model_params.get("entity_probability", 0.0)
        opacity = model_params.get("entity_opacity", 0.0)
        aggression = model_params.get("entity_aggression", 0.0)

        return cls(
            probability=prob,
            opacity=max(opacity, 0.4) if prob > 0.5 else opacity,
            aggression=aggression,
            morphology_distort=aggression * 0.8,
            aura_intensity=aggression * 0.6,
            height_scale=1.0 + aggression * 0.4,
            width_scale=0.6 + opacity * 0.5,
            movement_speed=max(0.1, (1.0 - opacity) * 0.8),
            eye_glow=aggression * 1.2,
            flicker_rate=max(0.0, (1.0 - opacity) * 1.2),
        )


ENTITY_PRESETS = {
    "shadow_pass": {
        "base_opacity": 0.4,
        "speed": 0.6,
        "height": 200,
        "width": 45,
        "has_eyes": False,
        "flicker": True,
        "aura": False,
        "limb_count": 0,
    },
    "silhouette_stare": {
        "base_opacity": 0.9,
        "speed": 0.05,
        "height": 240,
        "width": 60,
        "has_eyes": True,
        "flicker": False,
        "aura": True,
        "limb_count": 4,
    },
    "hallway_flash": {
        "base_opacity": 1.0,
        "speed": 0.0,
        "height": 220,
        "width": 55,
        "has_eyes": True,
        "flicker": True,
        "aura": False,
        "limb_count": 6,
    },
    "corner_lurk": {
        "base_opacity": 0.6,
        "speed": 0.15,
        "height": 170,
        "width": 35,
        "has_eyes": True,
        "flicker": False,
        "aura": False,
        "limb_count": 2,
    },
}


def select_entity_type(params: EntityParams, tick: int) -> str:
    if params.opacity > 0.6 and params.aggression > 0.4:
        return "silhouette_stare"
    elif params.flicker_rate > 1.0:
        return "hallway_flash"
    elif params.movement_speed > 0.5:
        return "shadow_pass"
    else:
        return "corner_lurk"


def should_spawn_entity(params: EntityParams, tick: int, rng: random.Random) -> bool:
    if params.probability >= 0.8:
        return True
    if params.probability < 0.05:
        return False
    cycle = 120
    window = int(cycle * params.probability)
    phase = tick % cycle
    return phase < window


def compute_entity_position(entity_type: str, params: EntityParams,
                            tick: int, canvas_w: int, canvas_h: int,
                            rng: random.Random) -> Tuple[int, int]:
    preset = ENTITY_PRESETS[entity_type]

    if entity_type == "shadow_pass":
        period = max(200, int(600 / max(0.1, params.movement_speed)))
        progress = (tick % period) / period
        x = int(progress * (canvas_w + 100)) - 50
        y = canvas_h // 2 + int(math.sin(tick * 0.01) * 15)
    elif entity_type == "corner_lurk":
        side = (tick // 180) % 4
        if side == 0:
            x, y = 15, canvas_h // 3
        elif side == 1:
            x, y = canvas_w - 30, canvas_h // 3
        elif side == 2:
            x, y = canvas_w // 3, 15
        else:
            x, y = canvas_w // 3, canvas_h - 30
    elif entity_type == "hallway_flash":
        x = canvas_w // 2 + int(math.sin(tick * 0.015) * 10)
        y = canvas_h // 3
    else:
        sway = math.sin(tick * 0.008) * 5
        x = canvas_w // 2 + int(sway)
        y = canvas_h // 3 + int(math.sin(tick * 0.005) * 4)

    return x, y


def compute_flicker_alpha(params: EntityParams, tick: int) -> float:
    if params.flicker_rate < 0.1:
        return params.opacity

    flicker = math.sin(tick * params.flicker_rate * 0.15)
    if flicker < -0.5:
        return params.opacity * 0.3
    return params.opacity


def get_entity_color(entity_type: str, params: EntityParams,
                     tick: int) -> Tuple[int, int, int]:
    if entity_type == "silhouette_stare" and params.aggression > 0.5:
        pulse = abs(math.sin(tick * 0.03))
        r = int(15 + pulse * 25 * params.aggression)
        return (r, 3, 6)
    return (10, 6, 12)


def get_eye_color(params: EntityParams, tick: int) -> Tuple[int, int, int, int]:
    pulse = abs(math.sin(tick * 0.04))
    intensity = params.eye_glow * (0.7 + 0.3 * pulse)
    r = int(min(255, 150 + 105 * intensity))
    g = int(min(255, 20 + 30 * (1 - intensity)))
    b = int(min(255, 10 + 20 * (1 - intensity)))
    a = int(min(255, 180 + 75 * intensity))
    return (r, g, b, a)


def get_aura_params(params: EntityParams, tick: int) -> Tuple[int, int, Tuple[int, int, int, int]]:
    pulse = abs(math.sin(tick * 0.025))
    radius = int(25 + 50 * params.aura_intensity * (0.8 + 0.2 * pulse))
    alpha = int(25 + 40 * params.aura_intensity * pulse)
    color = (60, 8, 8, alpha)
    return radius, alpha, color


@dataclass
class EntityRenderData:
    """All data needed to render an entity in any renderer."""
    visible: bool
    entity_type: str
    x: int
    y: int
    width: int
    height: int
    opacity: float
    body_color: Tuple[int, int, int]
    has_eyes: bool
    eye_color: Tuple[int, int, int, int]
    eye_positions: list
    aura_radius: int
    aura_color: Tuple[int, int, int, int]
    limb_points: list
    particle_positions: list
    tendrils: List[List[Tuple[int, int]]] = field(default_factory=list)
    rib_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    mouth_points: List[Tuple[int, int]] = field(default_factory=list)


def generate_entity_frame(params: EntityParams, tick: int,
                          canvas_w: int, canvas_h: int,
                          seed: int = 0) -> EntityRenderData:
    rng = random.Random(seed + (tick // 4) * 7)

    if not should_spawn_entity(params, tick, rng):
        return EntityRenderData(
            visible=False, entity_type="none",
            x=0, y=0, width=0, height=0, opacity=0,
            body_color=(0, 0, 0), has_eyes=False,
            eye_color=(0, 0, 0, 0), eye_positions=[],
            aura_radius=0, aura_color=(0, 0, 0, 0),
            limb_points=[], particle_positions=[],
        )

    entity_type = select_entity_type(params, tick)
    preset = ENTITY_PRESETS[entity_type]

    x, y = compute_entity_position(entity_type, params, tick, canvas_w, canvas_h, rng)
    alpha = compute_flicker_alpha(params, tick)

    height = int(preset["height"] * params.height_scale)
    width = int(preset["width"] * params.width_scale)

    if params.morphology_distort > 0.3:
        height += int(rng.uniform(-5, 10) * params.morphology_distort)
        width += int(rng.uniform(-3, 6) * params.morphology_distort)

    body_color = get_entity_color(entity_type, params, tick)

    has_eyes = preset["has_eyes"]
    eye_color = get_eye_color(params, tick) if has_eyes else (0, 0, 0, 0)

    eye_positions = []
    if has_eyes:
        head_y = y - height // 3
        eye_spread = int(width * 0.25)
        eye_positions = [(x - eye_spread, head_y), (x + eye_spread, head_y)]
        if params.aggression > 0.7 and rng.random() < 0.4:
            eye_positions.append((x, head_y - 10))

    aura_radius, _, aura_color = get_aura_params(params, tick)
    if not preset["aura"]:
        aura_radius = 0
        aura_color = (0, 0, 0, 0)

    limb_points = []
    for i in range(preset["limb_count"]):
        angle = (i / max(1, preset["limb_count"])) * math.pi + math.sin(tick * 0.02) * 0.15
        lx = x + int(math.cos(angle) * width * (0.9 + params.morphology_distort * 0.4))
        ly = y + int(math.sin(angle) * height * 0.35)
        limb_points.append((lx, ly))

    particle_positions = []
    if params.aura_intensity > 0.2:
        num_particles = int(params.aura_intensity * 18)
        for _ in range(num_particles):
            px = x + rng.randint(-aura_radius - 10, aura_radius + 10)
            py = y + rng.randint(-height // 2, height // 2)
            particle_positions.append((px, py))

    tendrils = []
    if params.aggression > 0.4 and entity_type in ("silhouette_stare", "hallway_flash"):
        num_tendrils = int(2 + params.aggression * 4)
        for i in range(num_tendrils):
            base_angle = (i / num_tendrils) * math.pi * 2 + math.sin(tick * 0.01) * 0.1
            tendril_pts = []
            cx, cy = x, y + height // 4
            for seg in range(8):
                t = seg / 7.0
                reach = width * (0.6 + params.morphology_distort * 0.8) * t
                wobble = math.sin(tick * 0.02 + seg * 0.5 + i) * 8 * t
                tx = cx + int(math.cos(base_angle) * reach + wobble)
                ty = cy + int(math.sin(base_angle) * reach * 0.6)
                tendril_pts.append((tx, ty))
            tendrils.append(tendril_pts)

    rib_lines = []
    if params.morphology_distort > 0.4 and entity_type == "silhouette_stare":
        body_top = y - height // 6
        body_mid = y + height // 6
        num_ribs = int(3 + params.morphology_distort * 4)
        for i in range(num_ribs):
            ry = body_top + int((body_mid - body_top) * (i / max(1, num_ribs - 1)))
            half_w = int(width * 0.4 * (1 - abs(i - num_ribs // 2) / max(1, num_ribs)))
            rib_lines.append(((x - half_w, ry), (x + half_w, ry)))

    mouth_points = []
    if has_eyes and params.aggression > 0.5:
        head_y = y - height // 3
        mouth_y = head_y + int(width * 0.35)
        mouth_w = int(width * 0.3 * (1 + math.sin(tick * 0.03) * 0.2))
        mouth_points = [
            (x - mouth_w, mouth_y),
            (x - mouth_w // 2, mouth_y + int(mouth_w * 0.4 * params.aggression)),
            (x, mouth_y + int(mouth_w * 0.6 * params.aggression)),
            (x + mouth_w // 2, mouth_y + int(mouth_w * 0.4 * params.aggression)),
            (x + mouth_w, mouth_y),
        ]

    return EntityRenderData(
        visible=True,
        entity_type=entity_type,
        x=x, y=y,
        width=width, height=height,
        opacity=alpha,
        body_color=body_color,
        has_eyes=has_eyes,
        eye_color=eye_color,
        eye_positions=eye_positions,
        aura_radius=aura_radius,
        aura_color=aura_color,
        limb_points=limb_points,
        particle_positions=particle_positions,
        tendrils=tendrils,
        rib_lines=rib_lines,
        mouth_points=mouth_points,
    )
