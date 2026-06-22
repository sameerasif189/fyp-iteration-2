"""Minimal 3D environment for viewing the GLB creatures in `new entities/`.

Loads every .glb under ``./new entities/`` into a single interactive Panda3D
scene with PBR lighting, a ground grid, an orbiting camera, and on-screen
labels for each entity. The camera auto-frames all entities on launch.

Controls
--------
    Left mouse drag    Orbit the camera around the focus point.
    Right mouse drag   Pan the focus point.
    Mouse wheel        Zoom in / out.
    1 .. 9             Focus a single entity (by load order).
    F                  Re-frame all entities.
    R                  Reset the camera to its default pose.
    Space              Toggle slow auto-rotation of the entities.
    G                  Toggle the ground grid.
    L                  Cycle a lighting preset (dim / standard / bright).
    Esc                Quit.

Usage
-----
    python view_entities_3d.py
    python view_entities_3d.py --entities-dir "new entities"
    python view_entities_3d.py --target-size 3.0 --spacing 4.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

import gltf
import simplepbr
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CardMaker,
    DirectionalLight,
    Fog,
    LineSegs,
    NodePath,
    TextNode,
    Vec3,
    Vec4,
    WindowProperties,
    loadPrcFileData,
)


DEFAULT_ENTITY_DIR = Path(__file__).parent / "new entities"


# Panda3D config has to be set before ShowBase is constructed.
loadPrcFileData("", "window-title Entity Viewer  -  new entities/")
loadPrcFileData("", "win-size 1280 800")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")
loadPrcFileData("", "show-frame-rate-meter true")
loadPrcFileData("", "sync-video false")


def _spherical_offset(yaw_deg: float, pitch_deg: float, radius: float) -> Vec3:
    """Camera offset from target using Panda3D's Z-up, +Y-forward convention.

    yaw=0, pitch=0 places the camera at (0, -r, 0) looking toward +Y.
    Positive pitch lifts the camera upward.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cp = math.cos(pitch)
    return Vec3(
        -radius * cp * math.sin(yaw),
        -radius * cp * math.cos(yaw),
        radius * math.sin(pitch),
    )


class EntityViewer(ShowBase):
    """Interactive Panda3D viewer for a folder of GLB entities."""

    def __init__(
        self,
        entity_paths: List[Path],
        target_size: float = 2.5,
        spacing: float = 3.5,
    ) -> None:
        super().__init__()
        self.disableMouse()

        # PBR pipeline so the GLB materials render with proper lighting.
        self.pbr = simplepbr.init(
            window=self.win,
            use_normal_maps=True,
            enable_shadows=False,
            max_lights=6,
        )

        self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.setBackgroundColor(0.045, 0.055, 0.085, 1.0)

        fog = Fog("scene-fog")
        fog.setColor(0.045, 0.055, 0.085)
        fog.setExpDensity(0.018)
        self.render.setFog(fog)

        self.target_size = float(target_size)
        self.spacing = float(spacing)

        self._setup_lights()
        self._build_ground()
        self._load_entities(entity_paths)
        self._setup_camera()
        self._setup_input()
        self._setup_hud()

        self.auto_rotate = True
        self.taskMgr.add(self._spin_task, "entity-spin")
        self.taskMgr.add(self._mouse_task, "orbit-mouse")

        self._frame_all()

    # ------------------------------------------------------------------ scene

    def _setup_lights(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.22, 0.24, 0.30, 1.0))
        self.amb_np = self.render.attachNewNode(ambient)
        self.render.setLight(self.amb_np)

        key = DirectionalLight("key")
        key.setColor(Vec4(1.05, 1.00, 0.90, 1.0))
        self.key_np = self.render.attachNewNode(key)
        self.key_np.setHpr(40, -50, 0)
        self.render.setLight(self.key_np)

        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.35, 0.42, 0.58, 1.0))
        self.fill_np = self.render.attachNewNode(fill)
        self.fill_np.setHpr(-130, -25, 0)
        self.render.setLight(self.fill_np)

        rim = DirectionalLight("rim")
        rim.setColor(Vec4(0.55, 0.42, 0.70, 1.0))
        self.rim_np = self.render.attachNewNode(rim)
        self.rim_np.setHpr(170, -15, 0)
        self.render.setLight(self.rim_np)

        self.light_preset = 1
        self._light_presets = [
            dict(ambient=0.10, key=0.60, fill=0.22, rim=0.35),
            dict(ambient=0.22, key=1.05, fill=0.42, rim=0.55),
            dict(ambient=0.45, key=1.50, fill=0.70, rim=0.80),
        ]

    def _build_ground(self) -> None:
        self.ground_np = self.render.attachNewNode("ground-root")

        size = 50
        cm = CardMaker("ground-plane")
        cm.setFrame(-size, size, -size, size)
        plane = self.ground_np.attachNewNode(cm.generate())
        plane.setHpr(0, -90, 0)
        plane.setColor(0.075, 0.080, 0.105, 1.0)
        plane.setZ(0)

        minor = LineSegs("grid-minor")
        minor.setThickness(1.0)
        minor.setColor(0.17, 0.19, 0.26, 1.0)
        for i in range(-size, size + 1):
            minor.moveTo(i, -size, 0.001)
            minor.drawTo(i, size, 0.001)
            minor.moveTo(-size, i, 0.001)
            minor.drawTo(size, i, 0.001)
        self.ground_np.attachNewNode(minor.create()).setLightOff()

        major = LineSegs("grid-major")
        major.setThickness(1.8)
        major.setColor(0.32, 0.36, 0.46, 1.0)
        for i in range(-size, size + 1, 5):
            major.moveTo(i, -size, 0.002)
            major.drawTo(i, size, 0.002)
            major.moveTo(-size, i, 0.002)
            major.drawTo(size, i, 0.002)
        self.ground_np.attachNewNode(major.create()).setLightOff()

        axes = LineSegs("axes")
        axes.setThickness(2.5)
        axes.setColor(0.80, 0.30, 0.30, 1.0)
        axes.moveTo(-size, 0, 0.003)
        axes.drawTo(size, 0, 0.003)
        axes.setColor(0.30, 0.80, 0.45, 1.0)
        axes.moveTo(0, -size, 0.003)
        axes.drawTo(0, size, 0.003)
        self.ground_np.attachNewNode(axes.create()).setLightOff()

    # ----------------------------------------------------------------- models

    def _load_entities(self, paths: List[Path]) -> None:
        settings = gltf.GltfSettings(skip_animations=True)
        self.entities: list[dict] = []
        n = len(paths)
        for i, p in enumerate(paths):
            try:
                root = gltf.load_model(str(p), gltf_settings=settings)
            except Exception as exc:
                print(f"[viewer] failed to load {p.name}: {exc}")
                continue

            holder = self.render.attachNewNode(f"entity-{i}")
            model = NodePath(root)
            model.reparentTo(holder)

            bounds = model.getTightBounds()
            if bounds is None:
                print(f"[viewer] {p.name}: empty bounds, skipping")
                holder.removeNode()
                continue

            bmin, bmax = bounds
            size_vec = bmax - bmin
            diag = max(size_vec.length(), 1e-6)
            scale = self.target_size / diag
            model.setScale(scale)

            bounds2 = model.getTightBounds()
            if bounds2 is not None:
                bmin2, bmax2 = bounds2
                center2 = (bmin2 + bmax2) * 0.5
                model.setPos(-center2.x, -center2.y, -bmin2.z)

            x = (i - (n - 1) / 2.0) * self.spacing
            holder.setPos(x, 0, 0)

            label_tn = TextNode(f"label-{i}")
            label_tn.setText(p.stem)
            label_tn.setAlign(TextNode.ACenter)
            label_tn.setTextColor(0.95, 0.97, 1.0, 1.0)
            label_tn.setCardColor(0.0, 0.0, 0.0, 0.55)
            label_tn.setCardAsMargin(0.30, 0.30, 0.12, 0.12)
            label_tn.setCardDecal(True)
            label = self.render.attachNewNode(label_tn)
            label.setBillboardPointEye()
            label.setScale(0.32)
            label.setPos(x, 0, self.target_size + 0.7)
            label.setLightOff()
            label.setDepthWrite(False)
            label.setBin("fixed", 50)

            index_tn = TextNode(f"index-{i}")
            index_tn.setText(f"{i + 1}")
            index_tn.setAlign(TextNode.ACenter)
            index_tn.setTextColor(0.55, 0.85, 1.0, 1.0)
            index_label = self.render.attachNewNode(index_tn)
            index_label.setBillboardPointEye()
            index_label.setScale(0.45)
            index_label.setPos(x, 0, self.target_size + 1.2)
            index_label.setLightOff()
            index_label.setDepthWrite(False)
            index_label.setBin("fixed", 50)

            self.entities.append(
                {
                    "index": i,
                    "name": p.stem,
                    "path": p,
                    "node": holder,
                    "label": label,
                    "index_label": index_label,
                    "center": Vec3(x, 0, self.target_size * 0.5),
                    "radius": self.target_size,
                }
            )
            print(
                f"[viewer] loaded {p.name}  scale={scale:.4f}  "
                f"x={x:+.2f}  size=({size_vec.x:.2f},{size_vec.y:.2f},{size_vec.z:.2f})"
            )

        if not self.entities:
            raise SystemExit("No entities could be loaded.")

    # ----------------------------------------------------------------- camera

    def _setup_camera(self) -> None:
        self.cam_target = Vec3(0.0, 0.0, self.target_size * 0.5)
        self.cam_yaw = 30.0
        self.cam_pitch = 18.0
        self.cam_distance = 12.0
        self._update_camera()

    def _update_camera(self) -> None:
        offset = _spherical_offset(self.cam_yaw, self.cam_pitch, self.cam_distance)
        self.camera.setPos(self.cam_target + offset)
        self.camera.lookAt(self.cam_target)

    def _reset_camera(self) -> None:
        self.cam_target = Vec3(0.0, 0.0, self.target_size * 0.5)
        self.cam_yaw = 30.0
        self.cam_pitch = 18.0
        self.cam_distance = 12.0
        self._update_camera()

    def _frame_all(self) -> None:
        if not self.entities:
            return
        xs = [e["center"].x for e in self.entities]
        cx = sum(xs) / len(xs)
        cz = sum(e["center"].z for e in self.entities) / len(self.entities)
        spread = (max(xs) - min(xs)) if len(xs) > 1 else self.target_size
        self.cam_target = Vec3(cx, 0.0, cz)
        self.cam_distance = max(8.0, spread * 1.25 + self.target_size * 1.5)
        self._update_camera()

    def _focus_index(self, idx: int) -> None:
        if 0 <= idx < len(self.entities):
            e = self.entities[idx]
            self.cam_target = Vec3(e["center"])
            self.cam_distance = max(4.0, e["radius"] * 2.4)
            self._update_camera()

    # ------------------------------------------------------------------ input

    def _setup_input(self) -> None:
        self.mouse_btn = {1: False, 2: False, 3: False}
        self.prev_mouse: Optional[tuple[float, float]] = None

        self.accept("escape", sys.exit)
        self.accept("mouse1", self._on_btn, [1, True])
        self.accept("mouse1-up", self._on_btn, [1, False])
        self.accept("mouse2", self._on_btn, [2, True])
        self.accept("mouse2-up", self._on_btn, [2, False])
        self.accept("mouse3", self._on_btn, [3, True])
        self.accept("mouse3-up", self._on_btn, [3, False])
        self.accept("wheel_up", self._on_zoom, [-1])
        self.accept("wheel_down", self._on_zoom, [1])

        self.accept("r", self._reset_camera)
        self.accept("f", self._frame_all)
        self.accept("space", self._toggle_spin)
        self.accept("g", self._toggle_ground)
        self.accept("l", self._cycle_lights)
        for i in range(1, 10):
            self.accept(str(i), self._focus_index, [i - 1])

    def _on_btn(self, btn: int, down: bool) -> None:
        self.mouse_btn[btn] = down
        if down and self.mouseWatcherNode is not None and self.mouseWatcherNode.hasMouse():
            self.prev_mouse = (
                self.mouseWatcherNode.getMouseX(),
                self.mouseWatcherNode.getMouseY(),
            )
        if not down:
            self.prev_mouse = None

    def _on_zoom(self, direction: int) -> None:
        factor = 1.12 if direction > 0 else 0.89
        self.cam_distance = max(2.0, min(80.0, self.cam_distance * factor))
        self._update_camera()

    def _mouse_task(self, task):
        mw = self.mouseWatcherNode
        if mw is None:
            return Task.cont
        if mw.hasMouse():
            mx, my = mw.getMouseX(), mw.getMouseY()
            if self.prev_mouse is not None:
                dx = mx - self.prev_mouse[0]
                dy = my - self.prev_mouse[1]
                if self.mouse_btn[1]:
                    self.cam_yaw += dx * 130.0
                    self.cam_pitch = max(-85.0, min(85.0, self.cam_pitch + dy * 90.0))
                    self._update_camera()
                elif self.mouse_btn[3] or self.mouse_btn[2]:
                    right = self.camera.getMat().getRow3(0)
                    up = self.camera.getMat().getRow3(2)
                    pan = right * (-dx * self.cam_distance) + up * (-dy * self.cam_distance)
                    self.cam_target += pan
                    self._update_camera()
            self.prev_mouse = (mx, my)
        else:
            self.prev_mouse = None
        return Task.cont

    # ----------------------------------------------------------------- tweaks

    def _spin_task(self, task):
        if self.auto_rotate:
            dt = globalClock.getDt()
            for e in self.entities:
                e["node"].setH(e["node"].getH() + 12.0 * dt)
        return Task.cont

    def _toggle_spin(self) -> None:
        self.auto_rotate = not self.auto_rotate

    def _toggle_ground(self) -> None:
        if self.ground_np.isHidden():
            self.ground_np.show()
        else:
            self.ground_np.hide()

    def _cycle_lights(self) -> None:
        self.light_preset = (self.light_preset + 1) % len(self._light_presets)
        p = self._light_presets[self.light_preset]
        a = p["ambient"]
        self.amb_np.node().setColor(Vec4(a, a + 0.02, a + 0.07, 1.0))
        k = p["key"]
        self.key_np.node().setColor(Vec4(k, k * 0.95, k * 0.85, 1.0))
        f = p["fill"]
        self.fill_np.node().setColor(Vec4(f * 0.85, f * 1.0, f * 1.35, 1.0))
        r = p["rim"]
        self.rim_np.node().setColor(Vec4(r * 1.05, r * 0.80, r * 1.25, 1.0))

    # -------------------------------------------------------------------- HUD

    def _setup_hud(self) -> None:
        OnscreenText(
            text=(
                "Entity Viewer  |  new entities/\n"
                "LMB orbit   RMB / MMB pan   wheel zoom\n"
                "[F] frame all   [R] reset   [Space] spin\n"
                "[G] grid   [L] lights   [1-9] focus   [Esc] quit"
            ),
            parent=self.a2dTopLeft,
            pos=(0.04, -0.10),
            scale=0.045,
            fg=(0.88, 0.92, 1.0, 1.0),
            shadow=(0, 0, 0, 0.65),
            align=TextNode.ALeft,
            mayChange=False,
        )

        names = "  |  ".join(f"[{i + 1}] {e['name']}" for i, e in enumerate(self.entities))
        OnscreenText(
            text=names,
            parent=self.a2dBottomCenter,
            pos=(0.0, 0.06),
            scale=0.038,
            fg=(0.78, 0.84, 0.95, 1.0),
            shadow=(0, 0, 0, 0.6),
            align=TextNode.ACenter,
            mayChange=False,
        )


def _collect_entities(entity_dir: Path) -> List[Path]:
    if not entity_dir.exists():
        raise SystemExit(f"Entity directory not found: {entity_dir}")
    paths = sorted(
        list(entity_dir.glob("*.glb"))
        + list(entity_dir.glob("*.gltf"))
    )
    if not paths:
        raise SystemExit(f"No .glb/.gltf files in {entity_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal 3D viewer for the GLB entities.")
    parser.add_argument(
        "--entities-dir",
        type=Path,
        default=DEFAULT_ENTITY_DIR,
        help="Folder containing .glb / .gltf files (default: ./new entities).",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=2.5,
        help="Diagonal length each entity is normalized to (world units).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=3.5,
        help="Horizontal spacing between entities (world units).",
    )
    args = parser.parse_args()

    paths = _collect_entities(args.entities_dir)
    print(f"[viewer] {len(paths)} entities in {args.entities_dir}")
    for p in paths:
        print(f"  - {p.name}")

    app = EntityViewer(
        paths,
        target_size=args.target_size,
        spacing=args.spacing,
    )
    app.run()


if __name__ == "__main__":
    main()
