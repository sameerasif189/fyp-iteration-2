"""
Peripheral Vision Camera Feed Manipulation.

Captures live webcam feed and applies stress-driven manipulations:
  - Face distortion (uncanny valley warping via facial landmarks)
  - Background darkening at edges
  - Fleeting shadow figures at frame borders
  - Gaslighting mechanic: distortions happen at LOW stress (0-2),
    feed normalizes at HIGH stress (3-5) -- reverse psychology

Uses OpenCV for capture and warping, with optional MediaPipe for landmarks.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


FEED_WIDTH = 320
FEED_HEIGHT = 240


@dataclass
class CameraFeedParams:
    """Parameters from fusion model controlling camera feed effects."""
    face_distort_amount: float = 0.0
    shadow_inject_prob: float = 0.0
    edge_darken: float = 0.0
    peripheral_figure_prob: float = 0.0

    @classmethod
    def from_stress(cls, stress_level: int, model_params: dict = None) -> "CameraFeedParams":
        """
        Gaslighting mechanic: effects are STRONGER at LOW stress,
        weaker at HIGH stress. Player sees weird things when calm.
        """
        if model_params:
            return cls(
                face_distort_amount=model_params.get("camera_feed_face_distort", 0.0),
                shadow_inject_prob=model_params.get("camera_feed_shadow", 0.0),
                edge_darken=model_params.get("camera_feed_darken", 0.0),
                peripheral_figure_prob=model_params.get("camera_feed_figure", 0.0),
            )

        inverse_stress = max(0, 3 - stress_level) / 3.0

        return cls(
            face_distort_amount=inverse_stress * 0.6,
            shadow_inject_prob=inverse_stress * 0.3,
            edge_darken=inverse_stress * 0.5,
            peripheral_figure_prob=inverse_stress * 0.15,
        )


class WebcamCapture:
    """Manages webcam access with graceful fallback."""

    def __init__(self, camera_id: int = 0):
        self.cap = None
        self.camera_id = camera_id
        self.frame_count = 0
        self._init_camera()

    def _init_camera(self):
        if not HAS_CV2:
            return
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.cap = None
        except Exception:
            self.cap = None

    @property
    def available(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from webcam, returns BGR image or None."""
        if not self.available:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_count += 1
        frame = cv2.resize(frame, (FEED_WIDTH, FEED_HEIGHT))
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


def apply_face_distortion(frame: np.ndarray, amount: float,
                          tick: int) -> np.ndarray:
    """
    Apply uncanny valley face distortion using mesh warping.
    Without face landmarks, applies general warping to center of frame.
    """
    if amount < 0.01 or not HAS_CV2:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy()

    cx, cy = w // 2, h // 3
    region_size = min(w, h) // 3

    y1 = max(0, cy - region_size)
    y2 = min(h, cy + region_size)
    x1 = max(0, cx - region_size)
    x2 = min(w, cx + region_size)

    region = result[y1:y2, x1:x2]
    rh, rw = region.shape[:2]

    map_x = np.zeros((rh, rw), dtype=np.float32)
    map_y = np.zeros((rh, rw), dtype=np.float32)

    for iy in range(rh):
        for ix in range(rw):
            dx = (ix - rw / 2) / (rw / 2)
            dy = (iy - rh / 2) / (rh / 2)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1.0:
                warp_str = amount * (1.0 - dist) * 8.0
                phase = tick * 0.05
                wx = math.sin(dy * 3.0 + phase) * warp_str
                wy = math.cos(dx * 3.0 + phase * 1.3) * warp_str
                map_x[iy, ix] = ix + wx
                map_y[iy, ix] = iy + wy
            else:
                map_x[iy, ix] = ix
                map_y[iy, ix] = iy

    warped = cv2.remap(region, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    result[y1:y2, x1:x2] = warped
    return result


def apply_edge_darkening(frame: np.ndarray, amount: float) -> np.ndarray:
    """Darken the edges of the camera feed (vignette)."""
    if amount < 0.01 or not HAS_CV2:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy().astype(np.float32)

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    dist_norm = dist / max_dist

    darken_mask = 1.0 - (dist_norm ** 1.5) * amount
    darken_mask = np.clip(darken_mask, 0.2, 1.0)

    for c in range(3):
        result[:, :, c] *= darken_mask

    return np.clip(result, 0, 255).astype(np.uint8)


def inject_shadow_figure(frame: np.ndarray, prob: float, tick: int,
                         rng: random.Random) -> np.ndarray:
    """Inject a fleeting dark figure at the edge of the frame."""
    if rng.random() > prob or not HAS_CV2:
        return frame

    h, w = frame.shape[:2]
    result = frame.copy()

    side = rng.choice(["left", "right", "top", "bottom"])
    figure_h = rng.randint(h // 4, h // 2)
    figure_w = rng.randint(15, 35)

    if side == "left":
        x_start = rng.randint(0, 20)
        y_start = rng.randint(0, h - figure_h)
    elif side == "right":
        x_start = w - figure_w - rng.randint(0, 20)
        y_start = rng.randint(0, h - figure_h)
    elif side == "top":
        x_start = rng.randint(0, w - figure_w)
        y_start = rng.randint(0, 15)
    else:
        x_start = rng.randint(0, w - figure_w)
        y_start = h - figure_h - rng.randint(0, 15)

    x_end = min(w, x_start + figure_w)
    y_end = min(h, y_start + figure_h)
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    shadow = np.zeros((y_end - y_start, x_end - x_start, 3), dtype=np.float32)
    alpha = rng.uniform(0.3, 0.7) * min(1.0, prob * 3)

    region = result[y_start:y_end, x_start:x_end].astype(np.float32)
    blended = region * (1.0 - alpha) + shadow * alpha
    result[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)

    return result


class CameraFeedProcessor:
    """
    Full camera feed processing pipeline.

    Captures webcam, applies effects based on stress level.
    Falls back to a synthetic 'static' feed if no webcam available.
    """

    def __init__(self, camera_id: int = 0):
        self.webcam = WebcamCapture(camera_id)
        self.tick = 0
        self.rng = random.Random(42)

    @property
    def has_webcam(self) -> bool:
        return self.webcam.available

    def get_raw_frame_rgb(self) -> Optional[np.ndarray]:
        """Return the latest *raw* webcam frame as RGB (no effects).

        Returns None when no webcam is connected -- the demo now refuses to
        synthesise a fake camera feed; the real webcam is the only source.
        """
        self.tick += 1
        frame = self.webcam.read_frame()
        if frame is None:
            return None
        if HAS_CV2:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_frame(self, params: CameraFeedParams) -> Optional[np.ndarray]:
        """Capture and process one frame with all effects.

        Returns None when no webcam is available (no synthetic fallback)."""
        self.tick += 1
        frame = self.webcam.read_frame()
        if frame is None:
            return None
        frame = apply_face_distortion(frame, params.face_distort_amount, self.tick)
        frame = apply_edge_darkening(frame, params.edge_darken)
        frame = inject_shadow_figure(frame, params.shadow_inject_prob,
                                     self.tick, self.rng)
        return frame

    def get_frame_rgb(self, params: CameraFeedParams) -> Optional[np.ndarray]:
        """Get processed frame in RGB format (for pygame), or None if no webcam."""
        frame_bgr = self.process_frame(params)
        if frame_bgr is None:
            return None
        if HAS_CV2:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_bgr

    def release(self):
        self.webcam.release()
