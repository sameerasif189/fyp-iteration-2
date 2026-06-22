"""
Face Distortion Module.

Applies uncanny valley facial warping using landmarks:
  - Eye widening/narrowing
  - Mouth stretching
  - Jawline sharpening
  - Skin color desaturation
  - Asymmetric feature displacement

Uses facial landmark detection (68-point model) when available,
falls back to general region-based warping otherwise.

Controlled by fusion model parameters for real-time generation.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class FaceDistortParams:
    """Controls for facial distortion effects."""
    eye_widen: float = 0.0
    eye_narrow: float = 0.0
    mouth_stretch: float = 0.0
    jaw_sharpen: float = 0.0
    skin_desaturate: float = 0.0
    asymmetry: float = 0.0
    overall_intensity: float = 0.0

    @classmethod
    def from_stress(cls, stress_level: int) -> "FaceDistortParams":
        """
        Generate distortion params inversely proportional to stress.
        More distortion when calm (gaslighting).
        """
        inverse = max(0, 3 - stress_level) / 3.0
        return cls(
            eye_widen=inverse * 0.4,
            eye_narrow=0.0,
            mouth_stretch=inverse * 0.3,
            jaw_sharpen=inverse * 0.25,
            skin_desaturate=inverse * 0.5,
            asymmetry=inverse * 0.2,
            overall_intensity=inverse,
        )


class FaceLandmarkDetector:
    """
    Detects facial landmarks using OpenCV's Haar cascade + DLib-style points.
    Falls back to center-of-frame estimation if no face detected.
    """

    def __init__(self):
        self.face_cascade = None
        if HAS_CV2:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box (x, y, w, h) or None."""
        if self.face_cascade is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

        if len(faces) == 0:
            return None

        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces_sorted[0])

    def estimate_landmarks(self, face_rect: Tuple[int, int, int, int]) -> dict:
        """
        Estimate approximate landmark positions from face bounding box.
        Returns dict with keys: left_eye, right_eye, nose, mouth, jaw_left, jaw_right
        """
        x, y, w, h = face_rect
        return {
            "left_eye": (x + int(w * 0.3), y + int(h * 0.35)),
            "right_eye": (x + int(w * 0.7), y + int(h * 0.35)),
            "nose": (x + int(w * 0.5), y + int(h * 0.55)),
            "mouth": (x + int(w * 0.5), y + int(h * 0.72)),
            "jaw_left": (x + int(w * 0.15), y + int(h * 0.85)),
            "jaw_right": (x + int(w * 0.85), y + int(h * 0.85)),
            "forehead": (x + int(w * 0.5), y + int(h * 0.15)),
        }


def apply_eye_warp(frame: np.ndarray, eye_pos: Tuple[int, int],
                   widen: float, tick: int) -> np.ndarray:
    """Warp region around eye to create widening/narrowing effect."""
    if not HAS_CV2 or widen < 0.01:
        return frame

    h, w = frame.shape[:2]
    ex, ey = eye_pos
    radius = 20

    y1 = max(0, ey - radius)
    y2 = min(h, ey + radius)
    x1 = max(0, ex - radius)
    x2 = min(w, ex + radius)

    region = frame[y1:y2, x1:x2].copy()
    rh, rw = region.shape[:2]
    if rh < 5 or rw < 5:
        return frame

    map_x = np.zeros((rh, rw), dtype=np.float32)
    map_y = np.zeros((rh, rw), dtype=np.float32)

    for iy in range(rh):
        for ix in range(rw):
            dx = (ix - rw / 2) / (rw / 2)
            dy = (iy - rh / 2) / (rh / 2)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1.0:
                scale = 1.0 + widen * (1.0 - dist) * 0.5
                map_x[iy, ix] = rw / 2 + dx * (rw / 2) / scale
                map_y[iy, ix] = rh / 2 + dy * (rh / 2) / scale
            else:
                map_x[iy, ix] = ix
                map_y[iy, ix] = iy

    warped = cv2.remap(region, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    frame[y1:y2, x1:x2] = warped
    return frame


def apply_mouth_stretch(frame: np.ndarray, mouth_pos: Tuple[int, int],
                        amount: float) -> np.ndarray:
    """Stretch mouth region horizontally."""
    if not HAS_CV2 or amount < 0.01:
        return frame

    h, w = frame.shape[:2]
    mx, my = mouth_pos
    radius_x, radius_y = 25, 15

    y1 = max(0, my - radius_y)
    y2 = min(h, my + radius_y)
    x1 = max(0, mx - radius_x)
    x2 = min(w, mx + radius_x)

    region = frame[y1:y2, x1:x2].copy()
    rh, rw = region.shape[:2]
    if rh < 5 or rw < 5:
        return frame

    stretch_factor = 1.0 + amount * 0.4
    new_w = int(rw * stretch_factor)
    stretched = cv2.resize(region, (new_w, rh))

    crop_start = (new_w - rw) // 2
    cropped = stretched[:, crop_start:crop_start + rw]

    if cropped.shape[1] == rw and cropped.shape[0] == rh:
        frame[y1:y2, x1:x2] = cropped

    return frame


def apply_skin_desaturation(frame: np.ndarray, face_rect: Tuple[int, int, int, int],
                            amount: float) -> np.ndarray:
    """Desaturate skin tones in face region for corpse-like appearance."""
    if not HAS_CV2 or amount < 0.01:
        return frame

    x, y, w, h = face_rect
    result = frame.copy()
    region = result[y:y + h, x:x + w]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 - amount * 0.7)
    hsv[:, :, 2] *= (1.0 - amount * 0.2)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    result[y:y + h, x:x + w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result


class FaceDistortionPipeline:
    """Complete face distortion pipeline combining all effects."""

    def __init__(self):
        self.detector = FaceLandmarkDetector()
        self.tick = 0
        self.last_face_rect = None

    def process(self, frame: np.ndarray, params: FaceDistortParams) -> np.ndarray:
        """Apply all face distortion effects to frame."""
        if params.overall_intensity < 0.01:
            return frame

        self.tick += 1

        face_rect = self.detector.detect_face_region(frame)
        if face_rect is None:
            if self.last_face_rect is not None:
                face_rect = self.last_face_rect
            else:
                h, w = frame.shape[:2]
                face_rect = (w // 4, h // 6, w // 2, h * 2 // 3)
        else:
            self.last_face_rect = face_rect

        landmarks = self.detector.estimate_landmarks(face_rect)

        result = frame.copy()

        if params.eye_widen > 0:
            result = apply_eye_warp(result, landmarks["left_eye"],
                                    params.eye_widen, self.tick)
            widen_r = params.eye_widen * (1.0 - params.asymmetry * 0.5)
            result = apply_eye_warp(result, landmarks["right_eye"],
                                    widen_r, self.tick)

        if params.mouth_stretch > 0:
            result = apply_mouth_stretch(result, landmarks["mouth"],
                                         params.mouth_stretch)

        if params.skin_desaturate > 0:
            result = apply_skin_desaturation(result, face_rect,
                                             params.skin_desaturate)

        return result
