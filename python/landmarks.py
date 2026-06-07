"""Step 5a: MediaPipe FaceMesh detection -> 468 landmarks.

Uses the Tasks API with models/face_landmarker.task. The model emits 478
points (468 face + 10 iris); we keep the first 468 to match the canonical
topology of face.obj / INDEX_MAP_468.

Coordinates are returned normalized: x, y in [0, 1] (image space, y down),
z is the relative depth MediaPipe reports (negative = toward camera), in the
same units as x. Use ``to_pixels`` to scale into image pixels.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task",
)

N_MP = 468


class FaceLandmarker:
    def __init__(self, model_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"face_landmarker.task not found at {path}")

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Return (468, 3) float32 normalized (x, y, z), or None if no face."""
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb)
        )
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0][:N_MP]
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass


def to_pixels(norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale normalized (x, y, z) landmarks into pixel space.

    x -> x*W, y -> y*H, z -> z*W (z shares x's scale per MediaPipe convention).
    """
    out = norm.astype(np.float64).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    out[:, 2] *= width
    return out


def detect_file(path: str) -> tuple[np.ndarray | None, tuple[int, int]]:
    """Convenience: detect from an image file. Returns (norm_landmarks, (W,H))."""
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    det = FaceLandmarker()
    try:
        return det.detect(rgb), (w, h)
    finally:
        det.close()
