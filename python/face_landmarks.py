"""MediaPipe FaceMesh wrapper returning 468 3D landmarks in [0,1] x/y space.

Uses the modern mediapipe.tasks.vision API. Requires models/face_landmarker.task
(downloaded once; see README).
"""
from __future__ import annotations
import os
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task",
)


class FaceLandmarker:
    def __init__(self, static_image_mode: bool = True, model_path: str | None = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"face_landmarker.task not found at {path}. "
                "Download it from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )

        running_mode = vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=running_mode,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Returns (468, 3) float32 of normalized x, y and relative z, or None.

        The task model produces 478 landmarks (468 face + 10 iris); we return
        only the first 468 to match the canonical FaceMesh topology used by
        the SDK's OBJ file.
        """
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0][:468]
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return arr

    def close(self):
        try:
            self._detector.close()
        except Exception:
            pass
