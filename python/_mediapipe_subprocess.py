"""Standalone MediaPipe FaceLandmarker runner used by web_service.py.

Running MediaPipe Tasks in the same process as the Flask dev server can
segfault under WSL (D3D12 EGL backend). Spawning a fresh subprocess per
request keeps the web server alive and lets us inject WSL-friendly env
vars before any mediapipe import.

Usage:
    python -m python._mediapipe_subprocess <image_path> <output_npy>
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from python.face_landmarks import FaceLandmarker  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python -m python._mediapipe_subprocess <image_path> <output_npy>",
            file=sys.stderr,
        )
        return 2

    image_path, output_path = sys.argv[1], sys.argv[2]

    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"could not read image: {image_path}", file=sys.stderr)
        return 3

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    landmarker = FaceLandmarker(static_image_mode=True)
    try:
        landmarks = landmarker.detect(rgb)
    finally:
        landmarker.close()

    if landmarks is None:
        print("no face detected", file=sys.stderr)
        return 4

    np.save(output_path, landmarks.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
