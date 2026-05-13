"""Quick mediapipe-only preview: render MP_TOP_ANCHORS on a face image.

Doesn't need torch / transformers — useful for verifying the anchor selection
before running the full pipeline.

Usage:
  py -3 python/preview_anchors.py path/to/image.jpg --out data/preview.png
"""
from __future__ import annotations
import argparse
import os
import sys

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    lmk = FaceLandmarker(static_image_mode=True)
    landmarks = lmk.detect(rgb)
    lmk.close()
    if landmarks is None:
        raise RuntimeError("no face detected")

    out = bgr.copy()

    # 1) Draw all 468 MP landmarks faintly so we can see the existing mesh boundary.
    for i in range(C.N_MP):
        x = int(landmarks[i, 0] * W)
        y = int(landmarks[i, 1] * H)
        cv2.circle(out, (x, y), 1, (180, 180, 180), -1)

    # 2) Highlight MP_TOP_ANCHORS in cyan and label index.
    for k, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        x = int(landmarks[mp_idx, 0] * W)
        y = int(landmarks[mp_idx, 1] * H)
        cv2.circle(out, (x, y), 5, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"{mp_idx}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # 3) Connect anchors left-to-right so we can visually check ordering.
    pts = [(int(landmarks[m, 0] * W), int(landmarks[m, 1] * H)) for m in C.MP_TOP_ANCHORS]
    for i in range(len(pts) - 1):
        cv2.line(out, pts[i], pts[i + 1], (0, 200, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}   (image {W}x{H})")


if __name__ == "__main__":
    main()
