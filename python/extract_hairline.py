"""Main CLI: image -> JSON of 502 3D points consumable by the SDK.

Pipeline:
  1. MediaPipe FaceMesh -> 468 normalized landmarks
  2. Face parsing (HF SegFormer) -> per-pixel class map
  3. Hairline curve detection + per-anchor ray casting -> 17 hairline 2D points
  4. Lift 2D hairline points to 3D using anchor Z + curvature offset
  5. Interpolate 17 middle-row 3D points
  6. Concatenate into (502, 3) and emit JSON

Output JSON schema:
  {
    "image": {"width": W, "height": H, "path": "..."},
    "n_total": 502,
    "n_mp": 468,
    "n_extension": 34,
    "layout": ["mp[0..468)", "middle[468..485)", "hairline[485..502)"],
    "points": [[x_norm, y_norm, z_relative], ...]   # length 502
    "valid_hairline": [true/false, ...]              # length 17
  }

Usage:
  py -3 python/extract_hairline.py path/to/image.jpg
  py -3 python/extract_hairline.py path/to/image.jpg --out data/out.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker
    from python.face_parsing  import FaceParser
    from python.hairline_2d   import sample_hairline, smooth_hairline
    from python.lift_3d       import lift_hairline_to_3d, build_middle_row, assemble_full
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker
    from .face_parsing  import FaceParser
    from .hairline_2d   import sample_hairline, smooth_hairline
    from .lift_3d       import lift_hairline_to_3d, build_middle_row, assemble_full


def run(image_path: str, out_path: str | None = None, device: str | None = None) -> dict:
    import cv2
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    t0 = time.perf_counter()
    lmk = FaceLandmarker(static_image_mode=True)
    landmarks = lmk.detect(rgb)
    lmk.close()
    if landmarks is None:
        raise RuntimeError("no face detected")
    t1 = time.perf_counter()
    print(f"  [time] mediapipe: {(t1 - t0)*1000:.0f} ms")

    parser = FaceParser(device=device)
    parse_map = parser.parse(rgb)
    t2 = time.perf_counter()
    print(f"  [time] parsing:   {(t2 - t1)*1000:.0f} ms")

    hairline_2d, valid = sample_hairline(landmarks, parse_map)
    hairline_2d = smooth_hairline(hairline_2d, valid)
    hairline_3d = lift_hairline_to_3d(landmarks, hairline_2d)
    middle_3d   = build_middle_row(landmarks, hairline_3d)
    points_full = assemble_full(landmarks, middle_3d, hairline_3d)
    t3 = time.perf_counter()
    print(f"  [time] hairline:  {(t3 - t2)*1000:.0f} ms")

    record = {
        "image": {"width": int(W), "height": int(H), "path": image_path},
        "n_total": C.N_TOTAL,
        "n_mp": C.N_MP,
        "n_extension": C.N_EXT,
        "layout": [
            f"mp[0..{C.N_MP})",
            f"middle[{C.MIDDLE_START}..{C.HAIRLINE_START})",
            f"hairline[{C.HAIRLINE_START}..{C.N_TOTAL})",
        ],
        "points": points_full.tolist(),
        "valid_hairline": valid.tolist(),
    }

    if out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join("data", base + ".json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out_path}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MediaPipe + hairline points from one image.")
    ap.add_argument("image", help="path to input image (jpg/png)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda); auto if omitted")
    args = ap.parse_args()
    run(args.image, args.out, args.device)


if __name__ == "__main__":
    main()
