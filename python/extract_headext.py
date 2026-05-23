"""Main CLI: image → JSON of 522 3D points (v2-headext) consumable by the SDK.

Pipeline
--------
  1. MediaPipe FaceMesh → 468 normalized landmarks
  2. Face parsing (HF SegFormer) → per-pixel class map
  3. Hairline curve detection (v1-hairline: lateral_extend_dense) → 17 hairline 2D pts
  4. Lift 2D hairline → 3D using anchor Z + curvature offset
  5. Interpolate 17 middle-row 3D pts
  6. Sample 10 lateral outer pts (silhouette ribbon, left 5 + right 5)
  7. Fit head ellipsoid from 468 MP pts → solve Z for lateral mid + out
  8. Concatenate into (522, 3) and emit JSON

Output JSON schema
------------------
  {
    "image": {"width": W, "height": H, "path": "..."},
    "version": "v2-headext",
    "n_total": 522,
    "n_mp": 468,
    "layout": [
      "mp[0..468)",
      "middle[468..485)",
      "hairline[485..502)",
      "lateral_mid[502..512)",
      "lateral_out[512..522)"
    ],
    "points": [[x_norm, y_norm, z_relative], ...],   # length 522
    "valid_hairline": [bool, ...],                   # length 17
    "valid_lateral":  [bool, ...],                   # length 10
    "lateral_in_envelope": [bool, ...],              # length 20 (mid+out)
    "ellipsoid": {"center":[cx,cy,cz], "axes":[a,b,c], "z_front_sign":±1, "residual":...}
  }

Usage
-----
  python -m python.extract_headext path/to/image.jpg
  python -m python.extract_headext path/to/image.jpg --out data/out_v2.json
  python -m python.extract_headext path/to/image.jpg --landmark-backend tasks
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from python.face_parsing import FaceParser
    from python.hairline_2d import (
        sample_hairline_lateral_extend_dense,
        sample_lateral_extension,
        smooth_hairline_corner_aware,
    )
    from python.head_ellipsoid import fit_head_ellipsoid
    from python.lift_3d import (
        assemble_full_v2,
        build_middle_row,
        lift_hairline_to_3d,
        lift_lateral_to_3d,
    )
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker, SolutionsFaceLandmarker
    from .face_parsing import FaceParser
    from .hairline_2d import (
        sample_hairline_lateral_extend_dense,
        sample_lateral_extension,
        smooth_hairline_corner_aware,
    )
    from .head_ellipsoid import fit_head_ellipsoid
    from .lift_3d import (
        assemble_full_v2,
        build_middle_row,
        lift_hairline_to_3d,
        lift_lateral_to_3d,
    )


LANDMARK_BACKENDS = ("subprocess", "tasks", "solutions")
SUBPROCESS_TIMEOUT_S = 120
WSL_FRIENDLY_ENV = {
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    "GALLIUM_DRIVER": "llvmpipe",
    "MEDIAPIPE_DISABLE_GPU": "1",
    "EGL_PLATFORM": "surfaceless",
}
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _detect_via_subprocess(image_path: str) -> np.ndarray:
    """Run MediaPipe FaceLandmarker in an isolated subprocess (WSL-safe)."""
    env = dict(os.environ)
    for key, value in WSL_FRIENDLY_ENV.items():
        env.setdefault(key, value)

    with tempfile.TemporaryDirectory(prefix="head3d_lmk_") as tmp:
        out_path = os.path.join(tmp, "landmarks.npy")
        cmd = [sys.executable, "-m", "python._mediapipe_subprocess", image_path, out_path]
        completed = subprocess.run(
            cmd, cwd=PROJECT_DIR, env=env,
            capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S,
        )
        if completed.returncode == 0 and os.path.isfile(out_path):
            return np.load(out_path)
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(
            f"MediaPipe subprocess failed (exit={completed.returncode})"
            + (f"\n{tail}" if tail else "")
        )


def detect_landmarks(image_path: str, rgb: np.ndarray, backend: str) -> np.ndarray:
    if backend == "subprocess":
        return _detect_via_subprocess(image_path)
    if backend == "tasks":
        lm = FaceLandmarker(static_image_mode=True)
        try:
            return lm.detect(rgb)
        finally:
            lm.close()
    if backend == "solutions":
        lm = SolutionsFaceLandmarker(static_image_mode=True)
        try:
            return lm.detect(rgb)
        finally:
            lm.close()
    raise ValueError(f"unknown landmark backend: {backend}")


def run(
    image_path: str,
    out_path: str | None = None,
    device: str | None = None,
    landmark_backend: str = "subprocess",
    lateral_max_walk_ratio: float = 0.15,
    hairline_intermediates: int = 1,
) -> dict:
    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    t0 = time.perf_counter()
    landmarks = detect_landmarks(image_path, rgb, landmark_backend)
    if landmarks is None:
        raise RuntimeError("no face detected")
    t1 = time.perf_counter()
    print(f"  [time] mediapipe ({landmark_backend}): {(t1 - t0) * 1000:.0f} ms")

    parse_map = FaceParser(device=device).parse(rgb)
    t2 = time.perf_counter()
    print(f"  [time] face parsing: {(t2 - t1) * 1000:.0f} ms")

    # v1-hairline: dense 17 hairline pts + 17 middle row
    hairline_dense, _ = sample_hairline_lateral_extend_dense(
        landmarks, parse_map, intermediates=hairline_intermediates,
    )
    # Lift back to 17 anchor positions (subsample dense → 17 by picking
    # the anchor positions; with intermediates=1 hairline_dense has 33 pts).
    step = hairline_intermediates + 1
    hairline_17 = hairline_dense[::step]
    if hairline_17.shape[0] != C.N_ANCHORS:
        hairline_17 = hairline_dense[:C.N_ANCHORS]  # safety net
    valid_17 = np.ones(C.N_ANCHORS, dtype=bool)
    hairline_smoothed = smooth_hairline_corner_aware(hairline_17.copy(), valid_17, iterations=2)
    hairline_3d = lift_hairline_to_3d(landmarks, hairline_smoothed)
    middle_3d = build_middle_row(landmarks, hairline_3d)
    t3 = time.perf_counter()
    print(f"  [time] hairline + middle: {(t3 - t2) * 1000:.0f} ms")

    # v2-headext: lateral ribbon
    lateral_mid_xy, lateral_out_xy, lateral_valid = sample_lateral_extension(
        landmarks, parse_map, max_walk_ratio=lateral_max_walk_ratio,
    )
    ellipsoid = fit_head_ellipsoid(landmarks)
    lateral_mid_3d, lateral_out_3d, in_envelope = lift_lateral_to_3d(
        landmarks, lateral_out_xy, lateral_mid_xy, ellipsoid,
    )
    t4 = time.perf_counter()
    print(f"  [time] lateral + ellipsoid: {(t4 - t3) * 1000:.0f} ms")
    print(f"  ellipsoid: {ellipsoid}")
    print(
        f"  lateral valid: {int(lateral_valid.sum())}/{lateral_valid.size}, "
        f"in_envelope: {int(in_envelope.sum())}/{in_envelope.size}"
    )

    pts_full = assemble_full_v2(landmarks, middle_3d, hairline_3d, lateral_mid_3d, lateral_out_3d)

    record = {
        "image": {"width": int(W), "height": int(H), "path": image_path},
        "version": "v2-headext",
        "n_total": C.N_TOTAL_V2,
        "n_mp": C.N_MP,
        "layout": [
            f"mp[0..{C.N_MP})",
            f"middle[{C.MIDDLE_START}..{C.HAIRLINE_START})",
            f"hairline[{C.HAIRLINE_START}..{C.N_TOTAL})",
            f"lateral_mid[{C.LATERAL_MID_START}..{C.LATERAL_OUT_START})",
            f"lateral_out[{C.LATERAL_OUT_START}..{C.N_TOTAL_V2})",
        ],
        "points": pts_full.tolist(),
        "valid_hairline": valid_17.tolist(),
        "valid_lateral": lateral_valid.tolist(),
        "lateral_in_envelope": in_envelope.tolist(),
        "ellipsoid": {
            "center": [ellipsoid.cx, ellipsoid.cy, ellipsoid.cz],
            "axes": [ellipsoid.a, ellipsoid.b, ellipsoid.c],
            "z_front_sign": int(ellipsoid.z_front_sign),
            "residual": float(ellipsoid.mean_residual),
        },
    }

    if out_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join("data", base + "_v2.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out_path}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MediaPipe + 522-point v2-headext from one image.")
    ap.add_argument("image", help="path to input image (jpg/png)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--device", default=None, help="torch device (cpu/cuda); auto if omitted")
    ap.add_argument(
        "--landmark-backend",
        choices=list(LANDMARK_BACKENDS),
        default="subprocess",
        help="subprocess (default, WSL-safe) | tasks | solutions",
    )
    ap.add_argument(
        "--lateral-max-walk-ratio", type=float, default=0.15,
        help="lateral ribbon max outward walk as fraction of image width (default 0.15)",
    )
    ap.add_argument(
        "--hairline-intermediates", type=int, default=1,
        help="hairline lateral_extend_dense intermediates (default 1 → 33 pts dense → 17 sampled)",
    )
    args = ap.parse_args()

    run(
        args.image,
        out_path=args.out,
        device=args.device,
        landmark_backend=args.landmark_backend,
        lateral_max_walk_ratio=args.lateral_max_walk_ratio,
        hairline_intermediates=args.hairline_intermediates,
    )


if __name__ == "__main__":
    main()
