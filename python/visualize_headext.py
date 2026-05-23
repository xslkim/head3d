"""Debug visualizations for the v2-headext pipeline.

Outputs a 3-panel side-by-side image:

  1. silhouette + lateral rays:
       head silhouette mask, with the 10 lateral rays drawn from each MP
       lateral anchor to its detected outer ribbon point.

  2. 522 points + new lateral triangles:
       Original image with all 522 vertices projected back to 2D,
       colored by group (MP / v1 hairline strip / v2 lateral mid+out),
       plus the lateral ribbon triangles drawn in green wireframe.

  3. Z color map:
       Each vertex drawn as a colored dot whose color encodes the
       depth (red = closer to camera, blue = further). Useful to see
       whether the ellipsoid is producing reasonable Z values for
       the new lateral vertices.

Usage
-----
  python -m python.visualize_headext path/to/image.jpg --out data/headext_overlay.png
  python -m python.visualize_headext path/to/image.jpg --out ... --landmark-backend tasks
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_parsing import FaceParser
    from python.hairline_2d import (
        build_head_silhouette_mask,
        sample_hairline_lateral_extend_dense,
        sample_lateral_extension,
        smooth_hairline_corner_aware,
    )
    from python.head_ellipsoid import fit_head_ellipsoid
    from python.lift_3d import (
        build_middle_row,
        lift_hairline_to_3d,
        lift_lateral_to_3d,
        assemble_full_v2,
    )
    from python.extract_headext import detect_landmarks, LANDMARK_BACKENDS
else:
    from . import constants as C
    from .face_parsing import FaceParser
    from .hairline_2d import (
        build_head_silhouette_mask,
        sample_hairline_lateral_extend_dense,
        sample_lateral_extension,
        smooth_hairline_corner_aware,
    )
    from .head_ellipsoid import fit_head_ellipsoid
    from .lift_3d import (
        build_middle_row,
        lift_hairline_to_3d,
        lift_lateral_to_3d,
        assemble_full_v2,
    )
    from .extract_headext import detect_landmarks, LANDMARK_BACKENDS


GROUP_COLOR = {
    "mp":       (200, 200, 200),  # gray
    "v1_mid":   (255, 160,  60),  # orange-ish
    "v1_hair":  (255, 200, 100),  # peach
    "lat_mid":  ( 60, 200, 255),  # cyan-blue
    "lat_out":  ( 60, 130, 255),  # darker blue
}


def _group_of(i: int) -> str:
    if i < C.N_MP:                 return "mp"
    if i < C.HAIRLINE_START:       return "v1_mid"
    if i < C.LATERAL_MID_START:    return "v1_hair"
    if i < C.LATERAL_OUT_START:    return "lat_mid"
    if i < C.N_TOTAL_V2:           return "lat_out"
    return "mp"


def _load_rgb(path: str) -> np.ndarray:
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _to_px(xy: np.ndarray, W: int, H: int) -> tuple[int, int]:
    return int(round(float(xy[0]) * W)), int(round(float(xy[1]) * H))


def render_silhouette_panel(
    rgb: np.ndarray,
    parse_map: np.ndarray,
    landmarks: np.ndarray,
    lateral_mid_xy: np.ndarray,
    lateral_out_xy: np.ndarray,
    lateral_valid: np.ndarray,
) -> np.ndarray:
    import cv2

    H, W = rgb.shape[:2]
    head = build_head_silhouette_mask(parse_map)
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[head] = (40, 200, 90)
    out = ((rgb.astype(np.float32) * 0.7) + (color.astype(np.float32) * 0.3)).astype(np.uint8)

    chain = (
        [(mp, "L", k) for k, mp in enumerate(C.MP_LATERAL_ANCHORS_LEFT)]
        + [(mp, "R", k) for k, mp in enumerate(C.MP_LATERAL_ANCHORS_RIGHT)]
    )

    for i, (mp_idx, side, k) in enumerate(chain):
        ax, ay = _to_px(landmarks[mp_idx, :2], W, H)
        mx, my = _to_px(lateral_mid_xy[i], W, H)
        ox, oy = _to_px(lateral_out_xy[i], W, H)
        line_color = (40, 220, 255) if lateral_valid[i] else (255, 90, 90)
        cv2.line(out, (ax, ay), (ox, oy), line_color, 2, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 3, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (mx, my), 4, GROUP_COLOR["lat_mid"], -1, cv2.LINE_AA)
        cv2.circle(out, (ox, oy), 5, GROUP_COLOR["lat_out"], -1, cv2.LINE_AA)
        cv2.putText(out, f"{side}{k}", (ox + 6, oy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(out, "silhouette + lateral rays", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_points_panel(
    rgb: np.ndarray,
    pts_522: np.ndarray,
    valid_lateral: np.ndarray,
) -> np.ndarray:
    """All 522 vertices + lateral ribbon wireframe overlaid on the image."""
    import cv2

    H, W = rgb.shape[:2]
    out = rgb.copy()
    radius_small = max(2, int(min(W, H) * 0.0025))
    radius_large = max(3, int(min(W, H) * 0.005))

    for i in range(pts_522.shape[0]):
        group = _group_of(i)
        if group == "mp":
            continue  # skip MP for clarity (too dense)
        x, y = _to_px(pts_522[i, :2], W, H)
        cv2.circle(out, (x, y), radius_large, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius_small, GROUP_COLOR[group], -1, cv2.LINE_AA)

    # Lateral ribbon wireframe (left + right)
    def draw_side(mid_start: int, out_start: int, mp_anchors: list[int], landmarks: np.ndarray):
        L = len(mp_anchors)
        for k in range(L - 1):
            a0 = _to_px(landmarks[mp_anchors[k], :2], W, H)
            a1 = _to_px(landmarks[mp_anchors[k + 1], :2], W, H)
            m0 = _to_px(pts_522[mid_start + k, :2], W, H)
            m1 = _to_px(pts_522[mid_start + k + 1, :2], W, H)
            o0 = _to_px(pts_522[out_start + k, :2], W, H)
            o1 = _to_px(pts_522[out_start + k + 1, :2], W, H)
            for (p, q) in [(a0, m0), (a1, m1), (m0, m1), (m0, o0), (m1, o1), (o0, o1),
                           (a0, m1), (m0, o1)]:
                cv2.line(out, p, q, (40, 230, 40), 1, cv2.LINE_AA)

    landmarks = pts_522[:C.N_MP]  # first 468 are MP (unchanged in v2)
    draw_side(C.LATERAL_MID_START, C.LATERAL_OUT_START,
              C.MP_LATERAL_ANCHORS_LEFT, landmarks)
    draw_side(C.LATERAL_MID_START + C.N_LATERAL_PER_SIDE,
              C.LATERAL_OUT_START + C.N_LATERAL_PER_SIDE,
              C.MP_LATERAL_ANCHORS_RIGHT, landmarks)

    cv2.putText(out, "522 verts + lateral ribbon wireframe", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def render_depth_panel(rgb: np.ndarray, pts_522: np.ndarray) -> np.ndarray:
    """Each vertex drawn as a dot colored by Z (red = near camera, blue = far)."""
    import cv2

    H, W = rgb.shape[:2]
    out = (rgb.astype(np.float32) * 0.5 + 60).astype(np.uint8)

    z = pts_522[:, 2]
    z_lo, z_hi = float(z.min()), float(z.max())
    z_span = max(z_hi - z_lo, 1e-6)

    radius = max(2, int(min(W, H) * 0.004))
    for i in range(pts_522.shape[0]):
        x, y = _to_px(pts_522[i, :2], W, H)
        t = (z[i] - z_lo) / z_span        # 0..1
        # red (near, high z) → green → blue (far, low z)
        # Note z_front_sign decides which extreme is "front"; we just visualize
        # the algebraic range so the user can spot outliers.
        r = int(255 * t)
        b = int(255 * (1.0 - t))
        g = int(255 * (1.0 - abs(2 * t - 1)))
        color = (r, g, b)
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)

    bar_w, bar_h = 200, 8
    for i in range(bar_w):
        t = i / max(bar_w - 1, 1)
        r = int(255 * t)
        b = int(255 * (1.0 - t))
        g = int(255 * (1.0 - abs(2 * t - 1)))
        cv2.line(out, (10 + i, H - 30), (10 + i, H - 30 + bar_h), (r, g, b), 1)
    cv2.putText(out, f"z={z_lo:+.3f}", (10, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"z={z_hi:+.3f}", (10 + bar_w - 70, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "Z depth coloring", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def hstack(panels: list[np.ndarray]) -> np.ndarray:
    import cv2

    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h))
        resized.append(p)
    return np.concatenate(resized, axis=1)


def run(
    image_path: str,
    out_path: str,
    device: str | None = None,
    landmark_backend: str = "subprocess",
    lateral_max_walk_ratio: float = 0.15,
    hairline_intermediates: int = 1,
) -> None:
    rgb = _load_rgb(image_path)
    landmarks = detect_landmarks(image_path, rgb, landmark_backend)
    if landmarks is None:
        raise RuntimeError("no face detected")
    parse_map = FaceParser(device=device).parse(rgb)

    hairline_dense, _ = sample_hairline_lateral_extend_dense(
        landmarks, parse_map, intermediates=hairline_intermediates,
    )
    step = hairline_intermediates + 1
    hairline_17 = hairline_dense[::step]
    if hairline_17.shape[0] != C.N_ANCHORS:
        hairline_17 = hairline_dense[:C.N_ANCHORS]
    valid_17 = np.ones(C.N_ANCHORS, dtype=bool)
    hairline_smoothed = smooth_hairline_corner_aware(hairline_17.copy(), valid_17, iterations=2)
    hairline_3d = lift_hairline_to_3d(landmarks, hairline_smoothed)
    middle_3d = build_middle_row(landmarks, hairline_3d)

    lateral_mid_xy, lateral_out_xy, lateral_valid = sample_lateral_extension(
        landmarks, parse_map, max_walk_ratio=lateral_max_walk_ratio,
    )
    ellipsoid = fit_head_ellipsoid(landmarks)
    lateral_mid_3d, lateral_out_3d, _ = lift_lateral_to_3d(
        landmarks, lateral_out_xy, lateral_mid_xy, ellipsoid,
    )

    pts_522 = assemble_full_v2(landmarks, middle_3d, hairline_3d, lateral_mid_3d, lateral_out_3d)

    p1 = render_silhouette_panel(rgb, parse_map, landmarks, lateral_mid_xy, lateral_out_xy, lateral_valid)
    p2 = render_points_panel(rgb, pts_522, lateral_valid)
    p3 = render_depth_panel(rgb, pts_522)
    out = hstack([p1, p2, p3])
    _save_rgb(out_path, out)
    print(f"wrote {out_path}  ({out.shape[1]}x{out.shape[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description="3-panel debug viz for v2-headext.")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--landmark-backend", choices=list(LANDMARK_BACKENDS), default="subprocess")
    ap.add_argument("--lateral-max-walk-ratio", type=float, default=0.15)
    ap.add_argument("--hairline-intermediates", type=int, default=1)
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
