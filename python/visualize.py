"""Debug visualizations for the hairline pipeline.

Subcommands:
  parse:     overlay the parsed hair / skin mask
  anchors:   draw MP_TOP_ANCHORS + detected hairline points
  mesh:      draw the full 502-point projection (wireframe of extension strip)
  all:       run all three and save side-by-side panels

Usage:
  py -3 python/visualize.py parse  path/to/image.jpg --out data/parse.png
  py -3 python/visualize.py anchors path/to/image.jpg --out data/anchors.png
  py -3 python/visualize.py mesh    path/to/image.jpg --out data/mesh.png
  py -3 python/visualize.py all     path/to/image.jpg --out data/all.png
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.face_landmarks import FaceLandmarker
    from python.face_parsing  import FaceParser
    from python.hairline_2d   import sample_hairline, smooth_hairline, build_hairline_mask
    from python.lift_3d       import lift_hairline_to_3d, build_middle_row
    from python._index_map_data import INDEX_MAP_468
    from python.build_extended_obj import build_inverse_index_map, build_extension_faces
else:
    from . import constants as C
    from .face_landmarks import FaceLandmarker
    from .face_parsing  import FaceParser
    from .hairline_2d   import sample_hairline, smooth_hairline, build_hairline_mask
    from .lift_3d       import lift_hairline_to_3d, build_middle_row
    from ._index_map_data import INDEX_MAP_468
    from .build_extended_obj import build_inverse_index_map, build_extension_faces


def load_rgb(path: str) -> np.ndarray:
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def palette_color(idx: int) -> tuple[int, int, int]:
    table = [
        (0, 0, 0), (220, 180, 150), (230, 120, 100), (60, 200, 200),
        (60, 100, 200), (60, 100, 200), (200, 100, 60), (200, 100, 60),
        (200, 200, 60), (200, 200, 60), (200, 60, 200), (240, 80, 80),
        (240, 60, 60), (90, 70, 50), (180, 180, 200), (240, 200, 50),
        (140, 140, 220), (160, 160, 230), (60, 60, 60),
    ]
    return table[idx] if idx < len(table) else (255, 255, 255)


def overlay_parse(rgb: np.ndarray, parse_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    H, W = rgb.shape[:2]
    color_map = np.zeros_like(rgb)
    for cls in np.unique(parse_map):
        color_map[parse_map == cls] = palette_color(int(cls))
    blended = (rgb.astype(np.float32) * (1 - alpha) + color_map.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_anchors(rgb: np.ndarray,
                 landmarks: np.ndarray,
                 hairline_2d: np.ndarray,
                 valid: np.ndarray) -> np.ndarray:
    import cv2
    out = rgb.copy()
    H, W = out.shape[:2]
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        ax = int(landmarks[mp_idx, 0] * W)
        ay = int(landmarks[mp_idx, 1] * H)
        hx = int(hairline_2d[i, 0] * W)
        hy = int(hairline_2d[i, 1] * H)
        color = (0, 200, 0) if valid[i] else (220, 80, 80)
        cv2.line(out, (ax, ay), (hx, hy), color, 1, cv2.LINE_AA)
        cv2.circle(out, (ax, ay), 3, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (hx, hy), 4, color, -1, cv2.LINE_AA)
    # connect hairline points
    pts = (hairline_2d * np.array([W, H])).astype(np.int32)
    for i in range(C.N_ANCHORS - 1):
        cv2.line(out, tuple(pts[i]), tuple(pts[i + 1]), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_mesh(rgb: np.ndarray,
              landmarks: np.ndarray,
              middle_3d: np.ndarray,
              hairline_3d: np.ndarray) -> np.ndarray:
    """Project the (502, 3) extended mesh onto image space and draw wireframe of the extension."""
    import cv2
    out = rgb.copy()
    H, W = out.shape[:2]
    inv = build_inverse_index_map()

    # Build vertex array length 502 in OBJ-index order. Original 468 verts
    # come from MediaPipe at their inverse-mapped slots; extension verts
    # come from middle_3d and hairline_3d (identity slots 468..501).
    verts = np.zeros((C.N_TOTAL, 3), dtype=np.float32)
    for obj_idx, mp_idx in enumerate(INDEX_MAP_468):
        verts[obj_idx] = landmarks[mp_idx]
    for i in range(C.N_ANCHORS):
        verts[C.MIDDLE_START + i]   = middle_3d[i]
        verts[C.HAIRLINE_START + i] = hairline_3d[i]

    faces = build_extension_faces(inv)
    # Each face entry is [(v,u,n)*3]; we only need v indices
    for tri in faces:
        idxs = [tri[k][0] for k in range(3)]
        pts2d = np.array([verts[i, :2] for i in idxs])
        pts_px = (pts2d * np.array([W, H])).astype(np.int32)
        for k in range(3):
            a = tuple(pts_px[k])
            b = tuple(pts_px[(k + 1) % 3])
            cv2.line(out, a, b, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    panels_p = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            new_w = int(p.shape[1] * scale)
            import cv2
            p = cv2.resize(p, (new_w, h))
        panels_p.append(p)
    return np.concatenate(panels_p, axis=1)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def cmd_parse(args):
    rgb = load_rgb(args.image)
    parser = FaceParser(device=args.device)
    parse_map = parser.parse(rgb)
    out = overlay_parse(rgb, parse_map)
    save_rgb(args.out, out)
    print(f"wrote {args.out}")


def _shared_run(args):
    rgb = load_rgb(args.image)
    lmk = FaceLandmarker(static_image_mode=True)
    landmarks = lmk.detect(rgb)
    lmk.close()
    if landmarks is None:
        raise RuntimeError("no face detected")
    parser = FaceParser(device=args.device)
    parse_map = parser.parse(rgb)
    hairline_2d, valid = sample_hairline(landmarks, parse_map)
    hairline_2d = smooth_hairline(hairline_2d, valid)
    hairline_3d = lift_hairline_to_3d(landmarks, hairline_2d)
    middle_3d   = build_middle_row(landmarks, hairline_3d)
    return rgb, parse_map, landmarks, hairline_2d, valid, middle_3d, hairline_3d


def cmd_anchors(args):
    rgb, parse_map, landmarks, hairline_2d, valid, *_ = _shared_run(args)
    out = draw_anchors(rgb, landmarks, hairline_2d, valid)
    save_rgb(args.out, out)
    print(f"wrote {args.out}")


def cmd_mesh(args):
    rgb, parse_map, landmarks, hairline_2d, valid, middle_3d, hairline_3d = _shared_run(args)
    out = draw_mesh(rgb, landmarks, middle_3d, hairline_3d)
    save_rgb(args.out, out)
    print(f"wrote {args.out}")


def cmd_all(args):
    rgb, parse_map, landmarks, hairline_2d, valid, middle_3d, hairline_3d = _shared_run(args)
    p_parse   = overlay_parse(rgb, parse_map)
    p_anchors = draw_anchors(rgb, landmarks, hairline_2d, valid)
    p_mesh    = draw_mesh(rgb, landmarks, middle_3d, hairline_3d)
    out = hstack_panels([p_parse, p_anchors, p_mesh])
    save_rgb(args.out, out)
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hairline pipeline visualizer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("parse", cmd_parse), ("anchors", cmd_anchors), ("mesh", cmd_mesh), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.add_argument("image")
        sp.add_argument("--out", required=True)
        sp.add_argument("--device", default=None)
        sp.set_defaults(_fn=fn)
    args = ap.parse_args()
    args._fn(args)


if __name__ == "__main__":
    main()
