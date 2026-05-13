"""Render the final extension mesh on top of the input image.

Reads a JSON produced by extract_hairline.py and draws:
  - Original MediaPipe boundary (cyan) — the OLD top of the face mesh
  - Middle row (orange)
  - Hairline row (yellow) with index labels
  - Triangle wireframe of the extension (faint white)

Use this to visually verify the JSON before piping into the SDK.

Usage:
  py -3 python/show_result.py data/test_face.png data/test_face.json --out data/test_face_result.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python._index_map_data import INDEX_MAP_468
    from python.build_extended_obj import build_inverse_index_map, build_extension_faces
else:
    from . import constants as C
    from ._index_map_data import INDEX_MAP_468
    from .build_extended_obj import build_inverse_index_map, build_extension_faces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    H, W = bgr.shape[:2]

    with open(args.json, "r", encoding="utf-8") as f:
        d = json.load(f)
    pts = np.array(d["points"], dtype=np.float32)  # (502, 3)
    assert pts.shape == (C.N_TOTAL, 3), pts.shape

    out = bgr.copy()

    # Helper to convert normalized to pixels
    def px(p):
        return (int(p[0] * W), int(p[1] * H))

    # 1) Triangle wireframe of the extension
    inv = build_inverse_index_map()
    # Build verts indexed by OBJ index for face lookup
    obj_verts = np.zeros((C.N_TOTAL, 2), dtype=np.float32)
    for obj_idx, mp_idx in enumerate(INDEX_MAP_468):
        obj_verts[obj_idx] = pts[mp_idx, :2]
    for k in range(C.N_EXT):
        obj_verts[C.N_MP + k] = pts[C.N_MP + k, :2]
    for tri in build_extension_faces(inv):
        idxs = [tri[k][0] for k in range(3)]
        for k in range(3):
            a = px(obj_verts[idxs[k]])
            b = px(obj_verts[idxs[(k + 1) % 3]])
            cv2.line(out, a, b, (240, 240, 240), 1, cv2.LINE_AA)

    # 2) Original MediaPipe top anchors (cyan)
    for mp_idx in C.MP_TOP_ANCHORS:
        cv2.circle(out, px(pts[mp_idx]), 4, (255, 200, 0), -1, cv2.LINE_AA)

    # 3) Middle row (orange)
    for i in range(C.N_ANCHORS):
        cv2.circle(out, px(pts[C.MIDDLE_START + i]), 3, (0, 150, 255), -1, cv2.LINE_AA)

    # 4) Hairline (yellow), with anchor index labels
    for i in range(C.N_ANCHORS):
        p = px(pts[C.HAIRLINE_START + i])
        cv2.circle(out, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p, 7, (0, 0, 0), 1, cv2.LINE_AA)
        if i % 2 == 0:
            cv2.putText(out, str(i), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # 5) Polyline connecting hairline points
    for i in range(C.N_ANCHORS - 1):
        cv2.line(out, px(pts[C.HAIRLINE_START + i]),
                      px(pts[C.HAIRLINE_START + i + 1]),
                      (0, 255, 255), 1, cv2.LINE_AA)

    # 6) Polyline connecting MP top anchors (original boundary)
    for i in range(C.N_ANCHORS - 1):
        cv2.line(out, px(pts[C.MP_TOP_ANCHORS[i]]),
                      px(pts[C.MP_TOP_ANCHORS[i + 1]]),
                      (255, 200, 0), 1, cv2.LINE_AA)

    # Legend
    legend = [
        ("cyan  (orig MP boundary)", (255, 200, 0)),
        ("orange (middle row)",      (0, 150, 255)),
        ("yellow (hairline)",        (0, 255, 255)),
    ]
    for i, (txt, color) in enumerate(legend):
        y = 12 + i * 16
        cv2.rectangle(out, (8, y), (28, y + 12), color, -1)
        cv2.putText(out, txt, (32, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, txt, (32, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
