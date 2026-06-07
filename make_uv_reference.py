#!/usr/bin/env python3
"""Render a UV reference image for face_ext.obj.

Produces a pixel-exact UV template the artist can paint over to design the
render texture: the base face mesh in light grey, the 27-point crown extension
(50 new faces) highlighted, mapped with the same convention as the renderer
(u -> u*W, v -> (1-v)*H), so it lines up with texture0.png.

    python make_uv_reference.py [-o imgs/uv_reference.png] [--size 1024]
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from python.obj_io import read_obj

ROOT = os.path.dirname(os.path.abspath(__file__))
N_BASE_FACES = 852          # canonical face.obj triangle count


def uv_to_px(uv: np.ndarray, w: int, h: int) -> np.ndarray:
    p = uv.copy()
    p[:, 0] *= w
    p[:, 1] = (1.0 - p[:, 1]) * h
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="face_ext.obj")
    ap.add_argument("-o", "--output", default="imgs/uv_reference.png")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--texture", default="imgs/texture0.png",
                    help="faint background reference (set '' to skip)")
    args = ap.parse_args()

    mesh = read_obj(os.path.join(ROOT, args.input))
    W = H = args.size
    uv = np.array(mesh.texcoords, dtype=np.float64)[:, :2]
    px = uv_to_px(uv, W, H)

    img = np.full((H, W, 3), 255, np.uint8)
    if args.texture and os.path.isfile(os.path.join(ROOT, args.texture)):
        tex = cv2.imread(os.path.join(ROOT, args.texture), cv2.IMREAD_COLOR)
        tex = cv2.resize(tex, (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.addWeighted(img, 0.60, tex, 0.40, 0)   # faint texture backdrop

    faces = mesh.faces
    base_col = (120, 120, 120)      # BGR medium grey
    ext_edge = (40, 150, 40)        # green edges for the extension
    ext_fill = (180, 240, 180)      # light green fill

    # 1) base face wireframe
    for f in faces[:N_BASE_FACES]:
        pts = np.array([px[v[1]] for v in f], np.int32)   # v[1] = uv index
        cv2.polylines(img, [pts], True, base_col, 1, cv2.LINE_AA)

    # 2) extension faces: translucent fill + bold edges
    overlay = img.copy()
    for f in faces[N_BASE_FACES:]:
        pts = np.array([px[v[1]] for v in f], np.int32)
        cv2.fillConvexPoly(overlay, pts, ext_fill, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    for f in faces[N_BASE_FACES:]:
        pts = np.array([px[v[1]] for v in f], np.int32)
        cv2.polylines(img, [pts], True, ext_edge, 2, cv2.LINE_AA)

    # 3) mark + number the 27 new vertices (495 - 27 = 468)
    for k in range(468, len(uv)):
        x, y = int(px[k, 0]), int(px[k, 1])
        cv2.circle(img, (x, y), 4, (0, 0, 220), -1, cv2.LINE_AA)

    cv2.putText(img, "face  (UV 0-467)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    cv2.putText(img, "crown extension  (UV 468-494, 50 faces)", (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 130, 40), 2, cv2.LINE_AA)

    out = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, img)
    print(f"wrote {out}  ({W}x{H})  faces={len(faces)} verts={len(uv)}")


if __name__ == "__main__":
    main()
