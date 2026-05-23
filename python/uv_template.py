"""Generate a UV-layout reference image for an .obj mesh.

Use this to author or upgrade effect textures (e.g. ``imgs/texture0.png``)
that need to stay compatible with both ``face.obj`` and the extended
meshes (``face_ext.obj`` and future ``face_ext_v2.obj``).

The output PNG is the same resolution as the target texture and shows:

* a translucent background — by default white, or the existing texture
  passed via ``--underlay`` so you can see how your current content
  aligns with the UV partitions;
* mesh wireframe in UV space, colored by vertex group:

    - MP 468 main face   → 浅灰   (vertex idx 0..467)
    - v1 middle row 17   → 蓝色   (idx 468..484)
    - v1 hairline 17     → 青色   (idx 485..501)
    - v2 lateral mid/out → 橙色   (idx 502..)   [reserved]

* labeled rectangles for each vertex zone with its image-y range and
  OBJ V_raw range, so you know where to paint new effect content;
* a faint grid + axis labels.

Examples
--------

    # Generate a template for the current extended mesh
    python python/uv_template.py face_ext.obj --out imgs/uv_template.png

    # Overlay on top of your existing texture for alignment check
    python python/uv_template.py face_ext.obj \
        --underlay imgs/texture0.png \
        --out imgs/uv_template_aligned.png

    # Higher resolution template for retouching detailed makeup textures
    python python/uv_template.py face_ext.obj --size 1024 \
        --out imgs/uv_template_1024.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# (start_idx, end_idx, label, BGR color)
DEFAULT_ZONES: tuple[tuple[int, int, str, tuple[int, int, int]], ...] = (
    (0,   468, "MP 468 main face",   (200, 200, 200)),
    (468, 485, "v1 middle row (17)", (255, 160,  60)),
    (485, 502, "v1 hairline (17)",   (255, 200, 100)),
    (502, 10_000, "v2 lateral (reserved)", (100, 165, 255)),
)


def _zone_of(idx: int, zones=DEFAULT_ZONES) -> tuple[str, tuple[int, int, int]] | None:
    for a, b, lbl, c in zones:
        if a <= idx < b:
            return lbl, c
    return None


def read_obj_uv_and_uvfaces(path: str) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Return (uvs[N,2] raw, uv_face_triplets[F, 3] 0-based)."""
    uvs: list[tuple[float, float]] = []
    uv_faces: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith("vt "):
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                p = line.split()[1:]
                vti = []
                for v in p:
                    parts = v.split("/")
                    if len(parts) < 2 or not parts[1]:
                        vti.append(-1)
                    else:
                        vti.append(int(parts[1]) - 1)
                if len(vti) == 3 and all(i >= 0 for i in vti):
                    uv_faces.append(tuple(vti))
    return np.asarray(uvs, dtype=np.float64), uv_faces


def uv_to_px(uv_raw: np.ndarray, size: int) -> np.ndarray:
    """OBJ raw UV (u, v_raw) → image pixel (x, y) with 1-v flip."""
    px = np.empty_like(uv_raw)
    px[:, 0] = uv_raw[:, 0] * size
    px[:, 1] = (1.0 - uv_raw[:, 1]) * size
    return np.clip(px, 0, size - 1).astype(np.int32)


def make_canvas(size: int, underlay_path: str | None) -> np.ndarray:
    import cv2

    if underlay_path:
        bgr = cv2.imread(underlay_path, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise FileNotFoundError(underlay_path)
        if bgr.shape[0] != size or bgr.shape[1] != size:
            bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        if bgr.shape[2] == 4:
            alpha = bgr[..., 3:4].astype(np.float32) / 255.0
            rgb = bgr[..., :3].astype(np.float32)
            white = np.full_like(rgb, 235.0)
            bgr = (alpha * rgb + (1.0 - alpha) * white).astype(np.uint8)
        return bgr
    return np.full((size, size, 3), 240, dtype=np.uint8)


def draw_grid(img: np.ndarray, step_frac: float = 0.1) -> None:
    import cv2

    h, w = img.shape[:2]
    step = max(1, int(round(w * step_frac)))
    for x in range(0, w + 1, step):
        cv2.line(img, (x, 0), (x, h - 1), (220, 220, 220), 1, cv2.LINE_AA)
    for y in range(0, h + 1, step):
        cv2.line(img, (0, y), (w - 1, y), (220, 220, 220), 1, cv2.LINE_AA)


def draw_wireframe(
    img: np.ndarray,
    uvs_raw: np.ndarray,
    uv_faces: Iterable[tuple[int, int, int]],
    zones=DEFAULT_ZONES,
    thickness: int = 1,
) -> None:
    import cv2

    px = uv_to_px(uvs_raw, img.shape[0])
    for tri in uv_faces:
        zone_ids = [_zone_of(i, zones) for i in tri]
        zone_ids = [z for z in zone_ids if z]
        if not zone_ids:
            continue
        zone_ids.sort(key=lambda z: z[0])
        color = zone_ids[-1][1]
        p0, p1, p2 = px[tri[0]], px[tri[1]], px[tri[2]]
        cv2.line(img, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)
        cv2.line(img, tuple(p2), tuple(p0), color, thickness, cv2.LINE_AA)


def draw_zone_legend(
    img: np.ndarray,
    uvs_raw: np.ndarray,
    zones=DEFAULT_ZONES,
) -> None:
    import cv2

    size = img.shape[0]
    legend_x = 10
    legend_y = 20
    line_h = max(14, size // 36)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, size / 1400.0)

    for a, b, label, color in zones:
        slice_ = uvs_raw[a: min(b, len(uvs_raw))]
        if len(slice_) == 0:
            continue
        v_lo, v_hi = float(slice_[:, 1].min()), float(slice_[:, 1].max())
        y_lo, y_hi = (1.0 - v_hi) * size, (1.0 - v_lo) * size
        text = f"{label}  V_raw={v_lo:.3f}..{v_hi:.3f}  img y={int(y_lo)}..{int(y_hi)}"
        cv2.rectangle(img, (legend_x, legend_y - 10), (legend_x + 16, legend_y + 4), color, -1)
        cv2.putText(img, text, (legend_x + 22, legend_y + 4), font, scale, (40, 40, 40), 1, cv2.LINE_AA)
        legend_y += line_h


def draw_zone_brackets(
    img: np.ndarray,
    uvs_raw: np.ndarray,
    zones=DEFAULT_ZONES,
) -> None:
    """Right-side colored brackets indicating image-y range of each zone."""
    import cv2

    size = img.shape[0]
    bracket_x = size - 14
    for a, b, label, color in zones:
        slice_ = uvs_raw[a: min(b, len(uvs_raw))]
        if len(slice_) == 0:
            continue
        v_lo, v_hi = float(slice_[:, 1].min()), float(slice_[:, 1].max())
        y_top = int((1.0 - v_hi) * size)
        y_bot = int((1.0 - v_lo) * size)
        y_top = max(0, min(size - 1, y_top))
        y_bot = max(0, min(size - 1, y_bot))
        if y_top == y_bot:
            y_top = max(0, y_top - 2)
            y_bot = min(size - 1, y_bot + 2)
        cv2.line(img, (bracket_x, y_top), (bracket_x, y_bot), color, 4, cv2.LINE_AA)


def draw_axes(img: np.ndarray) -> None:
    """Compass-style 'U →' / 'V ↑' labels in OBJ raw coords."""
    import cv2

    size = img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, size / 1400.0)
    cv2.putText(img, "U=0", (4, size - 6), font, scale, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "U=1", (size - 40, size - 6), font, scale, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "V_raw=1 (img top)", (4, 14), font, scale, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "V_raw=0 (img bot)", (4, size - 22), font, scale, (60, 60, 60), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a UV-layout reference / painting template for an OBJ mesh."
    )
    parser.add_argument("obj", help="Input .obj path, e.g. face_ext.obj")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--size", type=int, default=512, help="Output image size in px (default 512)")
    parser.add_argument(
        "--underlay",
        default=None,
        help="Optional existing texture (PNG) to draw underneath as light reference",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="Wireframe line thickness in px (default 1)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the light background grid",
    )
    args = parser.parse_args()

    import cv2  # imported here so --help works without cv2 installed

    uvs_raw, uv_faces = read_obj_uv_and_uvfaces(args.obj)
    if len(uvs_raw) == 0:
        raise SystemExit(f"{args.obj}: no UV (vt) entries found")
    print(f"loaded {args.obj}: {len(uvs_raw)} UVs, {len(uv_faces)} triangles")

    img = make_canvas(args.size, args.underlay)
    if not args.no_grid:
        draw_grid(img)
    draw_zone_brackets(img, uvs_raw)
    draw_wireframe(img, uvs_raw, uv_faces, thickness=args.thickness)
    draw_zone_legend(img, uvs_raw)
    draw_axes(img)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
