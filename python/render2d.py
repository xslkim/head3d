"""Step 6: CPU textured rasterizer (no GPU/EGL, WSL-safe).

Renders the deformed mesh with texture0.png by affine-warping each triangle's
texture patch onto its destination triangle in image space, back-to-front
(painter's algorithm by mean z), then alpha-composites the result over the
uploaded photo.
"""
from __future__ import annotations

import cv2
import numpy as np


def _warp_triangle(
    tex: np.ndarray,          # (TH, TW, 4) RGBA float32
    canvas: np.ndarray,       # (H, W, 3) float32 accumulation (RGB)
    cover: np.ndarray,        # (H, W) float32 alpha already laid down
    t_src: np.ndarray,        # (3, 2) texture pixel coords
    t_dst: np.ndarray,        # (3, 2) image pixel coords
    alpha: float,
) -> None:
    H, W = canvas.shape[:2]
    rs = cv2.boundingRect(t_src.astype(np.float32))
    rd = cv2.boundingRect(t_dst.astype(np.float32))
    # Clamp destination rect to the image.
    x0 = max(rd[0], 0); y0 = max(rd[1], 0)
    x1 = min(rd[0] + rd[2], W); y1 = min(rd[1] + rd[3], H)
    if x1 <= x0 or y1 <= y0 or rs[2] <= 0 or rs[3] <= 0:
        return
    dw, dh = x1 - x0, y1 - y0

    src_off = (t_src - [rs[0], rs[1]]).astype(np.float32)
    dst_off = (t_dst - [x0, y0]).astype(np.float32)
    if cv2.contourArea(dst_off) < 1e-3 or cv2.contourArea(src_off) < 1e-3:
        return

    src_crop = tex[rs[1]:rs[1] + rs[3], rs[0]:rs[0] + rs[2]]
    if src_crop.size == 0:
        return

    M = cv2.getAffineTransform(src_off[:3], dst_off[:3])
    warped = cv2.warpAffine(
        src_crop, M, (dw, dh),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )  # (dh, dw, 4)

    tri_mask = np.zeros((dh, dw), np.float32)
    cv2.fillConvexPoly(tri_mask, dst_off.astype(np.int32), 1.0, cv2.LINE_AA)

    a = tri_mask * (warped[:, :, 3] / 255.0) * alpha          # (dh, dw)
    a = a[:, :, None]
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = warped[:, :, :3] * a + region * (1.0 - a)
    cover[y0:y1, x0:x1] = np.maximum(cover[y0:y1, x0:x1], a[:, :, 0])


def render_overlay(
    photo_rgb: np.ndarray,        # (H, W, 3) uint8
    positions: np.ndarray,        # (V, 3) pixel space
    texcoords: np.ndarray,        # (V, 2) UV in [0,1]
    faces: list,                  # list of [(pi,ti,ni)x3]
    texture_rgba: np.ndarray,     # (TH, TW, 4) uint8
    alpha: float = 1.0,
    face_ids: list[int] | None = None,
) -> np.ndarray:
    """Composite the textured mesh over the photo. Returns (H, W, 3) uint8."""
    H, W = photo_rgb.shape[:2]
    TH, TW = texture_rgba.shape[:2]
    canvas = photo_rgb.astype(np.float32).copy()
    cover = np.zeros((H, W), np.float32)
    tex = texture_rgba.astype(np.float32)

    uv_px = texcoords.copy().astype(np.float64)
    uv_px[:, 0] *= TW
    uv_px[:, 1] = (1.0 - uv_px[:, 1]) * TH    # OBJ v origin bottom-left -> image top-left

    sel = range(len(faces)) if face_ids is None else face_ids
    # Painter's order: far (more +z) first. MediaPipe z<0 is toward camera.
    order = sorted(sel, key=lambda fi: -np.mean(
        [positions[v[0]][2] for v in faces[fi]]
    ))

    for fi in order:
        f = faces[fi]
        t_dst = np.array([positions[v[0]][:2] for v in f], dtype=np.float64)
        t_src = np.array([uv_px[v[1]] for v in f], dtype=np.float64)
        _warp_triangle(tex, canvas, cover, t_src, t_dst, alpha)

    return np.clip(canvas, 0, 255).astype(np.uint8)
