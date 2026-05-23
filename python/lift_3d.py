"""Lift 2D hairline samples to 3D in MediaPipe's normalized space.

Saggital-arc Z model
--------------------
MediaPipe / face.obj use ``-z = front of face`` (nose tip is the most
negative z), ``+z = back of head``. The frontal head surface curves
backward as you move up from the forehead to the crown, so any vertex
**above** an MP top anchor (smaller y in image coords) must have a
**larger** z than that anchor.

For each new hairline / middle vertex (x_h, y_h) attached to MP anchor a
located at (x_a, y_a, z_a), we model the local sagittal cross-section
of the head as a circular arc of radius ``R = HEAD_ARC_RADIUS_FRAC × face_h``
and derive

    z_new = z_a + (y_h - y_a)² / (2 R)

The squared dy term guarantees ``z_new ≥ z_a`` (the surface always
bulges backward as you walk up the head, never forward). x is taken
directly from the 2D hairline detection (we don't project x onto the
arc — the parsing tells us exactly where the hairline sits in x).
"""
from __future__ import annotations
import numpy as np

from . import constants as C


def _face_height(landmarks_norm: np.ndarray) -> float:
    """Range of MP y for the visible face (used for the arc radius)."""
    y = landmarks_norm[:, 1]
    return float(y.max() - y.min())


def _arc_dz(dy: float, R: float) -> float:
    """Backward (positive) z offset along a circular arc of radius R for
    a vertical displacement dy. Always non-negative."""
    return (dy * dy) / (2.0 * max(R, 1e-6))


def lift_hairline_to_3d(
    landmarks_norm: np.ndarray,
    hairline_norm_xy: np.ndarray,
) -> np.ndarray:
    """Combine 2D hairline points with a Z that follows the head's
    sagittal curvature.

    landmarks_norm:    (468, 3) MediaPipe normalized landmarks.
    hairline_norm_xy:  (N_ANCHORS, 2) 2D hairline samples in [0,1] image space.
    Returns:           (N_ANCHORS, 3) — (x_norm, y_norm, z_relative).
    """
    R = C.HEAD_ARC_RADIUS_FRAC * _face_height(landmarks_norm)
    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        z_a = float(landmarks_norm[mp_idx, 2])
        y_a = float(landmarks_norm[mp_idx, 1])
        x_h = float(hairline_norm_xy[i, 0])
        y_h = float(hairline_norm_xy[i, 1])
        dy = y_h - y_a
        out[i, 0] = x_h
        out[i, 1] = y_h
        out[i, 2] = z_a + _arc_dz(dy, R)
    return out


def build_middle_row(
    landmarks_norm: np.ndarray,
    hairline_3d: np.ndarray,
    bias: float = 0.5,
) -> np.ndarray:
    """Interpolate (x, y) between each MP anchor and its hairline point,
    then re-solve z on the same circular-arc model so the middle vertex
    lies on the head surface (not floating in front of it).
    """
    R = C.HEAD_ARC_RADIUS_FRAC * _face_height(landmarks_norm)
    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        a = landmarks_norm[mp_idx]
        h = hairline_3d[i]
        x_m = (1 - bias) * float(a[0]) + bias * float(h[0])
        y_m = (1 - bias) * float(a[1]) + bias * float(h[1])
        dy = y_m - float(a[1])
        out[i, 0] = x_m
        out[i, 1] = y_m
        out[i, 2] = float(a[2]) + _arc_dz(dy, R)
    return out


def assemble_full(landmarks_norm: np.ndarray,
                  middle_3d: np.ndarray,
                  hairline_3d: np.ndarray) -> np.ndarray:
    """Concatenate to a (N_TOTAL, 3) array in the SDK's expected order:
    [0..468) MediaPipe / [468..485) middle / [485..502) hairline."""
    assert landmarks_norm.shape == (C.N_MP, 3)
    assert middle_3d.shape   == (C.N_ANCHORS, 3)
    assert hairline_3d.shape == (C.N_ANCHORS, 3)
    return np.concatenate([landmarks_norm, middle_3d, hairline_3d], axis=0).astype(np.float32)
