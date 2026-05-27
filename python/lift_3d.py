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
from .hairline_2d import face_up_vector


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
    crown_lift_frac: float | None = None,
) -> np.ndarray:
    """Combine 2D hairline samples with a sagittal-arc Z, after lifting
    the (x, y) along the face-up direction by ``crown_lift_frac × face_h``
    so the ribbon's top row sits at the crown of the head rather than at
    the detected hair-skin boundary.

    landmarks_norm:    (468, 3) MediaPipe normalized landmarks.
    hairline_norm_xy:  (N_ANCHORS, 2) 2D hairline samples in [0,1] image space.
    crown_lift_frac:   how far above the detected hairline (in fractions of
                       face height) the mesh row should sit. ``None`` uses
                       ``C.HAIRLINE_CROWN_LIFT_FRAC``.
    Returns:           (N_ANCHORS, 3) — (x_norm_lifted, y_norm_lifted, z).
    """
    face_h = _face_height(landmarks_norm)
    R = C.HEAD_ARC_RADIUS_FRAC * face_h
    if crown_lift_frac is None:
        crown_lift_frac = C.HAIRLINE_CROWN_LIFT_FRAC
    up = face_up_vector(landmarks_norm)
    lift = up * (crown_lift_frac * face_h)        # 2D offset, face-up direction

    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        z_a = float(landmarks_norm[mp_idx, 2])
        y_a = float(landmarks_norm[mp_idx, 1])
        x_h = float(hairline_norm_xy[i, 0]) + float(lift[0])
        y_h = float(hairline_norm_xy[i, 1]) + float(lift[1])
        dy = y_h - y_a                              # < 0 (上移更多 → dz 更大)
        out[i, 0] = x_h
        out[i, 1] = y_h
        out[i, 2] = z_a + _arc_dz(dy, R)
    return out


def build_middle_row(
    landmarks_norm: np.ndarray,
    hairline_3d: np.ndarray,
    bias: float = 0.5,
) -> np.ndarray:
    """Place the middle row at the arc-length midpoint between each MP
    anchor and its hairline point.

    The old approach used bias=0.5 linear interpolation in XY and then
    independently recomputed Z via the arc formula.  Because dz ∝ dy²,
    the middle row only received 25 % of the hairline Z offset, creating
    a visible dent where the extension met the face mesh.

    New approach: for each anchor→hairline pair, parameterise the
    sagittal circular arc by the angle θ and place the middle vertex at
    θ_mid = bias × θ_hair.  This distributes both the Y displacement
    **and** the Z displacement smoothly along the arc, so the surface
    transitions from the face mesh through middle to hairline without
    any concavity.
    """
    R = C.HEAD_ARC_RADIUS_FRAC * _face_height(landmarks_norm)
    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        a = landmarks_norm[mp_idx]
        h = hairline_3d[i]
        # X: simple linear interpolation (no arc model in the coronal plane)
        out[i, 0] = (1 - bias) * float(a[0]) + bias * float(h[0])
        # Y, Z: interpolate along the circular arc in the sagittal plane.
        # The arc angle for the hairline point:
        #   θ_h = |dy_h| / R   (small-angle: arc-length ≈ R θ)
        # Middle sits at θ_m = bias × θ_h, giving:
        #   dy_m = R sin(θ_m) ≈ R θ_m   for small θ
        #   dz_m = R (1 − cos(θ_m))
        # For the parabolic regime (θ < 0.8 rad ≈ 46°) these simplify to
        # the same formula but we use the full trig for correctness.
        dy_h = float(h[1]) - float(a[1])          # negative (upward)
        sign = -1.0 if dy_h < 0 else 1.0
        theta_h = abs(dy_h) / max(R, 1e-9)
        theta_m = bias * theta_h
        dy_m = sign * R * np.sin(theta_m)          # same sign as dy_h
        dz_m = R * (1.0 - np.cos(theta_m))         # always ≥ 0
        out[i, 1] = float(a[1]) + dy_m
        out[i, 2] = float(a[2]) + dz_m
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
