"""Lift 2D hairline samples to 3D in MediaPipe's normalized coordinate system."""
from __future__ import annotations
import numpy as np

from . import constants as C


def lift_hairline_to_3d(
    landmarks_norm: np.ndarray,
    hairline_norm_xy: np.ndarray,
) -> np.ndarray:
    """Combine 2D hairline points with inherited Z from neighbor anchors.

    landmarks_norm:    (468, 3) MediaPipe normalized landmarks.
    hairline_norm_xy:  (N_ANCHORS, 2) 2D hairline samples in [0,1] image space.
    Returns:           (N_ANCHORS, 3) — (x_norm, y_norm, z_relative).
    """
    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        z_anchor = float(landmarks_norm[mp_idx, 2])
        out[i, 0] = hairline_norm_xy[i, 0]
        out[i, 1] = hairline_norm_xy[i, 1]
        out[i, 2] = z_anchor + C.curve_offset_z(i)
    return out


def build_middle_row(
    landmarks_norm: np.ndarray,
    hairline_3d: np.ndarray,
    bias: float = 0.5,
) -> np.ndarray:
    """Linear interp between each MP anchor and its hairline point + a forward bulge."""
    out = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        a = landmarks_norm[mp_idx]
        h = hairline_3d[i]
        out[i] = (1 - bias) * a + bias * h
        out[i, 2] += C.bulge_z(i)
    return out


def assemble_full(landmarks_norm: np.ndarray,
                  middle_3d: np.ndarray,
                  hairline_3d: np.ndarray) -> np.ndarray:
    """Concatenate to a (N_TOTAL, 3) array in the SDK's expected order."""
    assert landmarks_norm.shape == (C.N_MP, 3)
    assert middle_3d.shape   == (C.N_ANCHORS, 3)
    assert hairline_3d.shape == (C.N_ANCHORS, 3)
    return np.concatenate([landmarks_norm, middle_3d, hairline_3d], axis=0).astype(np.float32)
