"""Lift 2D hairline / lateral samples to 3D in MediaPipe's normalized space."""
from __future__ import annotations
import numpy as np

from . import constants as C
from .head_ellipsoid import HeadEllipsoid, solve_z_on_ellipsoid


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


# ---------------------------------------------------------------------------
# v2-headext: lateral extension lifting
# ---------------------------------------------------------------------------

def _nearest_lateral_mp_anchor_z(
    landmarks_norm: np.ndarray,
    xy: np.ndarray,
    anchors: list[int],
) -> float:
    """Inherit Z from the closest (in 2D) lateral MP anchor as a fallback."""
    pts = landmarks_norm[anchors, :2]
    d = np.linalg.norm(pts - xy[None, :], axis=1)
    nearest = int(np.argmin(d))
    return float(landmarks_norm[anchors[nearest], 2])


def lift_lateral_to_3d(
    landmarks_norm: np.ndarray,
    lateral_out_xy: np.ndarray,   # (2L, 2) from sample_lateral_extension
    lateral_mid_xy: np.ndarray,   # (2L, 2)
    ellipsoid: HeadEllipsoid,
    axes_clip_ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign a Z to every lateral mid/out vertex via ellipsoid intersection.

    Returns
    -------
    mid_3d:           (2L, 3) — (x_norm, y_norm, z_solved)
    out_3d:           (2L, 3)
    in_envelope_mask: (4L,)   — True where ellipsoid root was real (mid then out)
    """
    assert lateral_out_xy.shape == lateral_mid_xy.shape
    n = lateral_out_xy.shape[0]
    L_half = n // 2

    all_anchors = C.MP_LATERAL_ANCHORS_LEFT + C.MP_LATERAL_ANCHORS_RIGHT
    if len(all_anchors) != n:
        # Caller might pass a custom anchor list; fall back to MP anchors as
        # the inheritance set anyway.
        all_anchors = (C.MP_LATERAL_ANCHORS_LEFT + C.MP_LATERAL_ANCHORS_RIGHT)

    in_envelope = np.zeros(2 * n, dtype=bool)
    mid_3d = np.zeros((n, 3), dtype=np.float32)
    out_3d = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        for j, (xy, dst) in enumerate(((lateral_mid_xy[i], mid_3d), (lateral_out_xy[i], out_3d))):
            z, ok = solve_z_on_ellipsoid(
                ellipsoid, float(xy[0]), float(xy[1]),
                axes_clip_ratio=axes_clip_ratio,
            )
            if not ok:
                z = _nearest_lateral_mp_anchor_z(landmarks_norm, xy.astype(np.float64), all_anchors)
            dst[i, 0] = xy[0]
            dst[i, 1] = xy[1]
            dst[i, 2] = z
            in_envelope[i * 2 + j] = ok

    return mid_3d, out_3d, in_envelope


def assemble_full_v2(
    landmarks_norm: np.ndarray,
    middle_3d: np.ndarray,
    hairline_3d: np.ndarray,
    lateral_mid_3d: np.ndarray,
    lateral_out_3d: np.ndarray,
) -> np.ndarray:
    """Concatenate to a (N_TOTAL_V2, 3) array in the SDK's v2 layout.

    Layout
    ------
    [0       .. 468)   MediaPipe 468
    [468     .. 485)   v1 middle row 17
    [485     .. 502)   v1 hairline row 17
    [502     .. 512)   lateral mid row (left 5 + right 5)
    [512     .. 522)   lateral out row (left 5 + right 5)
    """
    assert landmarks_norm.shape == (C.N_MP, 3)
    assert middle_3d.shape == (C.N_ANCHORS, 3)
    assert hairline_3d.shape == (C.N_ANCHORS, 3)
    assert lateral_mid_3d.shape == (C.N_LATERAL, 3)
    assert lateral_out_3d.shape == (C.N_LATERAL, 3)
    return np.concatenate(
        [landmarks_norm, middle_3d, hairline_3d, lateral_mid_3d, lateral_out_3d],
        axis=0,
    ).astype(np.float32)
