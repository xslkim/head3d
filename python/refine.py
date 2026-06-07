"""Step 7: nudge the 27 raw extension points onto the head surface.

Finalised algorithm (chosen after comparing 6 candidates): **parabolic depth
recede, x/y preserved**.

The 2D overlay only uses each vertex's x/y, and plain linear extrapolation
already gets x/y right (it continues the brow -> top-edge direction, which
follows the face shape). So we keep x/y exactly at the linear extrapolation and
only refine the depth z: the forehead cap recedes back over the skull as a
smooth quadratic in the ring distance σ (= the ring multiplier t_k), measured
from the top edge b.

    z = z_linear + STRENGTH * σ² * A_xy

where A_xy is the horizontal length of one A-step (depth scale, px). STRENGTH
is fixed at 0.20. No tunable parameters.
"""
from __future__ import annotations

import numpy as np

from .extend_mesh import N_PAIRS, N_RINGS, T_DEFAULT, anchor_obj_indices

# Fixed recede strength (locked after live comparison).
STRENGTH = 0.20


def _pair_dxy(base: np.ndarray) -> np.ndarray:
    """Horizontal length of one A-step per anchor pair (depth scale, px)."""
    a_obj, b_obj = anchor_obj_indices()
    d = base[b_obj] - base[a_obj]
    return np.maximum(np.linalg.norm(d[:, :2], axis=1), 1e-6)


def refine(base: np.ndarray, linear_pts: np.ndarray,
           t: tuple[float, ...] = T_DEFAULT) -> np.ndarray:
    """Return the adjusted (27, 3) extension points.

    x/y stay at the linear extrapolation; z recedes quadratically with σ.
    """
    t = np.asarray(t, dtype=np.float64)
    dxy = _pair_dxy(base)                            # (9,)

    out = linear_pts.copy().reshape(N_RINGS, N_PAIRS, 3)
    for k in range(N_RINGS):
        out[k, :, 2] += STRENGTH * (t[k] ** 2) * dxy
    return out.reshape(-1, 3)
