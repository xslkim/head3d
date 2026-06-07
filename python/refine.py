"""Step 7: nudge the 27 raw extension points onto the head surface.

Lesson from the first attempt: the 2D overlay only uses each vertex's x/y, and
plain linear extrapolation already gets x/y right (it continues the brow ->
top-edge direction, which follows the face shape). Any algorithm that moves
x/y toward a guessed centre distorts the silhouette and looks *worse* than
linear.

So every algorithm here keeps **x/y exactly at the linear extrapolation** and
only refines the **depth (z)**: the forehead cap should roll back over the
skull instead of continuing as a flat tilted plane. z does not affect the 2D
silhouette at all, so these can only improve the 3D view, never hurt the 2D.

The six depth models differ in the *shape* of z along the ray (distance σ = the
ring multiplier t_k, measured from the top edge b). ``linear`` is kept as the
A/B reference. Each model exposes a single slider.

All operate on the raw points ``linear_pts`` (shape (27, 3), ring-major) and
the 468 base positions, returning an adjusted (27, 3) array.
"""
from __future__ import annotations

import numpy as np

from .extend_mesh import N_PAIRS, N_RINGS, T_DEFAULT, anchor_obj_indices

# ``linear`` (reference) + 6 depth-refinement algorithms.
ALGOS = ["linear", "parabolic", "cosine", "arc", "damped", "sphere", "flat"]

PARAM_SPECS: dict[str, list[dict]] = {
    "linear": [],
    "parabolic": [
        {"key": "strength", "label": "后收强度", "min": 0.0, "max": 1.5,
         "step": 0.05, "default": 0.5},
    ],
    "cosine": [
        {"key": "amp", "label": "滚动幅度", "min": 0.0, "max": 1.5,
         "step": 0.05, "default": 0.6},
    ],
    "arc": [
        {"key": "radius", "label": "圆弧半径系数", "min": 0.6, "max": 4.0,
         "step": 0.1, "default": 1.5},
    ],
    "damped": [
        {"key": "k", "label": "饱和速度", "min": 0.1, "max": 3.0,
         "step": 0.05, "default": 1.0},
    ],
    "sphere": [
        {"key": "blend", "label": "球面贴合度", "min": 0.0, "max": 1.5,
         "step": 0.05, "default": 1.0},
    ],
    "flat": [
        {"key": "tilt", "label": "后仰坡度", "min": -0.5, "max": 1.5,
         "step": 0.05, "default": 0.3},
    ],
}

ALGO_LABELS: dict[str, str] = {
    "linear": "线性外推（基准 · x/y/z 全线性）",
    "parabolic": "抛物后收：z 按距离² 平滑后退（保 x/y）",
    "cosine": "余弦滚动：眉端柔和、外环渐强后退（保 x/y）",
    "arc": "圆弧滚动：z 沿相切圆弧翻过头顶（保 x/y）",
    "damped": "坡度衰减：续眉坡但渐饱和，头顶变平（保 x/y）",
    "sphere": "球面深度：z 取自拟合颅骨球面（保 x/y）",
    "flat": "平整盖：z 设为可调后仰平面（保 x/y）",
}


def default_params(algo: str) -> dict[str, float]:
    return {s["key"]: s["default"] for s in PARAM_SPECS.get(algo, [])}


def _pair_geometry(base: np.ndarray):
    """Return (B, dxy, dz) for the 9 anchor pairs.

    B  : (9, 3) top-edge points (the ring origin).
    dxy: (9,)   horizontal length of one A-step (depth scale, in px).
    dz : (9,)   z change over one A-step (the brow slope, in px).
    """
    a_obj, b_obj = anchor_obj_indices()
    A = base[a_obj]
    B = base[b_obj]
    d = B - A
    dxy = np.linalg.norm(d[:, :2], axis=1)
    dxy = np.maximum(dxy, 1e-6)
    dz = d[:, 2]
    return B, dxy, dz


def _fit_sphere(pts: np.ndarray):
    """Least-squares sphere fit. Returns (centre(3,), R) or None if degenerate."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # x²+y²+z² = 2cx·x + 2cy·y + 2cz·z + c0
    Amat = np.column_stack([2 * x, 2 * y, 2 * z, np.ones_like(x)])
    b = x * x + y * y + z * z
    sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    cx, cy, cz, c0 = sol
    r2 = c0 + cx * cx + cy * cy + cz * cz
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return np.array([cx, cy, cz]), float(np.sqrt(r2))


def refine(base: np.ndarray, linear_pts: np.ndarray,
           t: tuple[float, ...] = T_DEFAULT,
           algo: str = "linear", params: dict | None = None) -> np.ndarray:
    """Return the adjusted (27, 3) extension points for ``algo``.

    x/y are always the linear extrapolation; only z is remodelled.
    """
    if algo not in ALGOS or algo == "linear":
        return linear_pts.copy()

    p = {**default_params(algo), **(params or {})}
    t = np.asarray(t, dtype=np.float64)
    sigma_max = float(t.max()) if t.max() > 0 else 1.0

    B, dxy, dz = _pair_geometry(base)        # (9,3), (9,), (9,)
    bz = B[:, 2]                             # (9,) top-edge depth

    out = linear_pts.copy().reshape(N_RINGS, N_PAIRS, 3)
    z_lin = out[..., 2].copy()               # (R, 9) linear depth per ring

    if algo == "parabolic":
        for k in range(N_RINGS):
            out[k, :, 2] = z_lin[k] + p["strength"] * (t[k] ** 2) * dxy

    elif algo == "cosine":
        for k in range(N_RINGS):
            w = (1.0 - np.cos(np.pi * t[k] / sigma_max)) * 0.5
            out[k, :, 2] = z_lin[k] + p["amp"] * w * dxy

    elif algo == "arc":
        R = p["radius"] * dxy                 # (9,) curvature radius per pair
        phi0 = np.arctan2(dz, dxy)            # continue the brow slope
        for k in range(N_RINGS):
            h = t[k] * dxy
            out[k, :, 2] = bz + R * (np.sin(phi0 + h / R) - np.sin(phi0))

    elif algo == "damped":
        k_sat = max(p["k"], 1e-3)
        for k in range(N_RINGS):
            # continues slope dz near 0, saturates to dz/k_sat (crown flattens)
            out[k, :, 2] = bz + (dz / k_sat) * (1.0 - np.exp(-k_sat * t[k]))

    elif algo == "sphere":
        fit = _fit_sphere(B)
        if fit is not None:
            c, R = fit
            for k in range(N_RINGS):
                px, py = out[k, :, 0], out[k, :, 1]
                disc = R * R - (px - c[0]) ** 2 - (py - c[1]) ** 2
                root = np.sqrt(np.clip(disc, 0.0, None))
                # pick the hemisphere root nearer the linear depth
                cand_hi, cand_lo = c[2] + root, c[2] - root
                z_sph = np.where(
                    np.abs(cand_hi - z_lin[k]) <= np.abs(cand_lo - z_lin[k]),
                    cand_hi, cand_lo,
                )
                out[k, :, 2] = z_lin[k] + p["blend"] * (z_sph - z_lin[k])

    elif algo == "flat":
        for k in range(N_RINGS):
            out[k, :, 2] = bz + p["tilt"] * t[k] * dxy

    return out.reshape(-1, 3)
