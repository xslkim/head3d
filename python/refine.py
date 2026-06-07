"""Step 7: nudge the 27 raw extension points onto the head surface.

The raw rings from ``extend_mesh.compute_extension`` are a flat radial
extrapolation (``b + t*(b-a)``). On a real head the forehead rolls back over
the crown, so the upper rings should both *converge* toward the crown centre
(visible in the 2D overlay, which only uses x/y) and *recede* in depth
(visible in the 3D view, which uses z).

Six selectable algorithms implement that wrap with different profiles so the
operator can compare them live and pick one (doc step 7). Each exposes a
single strength/curvature parameter, surfaced as one slider in the web UI.

Every algorithm takes the raw points ``linear_pts`` (shape (27, 3), ring-major
as produced by ``compute_extension``) plus the 468 base positions, and returns
an adjusted (27, 3) array. The ring multipliers ``t`` give each row's distance.
"""
from __future__ import annotations

import numpy as np

from .extend_mesh import N_PAIRS, N_RINGS, T_DEFAULT, anchor_obj_indices

ALGOS = ["linear", "parabolic", "spherical", "ellipsoid", "cosine", "arc"]

# One tunable parameter per algorithm, with the metadata the web UI needs to
# render a labelled slider. ``linear`` is the unmodified control (no params).
PARAM_SPECS: dict[str, list[dict]] = {
    "linear": [],
    "parabolic": [
        {"key": "strength", "label": "回收强度", "min": 0.0, "max": 1.0,
         "step": 0.05, "default": 0.6},
    ],
    "spherical": [
        {"key": "scale", "label": "球半径系数", "min": 0.4, "max": 2.0,
         "step": 0.05, "default": 1.0},
    ],
    "ellipsoid": [
        {"key": "back", "label": "后脑深度系数", "min": 0.0, "max": 1.5,
         "step": 0.05, "default": 0.5},
    ],
    "cosine": [
        {"key": "strength", "label": "贴合强度", "min": 0.0, "max": 1.0,
         "step": 0.05, "default": 0.7},
    ],
    "arc": [
        {"key": "radius", "label": "圆弧半径系数", "min": 0.6, "max": 4.0,
         "step": 0.1, "default": 1.5},
    ],
}

# Human-readable blurb per algorithm, shown beside the selector.
ALGO_LABELS: dict[str, str] = {
    "linear": "线性外推（基准，不贴合）",
    "parabolic": "抛物收拢：按距离² 向头顶收拢并后收",
    "spherical": "球面贴合：投影到拟合颅骨的球面",
    "ellipsoid": "椭球贴合：投影到头部包围盒椭球",
    "cosine": "余弦缓动：眉端柔和、头顶强贴合",
    "arc": "圆弧外扩：沿相切圆弧翻过头顶",
}


def default_params(algo: str) -> dict[str, float]:
    return {s["key"]: s["default"] for s in PARAM_SPECS.get(algo, [])}


def _frame(base: np.ndarray) -> dict:
    """Head reference frame estimated from the 9 top-edge anchors."""
    a_obj, b_obj = anchor_obj_indices()
    A = base[a_obj]                          # (9, 3) inner brow points
    B = base[b_obj]                          # (9, 3) outer/top edge points
    d = B - A
    Alen = np.linalg.norm(d, axis=1)         # (9,) per-pair length A
    cx = float(B[:, 0].mean())
    top_y = float(B[:, 1].min())             # smallest y == highest in image
    mean_z = float(B[:, 2].mean())
    headW = float(B[:, 0].max() - B[:, 0].min())
    headW = max(headW, 1.0)
    # Crown apex the rings should converge toward: above the brow by ~one head
    # radius, and receded in depth (larger z == farther from camera).
    apex = np.array([cx, top_y - 0.6 * headW, mean_z + 0.5 * headW])
    return {
        "A": A, "B": B, "d": d, "Alen": Alen,
        "cx": cx, "top_y": top_y, "mean_z": mean_z,
        "headW": headW, "apex": apex,
    }


def _converge(linear_pts: np.ndarray, fr: dict, t: np.ndarray,
              w_profile: np.ndarray, z_gain: float) -> np.ndarray:
    """Shared converge-to-apex + recede-in-z wrap.

    ``w_profile`` is a per-ring blend weight (N_RINGS,) in [0,1]; each ring's
    x/y is lerped that fraction toward the crown apex and pushed back in z.
    """
    out = linear_pts.copy().reshape(N_RINGS, N_PAIRS, 3)
    apex = fr["apex"]
    for k in range(N_RINGS):
        w = w_profile[k]
        out[k, :, 0] += (apex[0] - out[k, :, 0]) * w
        out[k, :, 1] += (apex[1] - out[k, :, 1]) * w
        out[k, :, 2] += z_gain * w * fr["headW"]
    return out.reshape(-1, 3)


def _project_quadric(linear_pts: np.ndarray, center: np.ndarray,
                     radii: np.ndarray) -> np.ndarray:
    """Project each point radially onto the sphere/ellipsoid (center, radii)."""
    out = linear_pts.copy()
    v = (out - center) / radii
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return center + (out - center) / norm


def refine(base: np.ndarray, linear_pts: np.ndarray,
           t: tuple[float, ...] = T_DEFAULT,
           algo: str = "linear", params: dict | None = None) -> np.ndarray:
    """Return the adjusted (27, 3) extension points for ``algo``."""
    if algo not in ALGOS:
        algo = "linear"
    p = {**default_params(algo), **(params or {})}
    t = np.asarray(t, dtype=np.float64)
    tmax = float(t.max()) if t.max() > 0 else 1.0
    u = t / tmax                              # normalized ring distance in (0,1]

    if algo == "linear":
        return linear_pts.copy()

    fr = _frame(base)

    if algo == "parabolic":
        w = np.clip(p["strength"] * u ** 2, 0.0, 1.0)
        return _converge(linear_pts, fr, t, w, z_gain=0.5)

    if algo == "cosine":
        # Smooth S-curve: gentle near the brow, strong over the crown.
        w = np.clip(p["strength"] * (1.0 - np.cos(np.pi * u)) * 0.5, 0.0, 1.0)
        return _converge(linear_pts, fr, t, w, z_gain=0.6)

    if algo == "spherical":
        center = np.array([
            fr["cx"], fr["B"][:, 1].mean() - 0.2 * fr["headW"],
            fr["mean_z"] + 0.8 * fr["headW"] * p["scale"],
        ])
        R = float(np.linalg.norm(fr["B"] - center, axis=1).mean())
        radii = np.array([R, R, R])
        return _project_quadric(linear_pts, center, radii)

    if algo == "ellipsoid":
        center = np.array([
            fr["cx"], fr["B"][:, 1].mean(),
            fr["mean_z"] + p["back"] * fr["headW"],
        ])
        radii = np.array([0.62, 0.78, 0.70]) * fr["headW"]
        return _project_quadric(linear_pts, center, radii)

    if algo == "arc":
        # Bend each straight ray into a circle tangent at b, curving toward the
        # crown apex. Radius = ``radius`` * A per pair.
        out = np.empty((N_RINGS * N_PAIRS, 3))
        B, d, Alen = fr["B"], fr["d"], fr["Alen"]
        apex = fr["apex"]
        for j in range(N_PAIRS):
            a_len = max(Alen[j], 1e-6)
            dhat = d[j] / a_len
            to_apex = apex - B[j]
            n = to_apex - np.dot(to_apex, dhat) * dhat       # component ⊥ dhat
            nn = np.linalg.norm(n)
            nhat = n / nn if nn > 1e-6 else np.array([0.0, 0.0, 1.0])
            R = max(p["radius"] * a_len, 1e-6)
            for k in range(N_RINGS):
                s = t[k] * a_len
                phi = s / R
                out[k * N_PAIRS + j] = (
                    B[j] + R * np.sin(phi) * dhat + R * (1 - np.cos(phi)) * nhat
                )
        return out

    return linear_pts.copy()
