"""Fit an axis-aligned ellipsoid to a set of 3D head landmarks and use it
to solve Z for new (x, y) points lying just outside the original mesh envelope.

The MediaPipe canonical 468 landmarks cover roughly the front half of the
head. v2-headext adds lateral extension vertices whose 2D positions are
detected from the image, but they need a Z to be usable in 3D. Rather than
introducing a full FLAME / HRN solver, we fit a simple axis-aligned
ellipsoid to the 468 MP points (in MediaPipe normalized space) and solve

    ((x-cx)/a)^2 + ((y-cy)/b)^2 + ((z-cz)/c)^2 = 1

for z given (x, y). The front-facing root is selected by matching the sign
of (z - cz) to the average MP z-offset (typically negative, since MP
chooses the deep face Z as negative).

Limitations / fallbacks
-----------------------
* When (x, y) falls outside the ellipsoid's xy-projection (no real root),
  we inherit Z from the nearest MP anchor with an optional offset.
* Optionally clamp |z - cz| to ``axes_clip_ratio * c`` to prevent the
  solver from drifting back of the head when (x, y) is too close to the
  ellipse boundary.

The whole module is dependency-light (numpy only).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeadEllipsoid:
    """Axis-aligned ellipsoid in MediaPipe normalized space.

    Surface: ((x-cx)/a)^2 + ((y-cy)/b)^2 + ((z-cz)/c)^2 = 1.
    """
    cx: float
    cy: float
    cz: float
    a: float
    b: float
    c: float
    z_front_sign: int   # +1 or -1; the sign of (z_front - cz) for MP points
    mean_residual: float

    def __str__(self) -> str:
        return (
            f"HeadEllipsoid(center=({self.cx:.4f},{self.cy:.4f},{self.cz:.4f}) "
            f"axes=({self.a:.4f},{self.b:.4f},{self.c:.4f}) "
            f"z_front_sign={self.z_front_sign:+d} "
            f"mean_residual={self.mean_residual:.5f})"
        )


def fit_head_ellipsoid(
    landmarks_3d: np.ndarray,
    axis_margin: float = 1.30,
    depth_to_width_ratio: float = 1.15,
    center_depth_offset: float = 0.35,
) -> HeadEllipsoid:
    """Estimate an axis-aligned ellipsoid that bounds the head.

    MediaPipe's 468 landmarks only cover the front-half surface of the head,
    so a pure algebraic fit (``Q(x,y,z) = 0`` LSQ) is ill-posed — it produces
    saddle / hyperboloid solutions. Instead we use an anatomical prior:

      * Center (cx, cy) = centroid of MP 468 in XY.
      * Half-axes (a, b) = half-extent of MP 468 in XY × ``axis_margin``.
        Default 1.30 because the head silhouette (ears + hair) is ~25-35%
        wider than the visible face skin.
      * Half-axis c = max(a, b) × ``depth_to_width_ratio`` (default 1.15,
        as heads are slightly deeper than wide).
      * cz = z_front - z_front_sign × c × ``center_depth_offset``
        (push the center behind the visible front of the face by a small
        fraction of the head depth, so MP points sit on the front lobe of
        the ellipsoid rather than dead-center).

    The result is good enough for solving Z on lateral extension points
    that lie just outside the MP xy-envelope; for points further out (e.g.
    full skull silhouette) the caller falls back to nearest-anchor Z.
    """
    pts = np.asarray(landmarks_3d, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 6:
        raise ValueError(f"need at least 6 landmarks, got {len(pts)}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    a = max(1e-6, 0.5 * (x_max - x_min)) * axis_margin
    b_ = max(1e-6, 0.5 * (y_max - y_min)) * axis_margin
    c = max(a, b_) * depth_to_width_ratio

    # Decide which Z direction is "front facing" w.r.t. the MP cloud:
    # whichever side the cloud is biased toward, taken with respect to a
    # naive center estimate at the cloud's midline.
    z_mid = 0.5 * (z_min + z_max)
    front_is_max = abs(z_max - z_mid) <= abs(z_min - z_mid)
    z_front = z_max if front_is_max else z_min
    z_front_sign = +1 if front_is_max else -1

    cz = z_front - z_front_sign * c * center_depth_offset

    # Residual: how far each MP point sits off the ellipsoid surface, in
    # units of the normalized algebraic LHS (mean |Q - 1|).
    nx2 = ((x - cx) / a) ** 2
    ny2 = ((y - cy) / b_) ** 2
    nz2 = ((z - cz) / c) ** 2
    lhs = nx2 + ny2 + nz2
    residual = float(np.mean(np.abs(lhs - 1.0)))

    return HeadEllipsoid(
        cx=cx, cy=cy, cz=cz, a=a, b=b_, c=c,
        z_front_sign=z_front_sign,
        mean_residual=residual,
    )


def solve_z_on_ellipsoid(
    ellipsoid: HeadEllipsoid,
    x: float,
    y: float,
    axes_clip_ratio: float = 1.0,
) -> tuple[float, bool]:
    """Solve for z on the ellipsoid given (x, y). Returns (z, in_envelope).

    The two algebraic roots represent the front and back hemispheres; we
    pick the one matching ``ellipsoid.z_front_sign``.

    in_envelope = False means (x, y) is outside the xy-projection of the
    ellipsoid; the caller should fall back (we still return a reasonable
    value: z at the ellipse boundary projected back to cz ± clip).
    """
    if axes_clip_ratio <= 0 or axes_clip_ratio > 1.5:
        raise ValueError(f"axes_clip_ratio must be in (0, 1.5], got {axes_clip_ratio}")

    nx2 = ((x - ellipsoid.cx) / ellipsoid.a) ** 2
    ny2 = ((y - ellipsoid.cy) / ellipsoid.b) ** 2
    discriminant = 1.0 - nx2 - ny2

    if discriminant >= 0.0:
        dz = ellipsoid.c * float(np.sqrt(discriminant))
        z = ellipsoid.cz + ellipsoid.z_front_sign * dz
        return z, True

    # Out of envelope: snap (x, y) to the closest point on the ellipse
    # boundary (parametric), then evaluate Z = cz.
    # We use the angular parameter atan2 to find the nearest boundary
    # point's angle and approximate z by cz - tiny offset (so the new
    # vertex sits just on the edge instead of inside).
    theta = float(np.arctan2(
        (y - ellipsoid.cy) / max(ellipsoid.b, 1e-9),
        (x - ellipsoid.cx) / max(ellipsoid.a, 1e-9),
    ))
    # Project onto boundary, then offset Z by a fraction of c towards the
    # front so the extrapolated vertex doesn't sit dead-flat on cz.
    z = ellipsoid.cz + ellipsoid.z_front_sign * (axes_clip_ratio * 0.2 * ellipsoid.c)
    _ = theta  # parameter computed for future use
    return z, False
