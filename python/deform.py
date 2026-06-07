"""Step 5: deform the mesh to a detected face.

Given 468 MediaPipe landmarks (pixel space) for an uploaded photo, place the
468 base OBJ vertices at the matching landmark (via INDEX_MAP_468), then
recompute the 18 extension vertices with the current t1/t2. The result is a
486-vertex position array aligned to the photo, ready for the 2D overlay and
the 3D viewer. UVs/faces are constant and come from the template.
"""
from __future__ import annotations

import numpy as np

from .extend_mesh import N_NEW, T1_DEFAULT, T2_DEFAULT, compute_extension
from .index_map import INDEX_MAP_468

N_MP = 468


def deform_positions(
    landmarks_px: np.ndarray,
    t1: float = T1_DEFAULT,
    t2: float = T2_DEFAULT,
) -> np.ndarray:
    """Return (486, 3) positions in pixel space for the deformed mesh.

    ``landmarks_px`` is (468, 3): MediaPipe landmarks scaled to pixels
    (see landmarks.to_pixels). OBJ vertex i takes landmark INDEX_MAP_468[i].
    """
    assert landmarks_px.shape[0] == N_MP, "expected 468 landmarks"
    idx = np.asarray(INDEX_MAP_468, dtype=np.int64)
    base = landmarks_px[idx].astype(np.float64)        # (468, 3) in OBJ order
    new = compute_extension(base, t1, t2)              # (18, 3)
    return np.vstack([base, new])


def n_total() -> int:
    return N_MP + N_NEW
