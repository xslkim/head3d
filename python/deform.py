"""Step 5: deform the mesh to a detected face.

Given 468 MediaPipe landmarks (pixel space) for an uploaded photo, place the
468 base OBJ vertices at the matching landmark (via INDEX_MAP_468), then
compute the 27 extension vertices (fixed ring multipliers) and refine their
depth onto the head surface. The result is a 495-vertex position array aligned
to the photo, ready for the 2D overlay and the 3D viewer. UVs/faces are
constant and come from the template.
"""
from __future__ import annotations

import numpy as np

from .extend_mesh import N_NEW, T_DEFAULT, compute_extension
from .index_map import INDEX_MAP_468
from .refine import refine

N_MP = 468


def deform_positions(landmarks_px: np.ndarray) -> np.ndarray:
    """Return (495, 3) positions in pixel space for the deformed mesh.

    ``landmarks_px`` is (468, 3): MediaPipe landmarks scaled to pixels
    (see landmarks.to_pixels). OBJ vertex i takes landmark INDEX_MAP_468[i].
    """
    assert landmarks_px.shape[0] == N_MP, "expected 468 landmarks"
    idx = np.asarray(INDEX_MAP_468, dtype=np.int64)
    base = landmarks_px[idx].astype(np.float64)        # (468, 3) in OBJ order
    new = compute_extension(base, T_DEFAULT)           # (27, 3) raw rings
    new = refine(base, new, T_DEFAULT)                 # (27, 3) head-fitted
    return np.vstack([base, new])


def n_total() -> int:
    return N_MP + N_NEW
