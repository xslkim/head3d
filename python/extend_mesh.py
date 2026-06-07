"""Core mesh-extension algorithm (docs/新算法.md, steps 1-4).

For each of the 9 anchor pairs ``[a, b]`` (MediaPipe canonical indices),
``b`` is the outer/upper point and the extension direction is ``b - a``.
Two new points are grown outward starting from ``b``:

    ring1 = b + t1 * (b - a)      (t1 default 0.8)
    ring2 = b + t2 * (b - a)      (t2 default 1.6)

t1 / t2 are global multipliers, the same for all 9 pairs, adjustable live
from the web UI. This gives 9 * 2 = 18 new vertices forming two rows above
the existing top edge. The 9 ``b`` anchors + ring1 + ring2 are stitched
into a triangle ribbon, and UVs are extrapolated the same way in UV space.
"""
from __future__ import annotations

import numpy as np

from .index_map import INDEX_MAP_468
from .obj_io import ObjMesh

# 9 anchor pairs in MediaPipe canonical landmark indices, ordered left->right.
# In each pair (a, b): b is the outer (upper) point, direction = b - a.
PAIRS: list[tuple[int, int]] = [
    (68, 54), (104, 103), (69, 67), (108, 109), (151, 10),
    (337, 338), (299, 297), (333, 332), (298, 284),
]
N_PAIRS = len(PAIRS)            # 9
N_NEW = 2 * N_PAIRS            # 18

T1_DEFAULT = 0.8
T2_DEFAULT = 1.6

# MediaPipe index -> OBJ vertex index (inverse of INDEX_MAP_468).
_MP_TO_OBJ = {mp: i for i, mp in enumerate(INDEX_MAP_468)}


def anchor_obj_indices() -> tuple[list[int], list[int]]:
    """Return (a_obj, b_obj): OBJ vertex indices for each pair's a and b."""
    a_obj = [_MP_TO_OBJ[a] for a, _ in PAIRS]
    b_obj = [_MP_TO_OBJ[b] for _, b in PAIRS]
    return a_obj, b_obj


def new_vertex_layout(n_base: int) -> tuple[list[int], list[int]]:
    """Return (ring1_idx, ring2_idx): global vertex ids for the 18 new points.

    Layout: [0, n_base) original, then ring1 (9), then ring2 (9).
    """
    ring1 = [n_base + j for j in range(N_PAIRS)]
    ring2 = [n_base + N_PAIRS + j for j in range(N_PAIRS)]
    return ring1, ring2


def compute_extension(
    values: np.ndarray, t1: float = T1_DEFAULT, t2: float = T2_DEFAULT
) -> np.ndarray:
    """Compute the 18 new rows from a per-OBJ-vertex array.

    ``values`` has shape (>=468, D) in OBJ vertex order (positions, texcoords,
    ...). Returns shape (18, D): the first 9 rows are ring1, the next 9 ring2.
    """
    a_obj, b_obj = anchor_obj_indices()
    A = values[a_obj]            # (9, D)
    B = values[b_obj]            # (9, D)
    direction = B - A            # b - a
    ring1 = B + t1 * direction
    ring2 = B + t2 * direction
    return np.vstack([ring1, ring2])


def ribbon_faces(n_base: int) -> list[tuple[int, int, int]]:
    """Triangle faces (vertex-id triples) stitching anchors -> ring1 -> ring2.

    Two strips of 8 quads = 16 quads = 32 triangles. Winding chosen so the
    outward face normal points the same way as the existing forehead faces
    (verified by the template build).
    """
    _, b_obj = anchor_obj_indices()
    ring1, ring2 = new_vertex_layout(n_base)
    faces: list[tuple[int, int, int]] = []
    for j in range(N_PAIRS - 1):
        b0, b1 = b_obj[j], b_obj[j + 1]
        m0, m1 = ring1[j], ring1[j + 1]
        h0, h1 = ring2[j], ring2[j + 1]
        # strip 1: anchors -> ring1
        faces.append((b0, b1, m1))
        faces.append((b0, m1, m0))
        # strip 2: ring1 -> ring2
        faces.append((m0, m1, h1))
        faces.append((m0, h1, h0))
    return faces


def _vertex_normals(positions: np.ndarray, tris: list[tuple[int, int, int]]) -> np.ndarray:
    """Area-weighted vertex normals for the given triangle list."""
    normals = np.zeros_like(positions, dtype=np.float64)
    for i0, i1, i2 in tris:
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        fn = np.cross(p1 - p0, p2 - p0)
        normals[i0] += fn
        normals[i1] += fn
        normals[i2] += fn
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    return normals / lens


def extend_obj_mesh(mesh: ObjMesh, t1: float = T1_DEFAULT, t2: float = T2_DEFAULT) -> ObjMesh:
    """Return a new ObjMesh = base mesh + 18 vertices + ribbon faces.

    Assumes the base mesh is 1:1 (pos idx == uv idx == normal idx), which
    holds for the canonical face.obj (468 v / 468 vt / 468 vn).
    """
    n_base = mesh.n_v()
    assert mesh.n_vt() == n_base and mesh.n_vn() == n_base, (
        "extend_obj_mesh expects a 1:1 base mesh"
    )

    pos = np.array(mesh.positions, dtype=np.float64)
    uv = np.array(mesh.texcoords, dtype=np.float64)

    new_pos = compute_extension(pos, t1, t2)              # (18, 3)
    new_uv = compute_extension(uv, t1, t2)               # (18, 2)

    out = ObjMesh()
    out.header_lines = list(mesh.header_lines)
    out.positions = list(mesh.positions) + [tuple(p) for p in new_pos]
    out.texcoords = list(mesh.texcoords) + [tuple(t) for t in new_uv]

    faces = ribbon_faces(n_base)

    # Normals for the 18 new verts: average of the ribbon faces touching them.
    full_pos = np.array(out.positions, dtype=np.float64)
    rib_normals = _vertex_normals(full_pos, faces)
    out.normals = list(mesh.normals) + [
        tuple(rib_normals[n_base + k]) for k in range(N_NEW)
    ]

    # Keep all original faces, then append the ribbon (1:1 pos/uv/normal idx).
    out.faces = [list(f) for f in mesh.faces]
    for i0, i1, i2 in faces:
        out.faces.append([(i0, i0, i0), (i1, i1, i1), (i2, i2, i2)])

    return out
