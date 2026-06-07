"""Core mesh-extension algorithm (docs/新算法.md, steps 1-4).

For each of the 9 anchor pairs ``[a, b]`` (MediaPipe canonical indices),
``b`` is the outer/upper point and the extension direction is ``b - a``;
its length defines ``A = |b - a|`` (doc step 1). Three new points are grown
outward starting from ``b`` along that direction (doc step 2):

    ring1 = b + t1 * (b - a)      (t1 default 0.8  -> 0.8A)
    ring2 = b + t2 * (b - a)      (t2 default 1.5  -> 1.5A)
    ring3 = b + t3 * (b - a)      (t3 default 2.2  -> 2.2A)

t1 / t2 / t3 are global multipliers, the same for all 9 pairs, adjustable
live from the web UI. This gives 9 * 3 = 27 new vertices forming three rows
above the existing top edge (doc step 3). The 9 ``b`` anchors + the three
rings are stitched into a triangle ribbon, and UVs are extrapolated the same
way in UV space (doc step 4).

The raw rings are a flat radial extrapolation; ``python/refine.py`` then
nudges them onto the head surface (doc step 7), with 6 selectable algorithms.
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

# Three rings at 0.8A / 1.5A / 2.2A (doc step 2).
T_DEFAULT: tuple[float, float, float] = (0.8, 1.5, 2.2)
N_RINGS = len(T_DEFAULT)        # 3
N_NEW = N_RINGS * N_PAIRS       # 27

# MediaPipe index -> OBJ vertex index (inverse of INDEX_MAP_468).
_MP_TO_OBJ = {mp: i for i, mp in enumerate(INDEX_MAP_468)}


def anchor_obj_indices() -> tuple[list[int], list[int]]:
    """Return (a_obj, b_obj): OBJ vertex indices for each pair's a and b."""
    a_obj = [_MP_TO_OBJ[a] for a, _ in PAIRS]
    b_obj = [_MP_TO_OBJ[b] for _, b in PAIRS]
    return a_obj, b_obj


def new_vertex_layout(n_base: int) -> list[list[int]]:
    """Return ``rings``: a list of ``N_RINGS`` lists, each ``N_PAIRS`` global
    vertex ids for that ring's new points.

    Layout: [0, n_base) original, then ring0 (9), ring1 (9), ring2 (9).
    """
    return [
        [n_base + r * N_PAIRS + j for j in range(N_PAIRS)]
        for r in range(N_RINGS)
    ]


def compute_extension(
    values: np.ndarray, t: tuple[float, ...] = T_DEFAULT
) -> np.ndarray:
    """Compute the 27 new rows from a per-OBJ-vertex array.

    ``values`` has shape (>=468, D) in OBJ vertex order (positions, texcoords,
    ...). Returns shape (27, D), stacked ring-major: rows [0:9) are ring0,
    [9:18) ring1, [18:27) ring2.
    """
    a_obj, b_obj = anchor_obj_indices()
    A = values[a_obj]            # (9, D)
    B = values[b_obj]            # (9, D)
    direction = B - A            # b - a
    rings = [B + tk * direction for tk in t]
    return np.vstack(rings)


def ribbon_faces(n_base: int) -> list[tuple[int, int, int]]:
    """Triangle faces (vertex-id triples) stitching anchors -> ring0 -> ring1
    -> ring2.

    Three strips of 8 quads = 24 quads = 48 triangles. Winding chosen so the
    outward face normal points the same way as the existing forehead faces
    (verified by the template build).
    """
    _, b_obj = anchor_obj_indices()
    rings = new_vertex_layout(n_base)
    rows = [b_obj] + rings        # [anchors, ring0, ring1, ring2]
    faces: list[tuple[int, int, int]] = []
    for r in range(len(rows) - 1):
        lo, hi = rows[r], rows[r + 1]
        for j in range(N_PAIRS - 1):
            l0, l1 = lo[j], lo[j + 1]
            h0, h1 = hi[j], hi[j + 1]
            faces.append((l0, l1, h1))
            faces.append((l0, h1, h0))

    # Two corner fill triangles closing the outer gaps (user-specified winding):
    #   (54, group[68,54] ring0 point, 21)   left side
    #   (284, group[298,284] ring0 point, 251) right side
    ring0 = rings[0]
    faces.append((_MP_TO_OBJ[54], ring0[0], _MP_TO_OBJ[21]))
    faces.append((_MP_TO_OBJ[284], ring0[N_PAIRS - 1], _MP_TO_OBJ[251]))
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


def extend_obj_mesh(mesh: ObjMesh, t: tuple[float, ...] = T_DEFAULT) -> ObjMesh:
    """Return a new ObjMesh = base mesh + 27 vertices + ribbon faces.

    Assumes the base mesh is 1:1 (pos idx == uv idx == normal idx), which
    holds for the canonical face.obj (468 v / 468 vt / 468 vn).
    """
    n_base = mesh.n_v()
    assert mesh.n_vt() == n_base and mesh.n_vn() == n_base, (
        "extend_obj_mesh expects a 1:1 base mesh"
    )

    pos = np.array(mesh.positions, dtype=np.float64)
    uv = np.array(mesh.texcoords, dtype=np.float64)

    new_pos = compute_extension(pos, t)                  # (27, 3)
    new_uv = compute_extension(uv, t)                    # (27, 2)

    out = ObjMesh()
    out.header_lines = list(mesh.header_lines)
    out.positions = list(mesh.positions) + [tuple(p) for p in new_pos]
    out.texcoords = list(mesh.texcoords) + [tuple(t_) for t_ in new_uv]

    faces = ribbon_faces(n_base)

    # Normals for the 27 new verts: average of the ribbon faces touching them.
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
