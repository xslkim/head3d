"""Generate face_ext_v2.obj from face_ext.obj by appending 20 lateral vertices.

Layout produced
---------------
  - 522 vertices: [0..502) from face_ext.obj (unchanged), plus
        [502..507) lateral_mid_left  (top→bottom: temple→ear)
        [507..512) lateral_mid_right
        [512..517) lateral_out_left
        [517..522) lateral_out_right
  - 522 texcoords (parallel, new UVs in unused image-top band y=[0,115])
  - 522 normals (placeholders)
  - 916 + 32 = 948 triangles (4 tris per adjacent lateral pair, 4 pairs ×
        2 sides × 4 tris = 32)

The canonical positions for the 20 lateral vertices are derived by pushing
the corresponding MP anchor outward along the OBJ's local +X / -X direction,
with a small backward Z offset. These are TEMPLATES only — at runtime the
SDK overwrites them with the per-image detected positions from
``python.extract_headext``.

Run
---
  python -m tools.gen_face_ext_v2
or
  python tools/gen_face_ext_v2.py
"""
from __future__ import annotations
import os
import sys

import numpy as np


if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python import constants as C
from python.obj_io import ObjMesh, read_obj, write_obj
from python._index_map_data import INDEX_MAP_468
from python.build_extended_obj import build_inverse_index_map


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
INPUT_OBJ = os.path.join(PROJECT_DIR, "face_ext.obj")
OUTPUT_OBJ = os.path.join(PROJECT_DIR, "face_ext_v2.obj")


def canonical_lateral_positions(
    base: ObjMesh,
    inv_index_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mid_pos[10, 3], out_pos[10, 3]) in OBJ local coords.

    Order matches ``MP_LATERAL_ANCHORS_LEFT + MP_LATERAL_ANCHORS_RIGHT``.

    Strategy: push outward (along OBJ +X or -X depending on side) by a small
    fraction of face width, and backward (along -Z, since face.obj's +Z
    points roughly forward) by a smaller fraction of face depth.
    """
    pos = np.array(base.positions[:C.N_MP], dtype=np.float32)
    x_min, x_max = float(pos[:, 0].min()), float(pos[:, 0].max())
    z_min, z_max = float(pos[:, 2].min()), float(pos[:, 2].max())
    face_w = x_max - x_min
    face_d = z_max - z_min

    mid_step = 0.04 * face_w
    out_step = 0.08 * face_w
    back_step_mid = 0.05 * face_d
    back_step_out = 0.12 * face_d

    mid = np.zeros((C.N_LATERAL, 3), dtype=np.float32)
    out = np.zeros((C.N_LATERAL, 3), dtype=np.float32)

    chain = (
        [(mp, -1.0) for mp in C.MP_LATERAL_ANCHORS_LEFT]
        + [(mp, +1.0) for mp in C.MP_LATERAL_ANCHORS_RIGHT]
    )
    for i, (mp_idx, side_sign) in enumerate(chain):
        obj_idx = inv_index_map[mp_idx]
        a = pos[obj_idx]
        mid[i] = (a[0] + side_sign * mid_step, a[1], a[2] - back_step_mid)
        out[i] = (a[0] + side_sign * out_step, a[1], a[2] - back_step_out)
    return mid, out


def lateral_uvs() -> tuple[
    list[tuple[float, float]], list[tuple[float, float]]
]:
    """Return UVs for (mid_row, out_row). Each list has length N_LATERAL."""
    mid_uv: list[tuple[float, float]] = []
    out_uv: list[tuple[float, float]] = []
    # left then right, matching the position ordering.
    for side in ("left", "right"):
        for col in range(C.N_LATERAL_PER_SIDE):
            mid_uv.append(C.lateral_uv_for(0, side, col))
            out_uv.append(C.lateral_uv_for(1, side, col))
    return mid_uv, out_uv


def lateral_normals() -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 1.0)] * C.N_LATERAL_EXT


def build_lateral_faces(inv_index_map: dict[int, int]) -> list[list[tuple[int, int, int]]]:
    """4 triangles per adjacent pair (k, k+1) per side. 4 pairs × 2 sides = 32 tris.

    Vertex layout (per side, with the side base offset already applied):
        A[k] = inv_index_map[mp_anchor[k]]   (in [0..468))
        M[k] = LATERAL_MID_START + side_base + k
        O[k] = LATERAL_OUT_START + side_base + k

    Triangles per (k, k+1):
        (A[k], M[k], A[k+1])
        (M[k], M[k+1], A[k+1])
        (M[k], O[k], M[k+1])
        (O[k], O[k+1], M[k+1])
    """
    faces: list[list[tuple[int, int, int]]] = []

    def tri(a: int, b: int, c: int) -> None:
        faces.append([(a, a, a), (b, b, b), (c, c, c)])

    sides = [
        (C.MP_LATERAL_ANCHORS_LEFT,  0),
        (C.MP_LATERAL_ANCHORS_RIGHT, C.N_LATERAL_PER_SIDE),
    ]
    for anchors, side_base in sides:
        A = [inv_index_map[mp] for mp in anchors]
        M = [C.LATERAL_MID_START + side_base + k for k in range(C.N_LATERAL_PER_SIDE)]
        O = [C.LATERAL_OUT_START + side_base + k for k in range(C.N_LATERAL_PER_SIDE)]
        for k in range(C.N_LATERAL_PER_SIDE - 1):
            tri(A[k],   M[k],     A[k + 1])
            tri(M[k],   M[k + 1], A[k + 1])
            tri(M[k],   O[k],     M[k + 1])
            tri(O[k],   O[k + 1], M[k + 1])
    return faces


def main() -> None:
    print(f"[gen-v2] reading {INPUT_OBJ}")
    base = read_obj(INPUT_OBJ)
    assert base.n_v() == C.N_TOTAL, f"expected {C.N_TOTAL} verts, got {base.n_v()}"
    assert base.n_vt() == C.N_TOTAL, f"expected {C.N_TOTAL} texcoords, got {base.n_vt()}"
    assert base.n_vn() == C.N_TOTAL, f"expected {C.N_TOTAL} normals, got {base.n_vn()}"
    base_faces = base.n_f()

    inv = build_inverse_index_map()
    mid_pos, out_pos = canonical_lateral_positions(base, inv)
    mid_uv, out_uv = lateral_uvs()
    ext_normals = lateral_normals()
    ext_faces = build_lateral_faces(inv)

    out = ObjMesh()
    out.header_lines = [
        "# head3d face_ext_v2.obj",
        f"# base = face_ext.obj ({C.N_TOTAL} v) + lateral extension ({C.N_LATERAL_EXT} v)",
        f"# total: {C.N_TOTAL_V2} vertices, {base_faces + len(ext_faces)} triangles",
        "# Layout: [0..468) MediaPipe, [468..485) v1 middle, [485..502) v1 hairline,",
        f"#         [{C.LATERAL_MID_START}..{C.LATERAL_OUT_START}) lateral mid (left 5 + right 5),",
        f"#         [{C.LATERAL_OUT_START}..{C.N_TOTAL_V2}) lateral out (left 5 + right 5)",
        "",
        "mtllib face_ext_v2.mtl",
        "",
        "g default",
    ]
    out.positions = list(base.positions)
    out.texcoords = list(base.texcoords)
    out.normals = list(base.normals)
    out.faces = list(base.faces)

    for i in range(C.N_LATERAL):
        out.positions.append(tuple(map(float, mid_pos[i])))
        out.texcoords.append(mid_uv[i])
        out.normals.append((0.0, 0.0, 1.0))
    for i in range(C.N_LATERAL):
        out.positions.append(tuple(map(float, out_pos[i])))
        out.texcoords.append(out_uv[i])
        out.normals.append((0.0, 0.0, 1.0))

    out.faces.extend(ext_faces)

    assert out.n_v() == C.N_TOTAL_V2, f"output v count: {out.n_v()} != {C.N_TOTAL_V2}"
    assert out.n_vt() == C.N_TOTAL_V2, f"output vt count: {out.n_vt()} != {C.N_TOTAL_V2}"
    assert out.n_vn() == C.N_TOTAL_V2, f"output vn count: {out.n_vn()} != {C.N_TOTAL_V2}"

    print(f"[gen-v2] writing {OUTPUT_OBJ}")
    write_obj(OUTPUT_OBJ, out)
    print(f"[gen-v2]   verts:   {out.n_v()}")
    print(f"[gen-v2]   uvs:     {out.n_vt()}")
    print(f"[gen-v2]   normals: {out.n_vn()}")
    print(f"[gen-v2]   faces:   {out.n_f()}  (added {len(ext_faces)})")
    print(f"[gen-v2]   lateral pos range: x∈[{mid_pos[:,0].min():.3f},{out_pos[:,0].max():.3f}], "
          f"z∈[{out_pos[:,2].min():.3f},{mid_pos[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
