"""Generate face_ext.obj from face.obj by appending 34 forehead-extension vertices.

Run once after editing constants.py / indexMap. The resulting OBJ has:
  - 502 vertices: original 468 (unchanged) + 17 middle-row + 17 hairline-row
  - 502 texcoords (parallel-indexed with positions)
  - 502 normals (placeholders; SDK recomputes at runtime)
  - 852 + 64 = 916 triangles

Vertex/UV/normal layout:
  [0   .. 467]  original MediaPipe-canonical vertices (3dsMax export order)
  [468 .. 484]  middle row, left-to-right, indexed identically to MP_TOP_ANCHORS
  [485 .. 501]  hairline row, same ordering as middle row

Usage:
  py -3 -m python.build_extended_obj
or:
  py -3 python/build_extended_obj.py
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np

# Allow running both as a module and as a script.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.obj_io import ObjMesh, read_obj, write_obj
    from python._index_map_data import INDEX_MAP_468
else:
    from . import constants as C
    from .obj_io import ObjMesh, read_obj, write_obj
    from ._index_map_data import INDEX_MAP_468


THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
INPUT_OBJ   = os.path.join(PROJECT_DIR, "face.obj")
OUTPUT_OBJ  = os.path.join(PROJECT_DIR, "face_ext.obj")


def build_inverse_index_map() -> dict[int, int]:
    """Return MP-canonical landmark index -> OBJ vertex index (0-based)."""
    inv: dict[int, int] = {}
    for obj_idx, mp_idx in enumerate(INDEX_MAP_468):
        if mp_idx in inv:
            raise ValueError(f"Duplicate MP index {mp_idx} in indexMap at OBJ {obj_idx}")
        inv[mp_idx] = obj_idx
    return inv


def canonical_extension_positions(
    mesh: ObjMesh,
    inv_index_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute default 3D positions for the 17 middle + 17 hairline vertices.

    Lift Y upward by a fraction of face height (face.obj has +Y growing
    DOWN, so "up" is -Y). Z is computed with the same sagittal-arc model
    used at runtime in ``lift_3d.lift_hairline_to_3d`` so the canonical
    mesh and the per-image detected mesh share the same surface law:

        z_new = z_anchor + (y_new - y_anchor)² / (2 R),
        R     = HEAD_ARC_RADIUS_FRAC × face_height

    This always pushes the new vertex BACKWARD (+Z in face.obj convention
    = back of head), so middle / hairline points sit on the head surface
    rather than floating in front of the face.
    """
    pos = np.array(mesh.positions, dtype=np.float32)  # (468, 3)
    min_y = float(pos[:, 1].min())
    max_y = float(pos[:, 1].max())
    face_h = max_y - min_y

    middle_lift  = 0.05 * face_h
    hairline_lift = 0.12 * face_h
    R = max(1e-6, C.HEAD_ARC_RADIUS_FRAC * face_h)

    middle   = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    hairline = np.zeros((C.N_ANCHORS, 3), dtype=np.float32)
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        obj_idx = inv_index_map[mp_idx]
        a = pos[obj_idx]
        y_m = a[1] - middle_lift
        y_h = a[1] - hairline_lift
        # dy is negative (above anchor); dy² / (2R) is positive.
        z_m = float(a[2]) + (y_m - float(a[1])) ** 2 / (2.0 * R)
        z_h = float(a[2]) + (y_h - float(a[1])) ** 2 / (2.0 * R)
        middle[i]   = (a[0], y_m, z_m)
        hairline[i] = (a[0], y_h, z_h)
    return middle, hairline


def extension_uvs() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return UVs for middle row and hairline row, in OBJ raw V format."""
    middle_uv   = [C.extension_uv_for(0, i) for i in range(C.N_ANCHORS)]
    hairline_uv = [C.extension_uv_for(1, i) for i in range(C.N_ANCHORS)]
    return middle_uv, hairline_uv


def extension_normals() -> list[tuple[float, float, float]]:
    """Placeholder normals; SDK recomputes from face geometry at runtime."""
    return [(0.0, 0.0, 1.0)] * C.N_EXT


def build_extension_faces(inv_index_map: dict[int, int]) -> list[list[tuple[int, int, int]]]:
    """Build triangle list for the forehead extension strip.

    For each adjacent anchor pair (i, i+1) we produce 4 triangles forming a
    quad-quad strip: anchor row -> middle row -> hairline row.

    Triangle vertex indices are 0-indexed into the OBJ's combined vertex
    array (length 502). UV and normal indices are parallel (same number).
    """
    A: list[int] = [inv_index_map[mp] for mp in C.MP_TOP_ANCHORS]
    M: list[int] = [C.MIDDLE_START   + i for i in range(C.N_ANCHORS)]
    H: list[int] = [C.HAIRLINE_START + i for i in range(C.N_ANCHORS)]

    faces: list[list[tuple[int, int, int]]] = []

    def tri(a: int, b: int, c: int) -> None:
        faces.append([(a, a, a), (b, b, b), (c, c, c)])

    for i in range(C.N_ANCHORS - 1):
        # Lower band: anchor row -> middle row
        tri(A[i],   M[i],   A[i + 1])
        tri(M[i],   M[i + 1], A[i + 1])
        # Upper band: middle row -> hairline row
        tri(M[i],   H[i],   M[i + 1])
        tri(H[i],   H[i + 1], M[i + 1])
    return faces


def main() -> None:
    print(f"[build] reading {INPUT_OBJ}")
    base = read_obj(INPUT_OBJ)
    assert base.n_v() == C.N_MP, f"expected 468 verts, got {base.n_v()}"
    assert base.n_vt() == C.N_MP, f"expected 468 texcoords, got {base.n_vt()}"
    assert base.n_vn() == C.N_MP, f"expected 468 normals, got {base.n_vn()}"
    assert base.n_f() == 852, f"expected 852 faces, got {base.n_f()}"

    inv = build_inverse_index_map()

    # Compute extension data
    middle_pos, hairline_pos = canonical_extension_positions(base, inv)
    middle_uv,  hairline_uv  = extension_uvs()
    ext_normals = extension_normals()
    ext_faces   = build_extension_faces(inv)

    # Assemble the extended mesh
    out = ObjMesh()
    out.header_lines = [
        "# head3d face_ext.obj",
        f"# base = face.obj (468 v) + forehead extension (middle {C.N_ANCHORS} + hairline {C.N_ANCHORS})",
        f"# total: {C.N_TOTAL} vertices, {852 + len(ext_faces)} triangles",
        "# Layout: [0..467] MediaPipe canonical, [468..484] middle row, [485..501] hairline row",
        "",
        "mtllib face_ext.mtl",
        "",
        "g default",
    ]
    out.positions = list(base.positions)
    out.texcoords = list(base.texcoords)
    out.normals   = list(base.normals)
    out.faces     = list(base.faces)

    # Append middle row, then hairline row
    for i in range(C.N_ANCHORS):
        out.positions.append(tuple(map(float, middle_pos[i])))
        out.texcoords.append(middle_uv[i])
        out.normals.append((0.0, 0.0, 1.0))
    for i in range(C.N_ANCHORS):
        out.positions.append(tuple(map(float, hairline_pos[i])))
        out.texcoords.append(hairline_uv[i])
        out.normals.append((0.0, 0.0, 1.0))

    # Append extension faces
    out.faces.extend(ext_faces)

    # Sanity check
    assert out.n_v() == C.N_TOTAL,  f"output v count mismatch: {out.n_v()}"
    assert out.n_vt() == C.N_TOTAL, f"output vt count mismatch: {out.n_vt()}"
    assert out.n_vn() == C.N_TOTAL, f"output vn count mismatch: {out.n_vn()}"

    print(f"[build] writing {OUTPUT_OBJ}")
    write_obj(OUTPUT_OBJ, out)
    print(f"[build]   verts:  {out.n_v()}")
    print(f"[build]   uvs:    {out.n_vt()}")
    print(f"[build]   normals:{out.n_vn()}")
    print(f"[build]   faces:  {out.n_f()}  (added {len(ext_faces)})")


if __name__ == "__main__":
    main()
