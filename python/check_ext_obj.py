"""Self-test: read face_ext.obj back and verify counts + topology integrity."""
from __future__ import annotations
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.obj_io import read_obj
    from python._index_map_data import INDEX_MAP_468
    from python.build_extended_obj import build_inverse_index_map
else:
    from . import constants as C
    from .obj_io import read_obj
    from ._index_map_data import INDEX_MAP_468
    from .build_extended_obj import build_inverse_index_map


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = read_obj(os.path.join(root, "face.obj"))
    ext  = read_obj(os.path.join(root, "face_ext.obj"))

    failures: list[str] = []

    if ext.n_v() != C.N_TOTAL:
        failures.append(f"v count: {ext.n_v()} != {C.N_TOTAL}")
    if ext.n_vt() != C.N_TOTAL:
        failures.append(f"vt count: {ext.n_vt()} != {C.N_TOTAL}")
    if ext.n_vn() != C.N_TOTAL:
        failures.append(f"vn count: {ext.n_vn()} != {C.N_TOTAL}")

    expected_faces = 852 + 4 * (C.N_ANCHORS - 1)
    if ext.n_f() != expected_faces:
        failures.append(f"face count: {ext.n_f()} != {expected_faces}")

    # First 468 positions must match the original exactly.
    for i in range(C.N_MP):
        if ext.positions[i] != base.positions[i]:
            failures.append(f"position[{i}] changed: {base.positions[i]} -> {ext.positions[i]}")
            break

    # All face indices must be in range.
    for fi, face in enumerate(ext.faces):
        for vi, (pi, ti, ni) in enumerate(face):
            if not (0 <= pi < C.N_TOTAL):
                failures.append(f"face[{fi}][{vi}] pos index out of range: {pi}")
            if ti >= 0 and not (0 <= ti < C.N_TOTAL):
                failures.append(f"face[{fi}][{vi}] uv index out of range: {ti}")
            if ni >= 0 and not (0 <= ni < C.N_TOTAL):
                failures.append(f"face[{fi}][{vi}] normal index out of range: {ni}")

    # Every extension vertex must appear in at least one face.
    used = set()
    for face in ext.faces:
        for pi, _, _ in face:
            used.add(pi)
    for i in range(C.N_MP, C.N_TOTAL):
        if i not in used:
            failures.append(f"extension vertex {i} not used in any face")

    # Each extension face must reference at least one extension vertex.
    inv = build_inverse_index_map()
    anchor_obj_ids = {inv[mp] for mp in C.MP_TOP_ANCHORS}
    ext_face_count = 0
    for face in ext.faces[852:]:
        verts = {face[0][0], face[1][0], face[2][0]}
        if not verts & {i for i in range(C.N_MP, C.N_TOTAL)} and not verts & anchor_obj_ids:
            failures.append(f"extension face has no extension/anchor verts: {face}")
        ext_face_count += 1
    if ext_face_count != 4 * (C.N_ANCHORS - 1):
        failures.append(f"extension face count: {ext_face_count} != {4 * (C.N_ANCHORS - 1)}")

    if failures:
        print("FAIL")
        for s in failures:
            print(" -", s)
        return 1
    print(f"OK  v={ext.n_v()} vt={ext.n_vt()} vn={ext.n_vn()} f={ext.n_f()}  (extension: 34 verts, {ext_face_count} tris)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
