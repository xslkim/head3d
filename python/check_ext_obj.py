"""Self-test: read face_ext.obj back and verify counts + topology integrity.

Usage:
  python python/check_ext_obj.py             # face_ext.obj, 502 verts
"""
from __future__ import annotations
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from python import constants as C
    from python.obj_io import read_obj
    from python._index_map_data import INDEX_MAP_468  # noqa: F401
    from python.build_extended_obj import build_inverse_index_map
else:
    from . import constants as C
    from .obj_io import read_obj
    from ._index_map_data import INDEX_MAP_468  # noqa: F401
    from .build_extended_obj import build_inverse_index_map


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _check_index_ranges(failures: list[str], mesh, n_total: int) -> None:
    for fi, face in enumerate(mesh.faces):
        for vi, (pi, ti, ni) in enumerate(face):
            if not (0 <= pi < n_total):
                failures.append(f"face[{fi}][{vi}] pos index out of range: {pi}")
            if ti >= 0 and not (0 <= ti < n_total):
                failures.append(f"face[{fi}][{vi}] uv index out of range: {ti}")
            if ni >= 0 and not (0 <= ni < n_total):
                failures.append(f"face[{fi}][{vi}] normal index out of range: {ni}")


def _check_first_n_positions_match(failures: list[str], a_pos, b_pos, n: int, label: str) -> None:
    for i in range(n):
        if a_pos[i] != b_pos[i]:
            failures.append(f"{label}[{i}] changed: {b_pos[i]} -> {a_pos[i]}")
            return


def _check(base_face_obj_path: str, ext_obj_path: str) -> int:
    base = read_obj(base_face_obj_path)
    ext = read_obj(ext_obj_path)

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

    _check_first_n_positions_match(failures, ext.positions, base.positions, C.N_MP, "position")
    _check_index_ranges(failures, ext, C.N_TOTAL)

    used = set()
    for face in ext.faces:
        for pi, _, _ in face:
            used.add(pi)
    for i in range(C.N_MP, C.N_TOTAL):
        if i not in used:
            failures.append(f"extension vertex {i} not used in any face")

    inv = build_inverse_index_map()
    anchor_obj_ids = {inv[mp] for mp in C.MP_TOP_ANCHORS}
    ext_face_count = 0
    for face in ext.faces[852:]:
        verts = {face[0][0], face[1][0], face[2][0]}
        if not verts & {i for i in range(C.N_MP, C.N_TOTAL)} and not verts & anchor_obj_ids:
            failures.append(f"extension face has no extension/anchor verts: {face}")
        ext_face_count += 1
    expected_ext = 4 * (C.N_ANCHORS - 1)
    if ext_face_count != expected_ext:
        failures.append(f"extension face count: {ext_face_count} != {expected_ext}")

    # Sanity check: every extension vertex's Z must be >= its source anchor's Z
    # (sagittal-arc model: head curves backward as you walk up, never forward).
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_obj_idx = inv[mp_idx]
        anchor_z = ext.positions[anchor_obj_idx][2]
        for slot, name in ((C.MIDDLE_START + i, f"middle[{i}]"),
                           (C.HAIRLINE_START + i, f"hairline[{i}]")):
            new_z = ext.positions[slot][2]
            if new_z < anchor_z - 1e-5:
                failures.append(
                    f"{name} (slot {slot}) z={new_z:+.4f} < anchor MP{mp_idx} "
                    f"z={anchor_z:+.4f}; should be >= anchor z (head curves backward)"
                )

    if failures:
        print("FAIL")
        for s in failures:
            print(" -", s)
        return 1
    print(f"OK  v={ext.n_v()} vt={ext.n_vt()} vn={ext.n_vn()} f={ext.n_f()}  "
          f"(forehead extension: {C.N_EXT} verts, {ext_face_count} tris)")
    return 0


def main() -> int:
    face_obj = os.path.join(PROJECT_DIR, "face.obj")
    ext_obj = os.path.join(PROJECT_DIR, "face_ext.obj")
    return _check(face_obj, ext_obj)


if __name__ == "__main__":
    raise SystemExit(main())
