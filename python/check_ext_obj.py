"""Self-test: read face_ext{,_v2}.obj back and verify counts + topology integrity.

Usage:
  python python/check_ext_obj.py             # v1 (face_ext.obj, 502 verts)
  python python/check_ext_obj.py --v2        # v2 (face_ext_v2.obj, 522 verts)
  python python/check_ext_obj.py --all       # both
"""
from __future__ import annotations
import argparse
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


def _check_v1(base_face_obj_path: str, ext_obj_path: str) -> int:
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

    if failures:
        print("[v1] FAIL")
        for s in failures:
            print(" -", s)
        return 1
    print(f"[v1] OK  v={ext.n_v()} vt={ext.n_vt()} vn={ext.n_vn()} f={ext.n_f()}  "
          f"(forehead extension: {C.N_EXT} verts, {ext_face_count} tris)")
    return 0


def _check_v2(v1_obj_path: str, v2_obj_path: str) -> int:
    v1 = read_obj(v1_obj_path)
    v2 = read_obj(v2_obj_path)
    failures: list[str] = []

    if v2.n_v() != C.N_TOTAL_V2:
        failures.append(f"v count: {v2.n_v()} != {C.N_TOTAL_V2}")
    if v2.n_vt() != C.N_TOTAL_V2:
        failures.append(f"vt count: {v2.n_vt()} != {C.N_TOTAL_V2}")
    if v2.n_vn() != C.N_TOTAL_V2:
        failures.append(f"vn count: {v2.n_vn()} != {C.N_TOTAL_V2}")

    base_face_count = v1.n_f()
    expected_lateral_tris = 4 * (C.N_LATERAL_PER_SIDE - 1) * 2   # 4 tris × 4 pairs × 2 sides = 32
    expected_total_faces = base_face_count + expected_lateral_tris
    if v2.n_f() != expected_total_faces:
        failures.append(f"face count: {v2.n_f()} != {expected_total_faces}")

    # First N_TOTAL positions (== full v1) must match exactly.
    _check_first_n_positions_match(failures, v2.positions, v1.positions, C.N_TOTAL, "position")

    _check_index_ranges(failures, v2, C.N_TOTAL_V2)

    # Every v2 lateral vertex must be used in some face.
    used = set()
    for face in v2.faces:
        for pi, _, _ in face:
            used.add(pi)
    for i in range(C.N_TOTAL, C.N_TOTAL_V2):
        if i not in used:
            failures.append(f"lateral vertex {i} not used in any face")

    # Each lateral face must reference at least one lateral vertex or
    # a lateral MP anchor.
    inv = build_inverse_index_map()
    lateral_anchor_obj_ids = {
        inv[mp] for mp in (C.MP_LATERAL_ANCHORS_LEFT + C.MP_LATERAL_ANCHORS_RIGHT)
    }
    lateral_ids = set(range(C.N_TOTAL, C.N_TOTAL_V2))
    lateral_face_count = 0
    for face in v2.faces[base_face_count:]:
        verts = {face[0][0], face[1][0], face[2][0]}
        if not (verts & lateral_ids) and not (verts & lateral_anchor_obj_ids):
            failures.append(f"lateral face has no lateral-extension/anchor verts: {face}")
        lateral_face_count += 1
    if lateral_face_count != expected_lateral_tris:
        failures.append(f"lateral face count: {lateral_face_count} != {expected_lateral_tris}")

    # Lateral UVs must land in the unused top band (V_raw > 0.77 == image y < 116).
    for i in range(C.N_TOTAL, C.N_TOTAL_V2):
        _, v_raw = v2.texcoords[i]
        if v_raw < 0.77:
            failures.append(
                f"lateral UV[{i}] v_raw={v_raw:.3f} not in reserved top band (>=0.77)"
            )

    if failures:
        print("[v2] FAIL")
        for s in failures:
            print(" -", s)
        return 1
    print(f"[v2] OK  v={v2.n_v()} vt={v2.n_vt()} vn={v2.n_vn()} f={v2.n_f()}  "
          f"(lateral extension: {C.N_LATERAL_EXT} verts, {lateral_face_count} tris)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify face_ext.obj / face_ext_v2.obj integrity.")
    ap.add_argument("--v2", action="store_true", help="check face_ext_v2.obj only")
    ap.add_argument("--all", action="store_true", help="check both face_ext.obj and face_ext_v2.obj")
    args = ap.parse_args()

    face_obj = os.path.join(PROJECT_DIR, "face.obj")
    ext_obj = os.path.join(PROJECT_DIR, "face_ext.obj")
    ext_v2_obj = os.path.join(PROJECT_DIR, "face_ext_v2.obj")

    rc = 0
    if args.v2:
        rc |= _check_v2(ext_obj, ext_v2_obj)
    elif args.all:
        rc |= _check_v1(face_obj, ext_obj)
        rc |= _check_v2(ext_obj, ext_v2_obj)
    else:
        rc |= _check_v1(face_obj, ext_obj)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
