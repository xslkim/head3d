#!/usr/bin/env python3
"""Step 4: build the extended OBJ template (486 verts) from canonical face.obj.

    python build_template.py [--t1 0.8] [--t2 1.6] [-o face_ext.obj]

The template is the canonical mesh with the 18 forehead-extension vertices,
their UVs, and the ribbon faces added. Per-photo deformation (step 5) reuses
the same algorithm; this file is just the static artifact + a sanity check.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.extend_mesh import N_NEW, T1_DEFAULT, T2_DEFAULT, extend_obj_mesh
from python.obj_io import read_obj, write_obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="face.obj")
    ap.add_argument("-o", "--output", default="face_ext.obj")
    ap.add_argument("--t1", type=float, default=T1_DEFAULT)
    ap.add_argument("--t2", type=float, default=T2_DEFAULT)
    args = ap.parse_args()

    base = read_obj(args.input)
    print(f"base: v={base.n_v()} vt={base.n_vt()} vn={base.n_vn()} f={base.n_f()}")

    ext = extend_obj_mesh(base, args.t1, args.t2)
    print(f"ext : v={ext.n_v()} vt={ext.n_vt()} vn={ext.n_vn()} f={ext.n_f()}")

    assert ext.n_v() == base.n_v() + N_NEW, "vertex count mismatch"
    assert ext.n_f() == base.n_f() + 32, "expected 32 new ribbon faces"

    write_obj(args.output, ext)
    print(f"wrote {args.output}  (t1={args.t1} t2={args.t2})")


if __name__ == "__main__":
    main()
