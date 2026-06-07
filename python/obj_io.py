"""Minimal Wavefront OBJ reader/writer.

Tailored to the project's face.obj:
- Latin-1 / GB-encoded comments allowed (we tolerate undecodable bytes).
- Vertices written with 4 decimal places (matches existing file).
- Preserves comment/mtllib/group lines if requested.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ObjMesh:
    """1-indexed in OBJ files; stored 0-indexed internally."""
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    texcoords: list[tuple[float, float]]        = field(default_factory=list)
    normals:   list[tuple[float, float, float]] = field(default_factory=list)
    # Each face is a list of 3 (pos_idx, uv_idx, normal_idx) tuples, 0-indexed.
    # -1 means "absent".
    faces:     list[list[tuple[int, int, int]]] = field(default_factory=list)
    header_lines: list[str] = field(default_factory=list)  # comments / mtllib

    def n_v(self) -> int:  return len(self.positions)
    def n_vt(self) -> int: return len(self.texcoords)
    def n_vn(self) -> int: return len(self.normals)
    def n_f(self) -> int:  return len(self.faces)


def read_obj(path: str) -> ObjMesh:
    """Read an OBJ file, ignoring undecodable bytes in comments."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin-1", errors="replace")
    mesh = ObjMesh()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("mtllib") or line.startswith("o ") or line.startswith("g ") or line.startswith("s "):
            mesh.header_lines.append(line)
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "v":
            mesh.positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif kind == "vt":
            mesh.texcoords.append((float(parts[1]), float(parts[2])))
        elif kind == "vn":
            mesh.normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif kind == "f":
            verts = []
            for spec in parts[1:]:
                seg = spec.split("/")
                pi = int(seg[0]) - 1 if seg[0] else -1
                ti = int(seg[1]) - 1 if len(seg) > 1 and seg[1] else -1
                ni = int(seg[2]) - 1 if len(seg) > 2 and seg[2] else -1
                verts.append((pi, ti, ni))
            # Triangulate fan if quad/n-gon (shouldn't happen for our mesh)
            for k in range(1, len(verts) - 1):
                mesh.faces.append([verts[0], verts[k], verts[k + 1]])
        # ignore everything else
    return mesh


def dumps_obj(mesh: ObjMesh, header: Sequence[str] | None = None) -> str:
    """Serialize an ObjMesh to OBJ text (1-indexed, 4dp)."""
    out: list[str] = []
    if header is not None:
        out.extend(header)
    else:
        out.append("# head3d extended face mesh")
        out.append(f"# vertices={mesh.n_v()} texcoords={mesh.n_vt()} normals={mesh.n_vn()} faces={mesh.n_f()}")

    out.append("")
    for x, y, z in mesh.positions:
        out.append(f"v  {x:.4f} {y:.4f} {z:.4f}")
    out.append("")
    for u, v in mesh.texcoords:
        out.append(f"vt {u:.4f} {v:.4f} 0.0000")
    out.append("")
    for nx, ny, nz in mesh.normals:
        out.append(f"vn {nx:.4f} {ny:.4f} {nz:.4f}")
    out.append("")
    for face in mesh.faces:
        toks = []
        for pi, ti, ni in face:
            a = str(pi + 1)
            b = str(ti + 1) if ti >= 0 else ""
            c = str(ni + 1) if ni >= 0 else ""
            toks.append(f"{a}/{b}/{c}")
        out.append("f " + " ".join(toks))

    return "\n".join(out) + "\n"


def write_obj(path: str, mesh: ObjMesh, header: Sequence[str] | None = None) -> None:
    """Write an OBJ file with the project's conventions (1-indexed, 4dp)."""
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(dumps_obj(mesh, header))
