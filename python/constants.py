"""Shared constants for the hairline-extension pipeline.

These define the topology of the extended mesh and must stay in sync with
the C++ side (see sdk/ExtensionConstants.h). If you change anything here,
regenerate face_ext.obj and update the C++ header.
"""
from __future__ import annotations

# Number of MediaPipe FaceMesh landmarks (no iris refinement).
N_MP = 468

# Anchors along the upper boundary of the MediaPipe face mesh,
# ordered left-to-right when viewing the face frontally.
# Each anchor will get a paired hairline sample directly "above" it.
#
# Verify visually with scripts/show_anchors.py before locking these in.
MP_TOP_ANCHORS: list[int] = [
    127, 234, 162,  21,  54, 103,  67, 109,  10,
    338, 297, 332, 284, 251, 389, 356, 454,
]
N_ANCHORS = len(MP_TOP_ANCHORS)  # 17

# Vertex ID layout in the extended array of length 468 + 2*N_ANCHORS.
#   [0       .. 468)         : MediaPipe canonical landmarks
#   [468     .. 468+N)       : middle row (between MP boundary and hairline)
#   [468+N   .. 468+2N)      : hairline row
N_EXT = 2 * N_ANCHORS  # 34
N_TOTAL = N_MP + N_EXT  # 502

MIDDLE_START   = N_MP             # 468
HAIRLINE_START = N_MP + N_ANCHORS # 485

# Z offset (in MediaPipe normalized Z units) applied to hairline points,
# producing a slight backward curvature at the top of the forehead.
# Center is deepest (most negative), tapering to 0 at the temples.
# Tune by viewing the mesh in Blender.
def curve_offset_z(i: int) -> float:
    """Symmetric forehead curvature offset per-anchor."""
    # i in [0, N_ANCHORS-1], center at i = N_ANCHORS//2
    n = N_ANCHORS
    t = (i - (n - 1) / 2.0) / ((n - 1) / 2.0)  # -1 at left, 0 at center, +1 at right
    # Cosine bell: deepest in the middle
    import math
    return -0.04 * math.cos(t * math.pi / 2.0)


# Middle-row bulge: additional inward Z offset so the forehead has
# convex shape. Same shape as curve_offset_z but smaller magnitude.
def bulge_z(i: int) -> float:
    n = N_ANCHORS
    t = (i - (n - 1) / 2.0) / ((n - 1) / 2.0)
    import math
    return -0.015 * math.cos(t * math.pi / 2.0)


# UV layout: the extension occupies a thin strip near the top edge of UV
# space (V close to 1.0). The strip is two rows (middle / hairline) by
# N_ANCHORS columns. Tune the band so it lands on empty pixels in your
# texture atlas. Each value is a pre-flip (raw OBJ) V; the loader does
# `1 - v` at load time, so the actual top of the texture in image space
# corresponds to a low V value here.
UV_STRIP_U_MIN = 0.05
UV_STRIP_U_MAX = 0.95
UV_MIDDLE_V    = 0.02   # raw OBJ value; near the top of the texture
UV_HAIRLINE_V  = 0.005  # very top edge


def extension_uv_for(row: int, col: int) -> tuple[float, float]:
    """Return (u, v) UV coords for the (row, col) extension vertex.

    row: 0 = middle, 1 = hairline.
    col: 0..N_ANCHORS-1 (left to right).
    """
    u = UV_STRIP_U_MIN + (UV_STRIP_U_MAX - UV_STRIP_U_MIN) * (col / (N_ANCHORS - 1))
    v = UV_MIDDLE_V if row == 0 else UV_HAIRLINE_V
    return (u, v)


# Face-parsing class indices for the jonathandinu/face-parsing
# SegFormer model (matches CelebAMask-HQ labels):
PARSE_BG       = 0
PARSE_SKIN     = 1
PARSE_NOSE     = 2
PARSE_EYE_G    = 3
PARSE_L_EYE    = 4
PARSE_R_EYE    = 5
PARSE_L_BROW   = 6
PARSE_R_BROW   = 7
PARSE_L_EAR    = 8
PARSE_R_EAR    = 9
PARSE_MOUTH    = 10
PARSE_U_LIP    = 11
PARSE_L_LIP    = 12
PARSE_HAIR     = 13
PARSE_HAT      = 14
PARSE_EAR_R    = 15
PARSE_NECK_L   = 16
PARSE_NECK     = 17
PARSE_CLOTH    = 18

HF_FACE_PARSER_MODEL = "jonathandinu/face-parsing"
