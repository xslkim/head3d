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


# ---------------------------------------------------------------------------
# v2-headext: lateral extension along temples / cheeks
# ---------------------------------------------------------------------------
#
# Lateral anchor chain on each side, ordered TOP -> BOTTOM
# (太阳穴上→颧骨外侧→耳前)。
#
#   MP 127 / 356 : 太阳穴上, 同时是 v1-hairline 的 1/17 号锚点
#   MP 234 / 454 : 太阳穴下, 同时是 v1-hairline 的 2/16 号锚点
#   MP  93 / 323 : 颧弓外侧上
#   MP 132 / 361 : 颧弓外侧中 / 耳前上
#   MP  58 / 288 : 耳前下 / 腮上 (ribbon 底部终点)
MP_LATERAL_ANCHORS_LEFT:  list[int] = [127, 234,  93, 132,  58]
MP_LATERAL_ANCHORS_RIGHT: list[int] = [356, 454, 323, 361, 288]
N_LATERAL_PER_SIDE = len(MP_LATERAL_ANCHORS_LEFT)  # 5
N_LATERAL = 2 * N_LATERAL_PER_SIDE  # 10 (per row, left+right concatenated)
N_LATERAL_EXT = 2 * N_LATERAL       # 20 (mid row + out row)

# Vertex ID layout (v2):
#   [0       .. 468)         MediaPipe canonical landmarks
#   [468     .. 485)         v1 middle row 17
#   [485     .. 502)         v1 hairline row 17
#   [502     .. 512)         lateral middle row (left 5 + right 5)
#   [512     .. 522)         lateral outer row (left 5 + right 5)
LATERAL_MID_START = N_TOTAL              # 502
LATERAL_OUT_START = N_TOTAL + N_LATERAL  # 512
N_TOTAL_V2 = N_TOTAL + N_LATERAL_EXT     # 522


# v2-headext UV: occupies the unused TOP region of the texture image
# (image y=[0, 116] in 512px = OBJ V_raw ∈ [0.77, 1.0]). Existing
# texture0.png has only transparent pixels there, so this band can be
# used without conflicting with current content. See PLAN_headext.md §5.2.
UV_LATERAL_OUT_V       = 0.92   # raw OBJ V; outer ribbon → image y ≈ 41
UV_LATERAL_MID_V       = 0.86   # raw OBJ V; middle ribbon → image y ≈ 72
UV_LATERAL_U_LEFT_MIN  = 0.05
UV_LATERAL_U_LEFT_MAX  = 0.45
UV_LATERAL_U_RIGHT_MIN = 0.55
UV_LATERAL_U_RIGHT_MAX = 0.95


def lateral_uv_for(row: int, side: str, col: int) -> tuple[float, float]:
    """Return (u, v_raw) UV for the lateral extension vertex.

    row:  0 = middle, 1 = outer.
    side: 'left' or 'right'.
    col:  0..N_LATERAL_PER_SIDE-1, top-to-bottom along the lateral chain
          (i.e. col=0 is closest to the temple top, col=4 is closest to ear).
    """
    if side == "left":
        u_min, u_max = UV_LATERAL_U_LEFT_MIN, UV_LATERAL_U_LEFT_MAX
    elif side == "right":
        u_min, u_max = UV_LATERAL_U_RIGHT_MIN, UV_LATERAL_U_RIGHT_MAX
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    u = u_min + (u_max - u_min) * (col / (N_LATERAL_PER_SIDE - 1))
    v = UV_LATERAL_MID_V if row == 0 else UV_LATERAL_OUT_V
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
