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

# Saggital head-curvature radius (in MediaPipe normalized-Y units),
# expressed as a fraction of face height. The head's mid-line cross
# section is treated locally as a circular arc; for a hairline / middle
# vertex located dy=(y_hair - y_anchor) above an MP top anchor (dy < 0
# since hairline y < anchor y) we compute its Z by
#
#     z_hair = z_anchor + dy² / (2 R),     R = HEAD_ARC_RADIUS_FRAC × face_h
#
# This always pushes the added vertex BACKWARD (toward +z in MP / face.obj
# convention, i.e. toward the back of the head), matching the actual
# anatomy. See README for the derivation.
#
# Smaller fraction = more pronounced backward bulge. 0.30 produces a
# moderate offset (~0.024 in normalized z for a 0.08-y hairline lift on
# a typical face).
HEAD_ARC_RADIUS_FRAC = 0.30


# UV layout for the 34 forehead-extension vertices.
#
# The texture (imgs/texture0.png, 512×512) is laid out with the original
# MediaPipe face skin in V_raw ≈ 0.00..0.77 (image y ≈ 117..511) and a
# horizontal stack of 5 hairline-design arcs at the TOP of the image
# (image y ≈ 30..170, i.e. V_raw ≈ 0.67..0.94). Those 5 arcs are the
# content that the extension strip is supposed to display: hairline row
# samples the topmost arc (blue), middle row samples the bottom arc
# (orange), and the 3 arcs in between fall out automatically because the
# ribbon triangle interpolates V linearly between the two rows.
#
# V conventions
# -------------
# uv_template.py / Three.js (with texture.flipY=true) treat V_raw=1 as
# the TOP of the image (image y=0) and V_raw=0 as the BOTTOM.
#
# U conventions — IMPORTANT
# -------------------------
# The 17 MP_TOP_ANCHORS in face.obj have NON-uniform u (≈ 0.00 at the
# temples, ≈ 0.50 at the forehead center, ≈ 1.00 at the other temple).
# Each ribbon triangle (anchor[i] → middle[i] → anchor[i+1] etc.) is a
# vertical column in UV space ONLY when the middle/hairline vertex
# inherits its U from the corresponding anchor. If we used a uniform
# 0.05..0.95 U for the strip the columns would slant relative to the
# anchor U values, warping the texture's 5 horizontal arcs into
# zig-zags. So `extension_uv_for` takes `anchor_u` and copies it.
UV_MIDDLE_V    = 0.820   # raw OBJ V → image y ≈  92 (magenta / middle arc)
UV_HAIRLINE_V  = 0.940   # raw OBJ V → image y ≈  31 (blue / top arc)
# face.obj 中 17 个 MP_TOP_ANCHORS 的 V_raw 最大值 ≈ 0.7724 (额头中央 MP 10).
# UV_MIDDLE_V 必须 > 0.7724, 否则 anchor → middle 这一段 V 方向会反向, 三角形
# 会被翻转, 贴图弧线在那 3 列出现锯齿/折叠。 当前 0.82 同时落在贴图品红弧线
# (V_raw ≈ 0.824) 上, 蓝/紫/品红 3 条弧线都会原位采到, 红/橙 2 条弧线通过
# anchor 行的 V 插值出现在 anchor 与 middle 之间的过渡带。


def extension_uv_for(row: int, anchor_u: float) -> tuple[float, float]:
    """Return (u, v_raw) UV for an extension vertex.

    row:       0 = middle, 1 = hairline.
    anchor_u:  U of the MP anchor this extension vertex sits above (copy
               it verbatim so the ribbon triangle is vertical in UV space).
    """
    v = UV_MIDDLE_V if row == 0 else UV_HAIRLINE_V
    return (anchor_u, v)


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
