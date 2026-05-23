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


# Extra lift applied to the detected hairline along the face-up direction,
# expressed as a fraction of the MP face height. The 2D hairline detector
# stops at the hair-skin boundary (start of the visible hair); the mesh
# ribbon's top row should sit at the crown of the head instead, so the
# texture-overlay band can cover the whole forehead → crown region.
#
# 0.10 ≈ moves the hairline row up by 10% of face height (~half the
# distance from the detected hairline to the visible crown on a typical
# frontal photo). Bump it to 0.15..0.20 for taller forehead / higher
# crown; drop to 0.05 to keep the ribbon close to the hair-skin boundary.
HAIRLINE_CROWN_LIFT_FRAC = 0.10


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
UV_MIDDLE_DV   = 0.110   # middle 行 V_raw 相对该列 anchor V_raw 上移这么多
UV_HAIRLINE_DV = 0.220   # hairline 行 V_raw 相对该列 anchor V_raw 上移这么多
# 为什么是"相对 anchor 平行偏移"而不是固定常数:
#
# face.obj 中 17 个 MP_TOP_ANCHORS 的 V_raw 是**非均匀弧形** (额头中央 MP 10
# = 0.7724, 太阳穴 MP 127 = 0.4668, 横跨 0.30 V 单位)。 如果 middle/hairline
# 用固定常数 V_raw (例如 0.82 / 0.998), 那每个 ribbon quad 的 V 跨度
# (= middle.V − anchor[i].V) 在 17 列之间差异巨大 (中央列 0.05, 两端列 0.35,
# 差了 7 倍)。 贴图最底部的弧线 (V_raw ≈ 0.76) 正好落在 V 跨度大的列上 →
# 被拉伸成粗大色块, 而顶部弧线 (V_raw ≈ 0.93) 落在 V 跨度小的列上 → 被压
# 缩成细线。 这就是"最下面那条线特别粗、上面 4 根都细"的根因。
#
# 把 middle/hairline 的 V 也设成"anchor V + 固定 Δ", 每列的 V 跨度变成恒定
# 的 Δm / (Δh − Δm), ribbon 在贴图上是上下都跟随 anchor 弧度的弯月形带,
# 贴图 5 条弧线在 mesh 上粗细均匀。
#
# 硬约束:
#   1. Δm > 0 且 Δm < Δh (顺序保持 anchor < middle < hairline, 防 V 反向)
#   2. anchor.V_max + Δh ≤ 1.0  (即 Δh ≤ 1 − 0.7724 = 0.2276)
#      否则中央列 hairline V 溢出, 采到贴图边缘的抗锯齿像素。
# 当前 Δh = 0.220 留 ≈ 0.008 V 单位 buffer; Δm = Δh / 2 让上下两段等宽。


def extension_uv_for(row: int, anchor_u: float, anchor_v: float) -> tuple[float, float]:
    """Return (u, v_raw) UV for an extension vertex.

    row:       0 = middle, 1 = hairline.
    anchor_u:  U of the corresponding MP anchor (copy verbatim → ribbon column
               is vertical in UV space).
    anchor_v:  V_raw of the corresponding MP anchor (we add a constant Δ to
               it → ribbon row stays parallel to anchor row in UV space, so
               each quad has the same V span and texture arcs render at the
               same thickness across all 17 columns).
    """
    dv = UV_MIDDLE_DV if row == 0 else UV_HAIRLINE_DV
    return (anchor_u, anchor_v + dv)


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
