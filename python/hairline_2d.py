"""2D hairline detection from a face-parsing label map.

Given the parser output, find the boundary curve between hair-above and
skin-below in the upper face region, then sample one point per anchor
along the face's local "up" direction.

This module also exposes several alternative strategies (see
``sample_hairline_*`` functions) that the Web service runs side-by-side
for visual comparison.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from . import constants as C


def face_up_vector(landmarks_norm: np.ndarray) -> np.ndarray:
    """Return a unit 2D vector pointing "up" in image space.

    landmarks_norm: (468, 3) MediaPipe output (normalized x,y in [0,1]).
    Uses chin (152) → forehead-top (10).
    """
    chin = landmarks_norm[152, :2]
    top  = landmarks_norm[10,  :2]
    v = top - chin   # in normalized image space
    n = np.linalg.norm(v) + 1e-9
    return v / n


def build_hair_mask(parse_map: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of pixels strictly classified as hair (or hat)."""
    return np.isin(parse_map, [C.PARSE_HAIR, C.PARSE_HAT])


def build_skin_mask(parse_map: np.ndarray) -> np.ndarray:
    """Boolean (H, W) mask of pixels classified as facial skin / ears / brows / nose / lips.

    Used to validate that a ray actually travels through the face before
    hitting hair — guards against the case where the ray exits laterally
    into background (e.g. from a temple anchor next to white background)
    and would otherwise stop at the first non-skin pixel.
    """
    face_classes = [
        C.PARSE_SKIN, C.PARSE_NOSE, C.PARSE_L_BROW, C.PARSE_R_BROW,
        C.PARSE_L_EYE, C.PARSE_R_EYE, C.PARSE_EYE_G,
        C.PARSE_L_EAR, C.PARSE_R_EAR, C.PARSE_EAR_R,
        C.PARSE_MOUTH, C.PARSE_U_LIP, C.PARSE_L_LIP,
    ]
    return np.isin(parse_map, face_classes)


def ray_first_hair_hit(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    hair_mask: np.ndarray,
    max_steps: int = 800,
    step_px: float = 1.0,
) -> np.ndarray | None:
    """March a ray until it hits a hair pixel. Returns float (x, y) or None.

    Unlike a generic boundary tracer, we explicitly target the HAIR class so
    a ray that drifts into background while traversing a temple anchor will
    continue searching rather than stopping at the face/background edge.
    """
    H, W = hair_mask.shape
    pos = origin_xy.astype(np.float64).copy()
    for _ in range(max_steps):
        pos += direction_xy * step_px
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return None
        if hair_mask[iy, ix]:
            return pos.copy()
    return None


def sample_hairline(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    fallback_extrapolation: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Find one hairline point per MP_TOP_ANCHORS, in normalized [0,1] image space.

    Returns:
        hairline_norm: (N_ANCHORS, 2) float — hairline points (x_norm, y_norm)
        valid_mask:    (N_ANCHORS,)  bool  — True where a hair-pixel was hit

    Algorithm:
      For each anchor, march a ray along the face's up-vector and return the
      first hair-classified pixel encountered. If the ray exits the image
      without ever finding hair (bald, hat, or hairline outside the frame),
      fall back to extrapolating along the up-vector by `fallback_extrapolation`.

      For lateral anchors (temples, ears) the ray may leave the face into
      background skin-less area before encountering hair on top of the
      head — that is OK because we only stop on hair, not background.
    """
    H, W = parse_map.shape
    hair_mask = build_hair_mask(parse_map)

    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    up_px = up_px / (np.linalg.norm(up_px) + 1e-9)

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        hit_px = ray_first_hair_hit(anchor_px, up_px, hair_mask)
        if hit_px is not None:
            hairline[i] = (hit_px / np.array([W, H])).astype(np.float32)
            valid[i] = True
        else:
            hairline[i] = (anchor_norm + up_norm * fallback_extrapolation).astype(np.float32)
            valid[i] = False
    return hairline, valid


# Backwards-compat alias for callers from the previous API.
def build_hairline_mask(parse_map: np.ndarray) -> np.ndarray:
    return build_hair_mask(parse_map)


def smooth_hairline(
    hairline_norm: np.ndarray,
    valid: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    """Light smoothing along the curve to absorb segmentation jitter.

    Uses a 1-2-1 binomial filter over neighbors with the same validity.
    Note: this also rounds off real corners. For data with sharp corners
    (e.g. M-shaped hairlines), use :func:`smooth_hairline_corner_aware` instead.
    """
    out = hairline_norm.copy()
    n = len(out)
    for _ in range(iterations):
        new = out.copy()
        for i in range(1, n - 1):
            if not valid[i]:
                continue
            new[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        out = new
    return out


def smooth_hairline_corner_aware(
    hairline_norm: np.ndarray,
    valid: np.ndarray,
    iterations: int = 2,
    corner_cos_threshold: float = 0.6,
) -> np.ndarray:
    """1-2-1 smoothing, but skip points that look like corners.

    A "corner" is detected by the cosine of the angle between (P[i] - P[i-1])
    and (P[i+1] - P[i]): smaller cosine = sharper bend. If
    ``cos(angle) < corner_cos_threshold`` (i.e. bend > ~53°), point i keeps
    its raw position; otherwise the binomial filter is applied.

    This preserves the user's expected M-shape / trapezoidal hairline whose
    "shoulders" are sharp corners.
    """
    out = hairline_norm.copy()
    n = len(out)
    for _ in range(iterations):
        new = out.copy()
        for i in range(1, n - 1):
            if not valid[i]:
                continue
            v1 = out[i] - out[i - 1]
            v2 = out[i + 1] - out[i]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-9 or n2 < 1e-9:
                new[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
                continue
            cos_a = float(np.dot(v1, v2)) / (n1 * n2)
            if cos_a >= corner_cos_threshold:
                # smooth: not a corner
                new[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
            # else: keep raw (preserve corner)
        out = new
    return out


# ---------------------------------------------------------------------------
# Alternative hairline sampling strategies.
#
# All functions share the signature ``(landmarks_norm, parse_map) -> (hairline,
# valid)`` so the caller can switch strategies uniformly. ``landmarks_norm``
# may be ``None`` for parsing-only strategies (e.g. ``sample_hairline_top_arc``).
# ---------------------------------------------------------------------------


def head_top_target(landmarks_norm: np.ndarray, scale: float = 0.6) -> np.ndarray:
    """Return a 2D point above the forehead used as a ray convergence target.

    The point is placed `scale * face_height` above the forehead-top landmark
    (10), measured along the chin->forehead axis. ``scale = 0.6`` roughly
    matches the human "crown" position relative to the chin->forehead line.
    """
    chin = landmarks_norm[152, :2]
    top = landmarks_norm[10, :2]
    face_height = float(np.linalg.norm(top - chin)) + 1e-9
    face_up_unit = (top - chin) / face_height
    return top + face_up_unit * face_height * scale


def build_face_adjacent_hair_mask(
    parse_map: np.ndarray,
    dilate_radius: int | None = None,
) -> np.ndarray:
    """Hair pixels that lie within ``dilate_radius`` px of any face-skin pixel.

    Used to filter out background blobs that the parser mis-classified as
    hair (chairs, dark walls, etc.) which would otherwise stop a ray prematurely.
    """
    import cv2

    hair = np.isin(parse_map, [C.PARSE_HAIR, C.PARSE_HAT]).astype(np.uint8)
    face = build_skin_mask(parse_map).astype(np.uint8)

    if dilate_radius is None:
        face_area = int(face.sum())
        dilate_radius = max(6, int(np.sqrt(max(face_area, 1)) * 0.05))

    k = 2 * dilate_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    face_dil = cv2.dilate(face, kernel)
    return (hair > 0) & (face_dil > 0)


def ray_first_mask_hit(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    mask: np.ndarray,
    max_steps: int = 1500,
    step_px: float = 1.0,
) -> np.ndarray | None:
    """Generic version of :func:`ray_first_hair_hit` that targets any boolean mask."""
    H, W = mask.shape
    pos = origin_xy.astype(np.float64).copy()
    for _ in range(max_steps):
        pos += direction_xy * step_px
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        if ix < 0 or iy < 0 or ix >= W or iy >= H:
            return None
        if mask[iy, ix]:
            return pos.copy()
    return None


def ray_first_dense_hit(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    mask: np.ndarray,
    run_length: int,
    max_steps: int = 2000,
) -> np.ndarray | None:
    """Density-based stopping: stop only after `run_length` consecutive hit pixels.

    Returns the position of the FIRST pixel of the qualifying run (i.e., the
    edge where the dense region begins). This filters out single-pixel
    classification noise and isolated wisps.

    The ray is allowed to start outside the image and walk in until it enters
    the image bounds (used by the "from-above" strategies). Once inside, if the
    ray exits the image again, the search stops.
    """
    H, W = mask.shape
    pos = np.asarray(origin_xy, dtype=np.float64).copy()
    direction = np.asarray(direction_xy, dtype=np.float64)
    consecutive = 0
    run_start: np.ndarray | None = None
    entered_image = False

    for _ in range(max_steps):
        pos += direction
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        in_image = (0 <= ix < W) and (0 <= iy < H)
        if not in_image:
            if entered_image:
                return None
            continue
        entered_image = True
        if mask[iy, ix]:
            if consecutive == 0:
                run_start = pos.copy()
            consecutive += 1
            if consecutive >= run_length:
                return run_start
        else:
            consecutive = 0
            run_start = None
    return None


def _face_height_px(landmarks_norm: np.ndarray, W: int, H: int) -> float:
    chin = landmarks_norm[152, :2] * np.array([W, H], dtype=np.float64)
    top = landmarks_norm[10, :2] * np.array([W, H], dtype=np.float64)
    return float(np.linalg.norm(top - chin))


def density_run_length(landmarks_norm: np.ndarray, W: int, H: int, ratio: float = 0.06) -> int:
    """Adaptive density threshold proportional to face size.

    Default ratio = 0.06 of the chin-to-forehead distance: for a 200 px face
    height that gives a 12-pixel run requirement; for 400 px → 24 pixels.
    """
    fh = _face_height_px(landmarks_norm, W, H)
    return max(6, int(round(fh * ratio)))


def _ray_sample_with_direction(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    mask: np.ndarray,
    direction_factory: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
    fallback_extrapolation: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Common ray-marching loop. ``direction_factory`` returns a unit pixel-space
    direction for a given anchor (passed `(landmarks, anchor_px, W, H)`)."""
    H, W = parse_map.shape
    up_norm_fallback = face_up_vector(landmarks_norm)

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        direction_px = direction_factory(landmarks_norm, anchor_px, W, H)
        hit_px = ray_first_mask_hit(anchor_px, direction_px, mask)
        if hit_px is not None:
            hairline[i] = (hit_px / np.array([W, H])).astype(np.float32)
            valid[i] = True
        else:
            hairline[i] = (anchor_norm + up_norm_fallback * fallback_extrapolation).astype(np.float32)
            valid[i] = False
    return hairline, valid


def _direction_face_up(landmarks_norm, anchor_px, W, H) -> np.ndarray:
    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    return up_px / (np.linalg.norm(up_px) + 1e-9)


def _direction_converging(landmarks_norm, anchor_px, W, H, scale: float = 0.6) -> np.ndarray:
    target_norm = head_top_target(landmarks_norm, scale=scale)
    target_px = target_norm * np.array([W, H], dtype=np.float64)
    d = target_px - anchor_px
    return d / (np.linalg.norm(d) + 1e-9)


def sample_hairline_baseline(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 1: 当前算法 - 沿 face-up 直上, 命中首个 hair/hat 像素。"""
    mask = build_hair_mask(parse_map)
    return _ray_sample_with_direction(landmarks_norm, parse_map, mask, _direction_face_up)


def sample_hairline_adjacent(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 2: baseline + hair 必须邻接 face skin, 过滤背景误分。"""
    mask = build_face_adjacent_hair_mask(parse_map)
    return _ray_sample_with_direction(landmarks_norm, parse_map, mask, _direction_face_up)


def _topmost_hair_y_per_col(mask: np.ndarray) -> np.ndarray:
    """For each column return the smallest y where ``mask[y, x]`` is True, or -1."""
    H, W = mask.shape
    cols_has = mask.any(axis=0)
    out = np.full(W, -1, dtype=np.int32)
    cols = np.where(cols_has)[0]
    for x in cols:
        out[x] = int(np.argmax(mask[:, x]))
    return out


def sample_hairline_window(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    window_lateral_ratio: float = 0.05,
    window_central_ratio: float = 0.015,
    use_adjacent: bool = False,
    fallback_extrapolation: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor, search a horizontal window biased OUTWARD and snap to
    the topmost hair pixel column inside it.

    Why this helps the 6/12-point problem:
      The lateral-middle anchors (MP 54/103 on the left, MP 332/284 on the
      right) sit on the forehead skin. A pure straight-up ray finds the top
      of the hair plateau at the anchor's x — but the real "shoulder corner"
      of the hairline is several pixels OUTWARD from the anchor. By widening
      the search to a window that extends further toward the image edge,
      these anchors can slide LATERALLY onto the actual shoulder.

    Per-anchor window:
      [anchor_x - lateral_w, anchor_x + central_w]   for LEFT anchors
      [anchor_x - central_w, anchor_x + lateral_w]   for RIGHT anchors
    Center anchor (anchor.x ≈ face center) uses symmetric window.

    Both ``x`` and ``y`` of each output point are FREELY chosen inside the
    window — the function returns the (col, top_y) of the topmost hair pixel.
    """
    H, W = parse_map.shape
    hair = (
        build_face_adjacent_hair_mask(parse_map)
        if use_adjacent
        else build_hair_mask(parse_map)
    )
    top_y = _topmost_hair_y_per_col(hair)
    up_norm = face_up_vector(landmarks_norm)
    face_center_x = float(landmarks_norm[10, 0])  # MP 10 = forehead top

    w_lat = max(1, int(round(W * window_lateral_ratio)))
    w_cen = max(1, int(round(W * window_central_ratio)))

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        ax = int(round(np.clip(anchor_norm[0] * W, 0, W - 1)))

        side = 1 if anchor_norm[0] > face_center_x else (-1 if anchor_norm[0] < face_center_x else 0)
        if side > 0:
            x_lo = max(0, ax - w_cen)
            x_hi = min(W, ax + w_lat + 1)
        elif side < 0:
            x_lo = max(0, ax - w_lat)
            x_hi = min(W, ax + w_cen + 1)
        else:
            x_lo = max(0, ax - w_cen)
            x_hi = min(W, ax + w_cen + 1)

        sub = top_y[x_lo:x_hi]
        valid_cols = sub >= 0
        if not valid_cols.any():
            hairline[i] = (anchor_norm + up_norm * fallback_extrapolation).astype(np.float32)
            continue

        sub_masked = np.where(valid_cols, sub, H + 1)
        best_off = int(np.argmin(sub_masked))
        best_x = x_lo + best_off
        best_y = int(sub[best_off])
        hairline[i, 0] = best_x / W
        hairline[i, 1] = best_y / H
        valid[i] = True

    return hairline, valid


def sample_hairline_window_adjacent(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """window 策略的 adjacent 变体: 候选 hair 必须邻接 face skin。"""
    return sample_hairline_window(landmarks_norm, parse_map, use_adjacent=True)


def sample_hairline_lateral_extend(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    use_adjacent: bool = True,
    outer_anchors_per_side: int = 3,
    max_walk_ratio: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """baseline (+adjacent) 之后, 把最外侧 ``outer_anchors_per_side`` 个锚点
    沿 hit 所在的横行向外侧推到真实的 hair 视觉边缘。

    动机
    ----
    1, 2, 3 号 / 15, 16, 17 号锚点 (MP 127/234/162 和 389/356/454) 本来就长在
    眉外角和耳前的皮肤上, 射线垂直向上停在 parser 给出的 hair 边界 —— 而
    parser 对侧鬓的边界比视觉上的 hair 边缘保守几个像素, 所以这些点看起来
    比红线略微偏向脸内。

    这一步仅修改 X 坐标 (Y 保持 baseline hit 不变): 从 hit 像素出发, 沿同一
    行向外 (远离脸部中心) 一步一步走, 只要下一像素还是 hair 就更新 X, 否则
    停止。走距上限为 ``max_walk_ratio * W`` (默认 1.5%, ~11 px@734 宽); 太大
    容易跑出可见 hair 之外 —— parser 会把侧鬓 / 头侧轮廓后面几像素也判成 hair,
    所以 1.5% 是相对保守的值。

    只对最外 ``outer_anchors_per_side`` 个锚点应用; 中间锚点保持 baseline。
    """
    if use_adjacent:
        raw, valid = sample_hairline_adjacent(landmarks_norm, parse_map)
        hair = build_face_adjacent_hair_mask(parse_map)
    else:
        raw, valid = sample_hairline_baseline(landmarks_norm, parse_map)
        hair = build_hair_mask(parse_map)

    H, W = parse_map.shape
    max_walk_px = max(2, int(round(W * max_walk_ratio)))
    n = C.N_ANCHORS

    targets = (
        [(i, -1) for i in range(0, outer_anchors_per_side)]
        + [(i, +1) for i in range(n - outer_anchors_per_side, n)]
    )

    for i, side in targets:
        if not valid[i]:
            continue
        x_px = int(round(raw[i, 0] * W))
        y_px = int(round(raw[i, 1] * H))
        if y_px < 0 or y_px >= H:
            continue

        x_outer = x_px
        for step in range(1, max_walk_px + 1):
            nx = x_px + step * side
            if nx < 0 or nx >= W:
                break
            if hair[y_px, nx]:
                x_outer = nx
            else:
                break

        raw[i, 0] = x_outer / W

    return raw, valid


def sample_hairline_lateral_extend_dense(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    intermediates: int = 1,
    use_adjacent: bool = True,
    max_walk_ratio: float = 0.015,
    outer_per_side: int | None = None,
    density_run_length: int = 0,
    fallback_extrapolation: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """`lateral_extend` 的加密版本: 每对相邻 MP 锚点之间插入 ``intermediates`` 个
    中间采样列, 再各自射线 + 横向延伸。

    总点数 = ``N_ANCHORS + (N_ANCHORS - 1) * intermediates``
    默认 17 + 16 = **33 点**, 是原始 17 点密度的 ~2 倍。

    做法
    ----
    1. 把 17 个 MP_TOP_ANCHORS 的归一化坐标按顺序线性插值, 在每对之间塞入
       ``intermediates`` 个等距中间种子。种子点 (x, y) 都在原 MP 锚点的连线上。
    2. 每个种子点沿 face-up 方向直上, 命中首个 hair 像素 (用 adjacent hair
       过滤背景误分)。中间点的 y 来自其相邻两个原锚点的 y 内插, 起射高度
       与真实锚点一致, 所以命中位置自然落在 hair 平台上。
    3. 对最外侧 ``(intermediates + 1) * 3`` 个点 (覆盖原始 1-3 号 / 15-17 号
       周围的加密点), 沿 hit 横行向外侧走最多 1.5% 图宽, 推到可见 hair 边缘。
    4. 配合 corner-aware 平滑使用, 转角自动被保留。

    加密带来的两个直接收益
    ----
    * **6 号点和 12 号点之间的几个新点** (原始 5-6 / 12-13 之间) 会落在
      hair 平台的左上 / 右上转角处, 让发际线在角落处不再被两个稀疏的 MP
      点跨过。
    * 整条发际线更平滑, 同时 corner-aware 还能定位出新的转角。
    """
    if use_adjacent:
        hair = build_face_adjacent_hair_mask(parse_map)
    else:
        hair = build_hair_mask(parse_map)

    H, W = parse_map.shape
    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    up_px /= np.linalg.norm(up_px) + 1e-9

    seeds: list[np.ndarray] = []
    for k in range(C.N_ANCHORS):
        p_anchor = landmarks_norm[C.MP_TOP_ANCHORS[k], :2].astype(np.float64).copy()
        seeds.append(p_anchor)
        if k + 1 < C.N_ANCHORS:
            p_next = landmarks_norm[C.MP_TOP_ANCHORS[k + 1], :2].astype(np.float64)
            for j in range(1, intermediates + 1):
                t = j / (intermediates + 1)
                seeds.append((1.0 - t) * p_anchor + t * p_next)

    n_dense = len(seeds)
    hairline = np.zeros((n_dense, 2), dtype=np.float32)
    valid = np.zeros(n_dense, dtype=bool)

    for i, sn in enumerate(seeds):
        anchor_px = sn * np.array([W, H], dtype=np.float64)
        if density_run_length > 0:
            hit_px = ray_first_dense_hit(anchor_px, up_px, hair, density_run_length)
        else:
            hit_px = ray_first_mask_hit(anchor_px, up_px, hair)
        if hit_px is not None:
            hairline[i] = (hit_px / np.array([W, H])).astype(np.float32)
            valid[i] = True
        else:
            hairline[i] = (sn + up_norm * fallback_extrapolation).astype(np.float32)

    if outer_per_side is None:
        outer = (intermediates + 1) * 3
    else:
        outer = max(0, min(outer_per_side, n_dense // 2))
    targets = (
        [(i, -1) for i in range(0, outer)]
        + [(i, +1) for i in range(n_dense - outer, n_dense)]
    )
    max_walk_px = max(2, int(round(W * max_walk_ratio))) if max_walk_ratio > 0 else 0

    if max_walk_px > 0:
        for i, side in targets:
            if not valid[i]:
                continue
            x_px = int(round(hairline[i, 0] * W))
            y_px = int(round(hairline[i, 1] * H))
            if y_px < 0 or y_px >= H:
                continue

            x_outer = x_px
            for step in range(1, max_walk_px + 1):
                nx = x_px + step * side
                if nx < 0 or nx >= W:
                    break
                if hair[y_px, nx]:
                    x_outer = nx
                else:
                    break

            hairline[i, 0] = x_outer / W

    return hairline, valid


def build_head_silhouette_mask(parse_map: np.ndarray) -> np.ndarray:
    """``(face_skin ∪ full_hair)`` 二值 mask, 用于 lateral silhouette 探测。

    左右两侧 lateral 锚点 (MP 127/234/93/132/58 等) 都长在脸侧的皮肤上,
    需要先穿过整个 head silhouette 才能停在最外侧的 hair / skin 边界。
    """
    return build_skin_mask(parse_map) | build_hair_mask(parse_map)


def sample_lateral_extension(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    mp_anchors_left: list[int] | None = None,
    mp_anchors_right: list[int] | None = None,
    max_walk_ratio: float = 0.15,
    use_full_silhouette: bool = True,
    mid_t: float = 0.5,
    fallback_extrapolation: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对每个 lateral 锚点, 沿"水平向外"方向探测到 head silhouette 外缘。

    返回
    ----
    mid_xy:   (2L, 2) float — 中间行点 = lerp(MP锚点, lateral_out, mid_t)
    out_xy:   (2L, 2) float — 外圈点 = silhouette 外缘像素的前一像素
    valid:    (2L,)   bool  — 是否真的命中 silhouette 边界 (False = 用 fallback)

    顺序: ``left_0 .. left_{L-1}, right_0 .. right_{L-1}`` (L = N_LATERAL_PER_SIDE)。

    方向定义
    --------
    face-up = chin (MP 152) → forehead top (MP 10), 单位 norm 向量。
    perp = (-up.y, up.x): 左脸为负 (perp 方向取负), 右脸为正 (perp 方向取正)。
    """
    if mp_anchors_left is None:
        mp_anchors_left = C.MP_LATERAL_ANCHORS_LEFT
    if mp_anchors_right is None:
        mp_anchors_right = C.MP_LATERAL_ANCHORS_RIGHT

    if len(mp_anchors_left) != len(mp_anchors_right):
        raise ValueError("left and right lateral anchor chains must have the same length")
    L = len(mp_anchors_left)
    n_total = 2 * L

    H, W = parse_map.shape

    if use_full_silhouette:
        head = build_head_silhouette_mask(parse_map)
    else:
        head = build_skin_mask(parse_map) | build_face_adjacent_hair_mask(parse_map)

    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    up_px /= np.linalg.norm(up_px) + 1e-9
    perp_px = np.array([-up_px[1], up_px[0]], dtype=np.float64)

    max_walk_px = max(2, int(round(W * max_walk_ratio)))

    mid_xy = np.zeros((n_total, 2), dtype=np.float32)
    out_xy = np.zeros((n_total, 2), dtype=np.float32)
    valid = np.zeros(n_total, dtype=bool)

    chain = [(mp, -1) for mp in mp_anchors_left] + [(mp, +1) for mp in mp_anchors_right]

    for i, (mp_idx, side_sign) in enumerate(chain):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        direction = perp_px * side_sign

        # March outward step-by-step, looking for the first OUTSIDE pixel.
        # The "outer" point is the previous (last inside) pixel, so the ribbon
        # hugs the silhouette without crossing into background.
        last_inside = anchor_px.copy()
        found = False
        pos = anchor_px.copy()
        for _ in range(max_walk_px):
            pos = pos + direction
            ix = int(round(pos[0]))
            iy = int(round(pos[1]))
            if ix < 0 or iy < 0 or ix >= W or iy >= H:
                found = True
                break
            if head[iy, ix]:
                last_inside = pos.copy()
            else:
                found = True
                break

        if not found:
            # Walked the full budget and still inside silhouette — use the
            # farthest point we reached. This usually only happens on
            # full-head shots where the head spans most of the image.
            last_inside = pos.copy()

        out_pxf = last_inside
        out_pxf[0] = float(np.clip(out_pxf[0], 0, W - 1))
        out_pxf[1] = float(np.clip(out_pxf[1], 0, H - 1))

        # If we never moved at all (anchor already on background), fall back
        # to a small geometric extrapolation along perp.
        moved = float(np.linalg.norm(out_pxf - anchor_px))
        if moved < 1.0:
            out_pxf = anchor_px + direction * (W * fallback_extrapolation)
            out_pxf[0] = float(np.clip(out_pxf[0], 0, W - 1))
            out_pxf[1] = float(np.clip(out_pxf[1], 0, H - 1))
            valid[i] = False
        else:
            valid[i] = True

        out_xy[i] = (out_pxf / np.array([W, H])).astype(np.float32)
        mid_xy[i] = ((1.0 - mid_t) * anchor_norm + mid_t * out_xy[i]).astype(np.float32)

    return mid_xy, out_xy, valid


def sample_hairline_converging(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 3: 每个锚点的射线方向都收敛到头顶上方目标点。"""
    mask = build_hair_mask(parse_map)
    return _ray_sample_with_direction(landmarks_norm, parse_map, mask, _direction_converging)


def sample_hairline_converging_adjacent(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 4: 收敛方向 + 邻接约束。"""
    mask = build_face_adjacent_hair_mask(parse_map)
    return _ray_sample_with_direction(landmarks_norm, parse_map, mask, _direction_converging)


def sample_hairline_boundary_proj(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 5: 计算 hair-skin 边界曲线, 每个锚点投影到带方向约束的最近边界点。

    "方向约束" 指: 候选边界点必须位于锚点的 face-up 方向之上 (cos(angle) >= 0),
    并且对横向偏移加额外惩罚, 避免吸到锚点正侧的边界。
    """
    import cv2

    H, W = parse_map.shape
    hair = np.isin(parse_map, [C.PARSE_HAIR, C.PARSE_HAT]).astype(np.uint8)
    face = build_skin_mask(parse_map).astype(np.uint8)
    face_dil = cv2.dilate(face, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    boundary = (hair > 0) & (face_dil > 0)

    bys, bxs = np.where(boundary)
    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    up_px /= np.linalg.norm(up_px) + 1e-9
    perp_px = np.array([-up_px[1], up_px[0]], dtype=np.float64)

    if bxs.size == 0:
        for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
            anchor_norm = landmarks_norm[mp_idx, :2]
            hairline[i] = (anchor_norm + up_norm * 0.18).astype(np.float32)
        return hairline, valid

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        dx = bxs - anchor_px[0]
        dy = bys - anchor_px[1]
        proj_up = dx * up_px[0] + dy * up_px[1]
        keep = proj_up > 1.0
        if not keep.any():
            hairline[i] = (anchor_norm + up_norm * 0.18).astype(np.float32)
            continue
        d_up = proj_up[keep]
        d_perp = np.abs(dx[keep] * perp_px[0] + dy[keep] * perp_px[1])
        cost = d_up + 2.5 * d_perp
        local_best = int(np.argmin(cost))
        chosen = np.where(keep)[0][local_best]
        hairline[i, 0] = bxs[chosen] / W
        hairline[i, 1] = bys[chosen] / H
        valid[i] = True

    return hairline, valid


def _ray_dense_sample(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    mask: np.ndarray,
    direction_factory: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
    fallback_extrapolation: float = 0.18,
    run_ratio: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Density-stopping variant of :func:`_ray_sample_with_direction`."""
    H, W = parse_map.shape
    run_length = density_run_length(landmarks_norm, W, H, run_ratio)
    up_norm_fallback = face_up_vector(landmarks_norm)

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        direction_px = direction_factory(landmarks_norm, anchor_px, W, H)
        hit_px = ray_first_dense_hit(anchor_px, direction_px, mask, run_length)
        if hit_px is not None:
            hairline[i] = (hit_px / np.array([W, H])).astype(np.float32)
            valid[i] = True
        else:
            hairline[i] = (anchor_norm + up_norm_fallback * fallback_extrapolation).astype(np.float32)
            valid[i] = False
    return hairline, valid


def sample_hairline_density(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 1: 沿 face-up 直上, 但要求连续 `density_run_length` 个 hair 像素才停。"""
    mask = build_hair_mask(parse_map)
    return _ray_dense_sample(landmarks_norm, parse_map, mask, _direction_face_up)


def sample_hairline_density_adjacent(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 2: face-up + density + 邻接约束。"""
    mask = build_face_adjacent_hair_mask(parse_map)
    return _ray_dense_sample(landmarks_norm, parse_map, mask, _direction_face_up)


def sample_hairline_density_converging(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 3: 收敛方向 + density。"""
    mask = build_hair_mask(parse_map)
    return _ray_dense_sample(landmarks_norm, parse_map, mask, _direction_converging)


def _ray_dense_inverse_sample(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    mask: np.ndarray,
    target_scale: float = 1.5,
    run_ratio: float = 0.06,
    fallback_extrapolation: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor, march FROM a target above the head DOWN toward the anchor.

    Density-based stopping returns the first dense hair pixel encountered. Since
    the ray comes from outside/above the head, the first dense hair = top edge
    of the head silhouette = actual hairline crown. This bypasses the side-hair
    interception problem entirely for lateral anchors.
    """
    H, W = parse_map.shape
    run_length = density_run_length(landmarks_norm, W, H, run_ratio)
    up_norm_fallback = face_up_vector(landmarks_norm)

    target_norm = head_top_target(landmarks_norm, scale=target_scale)
    target_px = target_norm * np.array([W, H], dtype=np.float64)

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        anchor_px = anchor_norm * np.array([W, H], dtype=np.float64)
        direction = anchor_px - target_px
        norm = float(np.linalg.norm(direction)) + 1e-9
        direction_unit = direction / norm
        max_steps = int(norm) + 200
        hit_px = ray_first_dense_hit(target_px, direction_unit, mask, run_length, max_steps=max_steps)
        accepted = False
        if hit_px is not None:
            t_along = float(np.dot(hit_px - target_px, direction_unit))
            if 0.0 < t_along < norm:
                hairline[i] = (hit_px / np.array([W, H])).astype(np.float32)
                valid[i] = True
                accepted = True
        if not accepted:
            hairline[i] = (anchor_norm + up_norm_fallback * fallback_extrapolation).astype(np.float32)
    return hairline, valid


def sample_hairline_density_inverse(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 4: 反向射线 - 从头顶上方目标点向锚点扫, density stop。

    完全绕开 baseline 的核心痛点: 不再从锚点向上找头发, 而是从头部上方"找下来",
    第一段 dense hair 自然就是头部的上边缘 (冠状发际线)。
    """
    mask = build_hair_mask(parse_map)
    return _ray_dense_inverse_sample(landmarks_norm, parse_map, mask)


def sample_hairline_density_inverse_adj(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 5: 反向射线 + adjacent + density。"""
    mask = build_face_adjacent_hair_mask(parse_map)
    return _ray_dense_inverse_sample(landmarks_norm, parse_map, mask)


def sample_hairline_silhouette_at_anchor_x(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    window_radius_ratio: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """密度策略 6 (head silhouette 锚点对齐): 对每个 MP 锚点, 取它 x 列附近的头部上沿 y。

    与 ``top_arc`` 相比, 17 个点的 x 仍然跟 MediaPipe 锚点的 x 一一对应,
    但 y 直接来自 (face ∪ adjacent_hair) 在该 x 列的最高点 (最小 y),
    所以点会自然贴到头顶轮廓上, 形成冠状弧, 而不是被侧鬓拽下来。

    window_radius_ratio: 用 anchor x 周围一小段窗口 (W·ratio) 取最小 y 做平滑。
    """
    H, W = parse_map.shape
    head = build_skin_mask(parse_map) | build_face_adjacent_hair_mask(parse_map)
    up_norm_fallback = face_up_vector(landmarks_norm)

    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    radius = max(1, int(round(W * window_radius_ratio)))
    for i, mp_idx in enumerate(C.MP_TOP_ANCHORS):
        anchor_norm = landmarks_norm[mp_idx, :2]
        cx = int(round(np.clip(anchor_norm[0] * W, 0, W - 1)))
        x0 = max(0, cx - radius)
        x1 = min(W, cx + radius + 1)
        sub = head[:, x0:x1]
        col_has = sub.any(axis=1)
        if not col_has.any():
            hairline[i] = (anchor_norm + up_norm_fallback * 0.18).astype(np.float32)
            continue
        top_y = int(np.argmax(col_has))
        hairline[i, 0] = anchor_norm[0]
        hairline[i, 1] = top_y / H
        valid[i] = True
    return hairline, valid


def _head_top_y_per_column(
    parse_map: np.ndarray,
    use_full_hair: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cols_with_head, top_y) where top_y[k] is the smallest y for which
    ``head_mask[top_y[k], cols_with_head[k]]`` is True.

    ``use_full_hair=True`` uses the FULL hair mask (proper head silhouette);
    pass ``False`` to fall back to the face-adjacent strip only.
    """
    if use_full_hair:
        head = build_skin_mask(parse_map) | build_hair_mask(parse_map)
    else:
        head = build_skin_mask(parse_map) | build_face_adjacent_hair_mask(parse_map)
    cols_with_head = np.where(head.any(axis=0))[0]
    if cols_with_head.size == 0:
        return cols_with_head, np.array([], dtype=np.int32)
    top_y = np.empty(cols_with_head.size, dtype=np.int32)
    for j, x in enumerate(cols_with_head):
        top_y[j] = int(np.argmax(head[:, x]))
    return cols_with_head, top_y


def sample_hairline_arch(
    landmarks_norm: np.ndarray,
    parse_map: np.ndarray,
    central_drop: int = 3,
    require_lift_px_ratio: float = 0.02,
    density_ratio: float = 0.03,
    use_adjacent_for_central: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """冠状弧 (coronal arch) 策略 — 从一只耳朵上方画到另一只耳朵上方。

    流程
    ----
      1. 中间 ``N_ANCHORS - 2*central_drop`` 个锚点（默认丢掉颧骨/太阳穴附近的 6 个），
         沿 face-up 做密度门限射线，命中 ``run_length`` 个连续 hair 像素的入口作为
         真实发际线点。``run_length = face_height * density_ratio``。
      2. 拒掉那些 hit 离锚点向上不到 ``face_height * require_lift_px_ratio``
         的，它们多半是侧鬓或假阳性。
      3. 取整个头部 (face ∪ 全部 hair) 的 silhouette：在最左 / 最右极限列上，
         记录 silhouette 上沿点。这两个点是耳朵上方的曲线终点。
      4. 控制点 = [左极限 silhouette] + [中间锚点真实 hit] + [右极限 silhouette]，
         按 x 排序，线性插值到 17 个等距输出 x。
      5. 钳位：任何输出 y 都不能比该 x 处头部 silhouette 上沿还小，避免曲线
         跑到头顶上方的虚空里去。
      6. 上层 web_service 会再调用 ``smooth_hairline`` 平滑一遍。

    生成的 17 个点 x 均匀覆盖整个头部宽度（从左耳上方到右耳上方），中间 y 贴近
    真实发际线、两侧 y 自然抬升到头部 silhouette 上沿，形成冠状弧。它们的 x
    不再与对应 MP_TOP_ANCHORS 一一对应，但顺序保持一致。
    """
    H, W = parse_map.shape
    hair_for_rays = (
        build_face_adjacent_hair_mask(parse_map)
        if use_adjacent_for_central
        else build_hair_mask(parse_map)
    )
    run_length = max(4, int(round(_face_height_px(landmarks_norm, W, H) * density_ratio)))
    up_norm = face_up_vector(landmarks_norm)
    up_px = up_norm * np.array([W, H], dtype=np.float64)
    up_px /= np.linalg.norm(up_px) + 1e-9

    face_h_px = _face_height_px(landmarks_norm, W, H)
    min_lift = face_h_px * require_lift_px_ratio

    # Step 1: central density-ray hits
    central_indices = range(central_drop, C.N_ANCHORS - central_drop)
    central_hits: list[tuple[float, float]] = []
    for i in central_indices:
        mp_idx = C.MP_TOP_ANCHORS[i]
        anchor_px = landmarks_norm[mp_idx, :2] * np.array([W, H], dtype=np.float64)
        hit = ray_first_dense_hit(anchor_px, up_px, hair_for_rays, run_length)
        if hit is None:
            continue
        # signed distance "up" from anchor to hit (positive when hit is above anchor)
        delta_up = float(np.dot(hit - anchor_px, up_px))
        if delta_up < min_lift:
            continue
        central_hits.append((float(hit[0]), float(hit[1])))

    # Step 2: head silhouette
    cols_with_head, top_y = _head_top_y_per_column(parse_map, use_full_hair=True)
    if cols_with_head.size < 2:
        return sample_hairline_baseline(landmarks_norm, parse_map)

    left_x = float(cols_with_head[0])
    right_x = float(cols_with_head[-1])

    def silhouette_y_at(x_px: float) -> float:
        idx = int(round(np.clip(x_px, left_x, right_x)))
        k = int(np.searchsorted(cols_with_head, idx))
        k = min(k, cols_with_head.size - 1)
        return float(top_y[k])

    if not central_hits:
        # No reliable central hits — fall back to silhouette top across the head
        # (still a coronal arc, just less precise on the forehead)
        output_xs = np.linspace(left_x, right_x, C.N_ANCHORS)
        output_ys = np.array([silhouette_y_at(float(x)) for x in output_xs], dtype=np.float64)
        hairline = np.column_stack([output_xs / W, output_ys / H]).astype(np.float32)
        return hairline, np.ones(C.N_ANCHORS, dtype=bool)

    # Step 3: control points = [left endpoint] + central hits + [right endpoint]
    control_pts: list[tuple[float, float]] = list(central_hits)
    control_pts.append((left_x, silhouette_y_at(left_x)))
    control_pts.append((right_x, silhouette_y_at(right_x)))

    # Step 4: sort + dedupe by x, then linear-interp 17 evenly-spaced output xs
    cps = sorted(control_pts, key=lambda p: p[0])
    cxs = [cps[0][0]]
    cys = [cps[0][1]]
    for x, y in cps[1:]:
        if x - cxs[-1] < 0.5:
            # collision: prefer the LARGER y (the actual hairline, not silhouette)
            cys[-1] = max(cys[-1], y)
        else:
            cxs.append(x)
            cys.append(y)
    cxs_arr = np.asarray(cxs, dtype=np.float64)
    cys_arr = np.asarray(cys, dtype=np.float64)
    output_xs = np.linspace(left_x, right_x, C.N_ANCHORS)
    output_ys = np.interp(output_xs, cxs_arr, cys_arr)

    # Step 5: hairline cannot be above the head silhouette top
    for k, x in enumerate(output_xs):
        sil_y = silhouette_y_at(float(x))
        if output_ys[k] < sil_y:
            output_ys[k] = sil_y

    hairline = np.column_stack([output_xs / W, output_ys / H]).astype(np.float32)
    valid = np.ones(C.N_ANCHORS, dtype=bool)
    return hairline, valid


def sample_hairline_arch_strict(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """冠状弧策略 (严格版): 只用中间 5 个最可靠的额头锚点; 同时要求 hit 离锚点更远。

    适合中间锚点也容易被刘海/眉毛干扰的场景，曲线更"瘦高"。
    """
    return sample_hairline_arch(
        landmarks_norm,
        parse_map,
        central_drop=6,                    # 只保留中间 5 个锚点
        require_lift_px_ratio=0.04,
        density_ratio=0.04,
    )


def sample_hairline_arch_adj(landmarks_norm: np.ndarray, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """冠状弧策略 (邻接版): 中间锚点的射线只命中"贴脸"的 hair 像素。

    更不容易被远离脸部的背景或后脑勺误命中，但要求 face-adjacent hair 必须
    形成连续 ``run_length`` 像素 (默认 ratio=0.025) 的密度。
    """
    return sample_hairline_arch(
        landmarks_norm,
        parse_map,
        central_drop=3,
        require_lift_px_ratio=0.015,
        density_ratio=0.025,
        use_adjacent_for_central=True,
    )


def sample_hairline_top_arc(landmarks_norm, parse_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """策略 6: 不用锚点, 取 (face ∪ adjacent_hair) 的上沿, 按 x 等距 17 点。

    ``landmarks_norm`` 仅用于 fallback 并允许传 None。
    """
    H, W = parse_map.shape
    head = build_skin_mask(parse_map) | build_face_adjacent_hair_mask(parse_map)
    hairline = np.zeros((C.N_ANCHORS, 2), dtype=np.float32)
    valid = np.zeros((C.N_ANCHORS,), dtype=bool)

    cols_with_head = np.where(head.any(axis=0))[0]
    if cols_with_head.size < 2:
        return hairline, valid

    top_y = np.empty(cols_with_head.size, dtype=np.int32)
    for j, x in enumerate(cols_with_head):
        top_y[j] = int(np.argmax(head[:, x]))

    left, right = int(cols_with_head[0]), int(cols_with_head[-1])
    xs = np.linspace(left, right, C.N_ANCHORS)
    ys = np.interp(xs, cols_with_head.astype(np.float64), top_y.astype(np.float64))
    hairline[:, 0] = xs / W
    hairline[:, 1] = ys / H
    valid[:] = True
    return hairline, valid
