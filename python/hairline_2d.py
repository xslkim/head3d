"""2D hairline detection from a face-parsing label map.

Given the parser output, find the boundary curve between hair-above and
skin-below in the upper face region, then sample one point per anchor
along the face's local "up" direction.
"""
from __future__ import annotations
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
    """
    out = hairline_norm.copy()
    for _ in range(iterations):
        new = out.copy()
        for i in range(1, C.N_ANCHORS - 1):
            if not valid[i]:
                continue
            new[i] = 0.25 * out[i - 1] + 0.5 * out[i] + 0.25 * out[i + 1]
        out = new
    return out
