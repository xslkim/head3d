"""Face parsing via HuggingFace SegFormer (jonathandinu/face-parsing).

First call downloads ~150 MB of weights into the HF cache.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from . import constants as C

if TYPE_CHECKING:  # avoid hard import at module load
    pass


class FaceParser:
    """Wraps a SegFormer face-parsing model and returns a (H, W) int label map."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SegformerImageProcessor.from_pretrained(C.HF_FACE_PARSER_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(C.HF_FACE_PARSER_MODEL)
        self.model.to(self.device).eval()
        self._torch = torch

    def parse(self, image_rgb: np.ndarray) -> np.ndarray:
        """image_rgb: (H, W, 3) uint8. Returns (H, W) int64 of class indices."""
        from PIL import Image

        H, W = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, h, w)
        # Upsample to original resolution
        up = self._torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        labels = up.argmax(dim=1).squeeze(0).to("cpu").numpy().astype(np.int32)
        return labels
