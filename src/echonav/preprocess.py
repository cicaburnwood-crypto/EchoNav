from __future__ import annotations
import numpy as np
import torch
from PIL import Image


class RGBResizePad:
    """Aspect-preserving RGB resize with white padding and [0, 1] output."""

    def __init__(self, height: int = 384, width: int = 512) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("output dimensions must be positive")
        self.height = height
        self.width = width
        self.version = "rgb-depth-resize-pad-v2"

    def parameters(
        self, source_height: int, source_width: int
    ) -> tuple[int, int, int, int]:
        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        return (resized_height, resized_width, top, left)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        resized_height, resized_width, top, left = self.parameters(
            rgb.height, rgb.width
        )
        resized = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
