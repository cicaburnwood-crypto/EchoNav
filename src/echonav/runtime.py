"""Inference only. No depth, pose, calibration or ground-truth inputs."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn

from .backbone.aggregator import Aggregator
from .config import BACKBONE_SHA256, HEAD_SHA256, HEAD_ARGUMENTS, MODEL_NAME, SCHEMA
from .model.head import BEVHead
from .preprocess import RGBResizePad


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class EchoNav(nn.Module):
    """EchoNav Gen.1. Caller chooses device; this class never manages GPUs."""

    def __init__(
        self,
        head_weights: str | Path,
        backbone_weights: str | Path,
        device: str = "cuda:0",
        history_batch_size: int = 9,
    ) -> None:
        super().__init__()
        if not isinstance(history_batch_size, int) or not 1 <= history_batch_size <= 9:
            raise ValueError(
                "History proposal batch size must be an integer from 1 to 9"
            )
        self.device = torch.device(device)
        torch.set_float32_matmul_precision("highest")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Use a CPU or CUDA device")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Reference CUDA inference requires BF16 support")
        if sha256(backbone_weights) != BACKBONE_SHA256:
            raise ValueError(
                "Backbone checksum does not match the released architecture"
            )
        if sha256(head_weights) != HEAD_SHA256:
            raise ValueError("Head checksum does not match the EchoNav Gen.1 release")
        state = torch.load(
            head_weights, map_location="cpu", weights_only=True, mmap=True
        )
        if (
            set(state) != {"model_name", "schema", "head"}
            or state["model_name"] != MODEL_NAME
            or state["schema"] != SCHEMA
        ):
            raise ValueError("Expected a clean EchoNav Gen.1 inference checkpoint")
        self.head = BEVHead(**HEAD_ARGUMENTS)
        self.head.history_proposal_batch_size = history_batch_size
        self.head.load_state_dict(state["head"], strict=True)
        del state
        self.aggregator = Aggregator(patch_size=16)
        backbone = torch.load(
            backbone_weights, map_location="cpu", weights_only=True, mmap=True
        )
        trunk = {
            k.removeprefix("aggregator."): v
            for k, v in backbone.items()
            if k.startswith("aggregator.")
        }
        self.aggregator.load_state_dict(trunk, strict=True)
        del backbone, trunk
        self.aggregator.to(
            device=self.device,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        self.head.to(device=self.device, dtype=torch.float32)
        self.head.channels_last_spatial = True
        self.preprocess = RGBResizePad(384, 512)
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Accept [B,T,3,384,512] float RGB in [0,1], oldest to latest."""
        if (
            images.ndim != 5
            or not 1 <= images.shape[1] <= 10
            or images.shape[2:] != (3, 384, 512)
            or images.shape[0] < 1
        ):
            raise ValueError("Expected [B,T,3,384,512], B>=1 and 1<=T<=10")
        if (
            not images.is_floating_point()
            or not torch.isfinite(images).all()
            or images.min() < 0
            or images.max() > 1
        ):
            raise ValueError("RGB must be finite floating-point values in [0,1]")
        images = images.to(self.device, dtype=torch.float32)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with amp:
            # The latest image must receive the special first-frame token role.
            levels, prefix_count = self.aggregator(images.flip(1))
            if prefix_count != 17:
                raise RuntimeError("Unexpected camera/register token layout")
            extraction = {
                "tokens": {
                    i: levels[i][:, :, prefix_count:]
                    for i in HEAD_ARGUMENTS["cached_layers"]
                },
                "camera_register_tokens": levels[-1][:, :, :prefix_count],
                "patch_grid": (24, 32),
            }
            # Residual streams can remain FP32 even under backbone autocast.
            # Match the released head boundary explicitly, not implicitly.
            if self.device.type == "cuda":
                extraction["tokens"] = {
                    i: value.to(torch.bfloat16)
                    for i, value in extraction["tokens"].items()
                }
                extraction["camera_register_tokens"] = extraction[
                    "camera_register_tokens"
                ].to(torch.bfloat16)
            result = self.head(extraction)["merged_bev"]
        return {
            "occupancy_probability": result["occupancy_probability"].float(),
            "support_probability": result["fov_support_probability"].float(),
            "observed_gate_probability": result["observed_gate_probability"].float(),
            "confidence": result["navigation_confidence"].float(),
            "completion_occupancy_probability": result["guessed"][
                "occupancy_probability"
            ].float(),
            "semantic": result["fov_complete_semantic"],
        }

    def predict(
        self, frames: Sequence[str | Path | Image.Image]
    ) -> dict[str, np.ndarray]:
        """One chronological RGB window; output arrays are each [256,256]."""
        if not 1 <= len(frames) <= 10:
            raise ValueError("Supply 1–10 frames in chronological order")
        tensors = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                tensors.append(self.preprocess(frame))
            else:
                with Image.open(frame) as image:
                    tensors.append(self.preprocess(image))
        outputs = self(torch.stack(tensors)[None])
        return {name: value[0].cpu().numpy() for name, value in outputs.items()}
