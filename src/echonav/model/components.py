from __future__ import annotations
import math
import torch
from torch import nn
from .attention import DirectDecoderBlock


def _native_query_coordinates(
    size: int, extent_vggt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if size <= 0 or extent_vggt <= 0.0:
        raise ValueError("EchoNav Gen.1 query size and extent must be positive")
    cell = extent_vggt / size
    pixel = torch.arange(size, dtype=torch.float32)
    x = -extent_vggt / 2.0 + (pixel + 0.5) * cell
    z = extent_vggt / 2.0 - (pixel + 0.5) * cell
    z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")
    normalized = torch.stack(
        (x_grid / (extent_vggt / 2.0), z_grid / (extent_vggt / 2.0)), dim=-1
    ).reshape(-1, 2)
    reference = torch.stack((normalized[:, 0], -normalized[:, 1]), dim=-1)
    return (normalized, reference)


class ParallelQueryBranch(nn.Module):
    """One independent latest or history query branch."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        feature_levels: int,
        self_attention_mode: str,
        cross_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.cross_attention_mode = str(cross_attention_mode)
        self.blocks = nn.ModuleList(
            [
                DirectDecoderBlock(
                    hidden_dim,
                    heads,
                    self_attention_mode=self_attention_mode,
                    cross_attention_mode=cross_attention_mode,
                    feature_levels=feature_levels,
                    deformable_samples=deformable_samples,
                    cross_query_chunk_size=cross_query_chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _flatten_context(pyramid: list[torch.Tensor]) -> torch.Tensor:
        batch = pyramid[0].shape[0]
        return torch.cat(
            [
                level.permute(0, 1, 3, 4, 2).reshape(batch, -1, level.shape[2])
                for level in pyramid
            ],
            dim=1,
        )

    @staticmethod
    def _flatten_weights(
        pyramid: list[torch.Tensor], frame_reliability: torch.Tensor
    ) -> torch.Tensor:
        batch, frames = frame_reliability.shape
        if any((level.shape[:2] != (batch, frames) for level in pyramid)):
            raise ValueError(
                "EchoNav Gen.1 pyramid and reliability frame counts disagree"
            )
        return torch.cat(
            [
                frame_reliability.repeat_interleave(
                    level.shape[-2] * level.shape[-1], dim=1
                )
                for level in pyramid
            ],
            dim=1,
        )

    def forward(
        self,
        query: torch.Tensor,
        pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
        *,
        frame_reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not pyramid:
            raise ValueError("EchoNav Gen.1 query branch requires a token pyramid")
        context_weight = None
        if self.cross_attention_mode == "deformable":
            context: torch.Tensor | list[torch.Tensor] = pyramid
            if frame_reliability is not None:
                raise ValueError("deformable latest branch does not use frame weights")
        else:
            context = self._flatten_context(pyramid)
            if frame_reliability is not None:
                context_weight = self._flatten_weights(pyramid, frame_reliability)
        for block in self.blocks:
            query = block(query, context, reference_grid, context_weight=context_weight)
        return self.output_norm(query)


class SpatialResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(self.norm(value))


class StructuredFrameReliabilityHead(nn.Module):
    """Predict frame reliability without averaging camera and register roles.

    VGGT-Omega supplies one camera token followed by sixteen learned register
    tokens per frame. The camera token remains an explicit feature, while the
    register bank is summarized by learned content- and type-aware attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        maximum_prefix_tokens: int = 17,
        minimum: float = 0.25,
        maximum: float = 1.75,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("frame reliability dimensions must be positive")
        if maximum_prefix_tokens < 2:
            raise ValueError("camera plus at least one register token are required")
        if not 0.0 < minimum < 1.0 < maximum:
            raise ValueError("reliability bounds must straddle one")
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.register_norm = nn.LayerNorm(input_dim)
        self.register_score = nn.Linear(input_dim, 1, bias=False)
        self.register_type_bias = nn.Parameter(torch.zeros(maximum_prefix_tokens - 1))
        self.register_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.register_score.weight)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _structured_feature(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.ndim != 4:
            raise ValueError("prefix tokens must have shape [B,N,P,C]")
        prefix_count = prefix_tokens.shape[2]
        if not 2 <= prefix_count <= self.maximum_prefix_tokens:
            raise ValueError("prefix token count is outside the structured contract")
        camera = prefix_tokens[:, :, 0]
        registers = prefix_tokens[:, :, 1:]
        score = self.register_score(self.register_norm(registers)).squeeze(-1)
        score = score + self.register_type_bias[: registers.shape[2]][None, None, :]
        weight = torch.softmax(score.float(), dim=2).to(dtype=registers.dtype)
        register_context = (registers * weight.unsqueeze(-1)).sum(dim=2)
        return torch.cat(
            (
                self.camera_projection(camera),
                self.register_projection(register_context),
            ),
            dim=-1,
        )

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        raw = self.network(self._structured_feature(prefix_tokens)).squeeze(-1)
        reliability = self.minimum + (self.maximum - self.minimum) * torch.sigmoid(raw)
        return reliability / reliability.mean(dim=1, keepdim=True).clamp_min(1e-06)


class DenseNativeQuery(nn.Module):
    """Learned content and Fourier position features for every metric BEV cell."""

    def __init__(
        self, *, size: int, extent_m: float, hidden_dim: int, fourier_bands: int = 4
    ) -> None:
        super().__init__()
        if fourier_bands <= 0:
            raise ValueError("EchoNav Gen.1 Fourier band count must be positive")
        self.size = int(size)
        self.extent_m = float(extent_m)
        normalized, reference = _native_query_coordinates(size, extent_m)
        self.register_buffer("normalized_coordinates", normalized, persistent=True)
        self.register_buffer("reference_grid", reference, persistent=True)
        self.register_buffer(
            "fourier_frequencies",
            2.0 ** torch.arange(fourier_bands, dtype=torch.float32),
            persistent=True,
        )
        self.query_content = nn.Parameter(torch.empty(size * size, hidden_dim))
        coordinate_dim = 2 + fourier_bands * 2 * 2
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.normal_(self.query_content, std=0.02)

    def _coordinate_features(self) -> torch.Tensor:
        coordinates = self.normalized_coordinates
        phase = (
            coordinates[:, :, None] * self.fourier_frequencies[None, None, :] * math.pi
        )
        return torch.cat(
            (coordinates, phase.sin().flatten(1), phase.cos().flatten(1)), dim=1
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("EchoNav Gen.1 query batch size must be positive")
        position = self.coordinate_projection(
            self._coordinate_features().to(dtype=self.query_content.dtype)
        )
        return (self.query_content + position).unsqueeze(0).expand(batch_size, -1, -1)
