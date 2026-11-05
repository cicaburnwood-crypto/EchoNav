from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import DeformableCrossAttention, MultiheadAttention
from .components import (
    ParallelQueryBranch,
    SpatialResidualBlock,
    DenseNativeQuery,
    StructuredFrameReliabilityHead,
)
from .probability import decode_binary_prediction, fuse_pixel_routing, compose_semantic


def assemble_bev(
    raw_evidence: torch.Tensor,
    routing_raw: torch.Tensor,
    *,
    assemble_runtime_outputs: bool,
) -> dict:
    guessed = decode_binary_prediction(
        raw_evidence, "evidential", include_diagnostics=assemble_runtime_outputs
    )
    routing_raw = routing_raw.float()
    observed_logit = routing_raw[:, 0]
    support_logit = routing_raw[:, 1]
    support_probability = torch.sigmoid(support_logit)
    output = {
        "guessed": guessed,
        "observed_gate_logit": observed_logit,
        "fov_support_logit": support_logit,
        "fov_support_probability": support_probability,
    }
    if not assemble_runtime_outputs:
        return output
    observed_probability = torch.sigmoid(observed_logit)
    guessed_region = 1.0 - observed_probability
    occupancy = guessed["occupancy_probability"].float()
    routing_probability = torch.stack(
        (
            observed_probability,
            guessed_region * (1.0 - occupancy),
            guessed_region * occupancy,
        ),
        dim=1,
    )
    fused = fuse_pixel_routing(
        routing_probability, guessed, support_probability, "evidential"
    )
    output.update(
        {
            "routing_probability": routing_probability,
            "routing_class": routing_probability.argmax(dim=1),
            "observed_gate_probability": observed_probability,
            "guessed_region_probability": guessed_region,
            "fused": fused,
            "occupancy_probability": fused["occupancy_probability"],
            "navigation_confidence": fused["navigation_confidence"],
            "fov_complete_semantic": compose_semantic(fused, support_probability),
        }
    )
    return output


class DPTLiteRefineBlock(nn.Module):
    """Small spatial residual block used by the patch-token top-down path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm_1 = nn.GroupNorm(1, channels)
        self.conv_1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm_2 = nn.GroupNorm(1, channels)
        self.conv_2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.conv_1(F.gelu(self.norm_1(feature)))
        residual = self.conv_2(F.gelu(self.norm_2(residual)))
        return feature + residual


class DPTLiteLocalGlobalPyramid(nn.Module):
    """Fuse cached VGGT depths while preserving local/global patch streams.

    VGGT-Omega concatenates a 1024D within-frame state and a 1024D
    inter-frame state at every cached layer.  Each half is normalized,
    projected and fused top-down independently.  They meet only after the
    multi-layer DPT-lite refinement at each output pyramid level.
    """

    def __init__(
        self,
        layers: tuple[int, ...],
        input_dim: int,
        hidden_dim: int,
        stream_dim: int,
        spatial_scales: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(layers) != len(spatial_scales) or not layers:
            raise ValueError("every cached token layer needs one spatial scale")
        if input_dim % 2:
            raise ValueError("local/global VGGT token width must be even")
        if hidden_dim <= 0 or stream_dim <= 0:
            raise ValueError("DPT-lite channel dimensions must be positive")
        if any((scale <= 0 for scale in spatial_scales)):
            raise ValueError("DPT-lite spatial scales must be positive")
        if tuple(layers) != tuple(sorted(layers)):
            raise ValueError("DPT-lite cached layers must run shallow to deep")
        if any(
            (
                shallow_scale < deep_scale
                for shallow_scale, deep_scale in zip(
                    spatial_scales[:-1], spatial_scales[1:]
                )
            )
        ):
            raise ValueError("DPT-lite spatial scales must run high to low resolution")
        self.layers = layers
        self.spatial_scales = spatial_scales
        self.input_dim = int(input_dim)
        self.input_stream_dim = input_dim // 2
        self.stream_dim = int(stream_dim)
        self.hidden_dim = int(hidden_dim)
        self.local_projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(self.input_stream_dim),
                    nn.Linear(self.input_stream_dim, stream_dim),
                )
                for layer in layers
            }
        )
        self.global_projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.LayerNorm(self.input_stream_dim),
                    nn.Linear(self.input_stream_dim, stream_dim),
                )
                for layer in layers
            }
        )
        self.local_level_embedding = nn.Parameter(
            torch.empty(len(layers), 1, 1, 1, stream_dim)
        )
        self.global_level_embedding = nn.Parameter(
            torch.empty(len(layers), 1, 1, 1, stream_dim)
        )
        self.local_refinement = nn.ModuleList(
            [DPTLiteRefineBlock(stream_dim) for _ in layers]
        )
        self.global_refinement = nn.ModuleList(
            [DPTLiteRefineBlock(stream_dim) for _ in layers]
        )
        self.stream_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(stream_dim * 2, hidden_dim, 1),
                    nn.GELU(),
                    DPTLiteRefineBlock(hidden_dim),
                )
                for _ in layers
            ]
        )
        nn.init.normal_(self.local_level_embedding, std=0.02)
        nn.init.normal_(self.global_level_embedding, std=0.02)

    @staticmethod
    def _resize_frames(feature: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        batch, frames, channels, _, _ = feature.shape
        return F.interpolate(
            feature.flatten(0, 1), size=size, mode="bilinear", align_corners=False
        ).view(batch, frames, channels, *size)

    @staticmethod
    def _refine_frames(feature: torch.Tensor, block: nn.Module) -> torch.Tensor:
        batch, frames, _, height, width = feature.shape
        refined = block(feature.flatten(0, 1))
        return refined.view(batch, frames, refined.shape[1], height, width)

    def _project_streams(
        self, tokens: dict[int, torch.Tensor], patch_grid: tuple[int, int]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        patch_height, patch_width = patch_grid
        local_levels: list[torch.Tensor] = []
        global_levels: list[torch.Tensor] = []
        for level_index, layer in enumerate(self.layers):
            if layer not in tokens:
                raise KeyError(f"missing cached VGGT layer {layer}")
            value = tokens[layer]
            if value.shape[-1] != self.input_dim:
                raise ValueError(
                    f"cached VGGT layer {layer} width must be {self.input_dim}"
                )
            batch, frames, patches, _ = value.shape
            if patches != patch_height * patch_width:
                raise ValueError(
                    f"layer {layer} has {patches} tokens but grid is {patch_height}x{patch_width}"
                )
            local = self.local_projections[str(layer)](
                value[..., : self.input_stream_dim]
            )
            global_ = self.global_projections[str(layer)](
                value[..., self.input_stream_dim :]
            )
            local = local + self.local_level_embedding[level_index]
            global_ = global_ + self.global_level_embedding[level_index]
            size = (
                max(1, round(patch_height * self.spatial_scales[level_index])),
                max(1, round(patch_width * self.spatial_scales[level_index])),
            )
            local = local.view(
                batch, frames, patch_height, patch_width, self.stream_dim
            ).permute(0, 1, 4, 2, 3)
            global_ = global_.view(
                batch, frames, patch_height, patch_width, self.stream_dim
            ).permute(0, 1, 4, 2, 3)
            local_levels.append(self._resize_frames(local, size))
            global_levels.append(self._resize_frames(global_, size))
        return (local_levels, global_levels)

    def forward(
        self, tokens: dict[int, torch.Tensor], patch_grid: tuple[int, int]
    ) -> list[torch.Tensor]:
        local_levels, global_levels = self._project_streams(tokens, patch_grid)
        outputs: list[torch.Tensor | None] = [None] * len(self.layers)
        local_state: torch.Tensor | None = None
        global_state: torch.Tensor | None = None
        for level_index in range(len(self.layers) - 1, -1, -1):
            size = local_levels[level_index].shape[-2:]
            local_state = (
                local_levels[level_index]
                if local_state is None
                else local_levels[level_index] + self._resize_frames(local_state, size)
            )
            global_state = (
                global_levels[level_index]
                if global_state is None
                else global_levels[level_index]
                + self._resize_frames(global_state, size)
            )
            local_state = self._refine_frames(
                local_state, self.local_refinement[level_index]
            )
            global_state = self._refine_frames(
                global_state, self.global_refinement[level_index]
            )
            fused = torch.cat((local_state, global_state), dim=2)
            outputs[level_index] = self._refine_frames(
                fused, self.stream_fusion[level_index]
            )
        if any((level is None for level in outputs)):
            raise RuntimeError("DPT-lite did not produce every pyramid level")
        return [level for level in outputs if level is not None]


class PrefixContextBlock(nn.Module):
    """Content-only self-attention over the ordered special-token sequence."""

    def __init__(self, hidden_dim: int, heads: int, expansion: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiheadAttention(hidden_dim, heads, mode="exact")
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(
        self, tokens: torch.Tensor, token_reliability: torch.Tensor | None
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, normalized, token_reliability)
        return tokens + self.ffn(self.ffn_norm(tokens))


class RoleSeparatedPrefixTrunk(nn.Module):
    """Contextualize Camera/Register tokens without pooling or geometry heads.

    Camera and Register inputs use different projections and remain distinct
    token slots throughout.  The module predicts no pose, depth, scale or
    calibration quantity; gradients arrive only through the BEV objective.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        maximum_history: int,
        maximum_prefix_tokens: int,
    ) -> None:
        super().__init__()
        if layers <= 0 or maximum_history <= 0 or maximum_prefix_tokens < 2:
            raise ValueError("prefix context dimensions must be positive")
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.hidden_dim = int(hidden_dim)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
        )
        self.register_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
        )
        self.frame_age_embedding = nn.Parameter(
            torch.empty(maximum_history, hidden_dim)
        )
        self.camera_type_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.register_type_embedding = nn.Parameter(
            torch.empty(maximum_prefix_tokens - 1, hidden_dim)
        )
        self.latest_reference_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.blocks = nn.ModuleList(
            [PrefixContextBlock(hidden_dim, heads) for _ in range(layers)]
        )
        self.camera_output_norm = nn.LayerNorm(hidden_dim)
        self.register_output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.frame_age_embedding, std=0.02)
        nn.init.normal_(self.camera_type_embedding, std=0.02)
        nn.init.normal_(self.register_type_embedding, std=0.02)
        nn.init.normal_(self.latest_reference_embedding, std=0.02)

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        *,
        frame_reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_tokens.ndim != 4:
            raise ValueError("prefix tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, _ = prefix_tokens.shape
        if not 1 <= frames <= self.maximum_history:
            raise ValueError("frame count exceeds prefix context contract")
        if prefix_count != self.maximum_prefix_tokens:
            raise ValueError("prefix context requires its exact token count")
        camera = self.camera_projection(prefix_tokens[:, :, 0])
        registers = self.register_projection(prefix_tokens[:, :, 1:])
        frame_age = self.frame_age_embedding[:frames][None, :, None, :]
        camera = (
            camera[:, :, None, :]
            + frame_age
            + self.camera_type_embedding[None, None, None, :]
        )
        registers = (
            registers
            + frame_age
            + self.register_type_embedding[: prefix_count - 1][None, None]
        )
        tokens = torch.cat((camera, registers), dim=2)
        tokens = tokens.clone()
        tokens[:, 0] = tokens[:, 0] + self.latest_reference_embedding
        tokens = tokens.reshape(batch, frames * prefix_count, self.hidden_dim)
        token_reliability = None
        if frame_reliability is not None:
            if frame_reliability.shape != (batch, frames):
                raise ValueError("frame_reliability must have shape [B,N]")
            token_reliability = frame_reliability.repeat_interleave(prefix_count, dim=1)
        for block in self.blocks:
            tokens = block(tokens, token_reliability)
        tokens = tokens.view(batch, frames, prefix_count, self.hidden_dim)
        camera_context = self.camera_output_norm(tokens[:, :, 0])
        register_context = self.register_output_norm(tokens[:, :, 1:])
        return (camera_context, register_context)


class PerCellPrefixReader(nn.Module):
    """Let each BEV cell read distinct Camera/Register token content."""

    def __init__(
        self, *, prefix_dim: int, hidden_dim: int, heads: int, query_chunk_size: int
    ) -> None:
        super().__init__()
        if query_chunk_size <= 0:
            raise ValueError("prefix query chunk size must be positive")
        self.query_chunk_size = int(query_chunk_size)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.camera_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, hidden_dim)
        )
        self.register_projection = nn.Sequential(
            nn.LayerNorm(prefix_dim), nn.Linear(prefix_dim, hidden_dim)
        )
        self.attention = MultiheadAttention(
            hidden_dim, heads, mode="exact", projection_fusion="kv"
        )

    def forward(
        self,
        query: torch.Tensor,
        camera_token: torch.Tensor,
        register_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if camera_token.ndim != 2 or register_tokens.ndim != 3:
            raise ValueError("per-cell prefix inputs have invalid ranks")
        if camera_token.shape[0] != query.shape[0]:
            raise ValueError("camera/query batches must match")
        if register_tokens.shape[0] != query.shape[0]:
            raise ValueError("register/query batches must match")
        memory = torch.cat(
            (
                self.camera_projection(camera_token)[:, None],
                self.register_projection(register_tokens),
            ),
            dim=1,
        )
        normalized = self.query_norm(query)
        chunks = []
        for start in range(0, query.shape[1], self.query_chunk_size):
            chunks.append(
                self.attention(
                    normalized[:, start : start + self.query_chunk_size], memory
                )
            )
        return torch.cat(chunks, dim=1)


class ProjectedDenseMetricQuery(nn.Module):
    """Full native cell identity with a wider shared decoder state."""

    def __init__(
        self,
        *,
        size: int,
        extent_m: float,
        content_dim: int,
        hidden_dim: int,
        fourier_bands: int,
    ) -> None:
        super().__init__()
        self.content = DenseNativeQuery(
            size=size,
            extent_m=extent_m,
            hidden_dim=content_dim,
            fourier_bands=fourier_bands,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(content_dim),
            nn.Linear(content_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def size(self) -> int:
        return self.content.size

    @property
    def extent_m(self) -> float:
        return self.content.extent_m

    @property
    def reference_grid(self) -> torch.Tensor:
        return self.content.reference_grid

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.projection(self.content(batch_size))


class TemporalFrameProposal(nn.Module):
    """Create one BEV residual proposal from one historical frame."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        heads: int,
        feature_levels: int,
        layers: int,
        deformable_samples: int,
        cross_query_chunk_size: int,
    ) -> None:
        super().__init__()
        self.state_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )
        self.context_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )
        self.attentions = nn.ModuleList(
            [
                DeformableCrossAttention(
                    hidden_dim,
                    heads,
                    levels=feature_levels,
                    samples=deformable_samples,
                    query_chunk_size=cross_query_chunk_size,
                )
                for _ in range(layers)
            ]
        )
        self.frame_conditions = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)]
        )
        self.delta_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim, elementwise_affine=False),
                    nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
                )
                for _ in range(layers)
            ]
        )
        for projection in self.delta_projections:
            nn.init.zeros_(projection[-1].weight)

    def forward(
        self,
        anchor: torch.Tensor,
        frame_pyramid: list[torch.Tensor],
        reference_grid: torch.Tensor,
        frame_context: torch.Tensor,
    ) -> torch.Tensor:
        proposal = anchor
        for state_norm, context_norm, attention, condition, projection in zip(
            self.state_norms,
            self.context_norms,
            self.attentions,
            self.frame_conditions,
            self.delta_projections,
        ):
            context = [
                context_norm(level.movedim(2, -1)).movedim(-1, 2)
                for level in frame_pyramid
            ]
            conditioned = condition(frame_context)
            if conditioned.ndim == 2:
                conditioned = conditioned[:, None, :].expand_as(proposal)
            if conditioned.shape != proposal.shape:
                raise ValueError(
                    "frame context must resolve to one feature per BEV query"
                )
            query = proposal + conditioned
            update = attention(state_norm(query), context, reference_grid)
            proposal = proposal + projection(update)
        return proposal - anchor


class PerQueryTemporalAttention(nn.Module):
    """Choose historical evidence independently at every BEV query.

    A learned null candidate lets a query preserve the latest anchor when no
    historical frame is useful. The softmax is only across homogeneous frame
    proposals; camera/register token roles remain separated upstream.
    """

    def __init__(
        self, hidden_dim: int, maximum_history: int, *, null_initial_probability: float
    ) -> None:
        super().__init__()
        if not 0.0 < null_initial_probability < 1.0:
            raise ValueError("null_initial_probability must be in (0,1)")
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.delta_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.frame_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1)
        self.null_score = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.frame_age_bias = nn.Parameter(torch.zeros(maximum_history - 1))
        initial_log_odds = math.log(
            null_initial_probability / (1.0 - null_initial_probability)
        )
        self.register_buffer(
            "null_initial_log_odds", torch.tensor(initial_log_odds, dtype=torch.float32)
        )
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.null_score[-1].weight)
        nn.init.zeros_(self.null_score[-1].bias)

    def forward(
        self,
        anchor: torch.Tensor,
        proposals: list[torch.Tensor] | torch.Tensor,
        frame_contexts: list[torch.Tensor] | torch.Tensor,
        frame_reliabilities: list[torch.Tensor] | torch.Tensor,
        frame_ages: list[int] | tuple[int, ...] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(proposals, list):
            if not proposals:
                empty = anchor.new_zeros((anchor.shape[0], 0))
                return (anchor, empty, anchor.new_ones((anchor.shape[0],)))
            proposal_tensor = torch.stack(proposals, dim=1)
            if not isinstance(frame_contexts, list) or not isinstance(
                frame_reliabilities, list
            ):
                raise ValueError("list proposals require list temporal metadata")
            context_tensor = torch.stack(frame_contexts, dim=1)
            reliability_tensor = torch.stack(frame_reliabilities, dim=1)
        else:
            proposal_tensor = proposals
            if not isinstance(frame_contexts, torch.Tensor) or not isinstance(
                frame_reliabilities, torch.Tensor
            ):
                raise ValueError("batched proposals require tensor temporal metadata")
            context_tensor = frame_contexts
            reliability_tensor = frame_reliabilities
        if proposal_tensor.ndim != 4 or proposal_tensor.shape[0] != anchor.shape[0]:
            raise ValueError("temporal proposals must have shape [B,H,Q,C]")
        if proposal_tensor.shape[2:] != anchor.shape[1:]:
            raise ValueError("temporal proposals must align with anchor queries")
        history_count = proposal_tensor.shape[1]
        if history_count == 0:
            empty = anchor.new_zeros((anchor.shape[0], 0))
            return (anchor, empty, anchor.new_ones((anchor.shape[0],)))
        if reliability_tensor.shape != (anchor.shape[0], history_count):
            raise ValueError("temporal reliability must have shape [B,H]")
        projected_context = self.frame_projection(context_tensor)
        if projected_context.ndim == 3:
            projected_context = projected_context[:, :, None, :]
        expected_context = (
            anchor.shape[0],
            history_count,
            anchor.shape[1],
            anchor.shape[2],
        )
        if projected_context.shape not in {
            expected_context,
            (anchor.shape[0], history_count, 1, anchor.shape[2]),
        }:
            raise ValueError("temporal frame context must be per-frame or per-query")
        ages = torch.as_tensor(
            frame_ages, device=self.frame_age_bias.device, dtype=torch.long
        )
        if ages.shape != (history_count,):
            raise ValueError("temporal frame ages must have shape [H]")
        if bool(((ages < 1) | (ages > self.frame_age_bias.numel())).any()):
            raise ValueError("temporal frame ages exceed the configured history")
        feature = torch.tanh(
            self.state_projection(anchor)[:, None]
            + self.delta_projection(proposal_tensor)
            + projected_context
        )
        history_scores = self.score(feature).squeeze(-1).float()
        history_scores = (
            history_scores
            + reliability_tensor.float().clamp_min(0.0001).log()[:, :, None]
        )
        history_scores = history_scores + self.frame_age_bias[ages - 1][None, :, None]
        null_score = self.null_score(anchor).squeeze(-1).unsqueeze(1).float()
        null_score = null_score + math.log(history_count) + self.null_initial_log_odds
        weights = torch.softmax(
            torch.cat((null_score, history_scores), dim=1), dim=1
        ).to(dtype=anchor.dtype)
        history_weights = weights[:, 1:]
        update = (history_weights.unsqueeze(-1) * proposal_tensor).sum(dim=1)
        merged = anchor + update
        return (
            merged,
            history_weights.float().mean(dim=2),
            weights[:, 0].float().mean(dim=1),
        )


class BEVHead(nn.Module):
    """EchoNav Gen.1 RGB-only, fixed-metric BEV prediction head."""

    def __init__(
        self,
        *,
        cached_layers: tuple[int, ...],
        spatial_scales: tuple[float, ...],
        vggt_token_dim: int,
        hidden_dim: int,
        query_content_dim: int,
        heads: int,
        latest_decoder_layers: int,
        temporal_proposal_layers: int,
        self_attention_mode: str,
        deformable_samples: int,
        cross_query_chunk_size: int,
        merged_bev_size: int,
        merged_extent_m: float,
        query_fourier_bands: int,
        shared_refinement_layers: int,
        routing_refinement_layers: int,
        evidence_refinement_layers: int,
        patch_stream_dim: int,
        prefix_context_hidden_dim: int,
        prefix_context_heads: int,
        prefix_context_layers: int,
        maximum_history: int,
        maximum_prefix_tokens: int,
        frame_reliability_hidden_dim: int,
        frame_reliability_minimum: float,
        frame_reliability_maximum: float,
        temporal_null_initial_probability: float = 0.9,
        history_proposal_batch_size: int = 1,
    ) -> None:
        super().__init__()
        if history_proposal_batch_size <= 0:
            raise ValueError("history proposal batch size must be positive")
        self.maximum_history = int(maximum_history)
        self.maximum_prefix_tokens = int(maximum_prefix_tokens)
        self.vggt_token_dim = int(vggt_token_dim)
        self.history_proposal_batch_size = int(history_proposal_batch_size)
        self.channels_last_spatial = False
        self.temporal_null_initial_probability = float(
            temporal_null_initial_probability
        )
        self.spatial_token_projector = DPTLiteLocalGlobalPyramid(
            cached_layers, vggt_token_dim, hidden_dim, patch_stream_dim, spatial_scales
        )
        self.prefix_context_trunk = RoleSeparatedPrefixTrunk(
            input_dim=vggt_token_dim,
            hidden_dim=prefix_context_hidden_dim,
            heads=prefix_context_heads,
            layers=prefix_context_layers,
            maximum_history=maximum_history,
            maximum_prefix_tokens=maximum_prefix_tokens,
        )
        self.prefix_reader = PerCellPrefixReader(
            prefix_dim=prefix_context_hidden_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            query_chunk_size=cross_query_chunk_size,
        )
        self.frame_reliability = StructuredFrameReliabilityHead(
            vggt_token_dim,
            frame_reliability_hidden_dim,
            maximum_prefix_tokens=maximum_prefix_tokens,
            minimum=frame_reliability_minimum,
            maximum=frame_reliability_maximum,
        )
        self.query = ProjectedDenseMetricQuery(
            size=merged_bev_size,
            extent_m=merged_extent_m,
            content_dim=query_content_dim,
            hidden_dim=hidden_dim,
            fourier_bands=query_fourier_bands,
        )
        self.latest_branch = ParallelQueryBranch(
            hidden_dim=hidden_dim,
            heads=heads,
            layers=latest_decoder_layers,
            feature_levels=len(cached_layers),
            self_attention_mode=self_attention_mode,
            cross_attention_mode="deformable",
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.history_proposal = TemporalFrameProposal(
            hidden_dim=hidden_dim,
            heads=heads,
            feature_levels=len(cached_layers),
            layers=temporal_proposal_layers,
            deformable_samples=deformable_samples,
            cross_query_chunk_size=cross_query_chunk_size,
        )
        self.temporal_attention = PerQueryTemporalAttention(
            hidden_dim,
            maximum_history,
            null_initial_probability=temporal_null_initial_probability,
        )
        self.shared_refinement = nn.Sequential(
            *[SpatialResidualBlock(hidden_dim) for _ in range(shared_refinement_layers)]
        )
        self.routing_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(routing_refinement_layers)
            ]
        )
        self.evidence_refinement = nn.Sequential(
            *[
                SpatialResidualBlock(hidden_dim)
                for _ in range(evidence_refinement_layers)
            ]
        )
        self.routing_head = nn.Conv2d(hidden_dim, 2, 1)
        self.evidence_head = nn.Conv2d(hidden_dim, 2, 1)

    @property
    def merged_bev_size(self) -> int:
        return self.query.size

    @property
    def merged_extent_m(self) -> float:
        return self.query.extent_m

    def _decode_state(
        self, state: torch.Tensor, *, assemble_runtime_outputs: bool
    ) -> dict:
        spatial = state.transpose(1, 2).reshape(
            state.shape[0], -1, self.merged_bev_size, self.merged_bev_size
        )
        if self.channels_last_spatial:
            spatial = spatial.contiguous(memory_format=torch.channels_last)
        shared = self.shared_refinement(spatial)
        routing = self.routing_refinement(shared)
        evidence = self.evidence_refinement(shared)
        return assemble_bev(
            self.evidence_head(evidence),
            self.routing_head(routing),
            assemble_runtime_outputs=assemble_runtime_outputs,
        )

    @classmethod
    def _split_decoded_batch(cls, value, batch: int):
        if isinstance(value, torch.Tensor):
            if value.shape[0] != batch * 2:
                raise ValueError("batched BEV decoder output has an invalid batch")
            return (value[:batch], value[batch:])
        if isinstance(value, dict):
            first = {}
            second = {}
            for key, child in value.items():
                first[key], second[key] = cls._split_decoded_batch(child, batch)
            return (first, second)
        raise TypeError("BEV decoder outputs must be tensors or dictionaries")

    def _unused_temporal_zero(self, anchor: torch.Tensor) -> torch.Tensor:
        zero = anchor.new_zeros(())
        for module in (self.history_proposal, self.temporal_attention):
            for parameter in module.parameters():
                zero = zero + parameter.reshape(-1)[0].to(anchor.dtype) * 0.0
        return zero

    def forward(
        self,
        extraction: dict,
        *,
        include_merged: bool = True,
        include_latest_auxiliary: bool = False,
        assemble_runtime_outputs: bool = True,
    ) -> dict:
        prefix = extraction["camera_register_tokens"]
        if prefix.ndim != 4:
            raise ValueError("EchoNav Gen.1 prefix tokens must have shape [B,N,P,C]")
        batch, frames, prefix_count, channels = prefix.shape
        if not 1 <= frames <= self.maximum_history:
            raise ValueError("EchoNav Gen.1 RGB history length is outside its contract")
        if prefix_count != self.maximum_prefix_tokens:
            raise ValueError(
                f"EchoNav Gen.1 requires exactly one camera token plus the configured register tokens ({self.maximum_prefix_tokens} total)"
            )
        if channels != self.vggt_token_dim:
            raise ValueError(
                f"EchoNav Gen.1 prefix width must be {self.vggt_token_dim}, got {channels}"
            )
        patch_height, patch_width = extraction["patch_grid"]
        tokens = extraction.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError(
                "EchoNav Gen.1 extraction tokens must be a layer dictionary"
            )
        for layer in self.spatial_token_projector.layers:
            value = tokens.get(layer)
            if value is None:
                raise KeyError(f"missing cached VGGT layer {layer}")
            expected = (batch, frames, patch_height * patch_width, channels)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"cached VGGT layer {layer} must have shape {expected}, got {tuple(value.shape)}"
                )
        reliability = self.frame_reliability(prefix)
        output: dict = {
            "frame_reliability": reliability,
            "effective_frame_reliability": reliability,
            "frame_keep_mask": torch.ones_like(reliability, dtype=torch.bool),
        }
        if include_merged:
            pyramid = self.spatial_token_projector(
                extraction["tokens"], extraction["patch_grid"]
            )
            camera_context, register_context = self.prefix_context_trunk(
                prefix, frame_reliability=reliability
            )
            query = self.query(batch)
            latest_prefix = self.prefix_reader(
                query, camera_context[:, 0], register_context[:, 0]
            )
            anchor = self.latest_branch(
                query + latest_prefix,
                [level[:, :1] for level in pyramid],
                self.query.reference_grid,
            )
            frame_order = tuple(range(1, frames))
            proposal_chunks: list[torch.Tensor] = []
            context_chunks: list[torch.Tensor] = []
            for history_start in range(1, frames, self.history_proposal_batch_size):
                history_end = min(
                    history_start + self.history_proposal_batch_size, frames
                )
                history_count = history_end - history_start
                batched_anchor = (
                    anchor[:, None]
                    .expand(-1, history_count, -1, -1)
                    .reshape(batch * history_count, anchor.shape[1], anchor.shape[2])
                )
                batched_context = self.prefix_reader(
                    batched_anchor,
                    camera_context[:, history_start:history_end].reshape(
                        batch * history_count, -1
                    ),
                    register_context[:, history_start:history_end].reshape(
                        batch * history_count,
                        register_context.shape[2],
                        register_context.shape[3],
                    ),
                )
                batched_pyramid = [
                    level[:, history_start:history_end].reshape(
                        batch * history_count,
                        1,
                        level.shape[2],
                        level.shape[3],
                        level.shape[4],
                    )
                    for level in pyramid
                ]
                proposal_chunks.append(
                    self.history_proposal(
                        batched_anchor,
                        batched_pyramid,
                        self.query.reference_grid,
                        batched_context,
                    ).view(batch, history_count, anchor.shape[1], anchor.shape[2])
                )
                context_chunks.append(
                    batched_context.view(
                        batch, history_count, anchor.shape[1], anchor.shape[2]
                    )
                )
            if proposal_chunks:
                proposal_tensor = torch.cat(proposal_chunks, dim=1)
                context_tensor = torch.cat(context_chunks, dim=1)
            else:
                proposal_tensor = anchor.new_empty(
                    batch, 0, anchor.shape[1], anchor.shape[2]
                )
                context_tensor = anchor.new_empty(
                    batch, 0, anchor.shape[1], anchor.shape[2]
                )
            merged, attention_mean, null_mean = self.temporal_attention(
                anchor, proposal_tensor, context_tensor, reliability[:, 1:], frame_order
            )
            if frames == 1:
                merged = merged + self._unused_temporal_zero(anchor)
            output["history_frame_indices_newest_to_oldest"] = frame_order
            output["history_original_indices_newest_to_oldest"] = tuple(
                range(frames - 2, -1, -1)
            )
            output["temporal_frame_attention_mean"] = attention_mean
            output["temporal_null_attention_mean"] = null_mean
            output["history_update_gate_mean"] = attention_mean
            if include_latest_auxiliary:
                decoded = self._decode_state(
                    torch.cat((anchor, merged), dim=0),
                    assemble_runtime_outputs=assemble_runtime_outputs,
                )
                output["latest_auxiliary_bev"], output["merged_bev"] = (
                    self._split_decoded_batch(decoded, batch)
                )
            else:
                output["merged_bev"] = self._decode_state(
                    merged, assemble_runtime_outputs=assemble_runtime_outputs
                )
        return output
