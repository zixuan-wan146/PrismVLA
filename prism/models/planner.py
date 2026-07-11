from __future__ import annotations

# --- migrated from src/prism/model/planner/action_segment_autoencoder.py ---
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionSegmentAutoencoderConfig:
    action_dim: int
    chunk_size: int
    latent_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int | None = None
    dropout: float = 0.05
    gripper_dim: int = 1


@dataclass(frozen=True)
class ActionSegmentAutoencoderOutput:
    latents: torch.Tensor
    reconstruction: torch.Tensor


class ActionSegmentAutoencoder(nn.Module):
    """Action-only autoencoder that defines a latent for one future action chunk."""

    def __init__(self, config: ActionSegmentAutoencoderConfig) -> None:
        super().__init__()
        if config.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {config.action_dim}")
        if config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")
        if config.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {config.latent_dim}")
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
        if config.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {config.num_layers}")
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {config.num_heads}")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.ffn_dim is not None and config.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {config.ffn_dim}")
        if float(config.dropout) < 0.0:
            raise ValueError(f"dropout must be non-negative, got {config.dropout}")
        if config.gripper_dim <= 0:
            raise ValueError(f"gripper_dim must be positive, got {config.gripper_dim}")
        if config.gripper_dim >= config.action_dim:
            raise ValueError("gripper_dim must be smaller than action_dim")

        self.config = config
        ffn_dim = config.ffn_dim or config.hidden_dim * 4
        self.motion_dim = config.action_dim - config.gripper_dim

        self.input_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        self.encoder_pos = nn.Parameter(torch.empty(1, config.chunk_size + 1, config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.time_queries = nn.Parameter(torch.empty(1, config.chunk_size, config.hidden_dim))
        self.decoder_pos = nn.Parameter(torch.empty(1, config.chunk_size + 1, config.hidden_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        self.motion_head = nn.Linear(config.hidden_dim, self.motion_dim)
        self.gripper_head = nn.Linear(config.hidden_dim, config.gripper_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.encoder_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_pos, mean=0.0, std=0.02)

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.input_proj.to(device=device, dtype=dtype)
        self.encoder.to(device=device, dtype=dtype)
        self.latent_head.to(device=device, dtype=dtype)
        self.latent_proj.to(device=device, dtype=dtype)
        self.decoder.to(device=device, dtype=dtype)
        self.motion_head.to(device=device, dtype=dtype)
        self.gripper_head.to(device=device, dtype=dtype)

    def encode(self, action_segments: torch.Tensor) -> torch.Tensor:
        segments, leading_shape = self._flatten_segments(action_segments)
        device = segments.device
        dtype = segments.dtype
        self._match_runtime_dtype(device=device, dtype=dtype)
        tokens = self.input_proj(segments)
        cls = self.cls_token.to(device=device, dtype=dtype).expand(segments.shape[0], -1, -1)
        pos = self.encoder_pos.to(device=device, dtype=dtype)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + pos)
        latents = self.latent_head(encoded[:, 0])
        return latents.reshape(*leading_shape, self.config.latent_dim)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        leading_shape = latents.shape[:-1]
        if latents.shape[-1] != self.config.latent_dim:
            raise ValueError(f"latent dim {latents.shape[-1]} != {self.config.latent_dim}")
        device = latents.device
        dtype = latents.dtype
        self._match_runtime_dtype(device=device, dtype=dtype)
        flat = latents.reshape(-1, self.config.latent_dim)
        latent_token = self.latent_proj(flat).unsqueeze(1)
        queries = self.time_queries.to(device=device, dtype=dtype).expand(flat.shape[0], -1, -1)
        pos = self.decoder_pos.to(device=device, dtype=dtype)
        decoded = self.decoder(torch.cat([latent_token, queries], dim=1) + pos)
        query_output = decoded[:, 1:]
        motion = self.motion_head(query_output)
        gripper = self.gripper_head(query_output)
        reconstruction = torch.cat([motion, gripper], dim=-1)
        return reconstruction.reshape(*leading_shape, self.config.chunk_size, self.config.action_dim)

    def forward(self, action_segments: torch.Tensor) -> ActionSegmentAutoencoderOutput:
        latents = self.encode(action_segments)
        reconstruction = self.decode(latents)
        return ActionSegmentAutoencoderOutput(latents=latents, reconstruction=reconstruction)

    def _flatten_segments(self, action_segments: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        if action_segments.shape[-2:] != (self.config.chunk_size, self.config.action_dim):
            raise ValueError(
                "action_segments must end with "
                f"[{self.config.chunk_size}, {self.config.action_dim}], got {tuple(action_segments.shape)}"
            )
        leading_shape = action_segments.shape[:-2]
        return action_segments.reshape(-1, self.config.chunk_size, self.config.action_dim), leading_shape


def action_segment_autoencoder_loss(
    model: ActionSegmentAutoencoder,
    action_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    gripper_indices: tuple[int, ...] | list[int] | None = (-1,),
    gripper_loss_weight: float = 1.0,
    distance_loss_weight: float = 0.0,
    dct_low_frequency: int = 4,
    endpoint_distance_weight: float = 1.0,
    gripper_distance_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(action_segments)
    rec_loss = action_segment_reconstruction_loss(
        output.reconstruction,
        action_segments,
        segment_mask,
        gripper_indices=gripper_indices,
        gripper_loss_weight=gripper_loss_weight,
    )
    dist_loss = action_segment_distance_loss(
        output.latents,
        action_segments,
        segment_mask,
        gripper_indices=gripper_indices,
        dct_low_frequency=dct_low_frequency,
        endpoint_distance_weight=endpoint_distance_weight,
        gripper_distance_weight=gripper_distance_weight,
    )
    loss = rec_loss + float(distance_loss_weight) * dist_loss
    return loss, {
        "segment_ae_rec_loss": rec_loss.detach(),
        "segment_ae_dist_loss": dist_loss.detach(),
    }


def action_segment_reconstruction_loss(
    predicted_segments: torch.Tensor,
    target_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    gripper_indices: tuple[int, ...] | list[int] | None = (-1,),
    gripper_loss_weight: float = 1.0,
) -> torch.Tensor:
    if predicted_segments.shape != target_segments.shape:
        raise ValueError(f"predicted_segments shape {predicted_segments.shape} != target_segments shape {target_segments.shape}")
    if predicted_segments.ndim != 4:
        raise ValueError(f"action segments must have shape [B, K, C, A], got {predicted_segments.shape}")

    mask = _normalize_segment_mask(segment_mask, predicted_segments)
    target = target_segments.to(device=predicted_segments.device, dtype=predicted_segments.dtype)
    action_dim = predicted_segments.shape[-1]
    gripper = _normalize_indices(gripper_indices, action_dim)
    motion = tuple(index for index in range(action_dim) if index not in gripper)

    rec_loss = predicted_segments.new_zeros(())
    weight_sum = predicted_segments.new_zeros(())
    if motion:
        motion_loss = F.smooth_l1_loss(predicted_segments[..., motion], target[..., motion], reduction="none").mean(dim=(-1, -2))
        rec_loss = rec_loss + (motion_loss * mask).sum()
        weight_sum = weight_sum + mask.sum()
    if gripper:
        grip_target = target[..., gripper].clamp(0.0, 1.0)
        grip_loss = F.binary_cross_entropy_with_logits(
            predicted_segments[..., gripper],
            grip_target,
            reduction="none",
        ).mean(dim=(-1, -2))
        rec_loss = rec_loss + float(gripper_loss_weight) * (grip_loss * mask).sum()
        weight_sum = weight_sum + float(gripper_loss_weight) * mask.sum()
    if weight_sum.item() == 0:
        raise ValueError("action_segment_mask.sum() is 0. All action segments are masked.")
    return rec_loss / weight_sum


def action_segment_distance_loss(
    latents: torch.Tensor,
    action_segments: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    gripper_indices: tuple[int, ...] | list[int] | None = (-1,),
    dct_low_frequency: int = 4,
    endpoint_distance_weight: float = 1.0,
    gripper_distance_weight: float = 1.0,
) -> torch.Tensor:
    if latents.shape[:-1] != action_segments.shape[:-2]:
        raise ValueError(f"latent leading shape {latents.shape[:-1]} != action segment shape {action_segments.shape[:-2]}")
    mask = _normalize_segment_mask(segment_mask, action_segments).reshape(-1).bool()
    if mask.sum().item() <= 1:
        return latents.new_zeros(())

    flat_latents = latents.reshape(-1, latents.shape[-1])[mask]
    flat_segments = action_segments.to(device=latents.device, dtype=latents.dtype).reshape(
        -1, action_segments.shape[-2], action_segments.shape[-1]
    )[mask]
    latent_dist = torch.cdist(flat_latents, flat_latents, p=2)
    action_dist = action_segment_distance_matrix(
        flat_segments,
        gripper_indices=gripper_indices,
        dct_low_frequency=dct_low_frequency,
        endpoint_distance_weight=endpoint_distance_weight,
        gripper_distance_weight=gripper_distance_weight,
    )
    latent_dist = _normalize_distance_matrix(latent_dist)
    action_dist = _normalize_distance_matrix(action_dist)
    pair_mask = ~torch.eye(latent_dist.shape[0], dtype=torch.bool, device=latent_dist.device)
    return F.smooth_l1_loss(latent_dist[pair_mask], action_dist[pair_mask])


def action_segment_distance_matrix(
    action_segments: torch.Tensor,
    *,
    gripper_indices: tuple[int, ...] | list[int] | None = (-1,),
    dct_low_frequency: int = 4,
    endpoint_distance_weight: float = 1.0,
    gripper_distance_weight: float = 1.0,
) -> torch.Tensor:
    if action_segments.ndim != 3:
        raise ValueError(f"action_segments must have shape [N, C, A], got {action_segments.shape}")
    action_dim = action_segments.shape[-1]
    gripper = _normalize_indices(gripper_indices, action_dim)
    motion = tuple(index for index in range(action_dim) if index not in gripper)
    pieces = []
    if motion:
        motion_segments = action_segments[..., motion]
        pieces.append(_dct_low_frequency(motion_segments, dct_low_frequency).reshape(action_segments.shape[0], -1))
        endpoint = motion_segments[:, -1]
        pieces.append(float(endpoint_distance_weight) * endpoint)
    if gripper:
        gripper_state = (action_segments[:, -1, gripper] > 0.5).to(action_segments.dtype)
        pieces.append(float(gripper_distance_weight) * gripper_state)
    if not pieces:
        raise ValueError("at least one motion or gripper index must be selected")
    features = torch.cat(pieces, dim=-1)
    return torch.cdist(features, features, p=2)


def _dct_low_frequency(values: torch.Tensor, num_frequency: int) -> torch.Tensor:
    chunk_size = values.shape[-2]
    count = max(1, min(int(num_frequency), chunk_size))
    n = torch.arange(chunk_size, device=values.device, dtype=values.dtype)
    k = torch.arange(count, device=values.device, dtype=values.dtype).unsqueeze(1)
    basis = torch.cos(torch.pi * (n + 0.5) * k / float(chunk_size))
    basis[0] = basis[0] * (1.0 / chunk_size) ** 0.5
    if count > 1:
        basis[1:] = basis[1:] * (2.0 / chunk_size) ** 0.5
    return torch.einsum("kc,nca->nka", basis, values)


def _normalize_distance_matrix(distance: torch.Tensor) -> torch.Tensor:
    pair_mask = ~torch.eye(distance.shape[0], dtype=torch.bool, device=distance.device)
    scale = distance[pair_mask].mean().clamp_min(1.0e-6)
    return distance / scale


def _normalize_segment_mask(segment_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = segment_mask.to(device=reference.device, dtype=reference.dtype)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.shape != reference.shape[:2]:
        raise ValueError(f"segment_mask shape {mask.shape} != segment shape {reference.shape[:2]}")
    if mask.sum().item() == 0:
        raise ValueError("action_segment_mask.sum() is 0. All action segments are masked.")
    return mask


def _normalize_indices(indices: tuple[int, ...] | list[int] | None, action_dim: int) -> tuple[int, ...]:
    if indices is None:
        return ()
    normalized = []
    for index in indices:
        value = int(index)
        if value < 0:
            value += action_dim
        if value < 0 or value >= action_dim:
            raise ValueError(f"action index {index} is out of range for action_dim {action_dim}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)

# --- migrated from src/prism/model/planner/progress_state.py ---
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProgressStateConfig:
    hidden_dim: int = 896
    state_dim: int = 8
    action_dim: int = 7
    replan_stride: int = 16
    latent_dim: int = 128
    action_summary_hidden_dim: int = 512
    state_hidden_dim: int = 512
    updater_hidden_dim: int = 1792
    planner_ffn_dim: int = 3584
    planner_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.05
    completed_gate_bias: float = -2.0
    stage_gate_bias: float = -1.0


@dataclass(frozen=True)
class ProgressState:
    completed_events: torch.Tensor
    current_stage: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        return torch.stack([self.completed_events, self.current_stage], dim=1)


@dataclass(frozen=True)
class ProgressPlannerOutput:
    progress_state: ProgressState
    planner_token: torch.Tensor
    progress_evidence: torch.Tensor


@dataclass(frozen=True)
class ProgressPretrainHeadOutput:
    planner_intent: torch.Tensor
    stage_intent: torch.Tensor
    memory_pool_intent: torch.Tensor
    progress_score: torch.Tensor


class ActionSummaryEncoder(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        input_dim = config.replan_stride * (config.action_dim + 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.action_summary_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.action_summary_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, executed_actions: torch.Tensor, executed_mask: torch.Tensor | None = None) -> torch.Tensor:
        actions = _ensure_rank3(executed_actions, "executed_actions")
        if actions.shape[-2:] != (self.config.replan_stride, self.config.action_dim):
            raise ValueError(
                "executed_actions must have shape "
                f"[B, {self.config.replan_stride}, {self.config.action_dim}], got {tuple(actions.shape)}"
            )
        if executed_mask is None:
            mask = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(executed_mask, device=actions.device).bool()
            if mask.shape != actions.shape[:2]:
                raise ValueError(f"executed_mask shape {tuple(mask.shape)} != {tuple(actions.shape[:2])}")
        self._match_runtime_dtype(device=actions.device, dtype=actions.dtype)
        mask_value = mask.to(dtype=actions.dtype).unsqueeze(-1)
        masked_actions = actions * mask_value
        features = torch.cat([masked_actions, mask_value], dim=-1).reshape(actions.shape[0], -1)
        summary = self.encoder(features)
        has_any_action = mask.any(dim=1).to(dtype=summary.dtype).unsqueeze(-1)
        return summary * has_any_action

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.encoder.to(device=device, dtype=dtype)


class ProgressEvidenceEncoder(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Linear(config.state_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.updater_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.updater_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, vl_summary: torch.Tensor, state: torch.Tensor, action_summary: torch.Tensor) -> torch.Tensor:
        vl_summary = _ensure_rank2(vl_summary, self.config.hidden_dim, "vl_summary")
        state = _ensure_rank2(state, self.config.state_dim, "state").to(device=vl_summary.device, dtype=vl_summary.dtype)
        action_summary = _ensure_rank2(action_summary, self.config.hidden_dim, "action_summary").to(
            device=vl_summary.device,
            dtype=vl_summary.dtype,
        )
        self._match_runtime_dtype(device=vl_summary.device, dtype=vl_summary.dtype)
        state_embedding = self.state_encoder(state)
        return self.fusion(torch.cat([vl_summary, state_embedding, action_summary], dim=-1))

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.state_encoder.to(device=device, dtype=dtype)
        self.fusion.to(device=device, dtype=dtype)


class ProgressStateUpdater(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.init_completed_events = nn.Parameter(torch.zeros(config.hidden_dim))
        self.init_current_stage = nn.Parameter(torch.zeros(config.hidden_dim))
        input_dim = config.hidden_dim * 3
        self.completed_delta = _mlp(input_dim, config.updater_hidden_dim, config.hidden_dim, config.dropout)
        self.completed_gate = nn.Linear(input_dim, config.hidden_dim)
        self.completed_norm = nn.LayerNorm(config.hidden_dim)
        self.stage_delta = _mlp(input_dim, config.updater_hidden_dim, config.hidden_dim, config.dropout)
        self.stage_gate = nn.Linear(input_dim, config.hidden_dim)
        self.stage_norm = nn.LayerNorm(config.hidden_dim)
        nn.init.constant_(self.completed_gate.bias, float(config.completed_gate_bias))
        nn.init.constant_(self.stage_gate.bias, float(config.stage_gate_bias))

    def initial_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ProgressState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        completed = self.init_completed_events
        stage = self.init_current_stage
        if device is not None or dtype is not None:
            completed = completed.to(device=device, dtype=dtype)
            stage = stage.to(device=device, dtype=dtype)
        return ProgressState(
            completed_events=completed.unsqueeze(0).expand(batch_size, -1),
            current_stage=stage.unsqueeze(0).expand(batch_size, -1),
        )

    def forward(self, previous_state: ProgressState | torch.Tensor, progress_evidence: torch.Tensor) -> ProgressState:
        completed_prev, stage_prev = _state_to_tokens(previous_state, self.config.hidden_dim)
        evidence = _ensure_rank2(progress_evidence, self.config.hidden_dim, "progress_evidence").to(
            device=completed_prev.device,
            dtype=completed_prev.dtype,
        )
        self._match_runtime_dtype(device=completed_prev.device, dtype=completed_prev.dtype)
        completed_input = torch.cat([completed_prev, stage_prev, evidence], dim=-1)
        completed_delta = self.completed_delta(completed_input)
        completed_gate = torch.sigmoid(self.completed_gate(completed_input))
        completed = self.completed_norm(completed_prev + completed_gate * completed_delta)

        stage_input = torch.cat([stage_prev, completed, evidence], dim=-1)
        stage_delta = self.stage_delta(stage_input)
        stage_gate = torch.sigmoid(self.stage_gate(stage_input))
        stage = self.stage_norm(stage_prev + stage_gate * stage_delta)
        return ProgressState(completed_events=completed, current_stage=stage)

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.completed_delta.to(device=device, dtype=dtype)
        self.completed_gate.to(device=device, dtype=dtype)
        self.completed_norm.to(device=device, dtype=dtype)
        self.stage_delta.to(device=device, dtype=dtype)
        self.stage_gate.to(device=device, dtype=dtype)
        self.stage_norm.to(device=device, dtype=dtype)


class ProgressPlanner(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.state_token = nn.Sequential(
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Linear(config.state_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.planner_ffn_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.planner_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, progress_state: ProgressState | torch.Tensor, vl_summary: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        completed, stage = _state_to_tokens(progress_state, self.config.hidden_dim)
        vl_summary = _ensure_rank2(vl_summary, self.config.hidden_dim, "vl_summary").to(
            device=completed.device,
            dtype=completed.dtype,
        )
        state = _ensure_rank2(state, self.config.state_dim, "state").to(device=completed.device, dtype=completed.dtype)
        self._match_runtime_dtype(device=completed.device, dtype=completed.dtype)
        state_token = self.state_token(state)
        context = torch.stack([vl_summary, completed, stage, state_token], dim=1)
        query = self.query.to(device=completed.device, dtype=completed.dtype).expand(completed.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(tgt=query, memory=context)
        return self.output_norm(query)

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.state_token.to(device=device, dtype=dtype)
        self.layers.to(device=device, dtype=dtype)
        self.output_norm.to(device=device, dtype=dtype)


class ProgressPretrainHeads(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.planner_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.stage_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.memory_pool_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self.progress_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, planner_token: torch.Tensor, progress_state: ProgressState | torch.Tensor) -> ProgressPretrainHeadOutput:
        completed, stage = _state_to_tokens(progress_state, self.config.hidden_dim)
        planner_token = _ensure_planner_token(planner_token, self.config.hidden_dim).to(device=stage.device, dtype=stage.dtype)
        memory_pool = 0.5 * (completed + stage)
        self.to(device=stage.device, dtype=stage.dtype)
        return ProgressPretrainHeadOutput(
            planner_intent=F.normalize(self.planner_proj(planner_token.squeeze(1)), dim=-1),
            stage_intent=F.normalize(self.stage_proj(stage), dim=-1),
            memory_pool_intent=F.normalize(self.memory_pool_proj(memory_pool), dim=-1),
            progress_score=self.progress_head(memory_pool),
        )


class ProgressStatePlanner(nn.Module):
    def __init__(self, config: ProgressStateConfig) -> None:
        super().__init__()
        self.config = config
        self.action_summary = ActionSummaryEncoder(config)
        self.evidence = ProgressEvidenceEncoder(config)
        self.updater = ProgressStateUpdater(config)
        self.planner = ProgressPlanner(config)

    def initial_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ProgressState:
        return self.updater.initial_state(batch_size, device=device, dtype=dtype)

    def forward_step(
        self,
        previous_state: ProgressState | torch.Tensor,
        vl_summary: torch.Tensor,
        robot_state: torch.Tensor,
        executed_actions: torch.Tensor,
        executed_mask: torch.Tensor | None = None,
    ) -> ProgressPlannerOutput:
        action_summary = self.action_summary(executed_actions, executed_mask)
        evidence = self.evidence(vl_summary, robot_state, action_summary)
        progress_state = self.updater(previous_state, evidence)
        planner_token = self.planner(progress_state, vl_summary, robot_state)
        return ProgressPlannerOutput(
            progress_state=progress_state,
            planner_token=planner_token,
            progress_evidence=evidence,
        )


def progress_intent_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target.detach().to(device=predicted.device, dtype=predicted.dtype), dim=-1)
    mse = F.mse_loss(predicted, target)
    cosine = 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()
    return mse + float(cosine_weight) * cosine


def progress_warmup_loss(
    heads: ProgressPretrainHeadOutput,
    target_intent: torch.Tensor,
    *,
    lambda_plan: float = 1.0,
    lambda_stage: float = 0.5,
    lambda_mem_pool: float = 0.1,
    lambda_order: float = 0.02,
    use_order_loss: bool = False,
    min_order_gap: int = 2,
    replan_indices: torch.Tensor | None = None,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = F.normalize(target_intent.detach().to(device=heads.planner_intent.device, dtype=heads.planner_intent.dtype), dim=-1)
    plan_loss = progress_intent_alignment_loss(heads.planner_intent, target, cosine_weight=cosine_weight)
    stage_loss = progress_intent_alignment_loss(heads.stage_intent, target, cosine_weight=cosine_weight)
    mem_pool_loss = progress_intent_alignment_loss(heads.memory_pool_intent, target, cosine_weight=cosine_weight)
    order_loss = heads.progress_score.new_zeros(())
    if use_order_loss:
        order_loss = progress_order_loss(heads.progress_score, replan_indices=replan_indices, min_order_gap=min_order_gap)
    loss = (
        float(lambda_plan) * plan_loss
        + float(lambda_stage) * stage_loss
        + float(lambda_mem_pool) * mem_pool_loss
        + (float(lambda_order) * order_loss if use_order_loss else order_loss)
    )
    metrics = {
        "plan_loss": plan_loss.detach(),
        "stage_loss": stage_loss.detach(),
        "mem_pool_loss": mem_pool_loss.detach(),
        "order_loss": order_loss.detach(),
    }
    return loss, metrics


def progress_order_loss(
    progress_score: torch.Tensor,
    *,
    replan_indices: torch.Tensor | None = None,
    min_order_gap: int = 2,
) -> torch.Tensor:
    scores = torch.as_tensor(progress_score)
    if scores.ndim == 2 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.ndim != 2:
        raise ValueError(f"progress_score must have shape [B, T] or [B, T, 1], got {tuple(progress_score.shape)}")
    batch_size, time_steps = scores.shape
    if time_steps <= 1:
        return scores.new_zeros(())
    if replan_indices is None:
        indices = torch.arange(time_steps, device=scores.device).expand(batch_size, -1)
    else:
        indices = torch.as_tensor(replan_indices, device=scores.device)
        if indices.shape != scores.shape:
            raise ValueError(f"replan_indices shape {tuple(indices.shape)} != progress_score shape {tuple(scores.shape)}")
    losses = []
    for start in range(time_steps):
        for end in range(start + 1, time_steps):
            valid = (indices[:, end] - indices[:, start]) >= int(min_order_gap)
            if valid.any():
                losses.append(F.softplus(-(scores[:, end] - scores[:, start]))[valid])
    if not losses:
        return scores.new_zeros(())
    return torch.cat(losses).mean()


def progress_diagnostics(planner_token: torch.Tensor, stage_token: torch.Tensor, planner_intent: torch.Tensor, stage_intent: torch.Tensor) -> dict[str, torch.Tensor]:
    planner = _ensure_planner_token(planner_token, stage_token.shape[-1]).squeeze(1)
    stage = _ensure_rank2(stage_token, planner.shape[-1], "stage_token").to(device=planner.device, dtype=planner.dtype)
    return {
        "cos_g_p": F.cosine_similarity(stage, planner, dim=-1).mean().detach(),
        "cos_stage_plan_intent": F.cosine_similarity(
            F.normalize(stage_intent, dim=-1),
            F.normalize(planner_intent, dim=-1),
            dim=-1,
        ).mean().detach(),
        "stage_batch_variance": stage.var(dim=0, unbiased=False).mean().detach(),
        "stage_effective_rank": effective_rank(stage).detach(),
    }


def effective_rank(features: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [B, D], got {tuple(features.shape)}")
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    probabilities = singular_values / singular_values.sum().clamp_min(float(eps))
    entropy = -(probabilities * (probabilities + float(eps)).log()).sum()
    return entropy.exp().to(device=features.device, dtype=features.dtype)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _state_to_tokens(state: ProgressState | torch.Tensor, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(state, ProgressState):
        completed = _ensure_rank2(state.completed_events, hidden_dim, "completed_events")
        stage = _ensure_rank2(state.current_stage, hidden_dim, "current_stage")
        return completed, stage
    tensor = torch.as_tensor(state)
    if tensor.ndim != 3 or tensor.shape[1:] != (2, hidden_dim):
        raise ValueError(f"progress state tensor must have shape [B, 2, {hidden_dim}], got {tuple(tensor.shape)}")
    return tensor[:, 0], tensor[:, 1]


def _ensure_rank2(tensor: torch.Tensor, last_dim: int, name: str) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, {last_dim}], got {tuple(tensor.shape)}")
    if tensor.shape[-1] != last_dim:
        raise ValueError(f"{name} last dim {tensor.shape[-1]} != {last_dim}")
    return tensor


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D], got {tuple(tensor.shape)}")
    return tensor


def _ensure_planner_token(tensor: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tensor.shape[1:] != (1, hidden_dim):
        raise ValueError(f"planner_token must have shape [B, 1, {hidden_dim}], got {tuple(tensor.shape)}")
    return tensor


def _validate_config(config: ProgressStateConfig) -> None:
    for name in (
        "hidden_dim",
        "state_dim",
        "action_dim",
        "replan_stride",
        "latent_dim",
        "action_summary_hidden_dim",
        "state_hidden_dim",
        "updater_hidden_dim",
        "planner_ffn_dim",
        "planner_layers",
        "num_heads",
    ):
        value = int(getattr(config, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if float(config.dropout) < 0.0:
        raise ValueError("dropout must be non-negative")

