from __future__ import annotations

# Action-segment autoencoder retained as an independent research component.
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
