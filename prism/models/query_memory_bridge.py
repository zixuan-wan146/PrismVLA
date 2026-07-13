from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class DualSourceBridgeAttention(nn.Module):
    """Apply independent current-query and gated history-memory attention."""

    def __init__(
        self,
        *,
        action_dim: int,
        current_dim: int,
        memory_dim: int,
        num_heads: int,
        memory_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if action_dim % num_heads != 0:
            raise ValueError("action_dim must be divisible by num_heads")
        self.action_norm_current = nn.LayerNorm(action_dim)
        self.action_norm_memory = nn.LayerNorm(action_dim)
        self.current_norm = nn.LayerNorm(current_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.current_projection = nn.Linear(current_dim, action_dim)
        self.memory_projection = nn.Linear(memory_dim, action_dim)
        self.current_attention = nn.MultiheadAttention(action_dim, num_heads, batch_first=True)
        self.memory_attention = nn.MultiheadAttention(action_dim, num_heads, batch_first=True)
        self.memory_gate = nn.Parameter(torch.tensor(float(memory_gate_init)))

    def forward(
        self,
        action_states: torch.Tensor,
        current_features: torch.Tensor,
        current_valid_mask: torch.Tensor,
        memory_features: torch.Tensor,
        memory_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_mask(current_valid_mask, current_features, "current_valid_mask")
        _validate_mask(memory_valid_mask, memory_features, "memory_valid_mask")
        if action_states.ndim != 3:
            raise ValueError("action_states must have shape [B, A, D]")
        if action_states.shape[0] != current_features.shape[0] or action_states.shape[0] != memory_features.shape[0]:
            raise ValueError("action, current, and memory batch sizes must match")
        block_device = self.current_projection.weight.device
        if action_states.device != block_device:
            raise ValueError(f"action_states must be on the Bridge device {block_device}, got {action_states.device}")
        current_valid_mask = current_valid_mask.to(device=block_device)
        memory_valid_mask = memory_valid_mask.to(device=block_device)
        if not current_valid_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid current-query token")

        current_features = current_features.to(
            device=block_device,
            dtype=self.current_projection.weight.dtype,
        )
        memory_features = memory_features.to(
            device=block_device,
            dtype=self.memory_projection.weight.dtype,
        )
        action_states = action_states.to(dtype=self.current_attention.in_proj_weight.dtype)
        current_context = self.current_projection(self.current_norm(current_features))
        current_update, _ = self.current_attention(
            self.action_norm_current(action_states),
            current_context,
            current_context,
            key_padding_mask=~current_valid_mask,
            need_weights=False,
        )
        action_states = action_states + current_update

        samples_with_memory = memory_valid_mask.any(dim=1)
        if samples_with_memory.any():
            selected_actions = action_states[samples_with_memory]
            memory_context = self.memory_projection(self.memory_norm(memory_features[samples_with_memory]))
            memory_update, _ = self.memory_attention(
                self.action_norm_memory(selected_actions),
                memory_context,
                memory_context,
                key_padding_mask=~memory_valid_mask[samples_with_memory],
                need_weights=False,
            )
            action_states = action_states.clone()
            action_states[samples_with_memory] = selected_actions + self.memory_gate * memory_update
        return action_states


class LayerwiseQueryMemoryBridge(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        action_dim: int,
        current_dim: int = 1024,
        memory_dim: int = 512,
        num_heads: int,
        memory_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.blocks = nn.ModuleList(
            DualSourceBridgeAttention(
                action_dim=action_dim,
                current_dim=current_dim,
                memory_dim=memory_dim,
                num_heads=num_heads,
                memory_gate_init=memory_gate_init,
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        action_states: torch.Tensor,
        layerwise_current_features: Sequence[torch.Tensor],
        current_valid_mask: torch.Tensor,
        memory_features: torch.Tensor,
        memory_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(layerwise_current_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} current feature levels, got {len(layerwise_current_features)}"
            )
        for block, current_features in zip(self.blocks, layerwise_current_features):
            action_states = block(
                action_states,
                current_features,
                current_valid_mask,
                memory_features,
                memory_valid_mask,
            )
        return action_states


def _validate_mask(mask: torch.Tensor, features: torch.Tensor, name: str) -> None:
    if features.ndim != 3:
        raise ValueError("conditioning features must have shape [B, T, D]")
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor matching the first two feature dimensions")
