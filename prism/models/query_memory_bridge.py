from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from prism.models.config import PrismArchitectureConfig


class DualSourceBridgeAttention(nn.Module):
    """One aligned action block with self, current, memory, and FFN residuals."""

    def __init__(
        self,
        *,
        action_hidden_size: int,
        current_dim: int,
        memory_dim: int,
        num_heads: int,
        ffn_ratio: int,
        memory_gate_init: float,
    ) -> None:
        super().__init__()
        if action_hidden_size <= 0 or action_hidden_size % num_heads != 0:
            raise ValueError("action_hidden_size must be positive and divisible by num_heads")
        if current_dim <= 0 or memory_dim <= 0:
            raise ValueError("conditioning dimensions must be positive")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")

        self.action_hidden_size = action_hidden_size
        self.action_self_norm = nn.LayerNorm(action_hidden_size)
        self.action_norm_current = nn.LayerNorm(action_hidden_size)
        self.action_norm_memory = nn.LayerNorm(action_hidden_size)
        self.action_ffn_norm = nn.LayerNorm(action_hidden_size)
        self.current_norm = nn.LayerNorm(current_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)

        self.action_self_attention = nn.MultiheadAttention(action_hidden_size, num_heads, batch_first=True)
        self.current_projection = nn.Linear(current_dim, action_hidden_size)
        self.memory_projection = nn.Linear(memory_dim, action_hidden_size)
        self.current_attention = nn.MultiheadAttention(action_hidden_size, num_heads, batch_first=True)
        self.memory_attention = nn.MultiheadAttention(action_hidden_size, num_heads, batch_first=True)
        ffn_hidden_size = action_hidden_size * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(action_hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Linear(ffn_hidden_size, action_hidden_size),
        )
        self.memory_gate = nn.Parameter(torch.tensor(float(memory_gate_init)))

    @property
    def device(self) -> torch.device:
        return self.action_self_norm.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_self_norm.weight.dtype

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
            raise ValueError("action_states must have shape [B, action_steps, hidden]")
        if action_states.shape[-1] != self.action_hidden_size:
            raise ValueError(f"action hidden width must be {self.action_hidden_size}, got {action_states.shape[-1]}")
        if action_states.shape[0] != current_features.shape[0] or action_states.shape[0] != memory_features.shape[0]:
            raise ValueError("action, current, and memory batch sizes must match")
        if action_states.device != self.device:
            raise ValueError(f"action_states must be on Bridge device {self.device}, got {action_states.device}")

        current_valid_mask = current_valid_mask.to(device=self.device)
        memory_valid_mask = memory_valid_mask.to(device=self.device)
        if not current_valid_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid current-query token")
        action_states = action_states.to(dtype=self.dtype)
        current_features = current_features.to(device=self.device, dtype=self.current_projection.weight.dtype)
        memory_features = memory_features.to(device=self.device, dtype=self.memory_projection.weight.dtype)

        normalized_actions = self.action_self_norm(action_states)
        self_update, _ = self.action_self_attention(
            normalized_actions,
            normalized_actions,
            normalized_actions,
            need_weights=False,
        )
        action_states = action_states + self_update

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

        return action_states + self.ffn(self.action_ffn_norm(action_states))


class LayerwiseQueryMemoryBridge(nn.Module):
    """Consume every retained VLM query level exactly once."""

    def __init__(self, architecture: PrismArchitectureConfig) -> None:
        super().__init__()
        architecture.validate_for_policy()
        action_config = architecture.action_head
        action_hidden_size = int(action_config.action_hidden_size)
        num_heads = int(action_config.num_attention_heads)
        ffn_ratio = int(action_config.ffn_ratio)

        self.architecture = architecture
        self.num_layers = architecture.num_bridge_layers
        self.action_hidden_size = action_hidden_size
        self.blocks = nn.ModuleList(
            DualSourceBridgeAttention(
                action_hidden_size=action_hidden_size,
                current_dim=architecture.backbone.hidden_size,
                memory_dim=architecture.history.hidden_size,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                memory_gate_init=architecture.memory_gate_init,
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
        for block, current_features in zip(self.blocks, layerwise_current_features, strict=True):
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
        raise ValueError("conditioning features must have shape [B, tokens, hidden]")
    if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor matching the feature token dimensions")


__all__ = ["DualSourceBridgeAttention", "LayerwiseQueryMemoryBridge"]
