from __future__ import annotations

import torch
import torch.nn as nn

from prism.models.config import PrismArchitectureConfig
from prism.models.query_memory_bridge import LayerwiseQueryMemoryBridge
from prism.models.vlm import QueryMemoryEncoderOutput


class DirectActionHead(nn.Module):
    """Parallel eight-step direct-regression action head."""

    def __init__(
        self,
        architecture: PrismArchitectureConfig,
        *,
        state_dim: int,
    ) -> None:
        super().__init__()
        architecture.validate_for_policy()
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")

        action_config = architecture.action_head
        hidden_size, _num_heads, _ffn_ratio = action_config.resolved_dimensions()
        self.architecture = architecture
        self.state_dim = state_dim
        self.action_horizon = architecture.temporal.action_horizon
        self.action_dim = action_config.action_dim
        self.hidden_size = hidden_size

        self.action_step_queries = nn.Parameter(torch.empty(self.action_horizon, hidden_size))
        self.temporal_position_embeddings = nn.Parameter(torch.empty(self.action_horizon, hidden_size))
        self.state_projection = nn.Linear(state_dim, hidden_size)
        self.bridge = LayerwiseQueryMemoryBridge(architecture)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, self.action_dim)
        self._reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.action_step_queries.device

    @property
    def dtype(self) -> torch.dtype:
        return self.action_step_queries.dtype

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.action_step_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.temporal_position_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        encoder_output: QueryMemoryEncoderOutput,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(f"state must have shape [B, {self.state_dim}], got {tuple(state.shape)}")
        if not torch.is_floating_point(state):
            raise ValueError("state must be a floating tensor")
        if state.device.type == "cpu" and not torch.isfinite(state).all():
            raise ValueError("state must contain only finite values")
        self._validate_encoder_output(encoder_output, state.shape[0])

        state = state.to(device=self.device, dtype=self.dtype)
        state_condition = self.state_projection(state).unsqueeze(1)
        action_states = (
            self.action_step_queries.unsqueeze(0) + self.temporal_position_embeddings.unsqueeze(0) + state_condition
        )
        action_states = self.bridge(
            action_states,
            encoder_output.layerwise_query_features,
            encoder_output.query_valid_mask,
            encoder_output.memory.tokens,
            encoder_output.memory.valid_mask,
        )
        return self.output_projection(self.output_norm(action_states))

    def _validate_encoder_output(
        self,
        output: QueryMemoryEncoderOutput,
        batch_size: int,
    ) -> None:
        architecture = self.architecture
        if len(output.layerwise_query_features) != architecture.num_bridge_layers:
            raise ValueError("encoder output must contain one query level per Bridge block")
        expected_query_shape = (
            batch_size,
            architecture.backbone.num_action_queries,
            architecture.backbone.hidden_size,
        )
        for level, features in enumerate(output.layerwise_query_features):
            if features.shape != expected_query_shape:
                raise ValueError(
                    f"query level {level} must have shape {expected_query_shape}, got {tuple(features.shape)}"
                )
        if output.query_valid_mask.shape != expected_query_shape[:2]:
            raise ValueError("query_valid_mask shape does not match query features")
        if output.query_valid_mask.dtype != torch.bool:
            raise ValueError("query_valid_mask must be boolean")

        expected_memory_shape = (
            batch_size,
            architecture.history.num_memory_tokens,
            architecture.history.hidden_size,
        )
        if output.memory.tokens.shape != expected_memory_shape:
            raise ValueError(
                f"memory tokens must have shape {expected_memory_shape}, got {tuple(output.memory.tokens.shape)}"
            )
        if output.memory.valid_mask.shape != expected_memory_shape[:2]:
            raise ValueError("memory valid mask shape does not match memory tokens")
        if output.memory.valid_mask.dtype != torch.bool:
            raise ValueError("memory valid mask must be boolean")


def decode_gripper_open(
    predicted_actions: torch.Tensor,
    *,
    gripper_index: int,
    threshold: float,
) -> torch.Tensor:
    """Return canonical open commands; equality with threshold is close."""

    if predicted_actions.ndim < 1:
        raise ValueError("predicted_actions must have an action dimension")
    if gripper_index < 0 or gripper_index >= predicted_actions.shape[-1]:
        raise ValueError("gripper_index is outside the action dimension")
    if predicted_actions.device.type == "cpu" and not torch.isfinite(predicted_actions).all():
        raise ValueError("predicted_actions must contain only finite values")
    return predicted_actions[..., gripper_index] > threshold


__all__ = ["DirectActionHead", "decode_gripper_open"]
