from __future__ import annotations

from collections.abc import Sequence

import torch


def gather_layerwise_action_queries(
    hidden_states: Sequence[torch.Tensor],
    action_query_mask: torch.Tensor,
    *,
    num_backbone_layers: int = 16,
    num_action_queries: int = 48,
    hidden_size: int = 1024,
) -> tuple[torch.Tensor, ...]:
    """Gather H1..HN action-query states while explicitly excluding H0."""

    expected_levels = num_backbone_layers + 1
    if len(hidden_states) != expected_levels:
        raise ValueError(
            f"Expected {expected_levels} hidden-state levels H0..H{num_backbone_layers}, got {len(hidden_states)}"
        )
    if action_query_mask.ndim != 2 or action_query_mask.dtype != torch.bool:
        raise ValueError("action_query_mask must be a boolean [B, T] tensor")
    counts = action_query_mask.sum(dim=1)
    if not torch.all(counts == num_action_queries):
        raise ValueError(f"Each sample must contain {num_action_queries} action queries, got {counts.tolist()}")

    batch_size, sequence_length = action_query_mask.shape
    gathered: list[torch.Tensor] = []
    for level, hidden_state in enumerate(hidden_states[1:], start=1):
        if hidden_state.shape != (batch_size, sequence_length, hidden_size):
            raise ValueError(
                f"H{level} must have shape {(batch_size, sequence_length, hidden_size)}, got {tuple(hidden_state.shape)}"
            )
        selected = hidden_state[action_query_mask].reshape(batch_size, num_action_queries, hidden_size)
        gathered.append(selected)
    return tuple(gathered)
