from __future__ import annotations

# --- migrated from src/prism/dataset/action_segments.py ---
from typing import Any

import numpy as np


def token_span_steps(*, planning_horizon: int, num_plan_steps: int) -> int:
    if num_plan_steps <= 0:
        raise ValueError(f"num_plan_steps must be positive, got {num_plan_steps}")
    if planning_horizon <= 0:
        raise ValueError(f"planning_horizon must be positive, got {planning_horizon}")
    if planning_horizon % num_plan_steps != 0:
        raise ValueError("planning_horizon must be divisible by num_plan_steps")
    return planning_horizon // num_plan_steps


def build_action_segment_target(
    actions: Any,
    *,
    num_plan_steps: int,
    planning_horizon: int,
    valid_action_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice future actions into K full action chunks.

    Each valid target keeps the full `[chunk_size, action_dim]` trajectory.
    Invalid tail chunks are zero-filled and masked out.
    """

    chunk_size = token_span_steps(planning_horizon=planning_horizon, num_plan_steps=num_plan_steps)

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError(f"actions must have shape [T, A], got {action_array.shape}")
    if action_array.shape[0] < planning_horizon:
        raise ValueError(
            f"actions length {action_array.shape[0]} is shorter than planning_horizon {planning_horizon}"
        )

    valid_count = planning_horizon if valid_action_count is None else max(0, min(int(valid_action_count), planning_horizon))
    action_dim = int(action_array.shape[1])
    segments = np.zeros((num_plan_steps, chunk_size, action_dim), dtype=np.float32)
    mask = np.zeros((num_plan_steps,), dtype=bool)

    for step in range(num_plan_steps):
        start = step * chunk_size
        end = start + chunk_size
        if end > valid_count:
            continue
        segments[step] = action_array[start:end]
        mask[step] = True

    return segments, mask

