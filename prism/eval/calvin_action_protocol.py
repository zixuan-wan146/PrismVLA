from __future__ import annotations

# --- migrated from src/prism/benchmarks/calvin/action_protocol.py ---
from collections.abc import Mapping, Sequence
import json
from typing import Any


CALVIN_CONTROL_DIM = 7
VALID_CALVIN_GRIPPER_MODES = {"openvla", "passthrough", "sign"}


def parse_action_response(
    message: str,
    horizon: int,
    min_action_dim: int = CALVIN_CONTROL_DIM,
) -> list[list[float]]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Action response is not valid JSON: {exc}") from exc

    if isinstance(payload, Mapping):
        if "error" in payload:
            raise RuntimeError(f"Prism server returned error: {payload['error']}")
        if "actions" not in payload:
            raise ValueError(f"Action response object must contain 'actions', got keys: {sorted(payload.keys())}")
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError(f"Action response must be a list, got {type(payload).__name__}")
    if len(payload) < horizon:
        raise ValueError(f"Action response has {len(payload)} step(s), expected at least horizon {horizon}")

    actions: list[list[float]] = []
    for step, row in enumerate(payload[:horizon]):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Action at step {step} must be a sequence, got {type(row).__name__}")
        if len(row) < min_action_dim:
            raise ValueError(
                f"Action at step {step} has dimension {len(row)}, expected at least {min_action_dim}"
            )
        actions.append([_to_float(value, step, dim) for dim, value in enumerate(row)])
    return actions


def to_calvin_action(
    action: Sequence[float],
    *,
    control_dim: int = CALVIN_CONTROL_DIM,
    gripper_mode: str = "passthrough",
) -> list[float]:
    if len(action) < control_dim:
        raise ValueError(f"Action dimension {len(action)} is smaller than CALVIN control dim {control_dim}")
    calvin_action = [float(value) for value in action[:control_dim]]
    if gripper_mode == "passthrough":
        return calvin_action
    if gripper_mode == "openvla":
        calvin_action[6] = -1.0 if calvin_action[6] > 0.5 else 1.0
    elif gripper_mode == "sign":
        calvin_action[6] = -1.0 if calvin_action[6] < 0.0 else 1.0
    else:
        raise ValueError(f"unsupported CALVIN gripper mode: {gripper_mode!r}")
    return calvin_action


def _to_float(value: Any, step: int, dim: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Action value at step {step}, dim {dim} is not numeric: {value!r}") from exc

