from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prism.eval.action_response import parse_action_response as parse_policy_action_response


CALVIN_CONTROL_DIM = 7
VALID_CALVIN_GRIPPER_MODES = {"openvla", "passthrough", "sign"}


def parse_action_response(
    message: Any,
    horizon: int,
    min_action_dim: int = CALVIN_CONTROL_DIM,
) -> list[list[float]]:
    return parse_policy_action_response(
        message,
        horizon=horizon,
        min_action_dim=min_action_dim,
    )


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
