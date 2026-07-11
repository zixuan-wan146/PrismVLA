from __future__ import annotations

import numpy as np

from prism.eval.calvin.history import CalvinObservationHistory
from prism.eval.calvin.request_builder import build_request_from_observation


def test_calvin_request_builder_uses_policy_request_contract_with_memory_and_actions():
    obs0 = _obs(0)
    obs16 = _obs(16)
    history = CalvinObservationHistory(max_offset=16)
    history.record(0, obs0)
    history.record(16, obs16)

    request = build_request_from_observation(
        obs16,
        "open the drawer",
        history=history,
        current_step=16,
        reset_memory=True,
        executed_actions=[[0.1] * 7],
        executed_action_mask=[True],
    )

    assert request["benchmark"] == "calvin"
    assert sorted(request["images_by_view"]) == ["image", "wrist_image"]
    assert request["state"] == [16.0] * 8
    assert request["action_dim"] == 7
    assert request["reset_memory"] is True
    assert "16" in request["short_memory_images_by_offset"]
    assert request["executed_action_mask"] == [1]


def _obs(value: int) -> dict:
    return {
        "rgb_obs": {
            "rgb_static": np.full((2, 2, 3), value, dtype=np.uint8),
            "rgb_gripper": np.full((2, 2, 3), value + 1, dtype=np.uint8),
        },
        "robot_obs": np.full((8,), value, dtype=np.float32),
    }
