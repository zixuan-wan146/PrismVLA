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

    assert request.benchmark == "calvin"
    assert sorted(request.images_by_view) == ["image", "wrist_image"]
    np.testing.assert_array_equal(request.state, np.full(8, 16.0, dtype=np.float32))
    assert request.action_dim == 7
    assert request.reset_memory is True
    assert request.short_memory_images_by_offset is not None
    assert 16 in request.short_memory_images_by_offset
    np.testing.assert_array_equal(request.executed_action_mask, np.array([True]))


def test_calvin_request_builder_projects_raw_simulator_state_to_training_layout():
    obs = _obs(0)
    obs["robot_obs"] = np.arange(15, dtype=np.float32)

    request = build_request_from_observation(obs, "open the drawer")

    np.testing.assert_array_equal(
        request.state,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 14.0], dtype=np.float32),
    )


def test_calvin_request_builder_rejects_short_robot_state():
    with np.testing.assert_raises_regex(ValueError, "at least 8 values"):
        build_request_from_observation({**_obs(0), "robot_obs": np.zeros(7)}, "open the drawer")


def _obs(value: int) -> dict:
    return {
        "rgb_obs": {
            "rgb_static": np.full((2, 2, 3), value, dtype=np.uint8),
            "rgb_gripper": np.full((2, 2, 3), value + 1, dtype=np.uint8),
        },
        "robot_obs": np.full((8,), value, dtype=np.float32),
    }
