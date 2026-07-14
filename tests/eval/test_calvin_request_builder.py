from __future__ import annotations

import numpy as np

from experiments.calvin.eval import build_request_from_observation


def test_calvin_request_builder_uses_policy_request_contract():
    obs16 = _obs(16)

    request = build_request_from_observation(
        obs16,
        "open the drawer",
        stream_id="calvin:0:0",
        memory_generation=2,
    )

    assert request.benchmark == "calvin"
    assert tuple(request.images_by_view) == ("primary", "wrist")
    assert not hasattr(request, "history_images_by_view")
    np.testing.assert_array_equal(
        request.state,
        np.asarray([16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 0.0, 16.0], dtype=np.float32),
    )
    assert request.action_dim == 7
    assert request.stream_id == "calvin:0:0"
    assert request.memory_generation == 2


def test_calvin_request_builder_projects_raw_simulator_state_to_training_layout():
    obs = _obs(0)
    obs["robot_obs"] = np.arange(15, dtype=np.float32)

    request = build_request_from_observation(
        obs,
        "open the drawer",
        stream_id="calvin:0:0",
        memory_generation=0,
    )

    np.testing.assert_array_equal(
        request.state,
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0], dtype=np.float32),
    )


def test_calvin_request_builder_rejects_short_robot_state():
    with np.testing.assert_raises_regex(ValueError, "at least 8 values"):
        build_request_from_observation(
            {**_obs(0), "robot_obs": np.zeros(7)},
            "open the drawer",
            stream_id="calvin:0:0",
            memory_generation=0,
        )


def _obs(value: int) -> dict:
    return {
        "rgb_obs": {
            "rgb_static": np.full((2, 2, 3), value, dtype=np.uint8),
            "rgb_gripper": np.full((2, 2, 3), value + 1, dtype=np.uint8),
        },
        "robot_obs": np.full((8,), value, dtype=np.float32),
    }
