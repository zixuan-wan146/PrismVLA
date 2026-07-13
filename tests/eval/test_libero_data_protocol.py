from __future__ import annotations

import numpy as np

from experiments.libero.eval import build_request_from_observation
from experiments.libero.eval import LIBERO_ACTION_DIM, LIBERO_STATE_DIM, LIBERO_VIEW_ORDER


def test_libero_protocol_contract():
    assert LIBERO_VIEW_ORDER == ("agentview_rgb", "eye_in_hand_rgb")
    assert LIBERO_ACTION_DIM == 7
    assert LIBERO_STATE_DIM == 8


def test_build_request_from_observation_uses_two_raw_libero_views():
    agent = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
    obs = {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }

    request = build_request_from_observation(obs, "put the mug away")

    assert request.prompt == "put the mug away"
    assert request.benchmark == "libero"
    assert tuple(request.images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    np.testing.assert_array_equal(request.images_by_view["agentview_rgb"], agent)
    np.testing.assert_array_equal(request.images_by_view["eye_in_hand_rgb"], wrist)
    assert request.history_valid_mask.tolist() == [False, False]
    assert request.history_step_ages.tolist() == [6, 3]
    np.testing.assert_array_equal(
        request.history_images_by_view["agentview_rgb"], np.zeros((2, 2, 2, 3), dtype=np.uint8)
    )
    assert request.action_dim == 7
    assert request.robot_key == "libero"
    assert request.state.shape == (8,)


def _obs_with_image_values(*, agent_value: int, wrist_value: int) -> dict:
    return {
        "agentview_image": np.full((2, 2, 3), agent_value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), wrist_value, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }
