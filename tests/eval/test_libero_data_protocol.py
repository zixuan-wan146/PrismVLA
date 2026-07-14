from __future__ import annotations

import numpy as np

from experiments.libero.data import LIBERO_IMAGE_TRANSFORM
from experiments.libero.eval import build_request_from_observation
from experiments.libero.eval import LIBERO_ACTION_DIM, LIBERO_STATE_DIM, LIBERO_VIEW_ORDER


def test_libero_protocol_contract():
    assert LIBERO_VIEW_ORDER == ("primary", "wrist")
    assert LIBERO_ACTION_DIM == 7
    assert LIBERO_STATE_DIM == 8
    assert LIBERO_IMAGE_TRANSFORM == "rotate_180"


def test_build_request_from_observation_uses_two_canonical_rotated_views():
    agent = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3)
    obs = {
        "agentview_image": agent,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }

    request = build_request_from_observation(
        obs,
        "put the mug away",
        stream_id="libero:spatial:0:0",
        memory_generation=3,
    )

    assert request.prompt == "put the mug away"
    assert request.benchmark == "libero"
    assert tuple(request.images_by_view) == ("primary", "wrist")
    np.testing.assert_array_equal(request.images_by_view["primary"], np.rot90(agent, 2))
    np.testing.assert_array_equal(request.images_by_view["wrist"], np.rot90(wrist, 2))
    assert request.images_by_view["primary"].flags.c_contiguous
    assert not hasattr(request, "history_images_by_view")
    assert request.action_dim == 7
    assert request.stream_id == "libero:spatial:0:0"
    assert request.memory_generation == 3
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
