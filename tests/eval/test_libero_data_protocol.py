from __future__ import annotations

import numpy as np

from prism.eval.libero.protocol import LIBERO_ACTION_DIM
from prism.eval.libero.protocol import LIBERO_SHORT_MEMORY_OFFSETS
from prism.eval.libero.protocol import LIBERO_STATE_DIM
from prism.eval.libero.protocol import LIBERO_VIEW_ORDER
from prism.eval.libero.data_protocol import build_request_from_observation
from prism.eval.libero.history import LiberoObservationHistory


def test_libero_protocol_contract():
    assert LIBERO_VIEW_ORDER == ("agentview_rgb", "eye_in_hand_rgb")
    assert LIBERO_ACTION_DIM == 7
    assert LIBERO_STATE_DIM == 8
    assert LIBERO_SHORT_MEMORY_OFFSETS == (16, 8)


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

    request = build_request_from_observation(obs, "put the mug away", reset_memory=True)

    assert request.prompt == "put the mug away"
    assert request.benchmark == "libero"
    assert tuple(request.images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    np.testing.assert_array_equal(request.images_by_view["agentview_rgb"], agent)
    np.testing.assert_array_equal(request.images_by_view["eye_in_hand_rgb"], wrist)
    assert request.action_dim == 7
    assert request.robot_key == "libero"
    assert request.reset_memory is True
    assert request.state.shape == (8,)


def test_build_request_from_observation_includes_offset_short_memory_when_history_has_frames():
    obs0 = _obs_with_image_values(agent_value=1, wrist_value=2)
    obs8 = _obs_with_image_values(agent_value=8, wrist_value=9)
    obs16 = _obs_with_image_values(agent_value=16, wrist_value=17)
    history = LiberoObservationHistory(max_offset=16)
    history.record(0, obs0)
    history.record(8, obs8)
    history.record(16, obs16)

    request = build_request_from_observation(obs16, "pick", history=history, current_step=16)

    short_memory = request.short_memory_images_by_offset
    assert short_memory is not None
    assert tuple(short_memory) == (16, 8)
    np.testing.assert_array_equal(short_memory[16]["agentview_rgb"], obs0["agentview_image"])
    np.testing.assert_array_equal(short_memory[8]["eye_in_hand_rgb"], obs8["robot0_eye_in_hand_image"])


def test_build_request_from_observation_emits_empty_short_memory_object_for_warmup_steps():
    obs = _obs_with_image_values(agent_value=1, wrist_value=2)
    history = LiberoObservationHistory(max_offset=16)
    history.record(0, obs)

    request = build_request_from_observation(obs, "pick", history=history, current_step=0)

    assert request.short_memory_images_by_offset == {}


def test_build_request_from_observation_includes_executed_actions():
    obs = _obs_with_image_values(agent_value=1, wrist_value=2)

    request = build_request_from_observation(
        obs,
        "pick",
        executed_actions=[[0, 1, 2, 3, 4, 5, 1]],
        executed_action_mask=[True],
    )

    np.testing.assert_array_equal(
        request.executed_actions,
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(request.executed_action_mask, np.array([True]))


def _obs_with_image_values(*, agent_value: int, wrist_value: int) -> dict:
    return {
        "agentview_image": np.full((2, 2, 3), agent_value, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((2, 2, 3), wrist_value, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }
