from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from prism.serve.protocol import (
    HistoryObservationRequest,
    HistoryResetRequest,
    PolicyRequest,
    history_observation_from_mapping,
    history_observation_to_mapping,
    history_reset_from_mapping,
    history_reset_to_mapping,
    policy_request_from_mapping,
    policy_request_to_mapping,
)


def tiny_rgb_image(value: int = 0) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def valid_request() -> dict:
    return {
        "benchmark": "libero",
        "prompt": "pick up the object",
        "images_by_view": {
            "agentview_rgb": tiny_rgb_image(1),
            "eye_in_hand_rgb": tiny_rgb_image(2),
        },
        "state": [0.1, 0.2, 0.3],
        "action_dim": 7,
        "stream_id": "libero:0:0",
        "memory_generation": 3,
        "robot_key": "libero",
    }


def valid_history_observation() -> dict:
    return {
        "benchmark": "libero",
        "images_by_view": {
            "agentview_rgb": tiny_rgb_image(3),
            "eye_in_hand_rgb": tiny_rgb_image(4),
        },
        "stream_id": "libero:0:0",
        "target_generation": 4,
        "slot": 1,
        "robot_key": "libero",
    }


def test_policy_request_round_trip_contains_current_observation_only():
    request = policy_request_from_mapping(valid_request())
    round_trip = policy_request_from_mapping(policy_request_to_mapping(request))

    assert isinstance(request, PolicyRequest)
    assert request.benchmark == "libero"
    assert request.prompt == "pick up the object"
    assert tuple(request.images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    assert request.state.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert request.action_dim == 7
    assert request.stream_id == "libero:0:0"
    assert request.memory_generation == 3
    assert request.robot_key == "libero"
    assert request.return_debug is False
    assert request.executed_actions.shape == (8, 7)
    assert not request.executed_actions.any()
    assert not request.executed_action_valid_mask.any()
    np.testing.assert_array_equal(round_trip.images_by_view["agentview_rgb"], request.images_by_view["agentview_rgb"])


def test_policy_request_round_trips_executed_action_history_and_rejects_invalid_padding():
    payload = valid_request()
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:3, :6] = 0.25
    actions[:3, 6] = [0.0, 1.0, 0.0]
    valid = np.asarray([True, True, True, False, False, False, False, False])
    payload["executed_actions"] = actions
    payload["executed_action_valid_mask"] = valid

    request = policy_request_from_mapping(payload)
    round_trip = policy_request_from_mapping(policy_request_to_mapping(request))

    np.testing.assert_array_equal(round_trip.executed_actions, actions)
    np.testing.assert_array_equal(round_trip.executed_action_valid_mask, valid)

    missing_mask = valid_request()
    missing_mask["executed_actions"] = actions
    with pytest.raises(ValueError, match="must be provided together"):
        policy_request_from_mapping(missing_mask)

    invalid_padding = valid_request()
    invalid_padding["executed_actions"] = np.ones((8, 7), dtype=np.float32)
    invalid_padding["executed_action_valid_mask"] = np.zeros(8, dtype=np.bool_)
    with pytest.raises(ValueError, match="zero at invalid positions"):
        policy_request_from_mapping(invalid_padding)


def test_policy_request_from_mapping_accepts_debug_flag():
    payload = valid_request()
    payload["return_debug"] = True

    assert policy_request_from_mapping(payload).return_debug is True

    payload["return_debug"] = "false"
    with pytest.raises(ValueError, match="return_debug must be boolean"):
        policy_request_from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("action_dim", 0, "action_dim must be positive"),
        ("action_dim", 7.0, "action_dim must be an integer"),
        ("memory_generation", -1, "memory_generation must be non-negative"),
        ("memory_generation", True, "memory_generation must be an integer"),
        ("stream_id", "", "stream_id must not be empty"),
    ],
)
def test_policy_request_rejects_invalid_scalar_fields(field, value, match):
    payload = valid_request()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        policy_request_from_mapping(payload)


def test_policy_request_rejects_unknown_or_missing_fields():
    unknown = valid_request()
    unknown["history_images_by_view"] = {}
    with pytest.raises(ValueError, match="Unsupported policy request fields"):
        policy_request_from_mapping(unknown)

    missing = valid_request()
    del missing["stream_id"]
    with pytest.raises(ValueError, match="Missing required policy request fields"):
        policy_request_from_mapping(missing)


def test_policy_request_rejects_invalid_images_or_state():
    empty_images = valid_request()
    empty_images["images_by_view"] = {}
    with pytest.raises(ValueError, match="at least one image"):
        policy_request_from_mapping(empty_images)

    non_rgb = valid_request()
    non_rgb["images_by_view"]["agentview_rgb"] = [[[1, 2], [3, 4]]]
    with pytest.raises(ValueError, match="3 channels"):
        policy_request_from_mapping(non_rgb)

    bad_state = valid_request()
    bad_state["state"] = [0.0, math.inf]
    with pytest.raises(ValueError, match="finite"):
        policy_request_from_mapping(bad_state)

    non_string_view = valid_request()
    non_string_view["images_by_view"] = {1: tiny_rgb_image()}
    with pytest.raises(ValueError, match="view names must be strings"):
        policy_request_from_mapping(non_string_view)


@pytest.mark.parametrize("dtype", [np.float32, np.bool_, np.int64])
def test_policy_and_history_requests_reject_non_uint8_images(dtype):
    image = np.ones((2, 2, 3), dtype=dtype)

    policy_payload = valid_request()
    policy_payload["images_by_view"]["agentview_rgb"] = image
    with pytest.raises(ValueError, match="must have dtype uint8"):
        policy_request_from_mapping(policy_payload)

    history_payload = valid_history_observation()
    history_payload["images_by_view"]["agentview_rgb"] = image
    with pytest.raises(ValueError, match="must have dtype uint8"):
        history_observation_from_mapping(history_payload)


def test_request_serializers_do_not_coerce_invalid_generation_or_slot():
    policy_request = policy_request_from_mapping(valid_request())
    with pytest.raises(ValueError, match="memory_generation must be an integer"):
        policy_request_to_mapping(replace(policy_request, memory_generation=3.5))

    history_request = history_observation_from_mapping(valid_history_observation())
    with pytest.raises(ValueError, match="slot must be an integer"):
        history_observation_to_mapping(replace(history_request, slot=1.5))


def test_history_observation_round_trip_and_slot_validation():
    request = history_observation_from_mapping(valid_history_observation())
    round_trip = history_observation_from_mapping(history_observation_to_mapping(request))

    assert isinstance(request, HistoryObservationRequest)
    assert request.target_generation == 4
    assert request.slot == 1
    np.testing.assert_array_equal(
        round_trip.images_by_view["eye_in_hand_rgb"],
        request.images_by_view["eye_in_hand_rgb"],
    )

    for invalid_slot in (-1, 2, 1.5, True):
        payload = valid_history_observation()
        payload["slot"] = invalid_slot
        with pytest.raises(ValueError, match="slot must"):
            history_observation_from_mapping(payload)


def test_history_observation_requires_positive_target_generation():
    payload = valid_history_observation()
    payload["target_generation"] = 0

    with pytest.raises(ValueError, match="target_generation must be positive"):
        history_observation_from_mapping(payload)


def test_history_reset_round_trip_and_strict_fields():
    request = history_reset_from_mapping({"stream_id": "calvin:2:1"})

    assert request == HistoryResetRequest(stream_id="calvin:2:1")
    assert history_reset_to_mapping(request) == {"stream_id": "calvin:2:1"}
    with pytest.raises(ValueError, match="Unsupported history reset fields"):
        history_reset_from_mapping({"stream_id": "x", "legacy": True})


def test_legacy_payload_is_rejected_by_canonical_runtime_contract():
    with pytest.raises(ValueError, match="Unsupported policy request fields"):
        policy_request_from_mapping(
            {
                "image": [tiny_rgb_image(1), tiny_rgb_image(2)],
                "state": [0.1, 0.2],
                "prompt": "legacy",
                "action_mask": [1, 1, 1, 0],
                "robot_key": "libero",
            }
        )
