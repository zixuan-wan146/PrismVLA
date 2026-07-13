from __future__ import annotations

import math

import pytest

from prism.serve.protocol import policy_request_from_mapping
from prism.schema import PolicyInput


def tiny_rgb_image(value: int = 0):
    return [
        [[value, value, value], [value, value, value]],
        [[value, value, value], [value, value, value]],
    ]


def valid_request() -> dict:
    return {
        "benchmark": "libero",
        "prompt": "pick up the object",
        "images_by_view": {
            "agentview_rgb": tiny_rgb_image(1),
            "eye_in_hand_rgb": tiny_rgb_image(2),
        },
        "history_images_by_view": {
            "agentview_rgb": [tiny_rgb_image(3), tiny_rgb_image(4)],
            "eye_in_hand_rgb": [tiny_rgb_image(5), tiny_rgb_image(6)],
        },
        "history_step_ages": [6, 3],
        "history_valid_mask": [True, True],
        "state": [0.1, 0.2, 0.3],
        "action_dim": 7,
        "robot_key": "libero",
    }


def test_policy_request_from_mapping_accepts_canonical_payload():
    request = policy_request_from_mapping(valid_request())

    assert isinstance(request, PolicyInput)
    assert request.benchmark == "libero"
    assert request.prompt == "pick up the object"
    assert tuple(request.images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    assert tuple(request.history_images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    assert request.history_step_ages.tolist() == [6, 3]
    assert request.history_valid_mask.tolist() == [True, True]
    assert request.state.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert request.action_dim == 7
    assert request.robot_key == "libero"
    assert request.return_debug is False


def test_policy_request_from_mapping_accepts_debug_flag():
    payload = valid_request()
    payload["return_debug"] = True

    request = policy_request_from_mapping(payload)

    assert request.return_debug is True


def test_policy_request_from_mapping_rejects_unknown_fields():
    payload = valid_request()
    payload["legacy_field"] = True

    with pytest.raises(ValueError, match="Unsupported policy request fields"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_missing_required_fields():
    payload = valid_request()
    del payload["action_dim"]

    with pytest.raises(ValueError, match="Missing required policy request fields"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_empty_images_by_view():
    payload = valid_request()
    payload["images_by_view"] = {}

    with pytest.raises(ValueError, match="at least one image"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_non_rgb_image_shape():
    payload = valid_request()
    payload["images_by_view"]["agentview_rgb"] = [[[1, 2], [3, 4]]]

    with pytest.raises(ValueError, match="3 channels"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_out_of_range_pixels():
    payload = valid_request()
    payload["images_by_view"]["agentview_rgb"] = [[[256, 0, 0]]]

    with pytest.raises(ValueError, match="0..255"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_nonfinite_state():
    payload = valid_request()
    payload["state"] = [0.0, math.inf]

    with pytest.raises(ValueError, match="finite"):
        policy_request_from_mapping(payload)


def test_policy_request_from_mapping_rejects_invalid_action_dim():
    payload = valid_request()
    payload["action_dim"] = 0

    with pytest.raises(ValueError, match="action_dim must be positive"):
        policy_request_from_mapping(payload)


def test_policy_request_rejects_wrong_history_age_schedule():
    payload = valid_request()
    payload["history_step_ages"] = [5, 2]

    with pytest.raises(ValueError, match=r"accepted \[6, 3\]"):
        policy_request_from_mapping(payload)


def test_policy_request_rejects_history_view_mismatch():
    payload = valid_request()
    del payload["history_images_by_view"]["eye_in_hand_rgb"]

    with pytest.raises(ValueError, match="same ordered view names"):
        policy_request_from_mapping(payload)


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
