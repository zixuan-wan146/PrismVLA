from __future__ import annotations

import math

import pytest

from prism.serve.protocol import checkpoint_normalizer_dim
from prism.serve.protocol import policy_request_from_mapping


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
        "state": [0.1, 0.2, 0.3],
        "action_dim": 7,
        "robot_key": "libero",
    }


def test_policy_request_from_mapping_accepts_canonical_payload():
    request = policy_request_from_mapping(valid_request())

    assert request.benchmark == "libero"
    assert request.prompt == "pick up the object"
    assert tuple(request.images_by_view) == ("agentview_rgb", "eye_in_hand_rgb")
    assert request.state.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert request.action_dim == 7
    assert request.robot_key == "libero"
    assert request.return_debug is False


def test_policy_request_from_mapping_accepts_optional_runtime_fields():
    payload = valid_request()
    payload["return_debug"] = True
    payload["reset_memory"] = True
    payload["short_memory_images_by_offset"] = {
        "16": {
            "agentview_rgb": tiny_rgb_image(3),
            "eye_in_hand_rgb": tiny_rgb_image(4),
        }
    }
    payload["executed_actions"] = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]]
    payload["executed_action_mask"] = [1]

    request = policy_request_from_mapping(payload)

    assert request.return_debug is True
    assert request.reset_memory is True
    assert tuple(request.short_memory_images_by_offset or {}) == (16,)
    assert tuple((request.short_memory_images_by_offset or {})[16]) == ("agentview_rgb", "eye_in_hand_rgb")
    assert request.executed_actions.shape == (1, 7)
    assert request.executed_action_mask.tolist() == [True]


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


def test_legacy_payload_is_rejected_by_canonical_runtime_contract():
    with pytest.raises(ValueError, match="Missing required policy request fields"):
        policy_request_from_mapping(
            {
                "image": [tiny_rgb_image(1), tiny_rgb_image(2)],
                "state": [0.1, 0.2],
                "prompt": "legacy",
                "action_mask": [1, 1, 1, 0],
                "robot_key": "libero",
            }
        )


def test_checkpoint_normalizer_dim_tracks_checkpoint_state_and_action_dims():
    assert checkpoint_normalizer_dim({"state_dim": 7, "per_action_dim": 7}) == 7
    assert checkpoint_normalizer_dim({"state_dim": 8, "per_action_dim": 7}) == 8
    assert checkpoint_normalizer_dim({"state_dim": 7, "per_action_dim": 9}) == 9


def test_checkpoint_normalizer_dim_falls_back_for_missing_or_invalid_values():
    assert checkpoint_normalizer_dim({}) == 24
    assert checkpoint_normalizer_dim({"state_dim": 0, "per_action_dim": "bad"}) == 24
