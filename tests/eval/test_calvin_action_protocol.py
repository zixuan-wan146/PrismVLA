from __future__ import annotations

import json

import pytest

from experiments.calvin.eval import parse_action_response, to_calvin_action


def test_parse_action_response_accepts_actions_object_and_horizon_prefix():
    message = json.dumps({"actions": [[0, 1, 2, 3, 4, 5, 0.25, 99], [6, 7, 8, 9, 10, 11, 0.75]]})

    actions = parse_action_response(message, horizon=1)

    assert actions == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.25, 99.0]]


def test_parse_action_response_rejects_server_error():
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        parse_action_response(json.dumps({"error": "bad checkpoint"}), horizon=1)


def test_to_calvin_action_openvla_gripper_maps_model_binary_to_env_sign():
    open_action = to_calvin_action([0, 0, 0, 0, 0, 0, 0.2], gripper_mode="openvla")
    close_action = to_calvin_action([0, 0, 0, 0, 0, 0, 0.8], gripper_mode="openvla")

    assert open_action[6] == 1.0
    assert close_action[6] == -1.0


def test_to_calvin_action_passthrough_preserves_gripper():
    action = to_calvin_action([1, 2, 3, 4, 5, 6, -0.7], gripper_mode="passthrough")

    assert action == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -0.7]


def test_to_calvin_action_defaults_to_valid_binary_gripper_sign():
    open_action = to_calvin_action([0, 0, 0, 0, 0, 0, 0.0])
    close_action = to_calvin_action([0, 0, 0, 0, 0, 0, -0.1])

    assert open_action[6] == 1.0
    assert close_action[6] == -1.0
