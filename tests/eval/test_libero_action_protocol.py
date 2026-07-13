from __future__ import annotations

import json

import pytest

from experiments.libero.eval import (
    parse_action_response,
    to_libero_action,
)


def test_parse_action_response_returns_horizon_prefix():
    message = json.dumps(
        [
            [0, 1, 2, 3, 4, 5, 0.6, 7],
            [8, 9, 10, 11, 12, 13, 0.4, 15],
            [16, 17, 18, 19, 20, 21, 0.2, 23],
        ]
    )

    actions = parse_action_response(message, horizon=2)

    assert actions == [
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6, 7.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 0.4, 15.0],
    ]


def test_parse_action_response_accepts_debug_payload():
    message = json.dumps(
        {
            "actions": [[0, 1, 2, 3, 4, 5, 0.6]],
        }
    )

    assert parse_action_response(message, horizon=1) == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6]]


def test_parse_action_response_rejects_server_error_payload():
    with pytest.raises(RuntimeError, match="server returned error"):
        parse_action_response(json.dumps({"error": "bad request"}), horizon=1)


def test_parse_action_response_rejects_short_horizon():
    with pytest.raises(ValueError, match="expected at least horizon"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5, 6]]), horizon=2)


def test_parse_action_response_rejects_debug_payload_without_actions():
    with pytest.raises(ValueError, match="must contain 'actions'"):
        parse_action_response(json.dumps({"metadata": {"ready": False}}), horizon=1)


def test_parse_action_response_rejects_short_action_dim():
    with pytest.raises(ValueError, match="expected at least 7"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5]]), horizon=1)


def test_parse_action_response_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="not numeric"):
        parse_action_response(json.dumps([[0, 1, 2, 3, 4, 5, "closed"]]), horizon=1)


def test_to_libero_action_clamps_motion_and_converts_gripper_sign():
    assert to_libero_action([0, 1, 2, 3, 4, 5, 0.6, 99]) == [0, 1, 1, 1, 1, 1, -1.0]
    assert to_libero_action([0, -2, 0.5, 0, 0, 0, 0.5, 99]) == [0, -1, 0.5, 0, 0, 0, 1.0]
    assert to_libero_action([0, 0, 0, 0, 0, 0, -0.1, 99]) == [0, 0, 0, 0, 0, 0, 1.0]


def test_to_libero_action_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        to_libero_action([0, 0, 0, float("nan"), 0, 0, 0.5])
