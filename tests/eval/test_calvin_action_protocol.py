from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.calvin.eval import parse_action_response, to_calvin_action


def test_parse_action_response_accepts_actions_object_and_horizon_prefix():
    message = json.dumps({"actions": [[0, 1, 2, 3, 4, 5, 0.25, 99], [6, 7, 8, 9, 10, 11, 0.75]]})

    actions = parse_action_response(message, horizon=1)

    assert actions == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.25, 99.0]]


def test_parse_action_response_rejects_server_error():
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        parse_action_response(json.dumps({"error": "bad checkpoint"}), horizon=1)


def test_to_calvin_action_maps_canonical_open_01_to_environment_sign():
    close_below = to_calvin_action([0, 0, 0, 0, 0, 0, 0.49])
    close_at_threshold = to_calvin_action([0, 0, 0, 0, 0, 0, 0.5])
    open_above = to_calvin_action([0, 0, 0, 0, 0, 0, 0.50001])

    assert close_below[6] == -1.0
    assert close_at_threshold[6] == -1.0
    assert open_above[6] == 1.0


def test_to_calvin_action_clamps_normalized_relative_motion():
    assert to_calvin_action([-3, -1, -0.5, 0.5, 1, 4, 0.5]) == [
        -1,
        -1,
        -0.5,
        0.5,
        1,
        1,
        -1.0,
    ]


def test_to_calvin_action_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        to_calvin_action([0, 0, float("inf"), 0, 0, 0, 0.5])


def test_to_calvin_action_does_not_mutate_or_require_writable_input():
    action = np.array([0, 2, -2, 0, 0, 0, 0.6], dtype=np.float32)
    expected = action.copy()

    assert to_calvin_action(action) == [0, 1, -1, 0, 0, 0, 1.0]
    np.testing.assert_array_equal(action, expected)

    action.setflags(write=False)
    assert to_calvin_action(action) == [0, 1, -1, 0, 0, 0, 1.0]
