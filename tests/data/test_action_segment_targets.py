import numpy as np
import pytest

from prism.data.segments import (
    build_action_segment_target,
    token_span_steps,
)


def test_action_segment_target_keeps_full_chunk_trajectory():
    actions = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    segments, mask = build_action_segment_target(actions, num_plan_steps=2, planning_horizon=4)

    np.testing.assert_allclose(segments[0], actions[:2])
    np.testing.assert_allclose(segments[1], actions[2:4])
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_action_segment_target_masks_tail_chunks():
    actions = np.ones((6, 3), dtype=np.float32)

    segments, mask = build_action_segment_target(
        actions,
        num_plan_steps=3,
        planning_horizon=6,
        valid_action_count=4,
    )

    np.testing.assert_array_equal(mask, np.array([True, True, False]))
    np.testing.assert_allclose(segments[2], np.zeros((2, 3), dtype=np.float32))


def test_action_segment_target_rejects_nondivisible_horizon():
    with pytest.raises(ValueError, match="divisible"):
        build_action_segment_target(np.ones((5, 3), dtype=np.float32), num_plan_steps=2, planning_horizon=5)


def test_single_token_target_can_cover_full_intent_chunk():
    actions = np.arange(32 * 3, dtype=np.float32).reshape(32, 3)

    segments, mask = build_action_segment_target(actions, num_plan_steps=1, planning_horizon=32)

    assert token_span_steps(planning_horizon=32, num_plan_steps=1) == 32
    assert segments.shape == (1, 32, 3)
    np.testing.assert_allclose(segments[0], actions)
    np.testing.assert_array_equal(mask, np.array([True]))
