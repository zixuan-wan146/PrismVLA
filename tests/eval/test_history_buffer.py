from __future__ import annotations

import pytest

from prism.serve.history import ConnectionHistoryState, HistoryCaptureTarget, HistoryPrecomputeSchedule


def test_history_schedule_emits_generation_targets_only_at_o2_and_o5():
    schedule = HistoryPrecomputeSchedule()

    targets = [schedule.target_for_step(step) for step in range(1, 9)]

    assert targets == [
        None,
        HistoryCaptureTarget(target_generation=1, slot=0),
        None,
        None,
        HistoryCaptureTarget(target_generation=1, slot=1),
        None,
        None,
        None,
    ]
    assert schedule.scheduled_slots == (0, 1)
    assert schedule.advance_generation() == 1
    assert schedule.target_for_step(2) == HistoryCaptureTarget(target_generation=2, slot=0)


def test_history_schedule_rejects_duplicates_and_incomplete_generation():
    schedule = HistoryPrecomputeSchedule()
    schedule.target_for_step(2)

    with pytest.raises(ValueError, match="scheduled more than once"):
        schedule.target_for_step(2)
    with pytest.raises(RuntimeError, match="both capture slots"):
        schedule.advance_generation()


def test_connection_history_state_builds_memory_and_releases_visual_slots():
    state: ConnectionHistoryState[str, str] = ConnectionHistoryState()
    state.reset("episode:1")

    assert state.memory_for_inference(
        stream_id="episode:1",
        generation=0,
        empty_memory=lambda: "empty-memory",
    ) == "empty-memory"
    state.mark_inference_complete(stream_id="episode:1", generation=0)

    built_from = []

    def build_memory(observations):
        built_from.append(observations)
        return "ready-memory"

    first = state.reserve_observation(
        stream_id="episode:1",
        target_generation=1,
        slot=0,
    )
    assert state.reserved_slots == (0,)
    assert not state.commit_observation(
        first,
        observation="visual-o2",
        build_memory=build_memory,
    )
    assert state.reserved_slots == ()
    assert state.cached_visual_slots == (0,)
    second = state.reserve_observation(
        stream_id="episode:1",
        target_generation=1,
        slot=1,
    )
    assert state.commit_observation(
        second,
        observation="visual-o5",
        build_memory=build_memory,
    )

    assert built_from == [("visual-o2", "visual-o5")]
    assert state.cached_visual_slots == ()
    assert state.ready_generation == 1
    assert state.memory_for_inference(
        stream_id="episode:1",
        generation=1,
        empty_memory=lambda: "unused",
    ) == "ready-memory"
    state.mark_inference_complete(stream_id="episode:1", generation=1)
    assert state.ready_generation is None
    assert state.last_inferred_generation == 1


def test_connection_history_state_requires_reset_and_complete_memory():
    state: ConnectionHistoryState[str, str] = ConnectionHistoryState()
    with pytest.raises(RuntimeError, match="reset_history"):
        state.memory_for_inference(stream_id="missing", generation=0, empty_memory=lambda: "empty")

    state.reset("episode:2")
    state.memory_for_inference(stream_id="episode:2", generation=0, empty_memory=lambda: "empty")
    state.mark_inference_complete(stream_id="episode:2", generation=0)
    reservation = state.reserve_observation(
        stream_id="episode:2",
        target_generation=1,
        slot=0,
    )
    state.commit_observation(
        reservation,
        observation="only-slot",
        build_memory=lambda observations: "unused",
    )
    with pytest.raises(RuntimeError, match=r"missing history slots \[1\]"):
        state.memory_for_inference(stream_id="episode:2", generation=1, empty_memory=lambda: "empty")
    with pytest.raises(ValueError, match="Active stream"):
        state.memory_for_inference(stream_id="other", generation=1, empty_memory=lambda: "empty")


def test_connection_history_reset_drops_partial_and_ready_tokens():
    state: ConnectionHistoryState[object, object] = ConnectionHistoryState()
    state.reset("old")
    state.memory_for_inference(stream_id="old", generation=0, empty_memory=object)
    state.mark_inference_complete(stream_id="old", generation=0)
    reservation = state.reserve_observation(
        stream_id="old",
        target_generation=1,
        slot=0,
    )
    state.commit_observation(
        reservation,
        observation=object(),
        build_memory=lambda observations: object(),
    )
    assert state.cached_visual_slots == (0,)

    state.reset("new")

    assert state.stream_id == "new"
    assert state.cached_visual_slots == ()
    assert state.ready_generation is None
    assert state.last_inferred_generation == -1


def test_connection_history_build_failure_drops_both_visual_slots():
    state: ConnectionHistoryState[str, str] = ConnectionHistoryState()
    state.reset("episode:3")
    state.memory_for_inference(stream_id="episode:3", generation=0, empty_memory=lambda: "empty")
    state.mark_inference_complete(stream_id="episode:3", generation=0)
    first = state.reserve_observation(
        stream_id="episode:3",
        target_generation=1,
        slot=0,
    )
    state.commit_observation(
        first,
        observation="o2",
        build_memory=lambda observations: "unused",
    )

    def fail_build(observations):
        raise RuntimeError(f"failed for {observations}")

    second = state.reserve_observation(
        stream_id="episode:3",
        target_generation=1,
        slot=1,
    )
    with pytest.raises(RuntimeError, match="failed for"):
        state.commit_observation(
            second,
            observation="o5",
            build_memory=fail_build,
        )
    assert state.cached_visual_slots == ()
    assert state.ready_generation is None


def test_connection_history_reservation_rolls_back_without_losing_committed_slot():
    state: ConnectionHistoryState[str, str] = ConnectionHistoryState()
    state.reset("episode:4")
    state.memory_for_inference(stream_id="episode:4", generation=0, empty_memory=lambda: "empty")
    state.mark_inference_complete(stream_id="episode:4", generation=0)

    first_attempt = state.reserve_observation(
        stream_id="episode:4",
        target_generation=1,
        slot=0,
    )
    assert state.rollback_observation(first_attempt)
    assert state.reserved_slots == ()

    first = state.reserve_observation(
        stream_id="episode:4",
        target_generation=1,
        slot=0,
    )
    state.commit_observation(
        first,
        observation="visual-o2",
        build_memory=lambda observations: "unused",
    )
    second_attempt = state.reserve_observation(
        stream_id="episode:4",
        target_generation=1,
        slot=1,
    )
    assert state.rollback_observation(second_attempt)
    assert state.cached_visual_slots == (0,)
    assert state.reserved_slots == ()
    assert state.reserve_observation(
        stream_id="episode:4",
        target_generation=1,
        slot=1,
    ).slot == 1
