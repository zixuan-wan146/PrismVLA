from __future__ import annotations

from prism.data.cache import DEFAULT_MEMORY_SHORT_OFFSETS
from prism.data.cache import build_memory_replay_manifest
from prism.data.cache import build_memory_replay_samples
from prism.data.cache import read_memory_replay_jsonl
from prism.data.cache import write_memory_replay_jsonl


def test_build_memory_replay_samples_uses_low_level_offsets_and_masks_missing_history(tmp_path):
    samples = build_memory_replay_samples(
        episode_id="episode0",
        episode_length=80,
        action_horizon=32,
        stride=16,
        short_offsets=(32, 16),
        include_tail=False,
        benchmark="synthetic",
        task_name="task",
        source_path="episode0.hdf5",
    )

    assert [sample.current_step for sample in samples] == [0, 16, 32, 48]
    assert samples[0].short_steps == (None, None)
    assert samples[0].short_mask == (False, False)
    assert samples[2].short_steps == (0, 16)
    assert samples[2].short_mask == (True, True)
    assert samples[3].long_steps == ()
    assert samples[-1].action_valid_count == 32
    assert samples[0].executed_action_start == 0
    assert samples[0].executed_action_valid_count == 0
    assert samples[1].executed_action_start == 0
    assert samples[1].executed_action_valid_count == 16

    output = write_memory_replay_jsonl(tmp_path / "index.jsonl", samples)
    rows = read_memory_replay_jsonl(output)

    assert rows[0]["action_start"] == 0
    assert rows[0]["action_end"] == 32
    assert rows[1]["executed_action_start"] == 0
    assert rows[1]["executed_action_end"] == 16
    assert rows[1]["executed_action_valid_count"] == 16
    assert rows[0]["source_path"] == "episode0.hdf5"
    assert rows[2]["short_steps"] == [0, 16]


def test_build_memory_replay_samples_can_include_tail_with_valid_count():
    samples = build_memory_replay_samples(
        episode_id="episode0",
        episode_length=40,
        action_horizon=32,
        stride=16,
        include_tail=True,
    )

    assert [sample.current_step for sample in samples] == [0, 16, 32]
    assert [sample.action_valid_count for sample in samples] == [32, 24, 8]
    assert DEFAULT_MEMORY_SHORT_OFFSETS == (16, 8)


def test_build_memory_replay_samples_can_offset_future_action_targets(tmp_path):
    samples = build_memory_replay_samples(
        episode_id="episode0",
        episode_length=6,
        action_horizon=2,
        stride=2,
        action_start_offset=1,
        include_tail=False,
    )

    assert [sample.current_step for sample in samples] == [0, 2]
    assert [sample.action_start for sample in samples] == [1, 3]

    rows = read_memory_replay_jsonl(write_memory_replay_jsonl(tmp_path / "memory_replay_offset_test.jsonl", samples))
    assert rows[0]["action_start"] == 1
    assert rows[0]["action_end"] == 3
    assert rows[1]["action_start"] == 3
    assert rows[1]["action_end"] == 5


def test_build_memory_replay_manifest_records_generation_policy():
    manifest = build_memory_replay_manifest(
        benchmark="CALVIN",
        action_horizon=32,
        stride=1,
        short_offsets=(16, 32),
        action_start_offset=1,
        long_capacity=0,
        include_tail=False,
        sample_count=10,
        episode_count=2,
        task_counts={"b": 4, "a": 6},
    )

    assert manifest["format"] == "memory_replay_index"
    assert manifest["short_offsets"] == [32, 16]
    assert manifest["executed_action_stride"] == 16
    assert manifest["action_start_offset"] == 1
    assert manifest["long_capacity"] == 0
    assert manifest["task_counts"] == {"a": 6, "b": 4}


def test_build_memory_replay_samples_rejects_deprecated_long_memory_inputs():
    try:
        build_memory_replay_samples(
            episode_id="episode0",
            episode_length=80,
            long_candidate_steps=(8,),
        )
    except ValueError as exc:
        assert "long_candidate_steps is deprecated" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected long_candidate_steps to be rejected")

    try:
        build_memory_replay_samples(
            episode_id="episode0",
            episode_length=80,
            long_capacity=1,
        )
    except ValueError as exc:
        assert "long_capacity must be 0" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected nonzero long_capacity to be rejected")
