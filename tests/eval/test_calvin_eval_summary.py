from __future__ import annotations

import json

from prism.eval.calvin.eval_summary import (
    SequenceResult,
    summarize_sequence_results,
    write_result_summary,
)


def sample_results() -> list[SequenceResult]:
    return [
        SequenceResult(
            sequence_id=0,
            initial_state="state-a",
            subtasks=["open_drawer", "turn_on_lightbulb", "move_slider_left"],
            successful_subtasks=3,
            success=True,
            decision_steps=6,
            control_steps=42,
            video_paths=["videos/seq0.mp4"],
        ),
        SequenceResult(
            sequence_id=1,
            initial_state="state-b",
            subtasks=["open_drawer", "turn_on_lightbulb", "move_slider_left"],
            successful_subtasks=1,
            success=False,
            decision_steps=5,
            control_steps=31,
            failed_subtask="turn_on_lightbulb",
            failure_reason="max_steps_exhausted",
        ),
    ]


def test_summarize_sequence_results_reports_chain_metrics_and_tasks():
    summary = summarize_sequence_results(sample_results())

    assert summary["total_sequences"] == 2
    assert summary["successful_sequences"] == 1
    assert summary["sequence_success_rate"] == 0.5
    assert summary["average_successful_subtasks"] == 2.0
    assert summary["chain_success_rates"] == {"1": 1.0, "2": 0.5, "3": 0.5}
    assert summary["task_info"]["open_drawer"]["success"] == 2
    assert summary["task_info"]["turn_on_lightbulb"]["success"] == 1
    assert summary["successful_sequence_ids"] == [0]


def test_write_result_summary_persists_config_summary_and_sequences(tmp_path):
    result_path = tmp_path / "nested" / "calvin_results.json"

    written_path = write_result_summary(
        result_path,
        config={"horizon": 32, "num_sequences": 2},
        results=sample_results(),
        metadata={"created_at_utc": "2026-07-07T00:00:00Z", "git": {"commit": "abc123"}},
    )

    payload = json.loads(written_path.read_text())
    assert written_path == result_path
    assert payload["config"]["horizon"] == 32
    assert payload["metadata"]["git"]["commit"] == "abc123"
    assert payload["summary"]["total_sequences"] == 2
    assert payload["sequences"][1]["failed_subtask"] == "turn_on_lightbulb"
