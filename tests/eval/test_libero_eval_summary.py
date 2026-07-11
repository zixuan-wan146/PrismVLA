from __future__ import annotations

import json

from prism.eval.libero.eval_summary import (
    EpisodeResult,
    build_run_metadata,
    summarize_episode_results,
    write_result_summary,
)


def sample_results() -> list[EpisodeResult]:
    return [
        EpisodeResult(
            task_suite="libero_spatial",
            task_id=0,
            episode_id=0,
            task_description="put the object in the bowl",
            success=True,
            decision_steps=3,
            control_steps=42,
            video_path="videos/task1_episode1.mp4",
        ),
        EpisodeResult(
            task_suite="libero_spatial",
            task_id=0,
            episode_id=1,
            task_description="put the object in the bowl",
            success=False,
            decision_steps=5,
            control_steps=70,
            failure_reason="max_steps_exhausted",
            video_path="videos/task1_episode2.mp4",
        ),
        EpisodeResult(
            task_suite="libero_goal",
            task_id=2,
            episode_id=0,
            task_description="open the drawer",
            success=False,
            decision_steps=1,
            control_steps=0,
            failure_reason="action_parse_error: bad response",
        ),
    ]


def test_summarize_episode_results_reports_overall_and_suite_metrics():
    summary = summarize_episode_results(sample_results())

    assert summary["total_episodes"] == 3
    assert summary["successful_episodes"] == 1
    assert summary["failed_episodes"] == 2
    assert summary["success_rate"] == 1 / 3
    assert summary["average_decision_steps"] == 3.0
    assert summary["average_success_decision_steps"] == 3.0
    assert summary["suites"]["libero_spatial"]["success_rate"] == 0.5
    assert summary["suites"]["libero_goal"]["success_rate"] == 0.0
    assert summary["successful_episode_ids"] == [
        {"task_suite": "libero_spatial", "task_id": 0, "episode_id": 0}
    ]


def test_write_result_summary_persists_config_summary_and_episodes(tmp_path):
    result_path = tmp_path / "nested" / "results.json"

    written_path = write_result_summary(
        result_path,
        config={"horizon": 14, "task_suites": ["libero_spatial"]},
        results=sample_results(),
        metadata={
            "created_at_utc": "2026-06-11T00:00:00Z",
            "git": {"commit": "abc123", "is_dirty": False},
            "environment": {"PRISM_SERVER_URI": "ws://127.0.0.1:9000"},
        },
    )

    payload = json.loads(written_path.read_text())
    assert written_path == result_path
    assert payload["config"]["horizon"] == 14
    assert payload["metadata"]["git"]["commit"] == "abc123"
    assert payload["summary"]["total_episodes"] == 3
    assert payload["episodes"][1]["failure_reason"] == "max_steps_exhausted"


def test_build_run_metadata_keeps_safe_environment_and_redacts_secret_like_keys(tmp_path):
    metadata = build_run_metadata(
        repo_root=tmp_path,
        environ={
            "PRISM_SERVER_URI": "ws://127.0.0.1:9000",
            "PRISM_API_TOKEN": "secret",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "UNRELATED": "ignored",
            "MUJOCO_GL": "osmesa",
        },
        argv=["libero_client_4tasks.py"],
        created_at_utc="2026-06-11T00:00:00Z",
    )

    assert metadata["created_at_utc"] == "2026-06-11T00:00:00Z"
    assert metadata["argv"] == ["libero_client_4tasks.py"]
    assert metadata["git"]["commit"] is None
    assert metadata["git"]["is_dirty"] is None
    assert metadata["environment"] == {
        "PRISM_SERVER_URI": "ws://127.0.0.1:9000",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "MUJOCO_GL": "osmesa",
    }
