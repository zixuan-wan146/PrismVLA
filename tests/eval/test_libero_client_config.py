from __future__ import annotations

from experiments.libero.config import (
    DEFAULT_TASK_SUITES,
    LiberoClientConfig,
    align_max_steps,
    configure_mujoco_environment,
)
from prism.serve.wire import DEFAULT_MAX_MESSAGE_SIZE_BYTES


def test_default_config_matches_documented_smoke_server():
    config = LiberoClientConfig.from_env({})

    assert config.server_url == "ws://127.0.0.1:9000"
    assert config.task_suites == DEFAULT_TASK_SUITES
    assert config.max_steps == [220, 280, 300, 520]
    assert config.horizon == 8
    assert config.episode_offset == 0
    assert config.mujoco_gl == "osmesa"
    assert config.result_file == "./log_file/Prism_libero_all_results.json"
    assert config.camera_resolution == 448
    assert config.video_fps == 30
    assert config.connect_timeout_seconds == 30.0
    assert config.inference_timeout_seconds == 120.0
    assert config.max_message_size_bytes == DEFAULT_MAX_MESSAGE_SIZE_BYTES


def test_single_max_steps_value_expands_to_all_task_suites():
    config = LiberoClientConfig.from_env(
        {
            "PRISM_LIBERO_TASK_SUITES": "libero_spatial,libero_goal",
            "PRISM_LIBERO_MAX_STEPS": "7",
        }
    )

    assert config.task_suites == ["libero_spatial", "libero_goal"]
    assert config.max_steps == [7, 7]


def test_task_suite_subset_uses_matching_default_control_budgets():
    config = LiberoClientConfig.from_env({"PRISM_LIBERO_TASK_SUITES": "libero_goal,libero_spatial"})

    assert config.max_steps == [300, 220]


def test_nondefault_task_suite_accepts_an_explicit_control_budget():
    config = LiberoClientConfig.from_env(
        {
            "PRISM_LIBERO_TASK_SUITES": "libero_90",
            "PRISM_LIBERO_MAX_STEPS": "400",
        }
    )

    assert config.max_steps == [400]


def test_server_url_prefers_shared_env_var():
    config = LiberoClientConfig.from_env(
        {
            "PRISM_SERVER_URI": "ws://server-uri:9000",
            "PRISM_LIBERO_SERVER_URL": "ws://libero-only:9000",
        }
    )

    assert config.server_url == "ws://server-uri:9000"


def test_result_file_can_be_overridden():
    config = LiberoClientConfig.from_env({"PRISM_LIBERO_RESULT_FILE": "run_outputs/results.json"})

    assert config.result_file == "run_outputs/results.json"


def test_rendering_and_policy_timeouts_can_be_overridden():
    config = LiberoClientConfig.from_env(
        {
            "PRISM_LIBERO_CAMERA_RESOLUTION": "320",
            "PRISM_LIBERO_VIDEO_FPS": "24",
            "PRISM_POLICY_CONNECT_TIMEOUT_SECONDS": "4.5",
            "PRISM_POLICY_INFERENCE_TIMEOUT_SECONDS": "19",
            "PRISM_POLICY_MAX_MESSAGE_SIZE_BYTES": "8388608",
        }
    )

    assert config.camera_resolution == 320
    assert config.video_fps == 24
    assert config.connect_timeout_seconds == 4.5
    assert config.inference_timeout_seconds == 19.0
    assert config.max_message_size_bytes == 8388608


def test_nonpositive_policy_timeout_is_rejected():
    try:
        LiberoClientConfig.from_env({"PRISM_POLICY_INFERENCE_TIMEOUT_SECONDS": "0"})
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("Expected zero inference timeout to raise ValueError")


def test_invalid_max_steps_count_is_rejected():
    try:
        align_max_steps([1, 2], ["libero_spatial", "libero_goal", "libero_10"])
    except ValueError as exc:
        assert "one integer per task suite" in str(exc)
    else:
        raise AssertionError("Expected max_steps mismatch to raise ValueError")


def test_negative_task_limit_is_rejected():
    try:
        LiberoClientConfig.from_env({"PRISM_LIBERO_TASK_LIMIT": "-1"})
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative task limit to raise ValueError")


def test_episode_offset_can_be_configured():
    config = LiberoClientConfig.from_env({"PRISM_LIBERO_EPISODE_OFFSET": "7"})

    assert config.episode_offset == 7


def test_negative_episode_offset_is_rejected():
    try:
        LiberoClientConfig.from_env({"PRISM_LIBERO_EPISODE_OFFSET": "-1"})
    except ValueError as exc:
        assert "PRISM_LIBERO_EPISODE_OFFSET" in str(exc)
    else:
        raise AssertionError("Expected negative episode offset to raise ValueError")


def test_nonbaseline_action_horizon_is_rejected():
    try:
        LiberoClientConfig.from_env({"PRISM_LIBERO_HORIZON": "4"})
    except ValueError as exc:
        assert "architecture horizon 8" in str(exc)
    else:
        raise AssertionError("Expected nonbaseline horizon to raise ValueError")


def test_configure_mujoco_environment_sets_egl_platform():
    config = LiberoClientConfig.from_env({"PRISM_MUJOCO_GL": "egl"})
    environ = {}

    configure_mujoco_environment(config, environ)

    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"
