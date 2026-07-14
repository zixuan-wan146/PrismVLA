from __future__ import annotations

from experiments.calvin.config import (
    DEFAULT_CALVIN_ROOT,
    CalvinClientConfig,
    configure_calvin_environment,
)
from prism.serve.wire import DEFAULT_MAX_MESSAGE_SIZE_BYTES


def test_default_calvin_config_matches_abc_d_eval_defaults():
    config = CalvinClientConfig.from_env({})

    assert config.server_url == "ws://127.0.0.1:9000"
    assert config.calvin_root == DEFAULT_CALVIN_ROOT
    assert config.dataset_path == f"{DEFAULT_CALVIN_ROOT}/dataset/task_ABC_D"
    assert config.num_sequences == 1000
    assert config.horizon == 8
    assert config.max_steps_per_subtask == 360
    assert config.save_video is False
    assert config.connect_timeout_seconds == 30.0
    assert config.inference_timeout_seconds == 120.0
    assert config.max_message_size_bytes == DEFAULT_MAX_MESSAGE_SIZE_BYTES


def test_calvin_config_prefers_shared_server_uri():
    config = CalvinClientConfig.from_env(
        {
            "PRISM_SERVER_URI": "ws://shared:9000",
            "PRISM_CALVIN_SERVER_URL": "ws://calvin-only:9000",
        }
    )

    assert config.server_url == "ws://shared:9000"


def test_calvin_config_can_override_paths_and_counts():
    config = CalvinClientConfig.from_env(
        {
            "PRISM_CALVIN_ROOT": "local_data/datasets/calvin/runtime",
            "PRISM_CALVIN_DATASET_PATH": "local_data/datasets/calvin/runtime/dataset/task_D_D",
            "PRISM_CALVIN_NUM_SEQUENCES": "3",
            "PRISM_CALVIN_SEQUENCE_OFFSET": "7",
            "PRISM_CALVIN_SAVE_VIDEO": "true",
            "PRISM_POLICY_CONNECT_TIMEOUT_SECONDS": "8.5",
            "PRISM_POLICY_INFERENCE_TIMEOUT_SECONDS": "45",
            "PRISM_POLICY_MAX_MESSAGE_SIZE_BYTES": "8388608",
        }
    )

    assert config.dataset_path == "local_data/datasets/calvin/runtime/dataset/task_D_D"
    assert config.num_sequences == 3
    assert config.sequence_offset == 7
    assert config.save_video is True
    assert config.connect_timeout_seconds == 8.5
    assert config.inference_timeout_seconds == 45.0
    assert config.max_message_size_bytes == 8388608


def test_calvin_rejects_nonpositive_policy_timeout():
    try:
        CalvinClientConfig.from_env({"PRISM_POLICY_CONNECT_TIMEOUT_SECONDS": "-1"})
    except ValueError as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("Expected negative connection timeout to raise ValueError")


def test_configure_calvin_environment_sets_calvin_root_and_egl_platform():
    config = CalvinClientConfig.from_env({"PRISM_MUJOCO_GL": "egl", "PRISM_CALVIN_ROOT": "local_data/calvin"})
    environ = {}

    configure_calvin_environment(config, environ)

    assert environ["CALVIN_ROOT"] == "local_data/calvin"
    assert environ["MUJOCO_GL"] == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"


def test_calvin_rejects_nonbaseline_action_horizon():
    try:
        CalvinClientConfig.from_env({"PRISM_CALVIN_HORIZON": "16"})
    except ValueError as exc:
        assert "architecture horizon 8" in str(exc)
    else:
        raise AssertionError("Expected nonbaseline horizon to raise ValueError")
