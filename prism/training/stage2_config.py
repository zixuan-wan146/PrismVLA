from __future__ import annotations

# --- migrated from src/prism/training/stage2/libero/validators.py ---
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism.utils.paths import project_path


STAGE2_REPLAY_INDEX_FORMAT = "libero_episode_replay_index"
STAGE2_DATASET_TYPE = "libero_raw_episode"


def enforce_stage2_contract(config: Mapping[str, Any]) -> None:
    dataset_type = str(config.get("dataset_type", ""))
    if dataset_type != STAGE2_DATASET_TYPE:
        raise ValueError(
            f"Stage2 full E2E training requires dataset_type={STAGE2_DATASET_TYPE}, got {dataset_type!r}"
        )

    required_true = (
        "load_vlm",
        "finetune_vlm",
        "finetune_action_head",
        "progress_planner_enabled",
        "finetune_progress_planner",
    )
    for key in required_true:
        if not bool(config.get(key, False)):
            raise ValueError(f"Stage2 full E2E training requires {key}=true")

    if bool(config.get("memory_token_cache_sequence_training", False)):
        raise ValueError("Stage2 full E2E training must not use token-cache sequence training")
    if bool(config.get("enable_bridge_aux_loss", False)):
        raise ValueError("Stage2 first pass uses action FM only; enable_bridge_aux_loss must be false")
    if config.get("min_cuda_memory_gb") is not None:
        raise ValueError("Stage2 VRAM target must come from real training workload, not min_cuda_memory_gb")

    sampling_mode = str(config.get("stage2_sampling_mode", "group"))
    if sampling_mode != "group":
        raise ValueError(f"Stage2 currently supports MemoryVLA-style group sampling only, got {sampling_mode!r}")

    sequence_len = _as_int(config.get("sequence_len", 0), "--sequence_len")
    if sequence_len <= 0:
        raise ValueError(f"--sequence_len must be positive, got {sequence_len}")

    loss = config.get("loss") or {}
    if loss:
        if not isinstance(loss, Mapping):
            raise ValueError("--loss must be a mapping")
        action_fm = float(loss.get("action_fm", 0.0))
        if action_fm <= 0.0:
            raise ValueError("Stage2 requires loss.action_fm > 0")
        for key in ("vlm_ce", "planner_aux", "gripper_bce"):
            if float(loss.get(key, 0.0)) != 0.0:
                raise ValueError(f"Stage2 action-FM-only profile requires loss.{key}=0.0")


def validate_stage2_replay_index_contract(config: Mapping[str, Any], *, repo_root: str | Path) -> None:
    index_path = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    index_format = payload.get("format")
    if index_format != STAGE2_REPLAY_INDEX_FORMAT:
        raise ValueError(
            f"Stage2 training requires {STAGE2_REPLAY_INDEX_FORMAT} index, got {index_format!r}. "
            "Build it with scripts/build_cache.py using a LIBERO experiment config."
        )

    benchmark = str(payload.get("benchmark", "")).upper()
    if benchmark != "LIBERO":
        raise ValueError(f"Stage2 LIBERO training requires benchmark=LIBERO, got {benchmark!r}")

    episodes = payload.get("episodes")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("Stage2 replay index must contain a non-empty episodes list")

    horizon = _as_int(config.get("horizon", 32), "--horizon")
    index_horizon = payload.get("action_horizon")
    if index_horizon is not None and int(index_horizon) != horizon:
        raise ValueError(f"Stage2 horizon {horizon} does not match replay index action_horizon {index_horizon}")

    valid_episode_count = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("Stage2 replay index episodes must be mappings")
        for key in ("episode_id", "episode_key", "source_path", "episode_length"):
            if episode.get(key) in (None, ""):
                raise ValueError(f"Stage2 replay index episode is missing {key!r}")
        episode_length = _as_int(episode["episode_length"], "episode_length")
        if episode_length >= horizon:
            valid_episode_count += 1

    if valid_episode_count <= 0:
        raise ValueError(f"Stage2 replay index has no episodes with at least horizon={horizon} steps")


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc

# --- migrated from src/prism/training/stage2/libero/config.py ---
from pathlib import Path
from typing import Any, Mapping

from prism.config import resolve_experiment_config
from prism.utils.paths import normalize_project_relative_path, project_path
from prism.config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)


STAGE2_ACTIVE_DEFAULTS: dict[str, Any] = {
    "dataset_type": "libero_raw_episode",
    "load_vlm": True,
    "finetune_vlm": True,
    "finetune_action_head": True,
    "progress_planner_enabled": True,
    "finetune_progress_planner": True,
    "enable_bridge_aux_loss": False,
    "memory_token_cache_sequence_training": False,
    "horizon": 32,
    "sequence_len": 16,
    "stage2_sampling_mode": "group",
    "sample_valid_future_horizon_only": True,
    "shuffle_episodes": True,
    "num_inference_timesteps": 15,
    "inference_tau_schedule": "midpoint",
    "avoid_endpoint_tau": True,
}


def build_stage2_config(
    args: Any,
    *,
    repo_root: str | Path,
    validate_external_artifacts: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    cli_overrides = vars(args).copy()
    config_path = cli_overrides.pop("config", None)
    if config_path:
        config_file = project_path(config_path, repo_root, label="--config")
        file_config = load_training_config(config_file)
        file_config["training_config_path"] = normalize_project_relative_path(
            config_file,
            repo_root,
            label="--config",
        )
    else:
        file_config = {}

    explicit_config_keys = set(STAGE2_ACTIVE_DEFAULTS) | _provided_keys(file_config) | _provided_keys(cli_overrides)
    config = merge_training_config(
        default_training_config(repo_root),
        file_config={**STAGE2_ACTIVE_DEFAULTS, **file_config},
        cli_overrides=cli_overrides,
    )
    config["_explicit_config_keys"] = sorted(explicit_config_keys)
    config["repo_root"] = "."
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    enforce_stage2_contract(config)
    validate_training_config(
        config,
        repo_root=repo_root,
        validate_external_paths=validate_external_artifacts,
    )
    _validate_stage2_external_paths(
        config,
        repo_root=repo_root,
        validate_external_artifacts=validate_external_artifacts,
    )
    replay_index = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    if validate_external_artifacts or replay_index.exists():
        validate_stage2_replay_index_contract(config, repo_root=repo_root)
    return config


def _provided_keys(mapping: Mapping[str, Any]) -> set[str]:
    return {str(key) for key, value in mapping.items() if value is not None}


def _resolve_stage2_paths(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in ("normalization_source_path",):
        value = resolved.get(key)
        if value in (None, ""):
            continue
        resolved[key] = normalize_project_relative_path(value, repo_root, label=f"--{key}")
    return resolved


def _validate_stage2_external_paths(
    config: Mapping[str, Any], *, repo_root: str | Path, validate_external_artifacts: bool
) -> None:
    if not validate_external_artifacts:
        return
    normalization_path = config.get("normalization_source_path")
    if normalization_path:
        path = project_path(normalization_path, repo_root, label="--normalization_source_path")
        if not path.exists():
            raise FileNotFoundError(f"Normalization source file not found: {normalization_path}")

# --- migrated from src/prism/training/stage2/libero/cli.py ---
import argparse
import logging
import os
import sys

from prism.utils.paths import find_repo_root


REPO_ROOT = find_repo_root(__file__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage2 full E2E LIBERO policy from raw episodes")
    parser.add_argument("--config", type=str, default=None, help="Project-relative Stage2 YAML config.")

    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_wandb", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_swanlab", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--bridge_prism_config", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_base_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--normalization_source_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--cache_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--save_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_checkpoint", type=str, default=argparse.SUPPRESS)

    parser.add_argument("--dataset_type", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--sequence_len", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--stage2_sampling_mode", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--sample_valid_future_horizon_only",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--shuffle_episodes", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--load_vlm", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_vlm", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_action_head", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_enabled", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--finetune_progress_planner", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--enable_bridge_aux_loss", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--horizon", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--per_action_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--state_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--memory_entry_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--short_memory_time_bins", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--action_head_ffn_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_plan_slots", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_vlm_tokens", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min_lr_ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad_clip_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--log_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_min_step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--resume_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--resume_pretrain", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--reset_best_loss_on_resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--num_inference_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inference_tau_schedule", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--avoid_endpoint_tau", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = build_arg_parser().parse_args(argv)
    config = build_stage2_config(args, repo_root=REPO_ROOT, validate_external_artifacts=True)
    from prism.training.trainer import train_stage2

    try:
        train_stage2(config, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Cleaning up Stage2 training...")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# --- migrated from src/prism/training/stage2/calvin/validators.py ---
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from prism.utils.paths import project_path


STAGE2_REPLAY_INDEX_FORMAT = "calvin_episode_replay_index"
STAGE2_DATASET_TYPE = "calvin_raw_episode"


def enforce_stage2_contract(config: Mapping[str, Any]) -> None:
    dataset_type = str(config.get("dataset_type", ""))
    if dataset_type != STAGE2_DATASET_TYPE:
        raise ValueError(
            f"Stage2 full E2E training requires dataset_type={STAGE2_DATASET_TYPE}, got {dataset_type!r}"
        )

    required_true = (
        "load_vlm",
        "finetune_vlm",
        "finetune_action_head",
        "progress_planner_enabled",
        "finetune_progress_planner",
    )
    for key in required_true:
        if not bool(config.get(key, False)):
            raise ValueError(f"Stage2 full E2E training requires {key}=true")

    if bool(config.get("memory_token_cache_sequence_training", False)):
        raise ValueError("Stage2 full E2E training must not use token-cache sequence training")
    if bool(config.get("enable_bridge_aux_loss", False)):
        raise ValueError("Stage2 first pass uses action FM only; enable_bridge_aux_loss must be false")
    if config.get("min_cuda_memory_gb") is not None:
        raise ValueError("Stage2 VRAM target must come from real training workload, not min_cuda_memory_gb")

    sampling_mode = str(config.get("stage2_sampling_mode", "group"))
    if sampling_mode != "group":
        raise ValueError(f"Stage2 currently supports MemoryVLA-style group sampling only, got {sampling_mode!r}")

    sequence_len = _as_int(config.get("sequence_len", 0), "--sequence_len")
    if sequence_len <= 0:
        raise ValueError(f"--sequence_len must be positive, got {sequence_len}")

    loss = config.get("loss") or {}
    if loss:
        if not isinstance(loss, Mapping):
            raise ValueError("--loss must be a mapping")
        action_fm = float(loss.get("action_fm", 0.0))
        if action_fm <= 0.0:
            raise ValueError("Stage2 requires loss.action_fm > 0")
        for key in ("vlm_ce", "planner_aux", "gripper_bce"):
            if float(loss.get(key, 0.0)) != 0.0:
                raise ValueError(f"Stage2 action-FM-only profile requires loss.{key}=0.0")


def validate_stage2_replay_index_contract(config: Mapping[str, Any], *, repo_root: str | Path) -> None:
    index_path = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    index_format = payload.get("format")
    if index_format != STAGE2_REPLAY_INDEX_FORMAT:
        raise ValueError(
            f"Stage2 training requires {STAGE2_REPLAY_INDEX_FORMAT} index, got {index_format!r}. "
            "Build it with scripts/build_cache.py using a CALVIN experiment config."
        )

    benchmark = str(payload.get("benchmark", "")).upper()
    if benchmark != "CALVIN":
        raise ValueError(f"Stage2 CALVIN training requires benchmark=CALVIN, got {benchmark!r}")

    episodes = payload.get("episodes")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("Stage2 replay index must contain a non-empty episodes list")

    horizon = _as_int(config.get("horizon", 32), "--horizon")
    index_horizon = payload.get("action_horizon")
    if index_horizon is not None and int(index_horizon) != horizon:
        raise ValueError(f"Stage2 horizon {horizon} does not match replay index action_horizon {index_horizon}")

    valid_episode_count = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("Stage2 replay index episodes must be mappings")
        for key in ("episode_id", "episode_index", "source_path", "episode_length"):
            if episode.get(key) in (None, ""):
                raise ValueError(f"Stage2 replay index episode is missing {key!r}")
        episode_length = _as_int(episode["episode_length"], "episode_length")
        if episode_length >= horizon:
            valid_episode_count += 1

    if valid_episode_count <= 0:
        raise ValueError(f"Stage2 replay index has no episodes with at least horizon={horizon} steps")

# --- migrated from src/prism/training/stage2/calvin/config.py ---
from pathlib import Path
from typing import Any, Mapping

from prism.config import resolve_experiment_config
from prism.utils.paths import normalize_project_relative_path, project_path
from prism.config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)


STAGE2_ACTIVE_DEFAULTS: dict[str, Any] = {
    "dataset_type": "calvin_raw_episode",
    "load_vlm": True,
    "finetune_vlm": True,
    "finetune_action_head": True,
    "progress_planner_enabled": True,
    "finetune_progress_planner": True,
    "enable_bridge_aux_loss": False,
    "memory_token_cache_sequence_training": False,
    "horizon": 32,
    "sequence_len": 16,
    "stage2_sampling_mode": "group",
    "sample_valid_future_horizon_only": True,
    "shuffle_episodes": True,
    "num_inference_timesteps": 15,
    "inference_tau_schedule": "midpoint",
    "avoid_endpoint_tau": True,
}


def build_stage2_config(
    args: Any,
    *,
    repo_root: str | Path,
    validate_external_artifacts: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    cli_overrides = vars(args).copy()
    config_path = cli_overrides.pop("config", None)
    if config_path:
        config_file = project_path(config_path, repo_root, label="--config")
        file_config = load_training_config(config_file)
        file_config["training_config_path"] = normalize_project_relative_path(
            config_file,
            repo_root,
            label="--config",
        )
    else:
        file_config = {}

    explicit_config_keys = set(STAGE2_ACTIVE_DEFAULTS) | _provided_keys(file_config) | _provided_keys(cli_overrides)
    config = merge_training_config(
        default_training_config(repo_root),
        file_config={**STAGE2_ACTIVE_DEFAULTS, **file_config},
        cli_overrides=cli_overrides,
    )
    config["_explicit_config_keys"] = sorted(explicit_config_keys)
    config["repo_root"] = "."
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    enforce_stage2_contract(config)
    validate_training_config(
        config,
        repo_root=repo_root,
        validate_external_paths=validate_external_artifacts,
    )
    _validate_stage2_external_paths(
        config,
        repo_root=repo_root,
        validate_external_artifacts=validate_external_artifacts,
    )
    replay_index = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    if validate_external_artifacts or replay_index.exists():
        validate_stage2_replay_index_contract(config, repo_root=repo_root)
    return config


def _provided_keys(mapping: Mapping[str, Any]) -> set[str]:
    return {str(key) for key, value in mapping.items() if value is not None}


def _resolve_stage2_paths(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in ("normalization_source_path",):
        value = resolved.get(key)
        if value in (None, ""):
            continue
        resolved[key] = normalize_project_relative_path(value, repo_root, label=f"--{key}")
    return resolved


def _validate_stage2_external_paths(
    config: Mapping[str, Any], *, repo_root: str | Path, validate_external_artifacts: bool
) -> None:
    if not validate_external_artifacts:
        return
    normalization_path = config.get("normalization_source_path")
    if normalization_path:
        path = project_path(normalization_path, repo_root, label="--normalization_source_path")
        if not path.exists():
            raise FileNotFoundError(f"Normalization source file not found: {normalization_path}")

# --- migrated from src/prism/training/stage2/calvin/cli.py ---
import logging
import os
import sys

from prism.utils.paths import find_repo_root


REPO_ROOT = find_repo_root(__file__)


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = build_arg_parser().parse_args(argv)
    config = build_stage2_config(args, repo_root=REPO_ROOT, validate_external_artifacts=True)
    from prism.training.trainer import train_stage2

    try:
        train_stage2(config, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Cleaning up CALVIN Stage2 training...")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


