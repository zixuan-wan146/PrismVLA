from __future__ import annotations

# --- migrated from src/prism/training_config.py ---
from pathlib import Path
from typing import Callable, Mapping, Any

from prism.utils.paths import normalize_project_relative_path, project_path


TRAINING_DEFAULTS: dict[str, Any] = {
    "device": "cuda",
    "run_name": "default_run",
    "vlm_name": "OpenGVLab/InternVL3-1B",
    "load_vlm": True,
    "action_head": "flowmatching",
    "bridge_prism_config": None,
    "seed": None,
    "deterministic": False,
    "return_cls_only": False,
    "disable_wandb": False,
    "disable_swanlab": False,
    "dataset_type": "simulation",
    "dataset_config_path": None,
    "dataset_config_base_dir": ".",
    "cache_dir": "run_outputs/training_data_cache",
    "image_size": 448,
    "binarize_gripper": False,
    "use_augmentation": False,
    "lr": 1e-5,
    "batch_size": 16,
    "max_steps": 600,
    "warmup_steps": 300,
    "grad_clip_norm": 1.0,
    "weight_decay": 1e-5,
    "min_lr_ratio": 0.0,
    "lr_groups": {},
    "enable_bridge_aux_loss": False,
    "log_interval": 10,
    "ckpt_interval": 10,
    "best_ckpt_interval": 1000,
    "best_ckpt_min_step": 1000,
    "save_dir": "checkpoints",
    "resume": False,
    "resume_path": None,
    "resume_pretrain": False,
    "finetune_vlm": False,
    "finetune_action_head": False,
    "progress_planner_enabled": False,
    "progress_planner_checkpoint": None,
    "finetune_progress_planner": False,
    "progress_planner_replan_stride": 16,
    "per_action_dim": 7,
    "state_dim": 7,
    "horizon": 32,
    "num_layers": 8,
    "action_head_ffn_dim": 3584,
    "num_plan_slots": 8,
    "visual_gate_lambda": 0.5,
    "plan_gate_lambda": 0.25,
    "short_memory_time_bins": 2,
    "memory_token_cache_sequence_training": False,
    "burnin_replan_steps": 8,
    "loss_replan_steps": 8,
    "allow_short_burnin": True,
    "trajectory_window_stride": 1,
    "shuffle_trajectory_windows": False,
    "max_vlm_tokens": None,
    "num_inference_timesteps": 15,
    "inference_tau_schedule": "midpoint",
    "avoid_endpoint_tau": True,
    "num_workers": 4,
    "dropout": 0.0,
    "boundary_loss_weight": 1.0,
    "progress_loss_weight": 0.2,
    "min_cuda_memory_gb": None,
}


INPUT_PATH_KEYS = (
    "dataset_config_path",
    "dataset_config_base_dir",
    "bridge_prism_config",
    "progress_planner_checkpoint",
    "resume_path",
)

OUTPUT_PATH_KEYS = (
    "save_dir",
    "cache_dir",
)

METADATA_PATH_KEYS = (
    "repo_root",
    "training_config_path",
    "bridge_prism_config_path",
)


def default_training_config(repo_root: str | Path | None = None) -> dict[str, Any]:
    config = dict(TRAINING_DEFAULTS)
    return config


def load_training_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load training YAML configs") from exc

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return dict(loaded)


def merge_training_config(
    defaults: Mapping[str, Any],
    file_config: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(defaults)
    if file_config:
        merged.update({key: value for key, value in file_config.items() if value is not None})
    if cli_overrides:
        merged.update({key: value for key, value in cli_overrides.items() if value is not None})
    return merged


def resolve_training_config_paths(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in (*INPUT_PATH_KEYS, *OUTPUT_PATH_KEYS, *METADATA_PATH_KEYS):
        value = resolved.get(key)
        if value in (None, ""):
            continue
        resolved[key] = normalize_project_relative_path(value, repo_root, label=f"--{key}")
    return resolved


POSITIVE_INT_KEYS = (
    "max_steps",
    "log_interval",
    "ckpt_interval",
    "best_ckpt_interval",
    "batch_size",
    "horizon",
    "per_action_dim",
    "state_dim",
    "progress_planner_replan_stride",
    "num_inference_timesteps",
    "num_layers",
    "action_head_ffn_dim",
    "num_plan_slots",
    "short_memory_time_bins",
    "loss_replan_steps",
    "trajectory_window_stride",
)

NON_NEGATIVE_INT_KEYS = (
    "num_workers",
    "warmup_steps",
    "best_ckpt_min_step",
    "burnin_replan_steps",
)

POSITIVE_FLOAT_KEYS = (
    "lr",
    "grad_clip_norm",
)

NON_NEGATIVE_FLOAT_KEYS = (
    "weight_decay",
    "dropout",
    "boundary_loss_weight",
    "progress_loss_weight",
    "visual_gate_lambda",
    "plan_gate_lambda",
    "min_lr_ratio",
)


def validate_training_config(
    config: Mapping[str, Any],
    *,
    cuda_available: bool | None = None,
    path_exists: Callable[[Path], bool] | None = None,
    repo_root: str | Path | None = None,
    validate_external_paths: bool = True,
) -> None:
    path_exists = _default_path_exists if path_exists is None else path_exists

    dataset_config_path = config.get("dataset_config_path")
    if not dataset_config_path:
        raise ValueError("--dataset_config_path is required")
    dataset_config = _validation_path(dataset_config_path, repo_root, label="--dataset_config_path")
    if validate_external_paths and not path_exists(dataset_config):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")

    dataset_config_base_dir = config.get("dataset_config_base_dir")
    if dataset_config_base_dir and not path_exists(
        _validation_path(dataset_config_base_dir, repo_root, label="--dataset_config_base_dir")
    ):
        raise FileNotFoundError(f"Dataset config base directory not found: {dataset_config_base_dir}")

    bridge_prism_config = config.get("bridge_prism_config")
    if bridge_prism_config and not path_exists(
        _validation_path(bridge_prism_config, repo_root, label="--bridge_prism_config")
    ):
        raise FileNotFoundError(f"Bridge-Prism config file not found: {bridge_prism_config}")

    progress_planner_checkpoint = config.get("progress_planner_checkpoint")
    if progress_planner_checkpoint and validate_external_paths and not path_exists(
        _validation_path(progress_planner_checkpoint, repo_root, label="--progress_planner_checkpoint")
    ):
        raise FileNotFoundError(f"Progress planner checkpoint not found: {progress_planner_checkpoint}")
    if (
        bool(config.get("progress_planner_enabled", False))
        and not progress_planner_checkpoint
        and not bool(config.get("finetune_progress_planner", False))
    ):
        raise ValueError(
            "--progress_planner_enabled=true with no --progress_planner_checkpoint and "
            "--finetune_progress_planner=false would use a random frozen progress planner"
        )

    for key in POSITIVE_INT_KEYS:
        value = _as_int(config.get(key, TRAINING_DEFAULTS.get(key, 0)), f"--{key}")
        if value <= 0:
            raise ValueError(f"--{key} must be positive, got {value}")

    if config.get("max_vlm_tokens") is not None:
        value = _as_int(config["max_vlm_tokens"], "--max_vlm_tokens")
        if value <= 0:
            raise ValueError(f"--max_vlm_tokens must be positive, got {value}")

    for key in NON_NEGATIVE_INT_KEYS:
        if key in config:
            value = _as_int(config[key], f"--{key}")
            if value < 0:
                raise ValueError(f"--{key} must be non-negative, got {value}")

    for key in POSITIVE_FLOAT_KEYS:
        if key in config:
            value = _as_float(config[key], f"--{key}")
            if value <= 0:
                raise ValueError(f"--{key} must be positive, got {value}")

    lr_groups = config.get("lr_groups", {})
    if lr_groups is None:
        lr_groups = {}
    if not isinstance(lr_groups, Mapping):
        raise ValueError("--lr_groups must be a mapping from group name to learning rate")
    for group_name, group_lr in lr_groups.items():
        value = _as_float(group_lr, f"--lr_groups.{group_name}")
        if value <= 0:
            raise ValueError(f"--lr_groups.{group_name} must be positive, got {value}")

    for key in NON_NEGATIVE_FLOAT_KEYS:
        if key in config:
            value = _as_float(config[key], f"--{key}")
            if value < 0:
                raise ValueError(f"--{key} must be non-negative, got {value}")

    if config.get("min_cuda_memory_gb") is not None:
        value = _as_float(config["min_cuda_memory_gb"], "--min_cuda_memory_gb")
        if value <= 0:
            raise ValueError(f"--min_cuda_memory_gb must be positive, got {value}")

    dropout = _as_float(config.get("dropout", 0.0), "--dropout")
    if dropout > 1:
        raise ValueError(f"--dropout must be <= 1, got {dropout}")

    min_lr_ratio = _as_float(config.get("min_lr_ratio", 0.0), "--min_lr_ratio")
    if min_lr_ratio > 1:
        raise ValueError(f"--min_lr_ratio must be <= 1, got {min_lr_ratio}")

    inference_tau_schedule = str(config.get("inference_tau_schedule", "midpoint")).lower()
    if inference_tau_schedule != "midpoint":
        raise ValueError("--inference_tau_schedule must be midpoint")

    warmup_steps = _as_int(config.get("warmup_steps", 0), "--warmup_steps")
    max_steps = _as_int(config.get("max_steps", 0), "--max_steps")
    if warmup_steps > max_steps:
        raise ValueError(f"--warmup_steps must be <= --max_steps, got {warmup_steps} > {max_steps}")

    device = str(config.get("device", "cuda"))
    if device.startswith("cuda"):
        if cuda_available is None:
            cuda_available = _torch_cuda_available()
        if not cuda_available:
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")

    resume = bool(config.get("resume", False))
    resume_path = config.get("resume_path")
    if resume != bool(resume_path):
        raise ValueError("Inconsistent resume configuration: --resume and --resume_path must be set together.")
    if bool(config.get("resume_pretrain", False)) and not resume:
        raise ValueError("--resume_pretrain requires --resume and --resume_path.")
    if resume and validate_external_paths and not path_exists(_validation_path(resume_path, repo_root, label="--resume_path")):
        raise FileNotFoundError(f"Resume checkpoint path not found: {resume_path}")

    if bool(config.get("finetune_vlm", False)) and not bool(config.get("load_vlm", True)):
        raise ValueError("--finetune_vlm=true requires --load_vlm=true")
    dataset_type = str(config.get("dataset_type", "simulation"))
    if dataset_type != "memory_token_cache" and not bool(config.get("load_vlm", True)):
        raise ValueError("--load_vlm=false is only supported with dataset_type=memory_token_cache")
    if bool(config.get("memory_token_cache_sequence_training", False)):
        if dataset_type != "memory_token_cache":
            raise ValueError("--memory_token_cache_sequence_training=true requires dataset_type=memory_token_cache")
        if not bool(config.get("progress_planner_enabled", False)) and not config.get("progress_planner_checkpoint"):
            raise ValueError("--memory_token_cache_sequence_training=true requires a progress planner")


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc


def _default_path_exists(path: Path) -> bool:
    return path.exists()


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _validation_path(value: Any, repo_root: str | Path | None, *, label: str) -> Path:
    if repo_root is None:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            raise ValueError(f"{label} must be project-relative, got {value!r}")
        return path
    return project_path(str(value), repo_root, label=label)

