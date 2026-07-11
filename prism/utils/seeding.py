from __future__ import annotations

# --- migrated from src/prism/reproducibility.py ---
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from prism.utils.paths import display_project_path, sanitize_project_paths


PACKAGE_VERSION_NAMES = (
    "accelerate",
    "diffusers",
    "einops",
    "fvcore",
    "numpy",
    "opencv-python",
    "pandas",
    "Pillow",
    "pyarrow",
    "PyYAML",
    "swanlab",
    "timm",
    "torch",
    "torchvision",
    "transformers",
    "wandb",
    "websockets",
)

SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_ENDPOINT",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "WANDB_MODE",
)


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass


def build_torch_generator(seed: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_data_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ModuleNotFoundError:
        pass


def write_experiment_snapshot(save_dir: str | Path, config: Mapping[str, Any]) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root(config) or Path.cwd()
    _write_json(save_path / "resolved_config.json", sanitize_project_paths(config, repo_root))
    _write_json(save_path / "environment.json", build_environment_metadata(repo_root))
    _write_json(save_path / "reproducibility.json", build_reproducibility_metadata(config))


def build_reproducibility_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = _repo_root(config) or Path.cwd()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sanitize_project_paths(sys.argv, repo_root),
        "cwd": display_project_path(os.getcwd(), repo_root),
        "repo_root": ".",
        "python": {
            "executable": display_project_path(sys.executable, repo_root),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "seed": config.get("seed"),
        "deterministic": bool(config.get("deterministic", False)),
        "git": _git_metadata(repo_root),
        "environment": build_environment_metadata(repo_root),
        "bridge_prism_config_path": sanitize_project_paths(config.get("bridge_prism_config_path"), repo_root),
        "training_config_path": sanitize_project_paths(config.get("training_config_path"), repo_root),
        "experiment_name": _experiment_name(config),
    }


def build_environment_metadata(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd() if repo_root in (None, "") else Path(repo_root).expanduser().resolve()
    return {
        "python": {
            "executable": display_project_path(sys.executable, root),
            "version": sys.version,
        },
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGE_VERSION_NAMES},
        "torch": _torch_environment(),
        "env": _safe_environment(root),
    }


def _experiment_name(config: Mapping[str, Any]) -> str | None:
    bridge_config = config.get("bridge_prism")
    if isinstance(bridge_config, Mapping):
        experiment_name = bridge_config.get("experiment_name")
        if experiment_name is not None:
            return str(experiment_name)
    return None


def _repo_root(config: Mapping[str, Any]) -> Path | None:
    value = config.get("repo_root")
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser().resolve()


def _git_metadata(repo_root: Path | None = None) -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"], cwd=repo_root),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
        "dirty": _git_dirty(repo_root),
    }


def _git_dirty(repo_root: Path | None = None) -> bool | None:
    status = _run_git(["status", "--porcelain"], cwd=repo_root)
    if status is None:
        return None
    return bool(status.strip())


def _run_git(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_environment(repo_root: Path) -> dict[str, str]:
    payload = {}
    for key in SAFE_ENV_KEYS:
        value = os.environ.get(key)
        if value not in (None, ""):
            payload[key] = str(sanitize_project_paths(value, repo_root))
    return payload


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {"available": False}

    cuda_available = bool(torch.cuda.is_available())
    payload: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        payload["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        payload["devices"] = []
    return payload

