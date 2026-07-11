from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from prism.data.cache_utils import *  # noqa: F403
from prism.data.memory_replay import *  # noqa: F403
from prism.data.replay_frames import *  # noqa: F403
from prism.data.replay_dataset import *  # noqa: F403
from prism.data.token_cache_build import *  # noqa: F403
from prism.data.token_cache_core import *  # noqa: F403
from prism.data.token_cache_dataset import *  # noqa: F403
from prism.data.token_cache_io import *  # noqa: F403

def build_cache_from_config(cfg):
    """Config-dispatched cache entry point used by scripts/build_cache.py."""

    raw = getattr(cfg, "raw", {})
    benchmark = getattr(getattr(cfg, "data", None), "benchmark", raw.get("benchmark", "libero"))
    cache_mode = str(raw.get("cache_mode", "")).lower()
    if cache_mode == "stage1_smoke":
        from prism.data.smoke import build_stage1_smoke_cache

        return build_stage1_smoke_cache(raw)
    raise NotImplementedError(
        f"Cache building for {benchmark!r} requires selecting a concrete cache mode in the experiment config"
    )


def build_stage1_smoke_cache(config: Mapping[str, Any]) -> Path:
    from prism.data.smoke import build_stage1_smoke_cache as _build_stage1_smoke_cache

    return _build_stage1_smoke_cache(config)


def _repo_relative_output_path(value: Any, *, repo_root: Path, label: str) -> Path:
    from prism.data.smoke import _repo_relative_output_path as _resolve_repo_relative_output_path

    return _resolve_repo_relative_output_path(value, repo_root=repo_root, label=label)
