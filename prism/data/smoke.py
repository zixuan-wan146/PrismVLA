from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from prism.data.token_cache_core import EPISODE_FEATURE_CACHE_FORMAT, EPISODE_FEATURE_CACHE_VERSION
from prism.data.token_cache_io import _require_torch
from prism.models.planner import ProgressStateConfig, ProgressStatePlanner
from prism.utils.paths import find_repo_root


def build_stage1_smoke_cache(config: Mapping[str, Any]) -> Path:
    """Create a tiny deterministic episode-feature cache for command smoke tests."""

    torch = _require_torch()
    repo_root = find_repo_root(__file__)
    manifest_path = _repo_relative_output_path(
        config.get("dataset_config_path", "local_data/smoke/stage1_cache/manifest.json"),
        repo_root=repo_root,
        label="dataset_config_path",
    )
    cache_root = manifest_path.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    progress_checkpoint = _repo_relative_output_path(
        config.get("progress_planner_checkpoint", "local_data/smoke/progress_planner_smoke.pt"),
        repo_root=repo_root,
        label="progress_planner_checkpoint",
    )
    progress_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    hidden_dim = int(config.get("embed_dim", config.get("hidden_dim", 32)))
    state_dim = int(config.get("state_dim", 8))
    action_dim = int(config.get("per_action_dim", 7))
    horizon = int(config.get("horizon", 32))
    stride = int(config.get("progress_planner_replan_stride", 16))
    planner_config = ProgressStateConfig(
        hidden_dim=hidden_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        replan_stride=stride,
        latent_dim=int(config.get("progress_planner_latent_dim", 16)),
        action_summary_hidden_dim=int(config.get("progress_planner_action_summary_hidden_dim", 32)),
        state_hidden_dim=int(config.get("progress_planner_state_hidden_dim", 32)),
        updater_hidden_dim=int(config.get("progress_planner_updater_hidden_dim", 64)),
        planner_ffn_dim=int(config.get("progress_planner_planner_ffn_dim", 64)),
        planner_layers=int(config.get("progress_planner_planner_layers", 1)),
        num_heads=int(config.get("progress_planner_num_heads", config.get("num_heads", 4))),
        dropout=float(config.get("progress_planner_dropout", 0.0)),
    )
    planner = ProgressStatePlanner(planner_config)
    torch.save(
        {
            "format": "progress_state_planner_warmup",
            "version": 1,
            "model_state_dict": planner.state_dict(),
            "model_config": asdict(planner_config),
        },
        progress_checkpoint,
    )

    torch.manual_seed(int(config.get("seed", 42)))
    required_hidden_state_layers = (3, 6, 9, 12)
    actions = torch.stack(
        [torch.linspace(-0.2, 0.2, action_dim, dtype=torch.float32) + step * 0.001 for step in range(horizon)]
    )
    visual_tokens = {
        "base": torch.randn(2, hidden_dim, dtype=torch.float32) * 0.01,
        "wrist": torch.randn(2, hidden_dim, dtype=torch.float32) * 0.01,
    }
    hidden_states = tuple(torch.randn(4, hidden_dim, dtype=torch.float32) * 0.01 for _ in required_hidden_state_layers)
    episode = {
        "episode_id": "smoke_episode_0",
        "prompt": "smoke task",
        "actions": actions,
        "visual_tokens_by_step": {0: visual_tokens},
        "state_by_step": {0: torch.linspace(-0.1, 0.1, state_dim, dtype=torch.float32)},
        "current_features_by_step": {
            0: {
                "hidden_states": hidden_states,
                "planner_vl_summary": torch.randn(hidden_dim, dtype=torch.float32) * 0.01,
            }
        },
        "nodes": [
            {
                "current_step": 0,
                "short_visual_steps": [None, None],
                "short_mask": [False, False],
                "executed_action_range": [0, 0],
                "executed_action_valid_count": 0,
                "future_action_range": [0, horizon],
                "action_valid_count": horizon,
            }
        ],
    }
    shard_name = "episodes_000000.pt"
    torch.save(
        {
            "format": EPISODE_FEATURE_CACHE_FORMAT,
            "version": EPISODE_FEATURE_CACHE_VERSION,
            "episodes": [episode],
        },
        cache_root / shard_name,
    )
    manifest = {
        "format": EPISODE_FEATURE_CACHE_FORMAT,
        "version": EPISODE_FEATURE_CACHE_VERSION,
        "benchmark": str(config.get("benchmark", "libero")).upper(),
        "data_root": "local_data/smoke/raw",
        "index_path": "local_data/smoke/index.json",
        "output_root": str(manifest_path.parent.relative_to(repo_root)),
        "encoder": "smoke",
        "hidden_state_encoder": "smoke",
        "hidden_state_layers": list(required_hidden_state_layers),
        "planner_vl_summary": {
            "enabled": True,
            "source": "smoke",
            "encoder": "smoke",
        },
        "hidden_dim": hidden_dim,
        "tokens_per_view": 2,
        "storage_dtype": "float32",
        "episode_count": 1,
        "node_count": 1,
        "source_executed_action_stride": stride,
        "view_names": ["base", "wrist"],
        "shards": [
            {
                "path": shard_name,
                "episode_count": 1,
                "start_index": 0,
                "end_index": 1,
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"stage1 smoke cache written: {manifest_path.relative_to(repo_root)}")
    print(f"stage1 smoke planner checkpoint written: {progress_checkpoint.relative_to(repo_root)}")
    return manifest_path


def _repo_relative_output_path(value: Any, *, repo_root: Path, label: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{label} must be project-relative, got {value!r}")
    return repo_root / path
