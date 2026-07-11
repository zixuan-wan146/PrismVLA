from pathlib import Path
from prism.utils.paths import find_repo_root

import pytest

torch = pytest.importorskip("torch")

from prism.data.libero import LIBERO_PROGRESS_WARMUP_FORMAT
from prism.data.libero import LIBERO_PROGRESS_WARMUP_VERSION
from prism.data.libero import build_libero_progress_windows
from prism.training import ProgressWarmupTrainingConfig
from prism.training import run_progress_warmup_training


def test_progress_warmup_training_writes_checkpoints_and_history(tmp_path: Path):
    cache_root = _write_tiny_progress_cache(tmp_path / "cache")
    output_dir = tmp_path / "run"

    result = run_progress_warmup_training(
        ProgressWarmupTrainingConfig(
            cache_manifest=str(cache_root),
            output_dir=str(output_dir),
            device="cpu",
            batch_size=2,
            max_steps=2,
            samples_per_epoch=4,
            hidden_dim=16,
            state_dim=5,
            action_dim=3,
            replan_stride=4,
            latent_dim=6,
            action_summary_hidden_dim=8,
            state_hidden_dim=8,
            updater_hidden_dim=32,
            planner_ffn_dim=32,
            planner_layers=1,
            num_heads=4,
            dropout=0.0,
            log_interval=0,
            repo_root=str(find_repo_root(__file__)),
        )
    )

    assert result.steps == 2
    assert result.checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert (output_dir / "train_history.json").exists()
    assert (output_dir / "resolved_config.json").exists()
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["format"] == "progress_state_planner_warmup"
    assert checkpoint["model_config"]["hidden_dim"] == 16


def test_progress_warmup_training_logs_episode_validation(tmp_path: Path):
    cache_root = _write_tiny_progress_cache(tmp_path / "cache")
    output_dir = tmp_path / "run_with_val"

    run_progress_warmup_training(
        ProgressWarmupTrainingConfig(
            cache_manifest=str(cache_root),
            output_dir=str(output_dir),
            device="cpu",
            batch_size=2,
            max_steps=1,
            samples_per_epoch=4,
            hidden_dim=16,
            state_dim=5,
            action_dim=3,
            replan_stride=4,
            latent_dim=6,
            action_summary_hidden_dim=8,
            state_hidden_dim=8,
            updater_hidden_dim=32,
            planner_ffn_dim=32,
            planner_layers=1,
            num_heads=4,
            dropout=0.0,
            val_fraction=0.5,
            eval_interval=1,
            log_interval=0,
            repo_root=str(find_repo_root(__file__)),
        )
    )

    history = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)["metrics"]
    assert history["train_window_count"] > 0
    assert history["val_window_count"] > 0
    assert "val_loss" in history
    assert "val_plan_loss" in history


def _write_tiny_progress_cache(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(123)
    steps = []
    for episode_index in range(2):
        episode_id = f"libero_spatial:tiny_task:demo_{episode_index}"
        for replan_index in range(6):
            step_index = len(steps)
            steps.append(
                {
                    "step_index": step_index,
                    "sample_index": step_index,
                    "episode_id": episode_id,
                    "suite": "libero_spatial",
                    "task_name": "tiny_task",
                    "prompt": "do the tiny task",
                    "current_step": replan_index * 4,
                    "replan_index": replan_index,
                    "state": torch.randn(5, generator=generator),
                    "executed_actions": torch.randn(4, 3, generator=generator),
                    "executed_action_mask": torch.ones(4, dtype=torch.bool) if replan_index > 0 else torch.zeros(4, dtype=torch.bool),
                    "target_intent": torch.nn.functional.normalize(torch.randn(6, generator=generator), dim=-1),
                    "vl_summary": torch.randn(16, generator=generator),
                }
            )
    windows = build_libero_progress_windows(
        steps,
        burnin_replan_steps=2,
        loss_replan_steps=2,
        allow_short_burnin=True,
    )
    torch.save(
        {
            "format": LIBERO_PROGRESS_WARMUP_FORMAT,
            "version": LIBERO_PROGRESS_WARMUP_VERSION,
            "steps": steps,
            "windows": windows,
        },
        cache_root / "data.pt",
    )
    (cache_root / "manifest.json").write_text(
        (
            "{\n"
            f'  "format": "{LIBERO_PROGRESS_WARMUP_FORMAT}",\n'
            f'  "version": {LIBERO_PROGRESS_WARMUP_VERSION},\n'
            '  "data_path": "data.pt",\n'
            '  "hidden_dim": 16,\n'
            '  "replan_stride": 4,\n'
            '  "step_count": 12,\n'
            f'  "window_count": {len(windows)}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    return cache_root
