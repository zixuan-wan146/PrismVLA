from __future__ import annotations

from prism.training.checkpointing import (
    _client_state_step as _client_state_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from prism.training.distributed import get_and_clip_grad_norm, unwrap_training_model
from prism.training.entrypoint import Trainer as Trainer
from prism.training.loggers import setup_file_logging
from prism.training.optim import build_param_groups
from prism.training.scheduler import get_lr_lambda
from prism.training.stage1 import (
    DEFAULT_STAGE1_HIDDEN_DIM,
    EPISODE_FEATURE_CACHE_FORMAT,
    REQUIRED_HIDDEN_STATE_LAYERS,
    REQUIRED_TRAJECTORY_STEP_KEYS,
    _check_numerical_stability as _check_numerical_stability,
    _detach_progress_state as _detach_progress_state,
    _get_autocast_context as _get_autocast_context,
    _log_training_step as _log_training_step,
    _run_trajectory_window_batch as _run_trajectory_window_batch,
    _scatter_progress_state as _scatter_progress_state,
    _slice_progress_state as _slice_progress_state,
    build_stage1_config,
    enforce_stage1_contract,
    prepare_stage1_dataloader,
    prepare_stage1_dataset,
    stage1_flow_matching_loss,
    train_stage1,
    validate_stage1_cache_contract,
    validate_stage1_step_batch,
    validate_stage1_window_batch,
)
from prism.training.stage2_config import (
    STAGE2_ACTIVE_DEFAULTS,
    STAGE2_DATASET_TYPE,
    STAGE2_REPLAY_INDEX_FORMAT,
    build_arg_parser,
    build_stage2_config,
    enforce_stage2_contract,
    main,
    validate_stage2_replay_index_contract,
)
from prism.training.stage2_data import (
    LiberoRawEpisodeSequenceDataset,
    RawEpisodeSequenceDataset,
    collate_libero_raw_episode_sequences,
    collate_raw_episode_sequences,
    load_stage2_normalization,
    prepare_stage2_dataloader,
    prepare_stage2_dataset,
)
from prism.training.stage2_loop import load_stage2_training_checkpoint, train_stage2

__all__ = [
    "DEFAULT_STAGE1_HIDDEN_DIM",
    "EPISODE_FEATURE_CACHE_FORMAT",
    "LiberoRawEpisodeSequenceDataset",
    "RawEpisodeSequenceDataset",
    "REQUIRED_HIDDEN_STATE_LAYERS",
    "REQUIRED_TRAJECTORY_STEP_KEYS",
    "STAGE2_ACTIVE_DEFAULTS",
    "STAGE2_DATASET_TYPE",
    "STAGE2_REPLAY_INDEX_FORMAT",
    "Trainer",
    "build_arg_parser",
    "build_param_groups",
    "build_stage1_config",
    "build_stage2_config",
    "collate_libero_raw_episode_sequences",
    "collate_raw_episode_sequences",
    "enforce_stage1_contract",
    "enforce_stage2_contract",
    "get_and_clip_grad_norm",
    "get_lr_lambda",
    "load_stage2_normalization",
    "load_stage2_training_checkpoint",
    "load_training_checkpoint",
    "main",
    "prepare_stage1_dataloader",
    "prepare_stage1_dataset",
    "prepare_stage2_dataloader",
    "prepare_stage2_dataset",
    "save_training_checkpoint",
    "setup_file_logging",
    "stage1_flow_matching_loss",
    "train_stage1",
    "train_stage2",
    "unwrap_training_model",
    "validate_stage1_cache_contract",
    "validate_stage1_step_batch",
    "validate_stage1_window_batch",
    "validate_stage2_replay_index_contract",
]
