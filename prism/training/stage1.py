from __future__ import annotations

from prism.training.checkpointing import load_training_checkpoint, save_training_checkpoint
from prism.training.distributed import get_and_clip_grad_norm, unwrap_training_model
from prism.training.loggers import setup_file_logging
from prism.training.optim import build_param_groups
from prism.training.scheduler import get_lr_lambda

# --- migrated from src/prism/training/stage1/common/batch_contract.py ---
from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_TRAJECTORY_STEP_KEYS = (
    "batch_indices",
    "loss_mask",
    "states",
    "actions",
    "action_mask",
    "fused_tokens",
)


def validate_stage1_window_batch(batch: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    steps = batch.get("trajectory_steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ValueError("Stage1 dataloader must return a non-empty trajectory_steps sequence")
    if int(batch.get("batch_size", 0)) <= 0:
        raise ValueError("Stage1 episode batch requires a positive batch_size")
    for index, step_batch in enumerate(steps):
        validate_stage1_step_batch(step_batch, index=index)
    return steps


def validate_stage1_step_batch(step_batch: Mapping[str, Any], *, index: int) -> None:
    missing = [key for key in REQUIRED_TRAJECTORY_STEP_KEYS if key not in step_batch]
    if missing:
        raise ValueError(f"Stage1 trajectory step {index} missing required keys: {', '.join(missing)}")

# --- migrated from src/prism/training/stage1/common/dataset.py ---
import logging
from functools import partial
import json
from pathlib import Path
from typing import Any

from prism.data import EPISODE_FEATURE_CACHE_FORMAT
from prism.data import EpisodeFeatureCacheTrajectoryDataset
from prism.data import collate_direct_bridge_token_cache_windows
from prism.utils.paths import display_project_path, project_path
from prism.utils.seeding import build_torch_generator, seed_data_worker


def prepare_stage1_dataset(
    config: dict[str, Any],
    *,
    repo_root: str | Path,
) -> EpisodeFeatureCacheTrajectoryDataset:
    manifest_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    manifest_format = _read_manifest_format(manifest_path)
    if manifest_format != EPISODE_FEATURE_CACHE_FORMAT:
        raise ValueError(
            f"Stage1 training requires {EPISODE_FEATURE_CACHE_FORMAT} manifest, got {manifest_format!r}. "
            "Build it with the benchmark-specific episode replay index and episode feature cache scripts."
        )
    dataset = EpisodeFeatureCacheTrajectoryDataset(
        manifest_path,
        action_horizon=int(config.get("horizon", 32)),
        max_episodes=config.get("max_samples_per_file"),
    )
    logging.info(
        "Loaded Stage1 episode feature cache: episodes=%s format=%s manifest=%s",
        len(dataset),
        manifest_format,
        display_project_path(manifest_path, repo_root),
    )
    return dataset


def prepare_stage1_dataloader(dataset: EpisodeFeatureCacheTrajectoryDataset, config: dict[str, Any]):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for Stage1 dataloading") from exc

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))
    shuffle = bool(config.get("shuffle_trajectory_windows", False))
    if len(dataset) == 0:
        raise ValueError("Stage1 dataset is empty")

    collate_fn = partial(
        collate_direct_bridge_token_cache_windows,
        memory_entry_tokens=int(config.get("memory_entry_tokens", 16)),
        action_horizon=int(config.get("horizon", 32)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(seed),
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Stage1 dataloader has no episode batches. Dataset size={len(dataset)}, "
            f"batch_size={batch_size}, drop_last=True."
        )
    logging.info(
        "Initialized Stage1 dataloader: episode_batch_size=%s num_workers=%s shuffle_episodes=%s",
        batch_size,
        num_workers,
        shuffle,
    )
    return dataloader


def _read_manifest_format(manifest_path: Path) -> str | None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return None if payload.get("format") is None else str(payload["format"])

# --- migrated from src/prism/training/stage1/common/loss.py ---
from typing import Any

from prism.training.loss import masked_flow_matching_mse


def stage1_flow_matching_loss(
    *,
    pred_velocity: Any,
    noise: Any,
    actions_gt: Any,
    action_mask: Any,
) -> Any:
    target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
    if pred_velocity.shape != target_velocity.shape:
        raise ValueError(
            f"pred_velocity shape {tuple(pred_velocity.shape)} != target_velocity shape {tuple(target_velocity.shape)}"
        )
    return masked_flow_matching_mse(pred_velocity, target_velocity, action_mask)

# --- migrated from src/prism/training/stage1/common/loop.py ---
import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from prism.config import resolve_experiment_config
from prism.utils.seeding import set_global_seed, write_experiment_snapshot
from prism.config import resolve_training_config_paths, validate_training_config
from prism.utils import cuda_memory_stats
from prism.utils import reserve_cuda_memory_floor


def train_stage1(config: dict[str, Any], *, repo_root: str | Path) -> None:
    runtime = _load_runtime()
    torch = runtime["torch"]
    Accelerator = runtime["Accelerator"]
    DistributedType = runtime["DistributedType"]
    AdamW = runtime["AdamW"]
    LambdaLR = runtime["LambdaLR"]
    tqdm = runtime["tqdm"]
    PrismPolicy = runtime["PrismPolicy"]

    repo_root = Path(repo_root)
    accelerator = Accelerator()
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    enforce_stage1_contract(config)
    validate_training_config(config, repo_root=repo_root)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)

    save_dir = str(config.get("save_dir", "local_data/runs/stage1/default"))
    setup_file_logging(save_dir, is_main_process=accelerator.is_main_process, filename_prefix="stage1_train_log")
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
        logging.info("Resolved Stage1 config written to %s", save_dir)
        logging.info("Seed=%s deterministic=%s", seed, deterministic)

    dataset = prepare_stage1_dataset(config, repo_root=repo_root)
    dataloader = prepare_stage1_dataloader(dataset, config)

    model = PrismPolicy(config)
    config = model.config
    enforce_stage1_contract(config)
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
    model.train()
    model.set_finetune_flags()

    lr = float(config.get("lr", 5e-5))
    weight_decay = float(config.get("weight_decay", 1e-3))
    param_groups = build_param_groups(
        model,
        weight_decay,
        base_lr=lr,
        lr_groups=config.get("lr_groups") or {},
    )
    optimizer = AdamW(param_groups, lr=lr, foreach=False)
    if accelerator.is_main_process:
        logging.info("Optimizer=AdamW base_lr=%s weight_decay=%s", lr, weight_decay)
        for index, group in enumerate(param_groups):
            params = sum(parameter.numel() for parameter in group["params"])
            logging.info(
                "Optimizer group %s | lr=%s | weight_decay=%s | params=%.3fM",
                group.get("name", f"group_{index}"),
                group.get("lr", lr),
                group.get("weight_decay", weight_decay),
                params / 1e6,
            )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model
    unwrapped_model = unwrap_training_model(accelerator, model)
    memory_floor = None

    max_steps = int(config.get("max_steps", 10000))
    warmup_steps = int(config.get("warmup_steps", 1000))
    scheduler = LambdaLR(
        optimizer,
        get_lr_lambda(
            warmup_steps,
            max_steps,
            resume_step=0,
            min_lr_ratio=float(config.get("min_lr_ratio", 0.1)),
        ),
    )

    os.makedirs(save_dir, exist_ok=True)
    norm_stats = getattr(dataset, "arm2stats_dict", None)
    step = 0
    best_loss = float("inf")

    if bool(config.get("resume", False)):
        resume_path = str(config.get("resume_path", "")).rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)
        step, client_state = load_training_checkpoint(
            torch,
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            optimizer=optimizer,
            load_optimizer_states=True,
            resume_pretrain=bool(config.get("resume_pretrain", False)),
        )
        best_loss = float(client_state.get("best_loss", float("inf")))
        scheduler = LambdaLR(
            optimizer,
            get_lr_lambda(
                warmup_steps,
                max_steps,
                resume_step=step,
                min_lr_ratio=float(config.get("min_lr_ratio", 0.1)),
            ),
        )
        if accelerator.is_main_process:
            logging.info("Resuming Stage1 from %s/%s at step %s", resume_dir, resume_tag, step)
    elif accelerator.is_main_process:
        logging.info("Starting fresh Stage1 training")

    if accelerator.is_main_process and config.get("min_cuda_memory_gb") is not None:
        memory_floor = reserve_cuda_memory_floor(
            torch,
            target_gb=float(config["min_cuda_memory_gb"]),
            device=accelerator.device,
        )
        stats = cuda_memory_stats(torch, accelerator.device)
        logging.info(
            "Stage1 CUDA memory floor active: target=%.2f GiB used=%.2f GiB floor_reserved=%.2f GiB",
            float(config["min_cuda_memory_gb"]),
            float(stats["used_gb"]),
            float(memory_floor.reserved_gb),
        )

    log_interval = int(config.get("log_interval", 10))
    ckpt_interval = int(config.get("ckpt_interval", 5000))
    best_ckpt_enabled = int(config.get("best_ckpt_interval", 1000)) != 0
    best_ckpt_min_step = int(config.get("best_ckpt_min_step", config.get("warmup_steps", 0)))
    max_norm = float(config.get("grad_clip_norm", 1.0))
    bridge_enabled = bool(unwrapped_model.use_bridge)
    last_loss = None

    while step < max_steps:
        for batch in tqdm(dataloader, desc="Stage1", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break
            validate_stage1_window_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            loss, extra_metrics, last_tensors = _run_trajectory_window_batch(
                torch=torch,
                model=model,
                unwrapped_model=unwrapped_model,
                batch=batch,
                config=config,
                accelerator=accelerator,
                bridge_enabled=bridge_enabled,
                step=step,
                backward_fn=accelerator.backward,
            )
            if not _check_numerical_stability(torch, step, loss=loss, **last_tensors):
                raise FloatingPointError(f"Non-finite tensor detected at Stage1 step {step}")
            last_tensors.clear()

            total_norm, clipped_norm = get_and_clip_grad_norm(torch, accelerator, model, loss, max_norm)
            if memory_floor is not None:
                memory_floor.trim_to_target()
            optimizer.step()
            if memory_floor is not None:
                stats = memory_floor.trim_to_target()
                if accelerator.is_main_process:
                    logging.debug(
                        "Stage1 CUDA memory floor trim: used=%.2f GiB floor_reserved=%.2f GiB",
                        float(stats["used_gb"]),
                        float(memory_floor.reserved_gb),
                    )
            scheduler.step()

            if step % log_interval == 0:
                _log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, extra_metrics)

            loss_value = float(loss.detach().cpu().item())
            checkpoint_loss = loss.detach()
            last_loss = checkpoint_loss
            is_best_tensor = torch.tensor(
                int(accelerator.is_main_process and step >= best_ckpt_min_step and loss_value < best_loss),
                device=accelerator.device,
            )
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)

            is_best = is_best_tensor.item() == 1
            if is_best:
                best_loss = loss_value

            should_save_best = is_best and best_ckpt_enabled
            if should_save_best:
                save_training_checkpoint(
                    torch,
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=checkpoint_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=norm_stats,
                    tag="step_best",
                    best_loss=best_loss,
                )
                if accelerator.is_main_process:
                    logging.info("Saved best Stage1 checkpoint at step %s loss %.6f", step, loss_value)

            step += 1
            if step % ckpt_interval == 0 and step > 0:
                save_training_checkpoint(
                    torch,
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss=checkpoint_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=norm_stats,
                    best_loss=best_loss if best_loss < float("inf") else None,
                )
            del loss, checkpoint_loss

    if last_loss is None:
        raise RuntimeError("Stage1 training loop did not run any steps")
    save_training_checkpoint(
        torch,
        save_dir,
        step=step,
        model_engine=model_engine,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=last_loss,
        accelerator=accelerator,
        config=config,
        norm_stats=norm_stats,
        tag="step_final",
        best_loss=best_loss if best_loss < float("inf") else None,
    )
    if accelerator.is_main_process:
        logging.info("Final Stage1 checkpoint saved to step_final/")


def _load_runtime() -> dict[str, Any]:
    try:
        import torch
        from accelerate import Accelerator
        from accelerate import DistributedType
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        from tqdm import tqdm

        from prism.models.policy import PrismPolicy
    except ModuleNotFoundError as exc:
        missing = exc.name or "a Stage1 training dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for Stage1 training. Run inside the prepared training environment."
        ) from exc

    return {
        "torch": torch,
        "Accelerator": Accelerator,
        "DistributedType": DistributedType,
        "AdamW": AdamW,
        "LambdaLR": LambdaLR,
        "tqdm": tqdm,
        "PrismPolicy": PrismPolicy,
    }


def _get_autocast_context(torch: Any, device: Any):
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _optional_batch_tensor(torch: Any, batch: dict[str, Any], key: str, device: Any, dtype: Any | None):
    value = batch.get(key)
    if value is None:
        return None
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _slice_progress_state(progress_state: Any, batch_indices: Any) -> Any:
    from prism.models.planner import ProgressState

    if isinstance(progress_state, ProgressState):
        return ProgressState(
            completed_events=progress_state.completed_events.index_select(0, batch_indices),
            current_stage=progress_state.current_stage.index_select(0, batch_indices),
        )
    return progress_state.index_select(0, batch_indices)


def _scatter_progress_state(progress_state: Any, batch_indices: Any, updated_state: Any) -> Any:
    from prism.models.planner import ProgressState

    if isinstance(progress_state, ProgressState):
        completed = progress_state.completed_events.clone()
        current = progress_state.current_stage.clone()
        completed.index_copy_(
            0,
            batch_indices,
            updated_state.completed_events.to(device=completed.device, dtype=completed.dtype),
        )
        current.index_copy_(
            0,
            batch_indices,
            updated_state.current_stage.to(device=current.device, dtype=current.dtype),
        )
        return ProgressState(completed_events=completed, current_stage=current)
    output = progress_state.clone()
    output.index_copy_(0, batch_indices, updated_state.to(device=output.device, dtype=output.dtype))
    return output


def _detach_progress_state(progress_state: Any) -> Any:
    from prism.models.planner import ProgressState

    if isinstance(progress_state, ProgressState):
        return ProgressState(
            completed_events=progress_state.completed_events.detach(),
            current_stage=progress_state.current_stage.detach(),
        )
    return progress_state.detach()


def _run_trajectory_window_batch(
    *,
    torch: Any,
    model: Any,
    unwrapped_model: Any,
    batch: dict[str, Any],
    config: dict[str, Any],
    accelerator: Any,
    bridge_enabled: bool,
    step: int,
    backward_fn: Callable[[Any], None] | None = None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    if unwrapped_model.progress_state_planner is None:
        raise ValueError("Stage1 trajectory training requires progress_state_planner")

    device = accelerator.device
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    batch_size = int(batch["batch_size"])
    progress_state = unwrapped_model.progress_state_planner.initial_state(batch_size, device=device, dtype=dtype)
    loss_terms = []
    action_loss_values = []
    loss_rows = 0
    last_tensors: dict[str, Any] = {}
    loss_step_count = sum(
        int(bool(step_batch["loss_mask"].bool().any().item()))
        for step_batch in batch["trajectory_steps"]
    )
    if loss_step_count <= 0:
        raise ValueError("Stage1 episode batch produced no full-horizon loss terms")

    for step_batch in batch["trajectory_steps"]:
        batch_indices = step_batch["batch_indices"].to(device=device)
        loss_mask = step_batch["loss_mask"].to(device=device).bool()
        active_progress_state = _slice_progress_state(progress_state, batch_indices)

        states = step_batch["states"].to(device=device, dtype=dtype)
        actions_gt = step_batch["actions"].to(device=device, dtype=dtype)
        action_mask = step_batch["action_mask"].to(device=device)
        fused_tokens = step_batch["fused_tokens"].to(device=device, dtype=dtype)
        raw_hidden_states = step_batch.get("vlm_hidden_states")
        hidden_states = None
        if raw_hidden_states is not None:
            hidden_states = [hidden_state.to(device=device, dtype=dtype) for hidden_state in raw_hidden_states]
        memory_context = _optional_batch_tensor(torch, step_batch, "memory_context", device, dtype)
        memory_context_mask = _optional_batch_tensor(torch, step_batch, "memory_context_mask", device, None)
        short_memory_time_ids = _optional_batch_tensor(torch, step_batch, "short_memory_time_ids", device, None)
        executed_actions = _optional_batch_tensor(torch, step_batch, "executed_actions", device, dtype)
        executed_action_mask = _optional_batch_tensor(torch, step_batch, "executed_action_mask", device, None)
        planner_vl_summary = _optional_batch_tensor(torch, step_batch, "planner_vl_summary", device, dtype)
        plan_token_mask = _optional_batch_tensor(torch, step_batch, "plan_token_mask", device, None)

        context = nullcontext() if bool(loss_mask.any().item()) else torch.no_grad()
        with context, _get_autocast_context(torch, device):
            pred_velocity, noise = model(
                fused_tokens,
                state=states,
                actions_gt=actions_gt,
                action_mask=action_mask,
                hidden_states=hidden_states if bridge_enabled else None,
                memory_context=memory_context,
                memory_context_mask=memory_context_mask,
                short_memory_time_ids=short_memory_time_ids,
                executed_actions=executed_actions,
                executed_action_mask=executed_action_mask,
                progress_state=active_progress_state,
                planner_vl_summary=planner_vl_summary,
                plan_token_mask=plan_token_mask,
            )

        planner_output = unwrapped_model.last_progress_planner_output
        if planner_output is None:
            raise RuntimeError("progress_state_planner did not produce an output during Stage1 trajectory training")
        progress_state = _scatter_progress_state(progress_state, batch_indices, planner_output.progress_state)
        progress_state = _detach_progress_state(progress_state)

        if bool(loss_mask.any().item()):
            if action_mask[loss_mask].sum() == 0:
                raise ValueError(f"[Step {step}] action_mask.sum() is 0 for a trajectory loss slice")
            action_loss = stage1_flow_matching_loss(
                pred_velocity=pred_velocity[loss_mask],
                noise=noise[loss_mask],
                actions_gt=actions_gt[loss_mask],
                action_mask=action_mask[loss_mask],
            )
            if backward_fn is not None:
                backward_fn(action_loss / float(loss_step_count))
                loss_terms.append(action_loss.detach())
            else:
                loss_terms.append(action_loss)
            action_loss_values.append(float(action_loss.detach().cpu().item()))
            loss_rows += int(loss_mask.sum().item())

        last_tensors = {
            "states": states.detach(),
            "actions_gt": actions_gt.detach(),
            "fused_tokens": fused_tokens.detach(),
            "pred_velocity": pred_velocity.detach(),
        }

    if not loss_terms:
        raise ValueError("Stage1 episode batch produced no full-horizon loss terms")
    loss = torch.stack(loss_terms).mean()
    extra_metrics = {
        "action_loss": float(loss.detach().cpu().item()),
        "trajectory_loss_steps": float(len(loss_terms)),
        "trajectory_loss_rows": float(loss_rows),
    }
    if action_loss_values:
        extra_metrics["action_loss_step_mean"] = float(sum(action_loss_values) / len(action_loss_values))
    return loss, extra_metrics, last_tensors


def _check_numerical_stability(torch: Any, step: int, **named_tensors: Any) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info("[Stage1 step %s] Non-finite detected in %s", step, name)
            return False
    return True


def _log_training_step(
    step: int,
    loss: Any,
    total_norm: Any,
    clipped_norm: Any,
    scheduler: Any,
    dataloader: Any,
    accelerator: Any,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    if not accelerator.is_main_process:
        return
    current_epoch = step / len(dataloader)
    logging.info("Estimated Stage1 epoch: %.2f", current_epoch)
    logging.info("[Stage1 step %s] loss=%.4f lr=%s", step, float(loss.item()), scheduler.get_last_lr()[0])
    logging.info("[Stage1 step %s] grad_norm=%s clipped_norm=%s", step, total_norm, clipped_norm)
    for name, value in sorted((extra_metrics or {}).items()):
        logging.info("[Stage1 step %s] %s=%.4f", step, name, value)

# --- migrated from src/prism/training/stage1/libero/validators.py ---
import json
from pathlib import Path
from typing import Any

from prism.utils.paths import project_path


EPISODE_FEATURE_CACHE_FORMAT = "libero_episode_feature_cache"
REQUIRED_HIDDEN_STATE_LAYERS = (3, 6, 9, 12)
DEFAULT_STAGE1_HIDDEN_DIM = 896


def enforce_stage1_contract(config: dict[str, Any]) -> None:
    """Reject non-active Stage1 routes before model or dataset construction."""

    if str(config.get("dataset_type")) != "memory_token_cache":
        raise ValueError("Stage1 requires dataset_type=memory_token_cache")
    if not bool(config.get("memory_token_cache_sequence_training", False)):
        raise ValueError("Stage1 requires episode-level fixed-replan-node token-cache training")
    if bool(config.get("load_vlm", True)):
        raise ValueError("Stage1 trains from token cache and requires load_vlm=false")
    if bool(config.get("finetune_vlm", False)):
        raise ValueError("Stage1 keeps the VLM frozen/offline and requires finetune_vlm=false")
    if not bool(config.get("finetune_action_head", False)):
        raise ValueError("Stage1 requires finetune_action_head=true")
    if bool(config.get("finetune_progress_planner", False)):
        raise ValueError("Stage1 uses a frozen ProgressPlanner and requires finetune_progress_planner=false")
    if bool(config.get("enable_bridge_aux_loss", False)):
        raise ValueError("Stage1 supports only masked flow-matching velocity loss; disable bridge aux loss")
    if not bool(config.get("progress_planner_enabled", False)):
        raise ValueError("Stage1 requires progress_planner.enabled=true")
    if not config.get("progress_planner_checkpoint"):
        raise ValueError("Stage1 requires a frozen progress_planner_checkpoint")
    if int(config.get("horizon", 0)) != 32:
        raise ValueError("Stage1 direct-bridge training is locked to horizon=32")
    if int(config.get("progress_planner_replan_stride", 0)) != 16:
        raise ValueError("Stage1 direct-bridge training is locked to replan stride=16")
    if int(config.get("num_inference_timesteps", 0)) != 15:
        raise ValueError("Stage1 rollout/smoke inference is locked to 15 Euler steps")
    if str(config.get("inference_tau_schedule", "")).lower() != "midpoint":
        raise ValueError("Stage1 requires midpoint inference tau schedule")
    if not bool(config.get("avoid_endpoint_tau", False)):
        raise ValueError("Stage1 requires avoid_endpoint_tau=true")


def validate_stage1_cache_contract(config: dict[str, Any], *, repo_root: str | Path) -> None:
    manifest_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_format = manifest.get("format")
    if manifest_format != EPISODE_FEATURE_CACHE_FORMAT:
        raise ValueError(
            f"Stage1 requires {EPISODE_FEATURE_CACHE_FORMAT} manifest, got {manifest_format!r}"
        )

    hidden_dim = int(manifest.get("hidden_dim", 0))
    expected_hidden_dim = int(config.get("embed_dim", DEFAULT_STAGE1_HIDDEN_DIM))
    if hidden_dim != expected_hidden_dim:
        raise ValueError(f"Stage1 cache hidden_dim {hidden_dim} != expected {expected_hidden_dim}")

    hidden_layers = tuple(int(layer) for layer in manifest.get("hidden_state_layers") or ())
    if hidden_layers != REQUIRED_HIDDEN_STATE_LAYERS:
        raise ValueError(
            f"Stage1 cache hidden_state_layers {hidden_layers!r} != required {REQUIRED_HIDDEN_STATE_LAYERS!r}"
        )
    if int(manifest.get("node_count", 0)) <= 0:
        raise ValueError("Stage1 episode feature cache must include at least one node")
    planner_summary = manifest.get("planner_vl_summary")
    if not isinstance(planner_summary, dict) or not bool(planner_summary.get("enabled", False)):
        raise ValueError("Stage1 cache must include planner_vl_summary generated from the VLM last valid token")

    stats = _manifest_robot_stats(manifest)
    action_max = stats.get("action", {}).get("max")
    state_max = stats.get("observation.state", {}).get("max")
    if action_max is not None and len(action_max) != int(config.get("per_action_dim", 0)):
        raise ValueError(
            f"Stage1 cache action dimension {len(action_max)} != per_action_dim {config.get('per_action_dim')}"
        )
    if state_max is not None and len(state_max) != int(config.get("state_dim", 0)):
        raise ValueError(f"Stage1 cache state dimension {len(state_max)} != state_dim {config.get('state_dim')}")


def _manifest_robot_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    normalization = manifest.get("normalization") or {}
    stats_by_robot = normalization.get("stats") or {}
    robot_key = normalization.get("robot_key")
    if robot_key and robot_key in stats_by_robot:
        return dict(stats_by_robot[robot_key])
    if len(stats_by_robot) == 1:
        return dict(next(iter(stats_by_robot.values())))
    return {}

# --- migrated from src/prism/training/stage1/libero/config.py ---
from pathlib import Path
from typing import Any

from prism.config import resolve_experiment_config
from prism.utils.paths import normalize_project_relative_path, project_path
from prism.config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)


def build_stage1_config(
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

    active_defaults = {
        "dataset_type": "memory_token_cache",
        "memory_token_cache_sequence_training": True,
        "load_vlm": False,
        "finetune_vlm": False,
        "finetune_action_head": True,
        "finetune_progress_planner": False,
        "enable_bridge_aux_loss": False,
        "horizon": 32,
        "progress_planner_replan_stride": 16,
        "num_inference_timesteps": 15,
        "inference_tau_schedule": "midpoint",
        "avoid_endpoint_tau": True,
    }
    explicit_config_keys = {
        key for key, value in file_config.items() if value is not None
    } | {
        key for key, value in cli_overrides.items() if value is not None
    }

    config = merge_training_config(
        default_training_config(repo_root),
        file_config={**active_defaults, **file_config},
        cli_overrides=cli_overrides,
    )
    config["_explicit_config_keys"] = sorted(explicit_config_keys)
    config["repo_root"] = "."
    config = resolve_training_config_paths(config, repo_root)
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    enforce_stage1_contract(config)
    validate_training_config(
        config,
        repo_root=repo_root,
        validate_external_paths=validate_external_artifacts,
    )
    manifest_path = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    if validate_external_artifacts or manifest_path.exists():
        validate_stage1_cache_contract(config, repo_root=repo_root)
    return config

# --- migrated from src/prism/training/stage1/libero/cli.py ---
import argparse
import logging
import os
import sys

from prism.utils.paths import find_repo_root


REPO_ROOT = find_repo_root(__file__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage1 direct-bridge policy from trajectory token cache")
    parser.add_argument("--config", type=str, default=None, help="Project-relative Stage1 YAML config.")

    parser.add_argument("--run_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_wandb", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--disable_swanlab", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--bridge_prism_config", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--dataset_config_base_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--cache_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--save_dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--progress_planner_checkpoint", type=str, default=argparse.SUPPRESS)

    parser.add_argument("--horizon", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--per_action_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--state_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--memory_entry_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--short_memory_time_bins", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_vlm_tokens", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--burnin_replan_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--loss_replan_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--allow_short_burnin", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--trajectory_window_stride", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--shuffle_trajectory_windows",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )

    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min_lr_ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grad_clip_norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--min_cuda_memory_gb", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--log_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_interval", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--best_ckpt_min_step", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--resume_path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--resume_pretrain", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    parser.add_argument("--num_inference_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--inference_tau_schedule", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--avoid_endpoint_tau", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.chdir(REPO_ROOT)
    args = build_arg_parser().parse_args(argv)
    config = build_stage1_config(args, repo_root=REPO_ROOT, validate_external_artifacts=True)
    from prism.training.trainer import train_stage1

    try:
        train_stage1(config, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Cleaning up Stage1 training...")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# --- migrated from src/prism/training/stage1/calvin/cli.py ---
import sys


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

