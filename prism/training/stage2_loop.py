from __future__ import annotations

from prism.training.checkpointing import _client_state_step, save_training_checkpoint
from prism.training.distributed import get_and_clip_grad_norm, unwrap_training_model
from prism.training.loggers import setup_file_logging
from prism.training.optim import build_param_groups
from prism.training.scheduler import get_lr_lambda
from prism.training.stage1 import (
    _check_numerical_stability,
    _detach_progress_state,
    _get_autocast_context,
    _scatter_progress_state,
    _slice_progress_state,
    stage1_flow_matching_loss,
)
from prism.training.stage2_data import prepare_stage2_dataloader, prepare_stage2_dataset

# --- migrated from src/prism/training/stage2/common/loop.py ---
import logging
import os
from pathlib import Path
from typing import Any, Callable

from prism.config import resolve_experiment_config
from prism.utils.seeding import set_global_seed, write_experiment_snapshot
from prism.config import resolve_training_config_paths, validate_training_config


def train_stage2(config: dict[str, Any], *, repo_root: str | Path) -> None:
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
    _enforce_stage2_contract(config)
    validate_training_config(config, repo_root=repo_root)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    set_global_seed(seed, deterministic=deterministic)

    save_dir = str(config.get("save_dir", "local_data/runs/stage2/default"))
    setup_file_logging(save_dir, is_main_process=accelerator.is_main_process, filename_prefix="stage2_train_log")
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
        logging.info("Resolved Stage2 config written to %s", save_dir)
        logging.info("Seed=%s deterministic=%s", seed, deterministic)

    dataset = prepare_stage2_dataset(config, repo_root=repo_root)
    dataloader = prepare_stage2_dataloader(dataset, config)

    model = PrismPolicy(config)
    config = model.config
    _enforce_stage2_contract(config)
    if accelerator.is_main_process:
        write_experiment_snapshot(save_dir, config)
    model.train()
    model.set_finetune_flags()

    lr = float(config.get("lr", 1e-6))
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

    max_steps = int(config.get("max_steps", 50000))
    warmup_steps = int(config.get("warmup_steps", 2000))
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
        step, client_state = load_stage2_training_checkpoint(
            torch,
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            optimizer=optimizer,
            load_optimizer_states=True,
            resume_pretrain=bool(config.get("resume_pretrain", False)),
        )
        best_loss = _resume_best_loss(config, client_state)
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
            logging.info("Resuming Stage2 from %s/%s at step %s", resume_dir, resume_tag, step)
            if best_loss == float("inf"):
                logging.info("Resetting Stage2 best checkpoint selection after resume")
    elif accelerator.is_main_process:
        logging.info("Starting fresh Stage2 training")

    log_interval = int(config.get("log_interval", 10))
    ckpt_interval = int(config.get("ckpt_interval", 5000))
    best_ckpt_enabled = int(config.get("best_ckpt_interval", 1000)) != 0
    best_ckpt_min_step = int(config.get("best_ckpt_min_step", config.get("warmup_steps", 0)))
    max_norm = float(config.get("grad_clip_norm", 1.0))
    last_loss = None

    while step < max_steps:
        for batch in tqdm(dataloader, desc="Stage2", disable=not accelerator.is_main_process):
            if step >= max_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            loss, extra_metrics, last_tensors = _run_stage2_episode_group_batch(
                torch=torch,
                model=model,
                unwrapped_model=unwrapped_model,
                batch=batch,
                accelerator=accelerator,
                backward_fn=accelerator.backward,
            )
            if not _check_numerical_stability(torch, step, loss=loss, **last_tensors):
                raise FloatingPointError(f"Non-finite tensor detected at Stage2 step {step}")
            last_tensors.clear()

            total_norm, clipped_norm = get_and_clip_grad_norm(torch, accelerator, model, loss, max_norm)
            optimizer.step()
            scheduler.step()

            if step % log_interval == 0:
                _log_stage2_step(step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, extra_metrics)

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

            if is_best and best_ckpt_enabled:
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
                    logging.info("Saved best Stage2 checkpoint at step %s loss %.6f", step, loss_value)

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
        raise RuntimeError("Stage2 training loop did not run any steps")
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
        logging.info("Final Stage2 checkpoint saved to step_final/")


def load_stage2_training_checkpoint(
    torch: Any,
    model_engine: Any,
    *,
    load_dir: str,
    accelerator: Any,
    tag: str,
    optimizer: Any,
    load_optimizer_states: bool,
    resume_pretrain: bool,
    allowed_missing_prefixes: tuple[str, ...] = ("embedder.",),
) -> tuple[int, dict[str, Any]]:
    checkpoint_path = os.path.join(load_dir, tag, "model.pt")
    payload = torch.load(checkpoint_path, map_location=accelerator.device, weights_only=False)
    if payload.get("format") != "stage1_torch_checkpoint":
        raise ValueError(f"unsupported torch checkpoint format in {checkpoint_path}")
    unwrapped = unwrap_training_model(accelerator, model_engine)
    missing_keys, unexpected_keys = unwrapped.load_state_dict(payload["model_state_dict"], strict=False)
    bad_missing = [key for key in missing_keys if not key.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected_keys:
        details = []
        if bad_missing:
            details.append(f"missing non-VLM keys={bad_missing[:20]}")
        if unexpected_keys:
            details.append(f"unexpected keys={list(unexpected_keys)[:20]}")
        raise RuntimeError("Stage2 checkpoint load is incompatible: " + "; ".join(details))
    if load_optimizer_states and not resume_pretrain and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    client_state = dict(payload.get("client_state") or {})
    if accelerator.is_main_process:
        logging.info(
            "Loaded Stage2 checkpoint from %s with %s allowed missing VLM keys",
            checkpoint_path,
            len(missing_keys),
        )
    return _client_state_step(client_state), client_state


def _resume_best_loss(config: dict[str, Any], client_state: dict[str, Any]) -> float:
    if bool(config.get("reset_best_loss_on_resume", True)):
        return float("inf")
    return float(client_state.get("best_loss", float("inf")))


def _run_stage2_episode_group_batch(
    *,
    torch: Any,
    model: Any,
    unwrapped_model: Any,
    batch: dict[str, Any],
    accelerator: Any,
    backward_fn: Callable[[Any], None] | None = None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    if unwrapped_model.progress_state_planner is None:
        raise ValueError("Stage2 full E2E training requires progress_state_planner")

    device = accelerator.device
    dtype = torch.bfloat16
    batch_size = int(batch["batch_size"])
    progress_state = unwrapped_model.progress_state_planner.initial_state(batch_size, device=device, dtype=dtype)
    loss_terms = []
    action_loss_values = []
    last_tensors: dict[str, Any] = {}
    loss_step_count = sum(int(step_batch["loss_mask"].sum().item()) for step_batch in batch["trajectory_steps"])
    if loss_step_count <= 0:
        raise ValueError("Stage2 episode group produced no loss terms")

    for step_batch in batch["trajectory_steps"]:
        batch_indices = step_batch["batch_indices"].to(device=device)
        loss_mask = step_batch["loss_mask"].to(device=device).bool()
        if not bool(loss_mask.any().item()):
            continue

        active_rows = loss_mask.nonzero(as_tuple=False).flatten()
        if active_rows.numel() <= 0:
            continue
        active_batch_indices = batch_indices.index_select(0, active_rows)
        active_progress_state = _slice_progress_state(progress_state, active_batch_indices)

        states = _select_stage2_rows(step_batch["states"], active_rows).to(device=device, dtype=dtype)
        actions_gt = _select_stage2_rows(step_batch["actions"], active_rows).to(device=device, dtype=dtype)
        action_mask = _select_stage2_rows(step_batch["action_mask"], active_rows).to(device=device)
        executed_actions = _select_stage2_rows(step_batch["executed_actions"], active_rows).to(device=device, dtype=dtype)
        executed_action_mask = _select_stage2_rows(step_batch["executed_action_mask"], active_rows).to(device=device)

        embedding_rows = []
        for row_index in active_rows.detach().cpu().tolist():
            images = step_batch["images"][row_index]
            image_mask = step_batch["image_mask"][row_index].to(device=device)
            prompt = step_batch["prompts"][row_index]
            short_images = step_batch["short_images"][row_index]
            short_image_masks = step_batch["short_image_masks"][row_index]
            with _get_autocast_context(torch, device):
                embedding_output = unwrapped_model.get_vl_embeddings(
                    images=images,
                    image_mask=image_mask,
                    prompt=prompt,
                    return_cls_only=False,
                    return_hidden_states=True,
                )
                fused_tokens = (
                    embedding_output.visual_tokens
                    if embedding_output.visual_tokens is not None
                    else embedding_output.fused_tokens
                )
                fused_tokens = fused_tokens.to(device=device, dtype=dtype)
                hidden_states = [
                    hidden_state.to(device=device, dtype=dtype)
                    for hidden_state in embedding_output.hidden_states
                ]
                planner_vl_summary = None
                if embedding_output.planner_vl_summary is not None:
                    planner_vl_summary = embedding_output.planner_vl_summary.to(device=device, dtype=dtype)
                memory_context, memory_context_mask, short_memory_time_ids = _build_short_memory_for_sample(
                    torch,
                    unwrapped_model,
                    short_images=short_images,
                    short_image_masks=short_image_masks,
                    device=device,
                    dtype=dtype,
                )
            embedding_rows.append(
                {
                    "fused_tokens": fused_tokens,
                    "hidden_states": hidden_states,
                    "planner_vl_summary": planner_vl_summary,
                    "memory_context": memory_context,
                    "memory_context_mask": memory_context_mask,
                    "short_memory_time_ids": short_memory_time_ids,
                }
            )

        fused_tokens = _cat_stage2_batch_tensors(
            [row["fused_tokens"] for row in embedding_rows],
            name="fused_tokens",
        )
        hidden_states = _cat_stage2_hidden_states([row["hidden_states"] for row in embedding_rows])
        planner_vl_summary = _cat_optional_stage2_batch_tensors(
            [row["planner_vl_summary"] for row in embedding_rows],
            name="planner_vl_summary",
        )
        memory_context = _cat_optional_stage2_batch_tensors(
            [row["memory_context"] for row in embedding_rows],
            name="memory_context",
        )
        memory_context_mask = _cat_optional_stage2_batch_tensors(
            [row["memory_context_mask"] for row in embedding_rows],
            name="memory_context_mask",
        )
        short_memory_time_ids = _cat_optional_stage2_batch_tensors(
            [row["short_memory_time_ids"] for row in embedding_rows],
            name="short_memory_time_ids",
        )

        with _get_autocast_context(torch, device):
            pred_velocity, noise = model(
                fused_tokens,
                state=states,
                actions_gt=actions_gt,
                action_mask=action_mask,
                hidden_states=hidden_states,
                memory_context=memory_context,
                memory_context_mask=memory_context_mask,
                short_memory_time_ids=short_memory_time_ids,
                executed_actions=executed_actions,
                executed_action_mask=executed_action_mask,
                planner_vl_summary=planner_vl_summary,
                progress_state=active_progress_state,
            )

        planner_output = unwrapped_model.last_progress_planner_output
        if planner_output is None:
            raise RuntimeError("progress_state_planner did not produce an output during Stage2 training")
        progress_state = _scatter_progress_state(progress_state, active_batch_indices, planner_output.progress_state)
        progress_state = _detach_progress_state(progress_state)

        if action_mask.sum() == 0:
            raise ValueError("Stage2 action_mask.sum() is 0 for a sampled timestep")
        action_loss = stage1_flow_matching_loss(
            pred_velocity=pred_velocity,
            noise=noise,
            actions_gt=actions_gt,
            action_mask=action_mask,
        )
        active_count = int(active_rows.numel())
        loss_weight = float(active_count) / float(loss_step_count)
        if backward_fn is not None:
            backward_fn(action_loss * loss_weight)
            loss_terms.append(action_loss.detach())
        else:
            loss_terms.append(action_loss * loss_weight)
        action_loss_values.append(float(action_loss.detach().cpu().item()))
        last_tensors = {
            "states": states.detach(),
            "actions_gt": actions_gt.detach(),
            "fused_tokens": fused_tokens.detach(),
            "pred_velocity": pred_velocity.detach(),
        }

    if not loss_terms:
        raise ValueError("Stage2 episode group produced no full-horizon loss terms")
    if backward_fn is not None:
        loss = torch.stack(loss_terms).mean()
    else:
        loss = torch.stack(loss_terms).sum()
    active_samples = sum(int(step_batch["loss_mask"].sum().item()) for step_batch in batch["trajectory_steps"])
    max_batch_rows = max(int(step_batch["loss_mask"].sum().item()) for step_batch in batch["trajectory_steps"])
    extra_metrics = {
        "action_loss": float(loss.detach().cpu().item()),
        "stage2_active_samples": float(active_samples),
        "stage2_batch_rows_max": float(max_batch_rows),
        "stage2_sequence_len": float(len(batch["trajectory_steps"])),
        "stage2_loss_terms": float(len(loss_terms)),
    }
    if action_loss_values:
        extra_metrics["action_loss_step_mean"] = float(sum(action_loss_values) / len(action_loss_values))
    return loss, extra_metrics, last_tensors


def _cat_stage2_hidden_states(hidden_states_by_sample: list[list[Any]]) -> list[Any]:
    if not hidden_states_by_sample:
        raise ValueError("hidden_states_by_sample must not be empty")
    layer_count = len(hidden_states_by_sample[0])
    for sample_index, hidden_states in enumerate(hidden_states_by_sample):
        if len(hidden_states) != layer_count:
            raise ValueError(
                f"hidden state layer count mismatch for sample {sample_index}: "
                f"{len(hidden_states)} != {layer_count}"
            )
    return [
        _cat_stage2_batch_tensors(
            [hidden_states[layer_index] for hidden_states in hidden_states_by_sample],
            name=f"hidden_states[{layer_index}]",
        )
        for layer_index in range(layer_count)
    ]


def _select_stage2_rows(tensor: Any, row_indices: Any) -> Any:
    return tensor.index_select(0, row_indices.to(device=tensor.device))


def _cat_optional_stage2_batch_tensors(tensors: list[Any | None], *, name: str) -> Any | None:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return None
    if len(present) != len(tensors):
        raise ValueError(f"{name} is present for only part of a Stage2 batch")
    return _cat_stage2_batch_tensors(present, name=name)


def _cat_stage2_batch_tensors(tensors: list[Any], *, name: str) -> Any:
    if not tensors:
        raise ValueError(f"{name} tensor list must not be empty")
    reference_shape = tuple(tensors[0].shape[1:])
    for index, tensor in enumerate(tensors):
        if tensor.ndim <= 0:
            raise ValueError(f"{name}[{index}] must include a batch dimension, got shape {tuple(tensor.shape)}")
        if tuple(tensor.shape[1:]) != reference_shape:
            raise ValueError(
                f"{name}[{index}] shape after batch dim {tuple(tensor.shape[1:])} "
                f"!= {reference_shape}"
            )
    import torch

    return torch.cat(tensors, dim=0)


def _build_short_memory_for_sample(
    torch: Any,
    model: Any,
    *,
    short_images: Any,
    short_image_masks: Any,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any, Any]:
    from prism.serve.engine import build_short_memory_inputs_from_visual_tokens
    from prism.serve.engine import empty_short_memory_inputs, short_memory_offsets

    memory, mask, time_ids = empty_short_memory_inputs(model, device=device, dtype=dtype)
    if memory is None or mask is None or time_ids is None:
        return None, None, None

    offsets = short_memory_offsets(model)
    visual_tokens_by_offset = {}
    for entry_index, images in enumerate(short_images):
        if images is None:
            continue
        image_mask = short_image_masks[entry_index]
        if image_mask is None:
            continue
        tokens = _encode_visual_tokens_for_images(
            torch,
            model,
            images=images,
            image_mask=image_mask.to(device=device),
        )
        if tokens.shape[1] <= 0:
            continue
        offset = int(offsets[entry_index]) if entry_index < len(offsets) else int(entry_index)
        visual_tokens_by_offset[offset] = tokens
    if not visual_tokens_by_offset:
        return memory, mask, time_ids
    return build_short_memory_inputs_from_visual_tokens(
        model,
        visual_tokens_by_offset,
        device=device,
        dtype=dtype,
    )


def _encode_visual_tokens_for_images(torch: Any, model: Any, *, images: list[Any], image_mask: Any) -> Any:
    from prism.models.vlm import _flatten_active_visual_tokens

    if getattr(model, "embedder", None) is None:
        raise RuntimeError("Stage2 short-memory encoding requires load_vlm=true")
    pixel_values, num_tiles_list = model.embedder._preprocess_images(list(images))
    vit_embeds = model.embedder.model.extract_feature(pixel_values)
    return _flatten_active_visual_tokens(vit_embeds, image_mask.to(device=vit_embeds.device), num_tiles_list)


def _log_stage2_step(
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
    logging.info("Estimated Stage2 epoch: %.2f", current_epoch)
    logging.info("[Stage2 step %s] loss=%.4f lr=%s", step, float(loss.item()), scheduler.get_last_lr()[0])
    logging.info("[Stage2 step %s] grad_norm=%s clipped_norm=%s", step, total_norm, clipped_norm)
    for name, value in sorted((extra_metrics or {}).items()):
        logging.info("[Stage2 step %s] %s=%.4f", step, name, value)


def _load_runtime() -> dict[str, Any]:
    try:
        import torch
        from accelerate import Accelerator, DistributedType
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR
        from tqdm import tqdm

        from prism.models.policy import PrismPolicy
    except ModuleNotFoundError as exc:
        missing = exc.name or "a Stage2 training dependency"
        raise ModuleNotFoundError(
            f"{missing} is required for Stage2 training. Run inside the prepared training environment."
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


def _enforce_stage2_contract(config: dict[str, Any]) -> None:
    from prism.training.stage2_config import enforce_stage2_contract

    enforce_stage2_contract(config)
