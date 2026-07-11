from __future__ import annotations

# --- migrated from src/prism/training/progress_warmup.py ---
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from prism.data import LiberoProgressWarmupDataset
from prism.data import TemperatureSuiteSampler
from prism.data import collate_libero_progress_warmup_windows
from prism.models.planner import ProgressPretrainHeads
from prism.models.planner import ProgressState
from prism.models.planner import ProgressStateConfig
from prism.models.planner import ProgressStatePlanner
from prism.models.planner import progress_diagnostics
from prism.models.planner import progress_intent_alignment_loss
from prism.models.planner import progress_order_loss
from prism.utils.seeding import build_torch_generator
from prism.utils.seeding import seed_data_worker
from prism.utils.seeding import set_global_seed
from prism.utils.seeding import write_experiment_snapshot


@dataclass(frozen=True)
class ProgressWarmupTrainingConfig:
    cache_manifest: str
    output_dir: str
    device: str = "cuda"
    batch_size: int = 64
    max_steps: int = 1000
    samples_per_epoch: int = 8192
    sampling_alpha: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    seed: int = 42
    deterministic: bool = False
    hidden_dim: int | None = None
    state_dim: int | None = None
    action_dim: int | None = None
    replan_stride: int | None = None
    latent_dim: int | None = None
    action_summary_hidden_dim: int = 512
    state_hidden_dim: int = 512
    updater_hidden_dim: int = 1792
    planner_ffn_dim: int = 3584
    planner_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.05
    lambda_plan: float = 1.0
    lambda_stage: float = 0.5
    lambda_mem_pool: float = 0.1
    lambda_order: float = 0.02
    use_order_loss: bool = False
    min_order_gap: int = 2
    cosine_weight: float = 0.1
    val_fraction: float = 0.1
    split_by_episode: bool = True
    eval_interval: int = 50
    eval_batch_size: int | None = None
    max_val_batches: int | None = None
    log_interval: int = 10
    ckpt_interval: int = 0
    repo_root: str | None = None


@dataclass(frozen=True)
class ProgressWarmupTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    final_loss: float
    best_loss: float
    steps: int


def run_progress_warmup_training(
    config: ProgressWarmupTrainingConfig | Mapping[str, Any],
) -> ProgressWarmupTrainingResult:
    config = _coerce_config(config)
    _validate_config(config)
    set_global_seed(config.seed, deterministic=config.deterministic)

    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config = asdict(config)
    if config.repo_root is not None:
        snapshot_config["repo_root"] = config.repo_root
    write_experiment_snapshot(output_dir, snapshot_config)

    dataset = LiberoProgressWarmupDataset(config.cache_manifest)
    first_loss_step = dataset[0]["loss"][0]
    model_config = _build_model_config(config, dataset, first_loss_step)
    train_indices, val_indices = split_progress_window_indices(
        dataset,
        val_fraction=config.val_fraction,
        seed=config.seed,
        split_by_episode=config.split_by_episode,
    )
    sampler = TemperatureSuiteSampler(
        dataset,
        samples_per_epoch=max(int(config.samples_per_epoch), int(config.batch_size)),
        alpha=config.sampling_alpha,
        seed=config.seed,
        indices=train_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_libero_progress_warmup_windows,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(config.seed),
    )
    val_loader = None
    if val_indices:
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=int(config.eval_batch_size or config.batch_size),
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
            collate_fn=collate_libero_progress_warmup_windows,
            worker_init_fn=seed_data_worker,
            generator=build_torch_generator(config.seed + 1),
        )

    model = ProgressStatePlanner(model_config).to(device)
    heads = ProgressPretrainHeads(model_config).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(heads.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    fallback_best_loss = float("inf")
    final_loss = float("nan")
    best_checkpoint_path = output_dir / "best.pt"
    data_iter = iter(dataloader)
    model.train()
    heads.train()
    for step in range(1, config.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = progress_warmup_batch_loss(model, heads, batch, config, device=device)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(heads.parameters()),
            config.grad_clip_norm,
        )
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())
        row = {
            "step": step,
            "loss": final_loss,
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            "train_window_count": len(train_indices),
            "val_window_count": len(val_indices),
            **{name: float(value.detach().cpu().item()) for name, value in metrics.items()},
        }
        if val_loader is not None and config.eval_interval > 0 and step % config.eval_interval == 0:
            row.update(
                {
                    f"val_{name}": value
                    for name, value in evaluate_progress_warmup(
                        model,
                        heads,
                        val_loader,
                        config,
                        device=device,
                        max_batches=config.max_val_batches,
                    ).items()
                }
            )
        history.append(row)
        if val_loader is None or "val_loss" in row:
            selection_loss = float(row.get("val_loss", final_loss))
            if selection_loss < best_loss:
                best_loss = selection_loss
                _save_checkpoint(best_checkpoint_path, model, heads, optimizer, model_config, config, row)
        elif not best_checkpoint_path.exists():
            fallback_best_loss = final_loss
            _save_checkpoint(best_checkpoint_path, model, heads, optimizer, model_config, config, row)
        if config.ckpt_interval > 0 and step % config.ckpt_interval == 0:
            _save_checkpoint(output_dir / f"step_{step:06d}.pt", model, heads, optimizer, model_config, config, row)
        if config.log_interval > 0 and step % config.log_interval == 0:
            print(_format_metrics(row), flush=True)

    checkpoint_path = output_dir / "last.pt"
    _save_checkpoint(checkpoint_path, model, heads, optimizer, model_config, config, history[-1])
    reported_best_loss = best_loss if best_loss < float("inf") else fallback_best_loss
    _write_json(
        output_dir / "train_history.json",
        {
            "steps": history,
            "final_loss": final_loss,
            "best_loss": reported_best_loss,
            "model_config": asdict(model_config),
            "training_config": asdict(config),
        },
    )
    return ProgressWarmupTrainingResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        final_loss=final_loss,
        best_loss=reported_best_loss,
        steps=config.max_steps,
    )


def split_progress_window_indices(
    dataset: LiberoProgressWarmupDataset,
    *,
    val_fraction: float,
    seed: int,
    split_by_episode: bool = True,
) -> tuple[list[int], list[int]]:
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    all_indices = list(range(len(dataset)))
    if not all_indices:
        raise ValueError("dataset has no windows")
    if float(val_fraction) == 0.0:
        return all_indices, []

    rng = random.Random(int(seed))
    if split_by_episode:
        by_episode: dict[str, list[int]] = {}
        for index, window in enumerate(dataset.windows):
            by_episode.setdefault(str(window["episode_id"]), []).append(index)
        episodes = sorted(by_episode)
        rng.shuffle(episodes)
        val_episode_count = max(1, int(round(len(episodes) * float(val_fraction))))
        val_episode_count = min(val_episode_count, len(episodes) - 1)
        val_episodes = set(episodes[:val_episode_count])
        val_indices = sorted(index for episode in val_episodes for index in by_episode[episode])
    else:
        shuffled = list(all_indices)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
        val_count = min(val_count, len(shuffled) - 1)
        val_indices = sorted(shuffled[:val_count])

    val_set = set(val_indices)
    train_indices = [index for index in all_indices if index not in val_set]
    if not train_indices or not val_indices:
        raise ValueError("train/val split produced an empty split")
    return train_indices, val_indices


@torch.no_grad()
def evaluate_progress_warmup(
    model: ProgressStatePlanner,
    heads: ProgressPretrainHeads,
    dataloader: DataLoader,
    config: ProgressWarmupTrainingConfig,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training_model = model.training
    was_training_heads = heads.training
    model.eval()
    heads.eval()
    totals: dict[str, float] = {}
    total_weight = 0
    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and int(max_batches) > 0 and batch_index >= int(max_batches):
                break
            loss, metrics = progress_warmup_batch_loss(model, heads, batch, config, device=device)
            weight = int(torch.as_tensor(batch["window_index"]).numel())
            total_weight += weight
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu().item()) * weight
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item()) * weight
    finally:
        model.train(was_training_model)
        heads.train(was_training_heads)
    if total_weight <= 0:
        raise ValueError("validation dataloader produced no batches")
    return {name: value / total_weight for name, value in totals.items()}


def progress_warmup_batch_loss(
    model: ProgressStatePlanner,
    heads: ProgressPretrainHeads,
    batch: Mapping[str, Any],
    config: ProgressWarmupTrainingConfig,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_window = _move_window(batch["loss"], device=device)
    burnin_window = _move_window(batch["burnin"], device=device)
    burnin_mask = torch.as_tensor(batch["burnin_mask"], device=device).bool()
    loss_replan_indices = torch.as_tensor(batch["loss_replan_indices"], device=device)

    batch_size = int(loss_window["vl_summary"].shape[0])
    state = model.initial_state(
        batch_size,
        device=device,
        dtype=loss_window["vl_summary"].dtype,
    )

    with torch.no_grad():
        for index in range(int(burnin_window["vl_summary"].shape[1])):
            valid = burnin_mask[:, index]
            if not valid.any():
                continue
            output = model.forward_step(
                state,
                burnin_window["vl_summary"][:, index],
                burnin_window["state"][:, index],
                burnin_window["executed_actions"][:, index],
                burnin_window["executed_action_mask"][:, index],
            )
            state = _where_progress_state(valid, output.progress_state, state)
    state = _detach_progress_state(state)

    losses: list[torch.Tensor] = []
    metric_accumulator: dict[str, list[torch.Tensor]] = {
        "plan_loss": [],
        "stage_loss": [],
        "mem_pool_loss": [],
    }
    progress_scores: list[torch.Tensor] = []
    last_output = None
    last_heads = None
    for index in range(int(loss_window["vl_summary"].shape[1])):
        output = model.forward_step(
            state,
            loss_window["vl_summary"][:, index],
            loss_window["state"][:, index],
            loss_window["executed_actions"][:, index],
            loss_window["executed_action_mask"][:, index],
        )
        state = output.progress_state
        head_output = heads(output.planner_token, output.progress_state)
        target = loss_window["target_intent"][:, index]
        plan_loss = progress_intent_alignment_loss(
            head_output.planner_intent,
            target,
            cosine_weight=config.cosine_weight,
        )
        stage_loss = progress_intent_alignment_loss(
            head_output.stage_intent,
            target,
            cosine_weight=config.cosine_weight,
        )
        mem_pool_loss = progress_intent_alignment_loss(
            head_output.memory_pool_intent,
            target,
            cosine_weight=config.cosine_weight,
        )
        step_loss = (
            float(config.lambda_plan) * plan_loss
            + float(config.lambda_stage) * stage_loss
            + float(config.lambda_mem_pool) * mem_pool_loss
        )
        losses.append(step_loss)
        metric_accumulator["plan_loss"].append(plan_loss.detach())
        metric_accumulator["stage_loss"].append(stage_loss.detach())
        metric_accumulator["mem_pool_loss"].append(mem_pool_loss.detach())
        progress_scores.append(head_output.progress_score.squeeze(-1))
        last_output = output
        last_heads = head_output

    if not losses:
        raise ValueError("loss window is empty")
    alignment_loss = torch.stack(losses).mean()
    order_loss = alignment_loss.new_zeros(())
    if config.use_order_loss:
        order_loss = progress_order_loss(
            torch.stack(progress_scores, dim=1),
            replan_indices=loss_replan_indices,
            min_order_gap=config.min_order_gap,
        )
    total_loss = alignment_loss + (float(config.lambda_order) * order_loss if config.use_order_loss else order_loss)
    metrics = {
        name: torch.stack(values).mean().detach()
        for name, values in metric_accumulator.items()
    }
    metrics["order_loss"] = order_loss.detach()
    metrics["alignment_loss"] = alignment_loss.detach()
    if last_output is not None and last_heads is not None:
        diagnostics = progress_diagnostics(
            last_output.planner_token,
            last_output.progress_state.current_stage,
            last_heads.planner_intent,
            last_heads.stage_intent,
        )
        metrics.update(diagnostics)
    return total_loss, metrics


def _build_model_config(
    config: ProgressWarmupTrainingConfig,
    dataset: LiberoProgressWarmupDataset,
    first_loss_step: Mapping[str, Any],
) -> ProgressStateConfig:
    executed_actions = torch.as_tensor(first_loss_step["executed_actions"])
    if executed_actions.ndim != 2:
        raise ValueError(f"executed_actions must have shape [R, A], got {tuple(executed_actions.shape)}")
    hidden_dim = int(config.hidden_dim or dataset.manifest["hidden_dim"])
    return ProgressStateConfig(
        hidden_dim=hidden_dim,
        state_dim=int(config.state_dim or torch.as_tensor(first_loss_step["state"]).shape[-1]),
        action_dim=int(config.action_dim or executed_actions.shape[-1]),
        replan_stride=int(config.replan_stride or dataset.manifest["replan_stride"]),
        latent_dim=int(config.latent_dim or torch.as_tensor(first_loss_step["target_intent"]).shape[-1]),
        action_summary_hidden_dim=int(config.action_summary_hidden_dim),
        state_hidden_dim=int(config.state_hidden_dim),
        updater_hidden_dim=int(config.updater_hidden_dim),
        planner_ffn_dim=int(config.planner_ffn_dim),
        planner_layers=int(config.planner_layers),
        num_heads=int(config.num_heads),
        dropout=float(config.dropout),
    )


def _move_window(window: Mapping[str, torch.Tensor], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in window.items()
    }


def _where_progress_state(mask: torch.Tensor, updated: ProgressState, previous: ProgressState) -> ProgressState:
    mask = mask.to(device=updated.completed_events.device, dtype=torch.bool).unsqueeze(-1)
    return ProgressState(
        completed_events=torch.where(mask, updated.completed_events, previous.completed_events),
        current_stage=torch.where(mask, updated.current_stage, previous.current_stage),
    )


def _detach_progress_state(state: ProgressState) -> ProgressState:
    return ProgressState(
        completed_events=state.completed_events.detach(),
        current_stage=state.current_stage.detach(),
    )


def _coerce_config(config: ProgressWarmupTrainingConfig | Mapping[str, Any]) -> ProgressWarmupTrainingConfig:
    if isinstance(config, ProgressWarmupTrainingConfig):
        return config
    return ProgressWarmupTrainingConfig(**dict(config))


def _validate_config(config: ProgressWarmupTrainingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")
    if config.sampling_alpha < 0.0:
        raise ValueError("sampling_alpha must be non-negative")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.log_interval < 0:
        raise ValueError("log_interval must be non-negative")
    if config.ckpt_interval < 0:
        raise ValueError("ckpt_interval must be non-negative")
    if config.min_order_gap <= 0:
        raise ValueError("min_order_gap must be positive")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.eval_interval < 0:
        raise ValueError("eval_interval must be non-negative")
    if config.eval_batch_size is not None and config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive when provided")
    if config.max_val_batches is not None and config.max_val_batches < 0:
        raise ValueError("max_val_batches must be non-negative when provided")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {value!r}, but CUDA is not available")
    return device


def _save_checkpoint(
    path: Path,
    model: ProgressStatePlanner,
    heads: ProgressPretrainHeads,
    optimizer: torch.optim.Optimizer,
    model_config: ProgressStateConfig,
    training_config: ProgressWarmupTrainingConfig,
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "progress_state_planner_warmup",
            "version": 1,
            "model_state_dict": model.state_dict(),
            "heads_state_dict": heads.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "metrics": dict(metrics),
        },
        path,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
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


def _format_metrics(row: Mapping[str, Any]) -> str:
    keys = [
        "step",
        "loss",
        "plan_loss",
        "stage_loss",
        "mem_pool_loss",
        "order_loss",
        "val_loss",
        "val_plan_loss",
        "val_stage_loss",
        "val_mem_pool_loss",
        "val_cos_g_p",
        "val_stage_effective_rank",
        "cos_g_p",
        "stage_batch_variance",
        "stage_effective_rank",
        "grad_norm",
    ]
    parts = []
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={float(value):.6f}")
    return "progress_warmup " + " ".join(parts)

