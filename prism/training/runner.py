"""Single explicit Accelerate training path for the direct-action policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from prism.data.dataset import (
    SingleVLADataset,
    VLAMixtureDataset,
    build_vla_dataloader,
    set_data_epoch,
)
from prism.data.lerobot import LeRobotDataset
from prism.data.normalization import DataSpecNormalizer
from prism.models.batch import PolicyBatch
from prism.models.factory import build_prism_policy
from prism.models.policy import ActionLossStatistics, ScalarStatistic
from prism.training.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint
from prism.training.config import ResolvedTrainConfig, load_train_config
from prism.training.optimization import build_optimizer
from prism.training.preprocessing import iter_preprocessed_batches


@dataclass(frozen=True)
class _LinearWarmupDecay:
    warmup_steps: int
    max_steps: int

    def __call__(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        decay_steps = self.max_steps - self.warmup_steps
        if decay_steps <= 0:
            return 0.0
        return max(
            0.0,
            float(self.max_steps - step) / float(decay_steps),
        )


def run_training(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
    project_root: str | Path | None = None,
) -> TrainingProgress:
    """Resolve a train YAML, construct the one accepted stack, and train it."""

    root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root).expanduser().resolve()
    config = load_train_config(config_path, project_root=root)
    resume_path = None
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser()
        if not resume_path.is_absolute():
            resume_path = (root / resume_path).resolve()
    return run_resolved_training(config, resume_from=resume_path)


def run_resolved_training(
    config: ResolvedTrainConfig,
    *,
    resume_from: str | Path | None = None,
) -> TrainingProgress:
    """Composition root for data, policy, optimization, resume, and the loop."""

    if not isinstance(config, ResolvedTrainConfig):
        raise TypeError(f"config must be ResolvedTrainConfig, got {type(config).__name__}")
    config.model.architecture.validate_for_policy()

    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError as exc:  # pragma: no cover - environment contract guard
        raise RuntimeError("training requires the Accelerate package in the remote project environment") from exc

    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        mixed_precision=config.trainer.mixed_precision,
        # The runner calls scheduler.step exactly once at each sync boundary.
        # Accelerate's default otherwise advances a prepared scheduler once per
        # process when batches are sharded, which would desynchronize it from
        # the single global optimizer step.
        step_scheduler_with_optimizer=False,
    )
    set_seed(config.experiment.seed, device_specific=False)

    normalizer = DataSpecNormalizer(
        data_spec=config.data.spec,
        statistics=config.data.normalization.statistics,
        statistics_group=config.data.normalization.group,
    )
    backends: list[LeRobotDataset] = []
    physical_datasets: list[SingleVLADataset] = []
    try:
        for dataset_config in config.data.datasets:
            backend = LeRobotDataset(
                dataset_config.path,
                config.data.spec,
                verify_files=True,
            )
            backends.append(backend)
            physical_datasets.append(
                SingleVLADataset(
                    name=dataset_config.name,
                    backend=backend,
                    normalizer=normalizer,
                    action_horizon=config.temporal.action_horizon,
                    history_step_ages=config.temporal.history_step_ages,
                    anchor_stride=config.data.anchor_stride,
                    include_tail=config.data.include_tail,
                )
            )

        dataset = VLAMixtureDataset(
            physical_datasets,
            [entry.weight for entry in config.data.datasets],
            samples_per_epoch=config.data.loader.global_samples_per_epoch,
            seed=config.experiment.seed,
        )
        dataloader = build_vla_dataloader(
            dataset,
            batch_size_per_rank=config.data.loader.batch_size_per_rank,
            num_workers=config.data.loader.num_workers,
            # Raw VLASample objects contain NumPy arrays, so DataLoader's tensor
            # pinning cannot see them. Final collated tensors are pinned by the
            # preprocessing pipeline below.
            pin_memory=False,
            drop_last=config.data.loader.drop_last,
            seed=config.experiment.seed,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )

        model = build_prism_policy(
            config.model.architecture,
            state_dim=config.data.spec.state_dim,
        )
        collator = model.make_collator()
        optimizer = build_optimizer(model, config.optimization)
        scheduler = LambdaLR(
            optimizer,
            _LinearWarmupDecay(
                warmup_steps=config.trainer.warmup_steps,
                max_steps=config.trainer.max_steps,
            ),
        )

        # The loader is already deterministically sharded by virtual index.
        # Preparing it again would let Accelerate shard the same stream twice.
        model, optimizer, scheduler = accelerator.prepare(
            model,
            optimizer,
            scheduler,
        )

        if accelerator.is_main_process:
            config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        progress = _initial_progress()
        if resume_from is not None:
            progress = load_checkpoint(
                resume_from,
                accelerator=accelerator,
                expected_config=config,
            )

        checkpoint_root = config.experiment.output_dir / "checkpoints"
        return run_training_loop(
            config=config,
            accelerator=accelerator,
            model=model,
            collator=collator,
            optimizer=optimizer,
            scheduler=scheduler,
            dataset=dataset,
            dataloader=dataloader,
            progress=progress,
            checkpoint_root=checkpoint_root,
        )
    finally:
        for backend in backends:
            backend.close()


def run_training_loop(
    *,
    config: ResolvedTrainConfig,
    accelerator: Any,
    model: torch.nn.Module,
    collator: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    dataset: Any,
    dataloader: Any,
    progress: TrainingProgress | None = None,
    checkpoint_root: str | Path | None = None,
) -> TrainingProgress:
    """Run direct-policy optimization; objective details remain inside policy."""

    progress = _initial_progress() if progress is None else progress
    _validate_start_progress(progress, config=config)
    epoch_batch_count = len(dataloader)
    if epoch_batch_count <= 0:
        raise ValueError("training dataloader has no batches; increase global_samples_per_epoch or disable drop_last")

    completed_steps = progress.completed_optimizer_steps
    micro_step = progress.gradient_accumulation_micro_step
    epoch = progress.epoch
    batch_cursor = progress.virtual_batch_cursor
    sample_cursor = progress.virtual_sample_cursor
    world_size = int(accelerator.num_processes)
    if world_size <= 0:
        raise ValueError("accelerator.num_processes must be positive")

    optimizer.zero_grad(set_to_none=True)
    loss_sum: torch.Tensor | None = None
    loss_normalizer_sum: torch.Tensor | None = None
    metric_sums: dict[str, ScalarStatistic] = {}

    while completed_steps < config.trainer.max_steps:
        set_data_epoch(dataset, dataloader, epoch)
        epoch_iterator = iter(dataloader)
        skipped_samples = 0
        for skipped_batch_index in range(batch_cursor):
            try:
                skipped_batch = next(epoch_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "checkpoint virtual_batch_cursor exceeds the epoch dataloader: "
                    f"cursor={batch_cursor}, batches={epoch_batch_count}"
                ) from exc
            skipped_samples += _raw_batch_size(skipped_batch) * world_size
        if skipped_samples != sample_cursor:
            raise ValueError(
                "checkpoint virtual cursors disagree with the deterministic loader: "
                f"batch_cursor={batch_cursor} resolves to {skipped_samples} global "
                f"samples, metadata records {sample_cursor}"
            )
        if batch_cursor == epoch_batch_count:
            # Runner-produced checkpoints already canonicalize this position to
            # the next epoch, but accepting the equivalent end cursor avoids an
            # empty-iterator loop for externally inspected/re-emitted metadata.
            epoch += 1
            batch_cursor = 0
            sample_cursor = 0
            continue

        preprocessed_batches = iter_preprocessed_batches(
            epoch_iterator,
            collator,
            preprocessing_workers=config.data.loader.preprocessing_workers,
            pin_memory=config.data.loader.pin_memory,
        )
        for raw_batch, prepared_batch in preprocessed_batches:
            raw_batch_size = _raw_batch_size(raw_batch)
            next_batch_cursor = batch_cursor + 1
            next_sample_cursor = sample_cursor + raw_batch_size * world_size
            epoch_ended = next_batch_cursor == epoch_batch_count
            next_epoch = epoch + 1 if epoch_ended else epoch
            if epoch_ended:
                next_batch_cursor = 0
                next_sample_cursor = 0

            with accelerator.accumulate(model):
                batch = _move_batch_to_device(
                    prepared_batch,
                    accelerator.device,
                )
                output = model(batch)
                statistics = getattr(output, "loss_statistics", None)
                if not isinstance(statistics, ActionLossStatistics):
                    raise TypeError("policy output must expose ActionLossStatistics as loss_statistics")
                _validate_loss_statistics(statistics)
                accelerator.backward(statistics.loss_sum)
                loss_sum, loss_normalizer_sum, metric_sums = _accumulate_statistics(
                    loss_sum,
                    loss_normalizer_sum,
                    metric_sums,
                    statistics,
                )
                micro_step += 1

                if accelerator.sync_gradients:
                    global_loss_sum, global_loss_normalizer, reduced_metrics = _reduce_statistics(
                        accelerator,
                        loss_sum,
                        loss_normalizer_sum,
                        metric_sums,
                    )
                    _normalize_accumulated_gradients(
                        model.parameters(),
                        global_loss_normalizer=global_loss_normalizer,
                        world_size=world_size,
                        gradient_accumulation_steps=int(accelerator.gradient_accumulation_steps),
                    )
                    gradient_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        config.trainer.max_grad_norm,
                    )
                    _require_finite_optimizer_boundary(
                        global_loss_sum=global_loss_sum,
                        gradient_norm=gradient_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    optimizer_step_was_skipped = bool(getattr(accelerator, "optimizer_step_was_skipped", False))
                    if not optimizer_step_was_skipped:
                        scheduler.step()
                        completed_steps += 1

                        if (
                            completed_steps % config.trainer.log_interval == 0
                            or completed_steps == config.trainer.max_steps
                        ):
                            _log_metrics(
                                accelerator,
                                completed_steps=completed_steps,
                                metrics=reduced_metrics,
                            )

                        current_progress = TrainingProgress(
                            completed_optimizer_steps=completed_steps,
                            gradient_accumulation_micro_step=micro_step,
                            epoch=next_epoch,
                            virtual_sample_cursor=next_sample_cursor,
                            virtual_batch_cursor=next_batch_cursor,
                        )
                        should_save = (
                            completed_steps % config.trainer.save_interval == 0
                            or completed_steps == config.trainer.max_steps
                        )
                        if checkpoint_root is not None and should_save:
                            checkpoint_path = Path(checkpoint_root) / (f"step-{completed_steps:08d}")
                            save_checkpoint(
                                checkpoint_path,
                                accelerator=accelerator,
                                config=config,
                                progress=current_progress,
                            )
                    loss_sum = None
                    loss_normalizer_sum = None
                    metric_sums = {}

            epoch = next_epoch
            batch_cursor = next_batch_cursor
            sample_cursor = next_sample_cursor
            if completed_steps >= config.trainer.max_steps or epoch_ended:
                break
        preprocessed_batches.close()

    return TrainingProgress(
        completed_optimizer_steps=completed_steps,
        gradient_accumulation_micro_step=micro_step,
        epoch=epoch,
        virtual_sample_cursor=sample_cursor,
        virtual_batch_cursor=batch_cursor,
    )


def _validate_start_progress(
    progress: TrainingProgress,
    *,
    config: ResolvedTrainConfig,
) -> None:
    if not isinstance(progress, TrainingProgress):
        raise TypeError(f"progress must be TrainingProgress, got {type(progress).__name__}")
    if progress.completed_optimizer_steps > config.trainer.max_steps:
        raise ValueError("checkpoint completed_optimizer_steps exceeds trainer.max_steps")
    if progress.gradient_accumulation_micro_step != 0:
        raise ValueError(
            "this runner saves and resumes only at synchronized optimizer "
            "boundaries, so gradient_accumulation_micro_step must be zero"
        )


def _initial_progress() -> TrainingProgress:
    return TrainingProgress(
        completed_optimizer_steps=0,
        gradient_accumulation_micro_step=0,
        epoch=0,
        virtual_sample_cursor=0,
        virtual_batch_cursor=0,
    )


def _raw_batch_size(raw_batch: Any) -> int:
    if isinstance(raw_batch, torch.Tensor):
        if raw_batch.ndim == 0:
            raise ValueError("raw batch tensor must have a batch dimension")
        size = int(raw_batch.shape[0])
    elif isinstance(raw_batch, Sequence) and not isinstance(raw_batch, (str, bytes)):
        size = len(raw_batch)
    else:
        raise TypeError("raw dataloader batches must be tensors or sized sample sequences")
    if size <= 0:
        raise ValueError("raw dataloader batch must contain at least one sample")
    return size


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    if not isinstance(batch, PolicyBatch):
        raise TypeError(f"model-owned collator must return PolicyBatch, got {type(batch).__name__}")
    return PolicyBatch(
        current_inputs={key: value.to(device=device, non_blocking=True) for key, value in batch.current_inputs.items()},
        history_inputs={key: value.to(device=device, non_blocking=True) for key, value in batch.history_inputs.items()},
        history_step_ages=batch.history_step_ages.to(device=device, non_blocking=True),
        history_valid_mask=batch.history_valid_mask.to(device=device, non_blocking=True),
        state=batch.state.to(device=device, non_blocking=True),
        executed_actions=batch.executed_actions.to(device=device, non_blocking=True),
        executed_action_valid_mask=batch.executed_action_valid_mask.to(device=device, non_blocking=True),
        target_actions=batch.target_actions.to(device=device, non_blocking=True),
        action_valid_mask=batch.action_valid_mask.to(device=device, non_blocking=True),
        action_dim_mask=(
            None if batch.action_dim_mask is None else batch.action_dim_mask.to(device=device, non_blocking=True)
        ),
    )


def _validate_loss_statistics(statistics: ActionLossStatistics) -> None:
    if not isinstance(statistics.loss_sum, torch.Tensor) or statistics.loss_sum.ndim != 0:
        raise TypeError("loss_sum must be a scalar tensor")
    if not isinstance(statistics.valid_element_count, torch.Tensor) or statistics.valid_element_count.ndim != 0:
        raise TypeError("valid_element_count must be a scalar tensor")
    if not isinstance(statistics.metrics, Mapping) or not statistics.metrics:
        raise TypeError("loss statistics metrics must be a non-empty mapping")
    for name, statistic in statistics.metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError("metric names must be non-empty strings")
        if not isinstance(statistic, ScalarStatistic):
            raise TypeError(f"metric {name!r} must be a ScalarStatistic")
        if statistic.numerator.ndim != 0 or statistic.denominator.ndim != 0:
            raise TypeError(f"metric {name!r} numerator and denominator must be scalar tensors")


def _accumulate_statistics(
    loss_sum: torch.Tensor | None,
    loss_normalizer_sum: torch.Tensor | None,
    metric_sums: dict[str, ScalarStatistic],
    statistics: ActionLossStatistics,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, ScalarStatistic]]:
    detached_loss = statistics.loss_sum.detach().float()
    normalizer = statistics.valid_element_count.detach().float()
    loss_sum = detached_loss.clone() if loss_sum is None else loss_sum + detached_loss
    if loss_normalizer_sum is None:
        loss_normalizer_sum = normalizer.clone()
    else:
        loss_normalizer_sum = loss_normalizer_sum + normalizer

    parsed = {
        name: ScalarStatistic(
            statistic.numerator.detach().float(),
            statistic.denominator.detach().float(),
        )
        for name, statistic in statistics.metrics.items()
    }
    if metric_sums and set(parsed) != set(metric_sums):
        raise ValueError("metric keys changed within an optimizer step")
    if not metric_sums:
        metric_sums = {
            name: ScalarStatistic(
                statistic.numerator.clone(),
                statistic.denominator.clone(),
            )
            for name, statistic in parsed.items()
        }
    else:
        metric_sums = {
            name: ScalarStatistic(
                metric_sums[name].numerator + statistic.numerator,
                metric_sums[name].denominator + statistic.denominator,
            )
            for name, statistic in parsed.items()
        }
    return loss_sum, loss_normalizer_sum, metric_sums


def _reduce_statistics(
    accelerator: Any,
    loss_sum: torch.Tensor | None,
    loss_normalizer_sum: torch.Tensor | None,
    metric_sums: Mapping[str, ScalarStatistic],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if loss_sum is None or loss_normalizer_sum is None or not metric_sums:
        raise RuntimeError("optimizer synchronization has no policy metrics to reduce")

    names = sorted(metric_sums)
    packed = torch.stack(
        [loss_sum, loss_normalizer_sum]
        + [metric_sums[name].numerator for name in names]
        + [metric_sums[name].denominator for name in names]
    )
    reduced = accelerator.reduce(packed, reduction="sum")
    metric_count = len(names)
    reduced_metrics = {
        name: ScalarStatistic(
            numerator=reduced[2 + index],
            denominator=reduced[2 + metric_count + index],
        ).value
        for index, name in enumerate(names)
    }
    return reduced[0], reduced[1], reduced_metrics


def _normalize_accumulated_gradients(
    parameters: Any,
    *,
    global_loss_normalizer: torch.Tensor,
    world_size: int,
    gradient_accumulation_steps: int,
) -> None:
    if world_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("world size and gradient accumulation steps must be positive")
    normalizer = float(global_loss_normalizer.detach().cpu().item())
    if normalizer <= 0.0:
        raise RuntimeError("global masked loss has no valid action elements")

    gradient_scale = (world_size * gradient_accumulation_steps) / normalizer
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)


def _require_finite_optimizer_boundary(
    *,
    global_loss_sum: torch.Tensor,
    gradient_norm: torch.Tensor,
) -> None:
    """Synchronize once at an optimizer boundary and reject poisoned updates."""

    if global_loss_sum.ndim != 0:
        raise TypeError("global loss sum must be a scalar tensor")
    if not isinstance(gradient_norm, torch.Tensor) or gradient_norm.ndim != 0:
        raise TypeError("gradient norm must be a scalar tensor")
    values = torch.stack(
        (
            global_loss_sum.detach().to(dtype=torch.float32),
            gradient_norm.detach().to(device=global_loss_sum.device, dtype=torch.float32),
        )
    )
    if not bool(torch.isfinite(values).all().item()):
        raise FloatingPointError("non-finite global loss or gradient norm before optimizer.step()")


def _log_metrics(
    accelerator: Any,
    *,
    completed_steps: int,
    metrics: Mapping[str, torch.Tensor],
) -> None:
    payload = {
        "optimizer_step": completed_steps,
        "metrics": {name: float(value.detach().float().cpu().item()) for name, value in sorted(metrics.items())},
    }
    accelerator.print(json.dumps(payload, sort_keys=True))


__all__ = [
    "run_resolved_training",
    "run_training",
    "run_training_loop",
]
