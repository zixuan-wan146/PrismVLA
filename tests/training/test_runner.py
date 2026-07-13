from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest
import torch

import prism.training.runner as runner
from prism.training.checkpoint import TrainingProgress


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen: list[float] = []

    def forward(self, batch: torch.Tensor) -> SimpleNamespace:
        self.seen.extend(float(value) for value in batch.tolist())
        prediction = self.weight * batch.float().mean()
        loss = prediction.square()
        return SimpleNamespace(
            loss=loss,
            metrics={
                "total_l1": prediction.detach().abs(),
                "batch_mean": batch.float().mean(),
            },
        )


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: Any) -> None:
        super().__init__(parameters, lr=0.01)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, closure: Any = None) -> Any:
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_calls += 1
        super().zero_grad(set_to_none=set_to_none)


class _CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class _FakeAccelerator:
    def __init__(self, sync_pattern: list[bool], *, num_processes: int = 1) -> None:
        self._sync_pattern = sync_pattern
        self._accumulation_index = 0
        self.sync_gradients = False
        self.num_processes = num_processes
        self.process_index = 0
        self.device = torch.device("cpu")
        self.optimizer_step_was_skipped = False
        self.clip_calls = 0
        self.reduce_calls: list[tuple[float, str]] = []
        self.printed: list[str] = []

    @contextmanager
    def accumulate(self, model: torch.nn.Module):
        del model
        if self._accumulation_index >= len(self._sync_pattern):
            raise AssertionError("fake sync pattern was exhausted")
        self.sync_gradients = self._sync_pattern[self._accumulation_index]
        self._accumulation_index += 1
        yield

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        self.clip_calls += 1
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        self.reduce_calls.append((float(value.item()), reduction))
        return value

    def print(self, message: str) -> None:
        self.printed.append(message)


def _config(*, max_steps: int, log_interval: int = 100, save_interval: int = 100) -> Any:
    return SimpleNamespace(
        trainer=SimpleNamespace(
            max_steps=max_steps,
            max_grad_norm=1.0,
            log_interval=log_interval,
            save_interval=save_interval,
        )
    )


def _loader(values: list[float], *, batch_size: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        values,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: torch.tensor(samples, dtype=torch.float32),
    )


def test_gradient_accumulation_steps_optimizer_scheduler_and_reduce_only_at_sync() -> None:
    model = _TinyPolicy()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator(
        [False, True, False, True],
        num_processes=2,
    )

    progress = runner.run_training_loop(
        config=_config(max_steps=2, log_interval=2),
        accelerator=accelerator,
        model=model,
        collator=lambda raw: raw,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=list(range(8)),
        dataloader=_loader([1, 2, 3, 4, 5, 6, 7, 8], batch_size=2),
    )

    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 2
    assert accelerator.clip_calls == 2
    assert len(accelerator.reduce_calls) == 4
    assert all(reduction == "mean" for _, reduction in accelerator.reduce_calls)
    assert len(accelerator.printed) == 1
    assert json.loads(accelerator.printed[0])["optimizer_step"] == 2
    assert progress == TrainingProgress(
        completed_optimizer_steps=2,
        gradient_accumulation_micro_step=0,
        epoch=1,
        virtual_sample_cursor=0,
        virtual_batch_cursor=0,
    )


def test_smoke_checkpoint_is_saved_only_after_synchronized_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyPolicy()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator([False, True])
    saved: list[tuple[Path, TrainingProgress]] = []

    def fake_save(path: Path, **kwargs: Any) -> Path:
        saved.append((path, kwargs["progress"]))
        assert kwargs["accelerator"] is accelerator
        return path

    monkeypatch.setattr(runner, "save_checkpoint", fake_save)
    config = _config(max_steps=1, save_interval=100)

    progress = runner.run_training_loop(
        config=config,
        accelerator=accelerator,
        model=model,
        collator=lambda raw: raw,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=list(range(4)),
        dataloader=_loader([1, 2, 3, 4], batch_size=1),
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert optimizer.step_calls == scheduler.step_calls == accelerator.clip_calls == 1
    assert saved == [
        (
            tmp_path / "checkpoints" / "step-00000001",
            progress,
        )
    ]
    assert progress.gradient_accumulation_micro_step == 0
    assert progress.virtual_batch_cursor == 2
    assert progress.virtual_sample_cursor == 2


def test_scheduler_and_completed_step_do_not_advance_on_skipped_optimizer_step() -> None:
    model = _TinyPolicy()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator([True, True])

    @contextmanager
    def accumulate_with_first_step_skipped(model: torch.nn.Module):
        with _FakeAccelerator.accumulate(accelerator, model):
            accelerator.optimizer_step_was_skipped = accelerator._accumulation_index == 1
            yield

    accelerator.accumulate = accumulate_with_first_step_skipped

    progress = runner.run_training_loop(
        config=_config(max_steps=1),
        accelerator=accelerator,
        model=model,
        collator=lambda raw: raw,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=list(range(2)),
        dataloader=_loader([1, 2], batch_size=1),
    )

    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 1
    assert progress.completed_optimizer_steps == 1


def test_resume_cursor_skips_deterministic_batches_before_forward() -> None:
    model = _TinyPolicy()
    optimizer = _CountingSGD(model.parameters())
    scheduler = _CountingScheduler()
    accelerator = _FakeAccelerator([True], num_processes=2)
    initial = TrainingProgress(
        completed_optimizer_steps=0,
        gradient_accumulation_micro_step=0,
        epoch=3,
        virtual_sample_cursor=4,
        virtual_batch_cursor=1,
    )

    progress = runner.run_training_loop(
        config=_config(max_steps=1),
        accelerator=accelerator,
        model=model,
        collator=lambda raw: raw,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=list(range(6)),
        dataloader=_loader([1, 2, 3, 4, 5, 6], batch_size=2),
        progress=initial,
    )

    assert model.seen == [3.0, 4.0]
    assert progress.virtual_batch_cursor == 2
    assert progress.virtual_sample_cursor == 8
    assert progress.epoch == 3


def test_non_boundary_micro_step_resume_is_rejected() -> None:
    model = _TinyPolicy()
    progress = TrainingProgress(
        completed_optimizer_steps=0,
        gradient_accumulation_micro_step=1,
        epoch=0,
        virtual_sample_cursor=1,
        virtual_batch_cursor=1,
    )

    with pytest.raises(ValueError, match="synchronized optimizer boundaries"):
        runner.run_training_loop(
            config=_config(max_steps=1),
            accelerator=_FakeAccelerator([True]),
            model=model,
            collator=lambda raw: raw,
            optimizer=_CountingSGD(model.parameters()),
            scheduler=_CountingScheduler(),
            dataset=[0],
            dataloader=_loader([1], batch_size=1),
            progress=progress,
        )


def test_direct_train_script_help_imports_project_from_an_unrelated_cwd(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "train.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--resume" in result.stdout
