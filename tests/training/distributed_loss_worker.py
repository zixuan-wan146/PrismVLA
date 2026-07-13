"""Two-process worker used by the real DDP masked-loss integration test."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import torch
from accelerate import Accelerator

from prism.models.policy import ActionLossStatistics, ScalarStatistic
from prism.training.runner import run_training_loop


class _WeightedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: torch.Tensor) -> SimpleNamespace:
        target, count, transition_true_positive, transition_positive = batch[0]
        loss_sum = torch.abs(self.weight - target) * count
        return SimpleNamespace(
            loss_statistics=ActionLossStatistics(
                loss_sum=loss_sum,
                valid_element_count=count,
                metrics={
                    "total_l1": ScalarStatistic(loss_sum.detach(), count),
                    "gripper_transition_recall": ScalarStatistic(
                        transition_true_positive,
                        transition_positive,
                    ),
                },
            )
        )


class _RecordingAccelerator:
    def __init__(self, accelerator: Accelerator) -> None:
        self._accelerator = accelerator
        self.logged_metrics: dict[str, float] | None = None

    def __getattr__(self, name: str):
        return getattr(self._accelerator, name)

    def print(self, message: str) -> None:
        if self._accelerator.is_main_process:
            self.logged_metrics = json.loads(message)["metrics"]


def main(output_path: str) -> None:
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    model = _WeightedPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    proxy = _RecordingAccelerator(accelerator)

    if accelerator.process_index == 0:
        rows = [
            [0.0, 7.0, 0.0, 0.0],
            [1.0, 56.0, 1.0, 1.0],
        ]
    else:
        rows = [
            [1.0, 56.0, 0.0, 0.0],
            [1.0, 56.0, 0.0, 0.0],
        ]
    dataloader = torch.utils.data.DataLoader(
        rows,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda samples: torch.tensor(samples, dtype=torch.float32),
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            loader=SimpleNamespace(preprocessing_workers=0, pin_memory=False),
        ),
        trainer=SimpleNamespace(
            max_steps=1,
            max_grad_norm=100.0,
            log_interval=1,
            save_interval=100,
        ),
    )
    run_training_loop(
        config=config,
        accelerator=proxy,
        model=model,
        collator=lambda raw: raw,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=rows,
        dataloader=dataloader,
    )

    local_weight = accelerator.unwrap_model(model).weight.detach().reshape(1)
    gathered_weights = accelerator.gather(local_weight)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if proxy.logged_metrics is None:
            raise RuntimeError("runner did not emit synchronized metrics")
        Path(output_path).write_text(
            json.dumps(
                {
                    "weights": gathered_weights.cpu().tolist(),
                    "metrics": proxy.logged_metrics,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_loss_worker.py OUTPUT_JSON")
    main(sys.argv[1])
