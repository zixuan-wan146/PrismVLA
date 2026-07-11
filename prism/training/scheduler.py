from __future__ import annotations

import math


def get_lr_lambda(warmup_steps: int, total_steps: int, *, resume_step: int = 0, min_lr_ratio: float = 0.0):
    min_lr_ratio = float(min_lr_ratio)
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(current_step: int) -> float:
        step = int(current_step) + int(resume_step)
        if step < int(warmup_steps):
            return step / max(1, int(warmup_steps))
        progress = (step - int(warmup_steps)) / max(1, int(total_steps) - int(warmup_steps))
        cosine = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda
