from __future__ import annotations

from typing import Any


def unwrap_training_model(accelerator: Any, model: Any):
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(model)
    return getattr(model, "module", model)


def get_and_clip_grad_norm(torch: Any, accelerator: Any, model: Any, loss: Any, max_norm: float = 1.0):
    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        total_norm = torch.tensor(0.0, device=loss.device) if not grad_norms else torch.norm(torch.stack(grad_norms), 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        clipped = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        clipped_norm = torch.tensor(0.0, device=loss.device) if not clipped else torch.norm(torch.stack(clipped), 2)
    return total_norm, clipped_norm
