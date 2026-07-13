"""Single construction path for the accepted PrismVLA policy architecture."""

from __future__ import annotations

from prism.models.config import PrismArchitectureConfig
from prism.models.history_qformer import HistoryQFormer
from prism.models.policy import PrismPolicy
from prism.models.vlm import Qwen35ActionQueryBackbone, Qwen35QueryMemoryEncoder


def build_prism_policy(
    architecture: PrismArchitectureConfig,
    *,
    state_dim: int,
    local_files_only: bool | None = None,
) -> PrismPolicy:
    """Construct training and inference policies through the same model graph."""

    if not isinstance(architecture, PrismArchitectureConfig):
        raise TypeError(
            f"architecture must be PrismArchitectureConfig, got {type(architecture).__name__}"
        )
    architecture.validate_for_policy()
    if type(state_dim) is not int or state_dim <= 0:
        raise ValueError("state_dim must be a positive integer")
    backbone = Qwen35ActionQueryBackbone.from_pretrained(
        architecture.backbone,
        local_files_only=local_files_only,
    )
    query_memory_encoder = Qwen35QueryMemoryEncoder(
        backbone,
        HistoryQFormer(architecture.history),
    )
    return PrismPolicy(
        architecture,
        query_memory_encoder,
        state_dim=state_dim,
    )


__all__ = ["build_prism_policy"]
