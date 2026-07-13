"""Explicit parameter freezing and AdamW group construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn
from torch.optim import AdamW

from prism.training.config import ResolvedOptimizationConfig
from prism.training.config import ResolvedOptimizationGroupConfig


def build_optimizer(
    model: nn.Module,
    config: ResolvedOptimizationConfig,
) -> AdamW:
    """Apply the resolved tuning scope and build complete named AdamW groups."""

    if not isinstance(config, ResolvedOptimizationConfig):
        raise TypeError(f"config must be ResolvedOptimizationConfig, got {type(config).__name__}")
    if config.optimizer != "adamw" or config.no_decay_rule != "bias_and_low_dimensional":
        raise ValueError("unsupported resolved optimization contract")

    named_parameters = dict(model.named_parameters())
    if not named_parameters:
        raise ValueError("policy has no parameters")
    name_by_parameter_id = {id(parameter): name for name, parameter in named_parameters.items()}
    grouped = _optimization_targets(model)
    assigned: dict[int, str] = {}
    optimizer_groups: list[dict[str, Any]] = []

    for group_name, group_config in config.named_groups():
        target = grouped[group_name]
        module = target if isinstance(target, nn.Module) else None
        if module is not None:
            module.train(group_config.trainable)
        group_parameters = list(_parameters(target))
        if not group_parameters:
            raise ValueError(f"optimization target {group_name!r} has no parameters")

        named_group_parameters: list[tuple[str, nn.Parameter]] = []
        for parameter in group_parameters:
            parameter_id = id(parameter)
            parameter_name = name_by_parameter_id.get(parameter_id)
            if parameter_name is None:
                raise ValueError(f"optimization target {group_name!r} contains a parameter outside the policy")
            if parameter_id in assigned:
                raise ValueError(
                    f"parameter {parameter_name!r} belongs to both {assigned[parameter_id]!r} and {group_name!r}"
                )
            assigned[parameter_id] = group_name
            parameter.requires_grad_(group_config.trainable)
            named_group_parameters.append((parameter_name, parameter))

        if group_config.trainable:
            optimizer_groups.extend(
                _adamw_groups(
                    group_name,
                    named_group_parameters,
                    group_config,
                )
            )

    missing = sorted(name for name, parameter in named_parameters.items() if id(parameter) not in assigned)
    if missing:
        raise ValueError(f"optimization config does not classify policy parameters: {missing}")
    if not optimizer_groups:
        raise ValueError("optimization config leaves no trainable parameters")

    return AdamW(
        optimizer_groups,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )


def _optimization_targets(model: nn.Module) -> dict[str, nn.Module | nn.Parameter]:
    try:
        encoder = model.query_memory_encoder
        backbone = encoder.backbone
        qwen_model = backbone.model
        return {
            "language_model": qwen_model.language_model,
            "vision_encoder": qwen_model.visual,
            "action_queries": backbone.action_queries,
            "history_qformer": encoder.history_qformer,
            "action_head": model.action_head,
        }
    except AttributeError as exc:
        raise TypeError("policy does not expose the accepted query-memory optimization boundaries") from exc


def _parameters(target: nn.Module | nn.Parameter) -> Iterable[nn.Parameter]:
    if isinstance(target, nn.Parameter):
        return (target,)
    return target.parameters()


def _adamw_groups(
    group_name: str,
    named_parameters: list[tuple[str, nn.Parameter]],
    config: ResolvedOptimizationGroupConfig,
) -> list[dict[str, Any]]:
    if config.learning_rate is None or config.weight_decay is None:
        raise ValueError(f"trainable group {group_name!r} has unresolved optimizer values")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter_name, parameter in named_parameters:
        target = no_decay if parameter_name.endswith(".bias") or parameter.ndim <= 1 else decay
        target.append(parameter)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "group_name": f"{group_name}.decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": config.learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{group_name}.no_decay",
            }
        )
    return groups


__all__ = ["build_optimizer"]
