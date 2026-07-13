from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from prism.training.config import ResolvedOptimizationConfig
from prism.training.config import ResolvedOptimizationGroupConfig
from prism.training.optimization import build_optimizer


class _QwenFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.visual = nn.Linear(4, 4)


class _BackboneFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _QwenFixture()
        self.action_queries = nn.Parameter(torch.randn(2, 4))


class _EncoderFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _BackboneFixture()
        self.history_qformer = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


class _PolicyFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_memory_encoder = _EncoderFixture()
        self.action_head = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def _group(
    trainable: bool,
    *,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> ResolvedOptimizationGroupConfig:
    return ResolvedOptimizationGroupConfig(
        trainable=trainable,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _config() -> ResolvedOptimizationConfig:
    return ResolvedOptimizationConfig(
        optimizer="adamw",
        beta1=0.9,
        beta2=0.95,
        epsilon=1.0e-8,
        no_decay_rule="bias_and_low_dimensional",
        language_model=_group(False),
        vision_encoder=_group(False),
        action_queries=_group(True, learning_rate=1.0e-4, weight_decay=0.0),
        history_qformer=_group(True, learning_rate=2.0e-4, weight_decay=0.01),
        action_head=_group(True, learning_rate=3.0e-4, weight_decay=0.02),
    )


def test_build_optimizer_applies_explicit_scope_and_named_groups() -> None:
    model = _PolicyFixture()

    optimizer = build_optimizer(model, _config())

    assert not model.query_memory_encoder.backbone.model.language_model.training
    assert not model.query_memory_encoder.backbone.model.visual.training
    assert all(
        not parameter.requires_grad
        for parameter in model.query_memory_encoder.backbone.model.parameters()
    )
    assert model.query_memory_encoder.backbone.action_queries.requires_grad
    assert all(parameter.requires_grad for parameter in model.query_memory_encoder.history_qformer.parameters())
    assert all(parameter.requires_grad for parameter in model.action_head.parameters())

    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        "action_queries.decay",
        "history_qformer.decay",
        "history_qformer.no_decay",
        "action_head.decay",
        "action_head.no_decay",
    }
    assert groups["action_queries.decay"]["lr"] == pytest.approx(1.0e-4)
    assert groups["history_qformer.decay"]["lr"] == pytest.approx(2.0e-4)
    assert groups["history_qformer.decay"]["weight_decay"] == pytest.approx(0.01)
    assert groups["history_qformer.no_decay"]["weight_decay"] == 0.0
    assert groups["action_head.decay"]["lr"] == pytest.approx(3.0e-4)
    assert groups["action_head.decay"]["weight_decay"] == pytest.approx(0.02)


def test_build_optimizer_rejects_an_unclassified_parameter() -> None:
    model = _PolicyFixture()
    model.unclassified = nn.Parameter(torch.ones(()))

    with pytest.raises(ValueError, match="does not classify.*unclassified"):
        build_optimizer(model, _config())
