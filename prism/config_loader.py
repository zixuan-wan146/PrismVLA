from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from prism.config_training import load_training_config

@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"


@dataclass
class PrismDataConfig:
    benchmark: Literal["libero", "calvin"] = "libero"
    dataset_type: str = "simulation"
    dataset_config_path: str | None = None
    cache_manifest: str | None = None


@dataclass
class PrismTrainingConfig:
    stage: Literal["warmup", "stage1", "stage2", "eval", "serve", "cache"] = "stage1"
    batch_size: int = 1
    lr: float = 1e-5
    max_steps: int = 1


@dataclass
class PrismModelConfig:
    architecture_version: str = "legacy_bridge_himem"
    action_horizon: int = 32
    replan_stride: int = 16
    hidden_dim: int = 896
    action_dim: int = 7
    vlm_layers: Sequence[int | str] = field(default_factory=lambda: (3, 6, 9, 12))


@dataclass
class PrismConfig:
    model: PrismModelConfig = field(default_factory=PrismModelConfig)
    data: PrismDataConfig = field(default_factory=PrismDataConfig)
    training: PrismTrainingConfig = field(default_factory=PrismTrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> PrismConfig:
    loaded = load_training_config(path)
    raw = dict(loaded)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        raw[key] = _parse_override_value(value)
    raw.setdefault("_explicit_config_keys", sorted(key for key in raw if not str(key).startswith("_")))
    benchmark = str(raw.get("benchmark") or raw.get("dataset") or raw.get("suite") or "libero").lower()
    if "calvin" in benchmark:
        benchmark = "calvin"
    elif "libero" in benchmark:
        benchmark = "libero"
    else:
        raise ValueError(f"Unsupported benchmark {benchmark!r}; PrismVLA targets LIBERO and CALVIN")
    stage = str(raw.get("stage") or raw.get("training_stage") or "stage1").lower()
    if stage not in {"warmup", "stage1", "stage2", "eval", "serve", "cache"}:
        raise ValueError(f"Unsupported training stage {stage!r}")
    model = PrismModelConfig(
        architecture_version=str(raw.get("architecture_version", "legacy_bridge_himem")),
        action_horizon=int(raw.get("action_horizon", raw.get("horizon", 32))),
        replan_stride=int(raw.get("replan_stride", raw.get("progress_planner_replan_stride", 16))),
        hidden_dim=int(raw.get("hidden_dim", raw.get("embed_dim", 896))),
        action_dim=int(raw.get("per_action_dim", raw.get("action_dim", 7))),
    )
    data = PrismDataConfig(
        benchmark=benchmark,  # type: ignore[arg-type]
        dataset_type=str(raw.get("dataset_type", "simulation")),
        dataset_config_path=raw.get("dataset_config_path"),
        cache_manifest=raw.get("cache_manifest"),
    )
    training = PrismTrainingConfig(
        stage=stage,  # type: ignore[arg-type]
        batch_size=int(raw.get("batch_size", 1)),
        lr=float(raw.get("lr", 1e-5)),
        max_steps=int(raw.get("max_steps", 1)),
    )
    runtime = RuntimeConfig(seed=int(raw.get("seed") or 42), device=str(raw.get("device", "cuda")))
    return PrismConfig(model=model, data=data, training=training, runtime=runtime, raw=raw)


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
