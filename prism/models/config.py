from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class Qwen35BackboneConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    num_hidden_layers: int = 16
    hidden_size: int = 1024
    num_action_queries: int = 48
    image_size: int = 384
    torch_dtype: str = "bfloat16"
    local_files_only: bool = False

    def validate(self) -> None:
        if self.num_hidden_layers != 16:
            raise ValueError("The accepted Qwen3.5 baseline requires exactly 16 retained layers")
        if self.hidden_size != 1024:
            raise ValueError("The accepted Qwen3.5-0.8B hidden size is 1024")
        if self.num_action_queries != 48:
            raise ValueError("The accepted baseline requires exactly 48 action queries")
        if self.image_size <= 0 or self.image_size % 32 != 0:
            raise ValueError("image_size must be positive and aligned to 32 pixels")
        if self.torch_dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Unsupported torch_dtype {self.torch_dtype!r}")


@dataclass(frozen=True)
class HistoryQFormerConfig:
    input_dim: int = 1024
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    num_memory_tokens: int = 24
    num_history_frames: int = 2
    max_relative_age: int = 8
    dropout: float = 0.0

    def validate(self) -> None:
        if self.input_dim != 1024 or self.hidden_size != 512:
            raise ValueError("The accepted history widths are input_dim=1024 and hidden_size=512")
        if self.num_layers != 2 or self.num_heads != 4:
            raise ValueError("The accepted History Q-Former uses 2 layers and 4 heads")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.num_memory_tokens != 24 or self.num_history_frames != 2:
            raise ValueError("The accepted history contract uses 2 frames and 24 memory tokens")
        if self.max_relative_age < 6:
            raise ValueError("max_relative_age must represent the accepted age 6 history frame")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TemporalContextConfig:
    action_horizon: int = 8
    replan_stride: int = 8
    history_capture_offsets: tuple[int, int] = (2, 5)

    @property
    def history_step_ages(self) -> tuple[int, int]:
        return tuple(self.replan_stride - offset for offset in self.history_capture_offsets)

    def validate(self) -> None:
        if self.action_horizon != 8 or self.replan_stride != 8:
            raise ValueError("The accepted runtime contract requires action_horizon=replan_stride=8")
        if self.history_capture_offsets != (2, 5):
            raise ValueError("The accepted sparse history capture offsets are (2, 5)")


@dataclass(frozen=True)
class DirectActionHeadConfig:
    objective: str = "direct_masked_l1"
    action_dim: int = 7
    gripper_index: int = 6
    gripper_threshold: float = 0.5
    action_hidden_size: int | None = None
    num_attention_heads: int | None = None
    ffn_ratio: int | None = None

    def validate(self) -> None:
        if self.objective != "direct_masked_l1":
            raise ValueError("The accepted action objective is direct_masked_l1")
        if self.action_dim != 7 or self.gripper_index != 6:
            raise ValueError("The accepted action contract is 7-dimensional with gripper at index 6")
        if self.gripper_threshold != 0.5:
            raise ValueError("The accepted canonical gripper threshold is exactly 0.5")
        for name, value in (
            ("action_hidden_size", self.action_hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("ffn_ratio", self.ffn_ratio),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when specified")
        if (
            self.action_hidden_size is not None
            and self.num_attention_heads is not None
            and self.action_hidden_size % self.num_attention_heads != 0
        ):
            raise ValueError("action_hidden_size must be divisible by num_attention_heads")

    def require_resolved(self) -> None:
        self.validate()
        missing = [
            name for name in ("action_hidden_size", "num_attention_heads", "ffn_ratio") if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"Action policy dimensions are not yet accepted in the architecture config: {missing}")


@dataclass(frozen=True)
class PrismArchitectureConfig:
    backbone: Qwen35BackboneConfig = field(default_factory=Qwen35BackboneConfig)
    history: HistoryQFormerConfig = field(default_factory=HistoryQFormerConfig)
    temporal: TemporalContextConfig = field(default_factory=TemporalContextConfig)
    action_head: DirectActionHeadConfig = field(default_factory=DirectActionHeadConfig)
    num_bridge_layers: int = 16
    memory_gate_init: float = 0.1

    def validate(self) -> None:
        self.backbone.validate()
        self.history.validate()
        self.temporal.validate()
        self.action_head.validate()
        if self.num_bridge_layers != self.backbone.num_hidden_layers:
            raise ValueError("Bridge depth must match retained Qwen depth")
        if self.memory_gate_init != 0.1:
            raise ValueError("The accepted memory gate initialization is 0.1")

    def validate_for_policy(self) -> None:
        self.validate()
        self.action_head.require_resolved()


def load_architecture_config(path: str | Path) -> PrismArchitectureConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Architecture config root must be a mapping: {config_path}")
    allowed = {"backbone", "history", "temporal", "action_head", "bridge"}
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        raise ValueError(f"Unsupported architecture config sections: {unknown}")

    backbone = Qwen35BackboneConfig(**_mapping(raw.get("backbone"), "backbone"))
    history = HistoryQFormerConfig(**_mapping(raw.get("history"), "history"))
    temporal_values = _mapping(raw.get("temporal"), "temporal")
    if "history_capture_offsets" in temporal_values:
        temporal_values["history_capture_offsets"] = tuple(temporal_values["history_capture_offsets"])
    temporal = TemporalContextConfig(**temporal_values)
    action_head = DirectActionHeadConfig(**_mapping(raw.get("action_head"), "action_head"))
    bridge = _mapping(raw.get("bridge"), "bridge")
    config = PrismArchitectureConfig(
        backbone=backbone,
        history=history,
        temporal=temporal,
        action_head=action_head,
        num_bridge_layers=int(bridge.get("num_layers", 16)),
        memory_gate_init=float(bridge.get("memory_gate_init", 0.1)),
    )
    extra_bridge = sorted(set(bridge) - {"num_layers", "memory_gate_init"})
    if extra_bridge:
        raise ValueError(f"Unsupported bridge config fields: {extra_bridge}")
    config.validate()
    return config


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)
