"""Strict, resolved training configuration for the rebuilt PrismVLA policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import importlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from prism.data.benchmark_contracts import CALVIN_EVAL_SPLITS
from prism.data.benchmark_contracts import CALVIN_STATISTICS_GROUP
from prism.data.benchmark_contracts import CALVIN_TRAIN_SPLITS
from prism.data.benchmark_contracts import LIBERO_DATASET_NAMES
from prism.data.benchmark_contracts import LIBERO_STATISTICS_GROUP
from prism.data.benchmark_contracts import (
    validate_benchmark_data_contract as _validate_shared_benchmark_data_contract,
)
from prism.data.normalization import DataSpecNormalizer
from prism.data.normalization import GRIPPER_SOURCE_ENCODINGS
from prism.data.normalization import canonical_sha256
from prism.data.normalization import load_statistics
from prism.data.schema import DataSpec
from prism.models.config import PrismArchitectureConfig
from prism.models.config import load_architecture_config


TRAIN_CONFIG_SNAPSHOT_FORMAT = "prism-resolved-train-config-v2"
ORDERED_VIEW_NAMES = ("primary", "wrist")
STATE_DIM = 8
ACTION_DIM = 7
ACTION_MOTION_DIM = 6
ACTION_OBJECTIVE = "direct_masked_l1"
OPTIMIZATION_GROUP_NAMES = (
    "language_model",
    "vision_encoder",
    "action_queries",
    "history_qformer",
    "action_head",
)


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    name: str
    output_dir: Path
    seed: int


@dataclass(frozen=True)
class ResolvedModelConfig:
    architecture_config: Path
    architecture: PrismArchitectureConfig
    architecture_sha256: str


@dataclass(frozen=True)
class ResolvedDatasetConfig:
    name: str
    path: Path
    weight: float
    splits: tuple[str, ...] | None


@dataclass(frozen=True)
class ResolvedNormalizationConfig:
    group: str
    statistics_path: Path
    statistics: Mapping[str, Any]
    content_sha256: str


@dataclass(frozen=True)
class ResolvedOptimizationGroupConfig:
    trainable: bool
    learning_rate: float | None
    weight_decay: float | None


@dataclass(frozen=True)
class ResolvedOptimizationConfig:
    optimizer: str
    beta1: float
    beta2: float
    epsilon: float
    no_decay_rule: str
    language_model: ResolvedOptimizationGroupConfig
    vision_encoder: ResolvedOptimizationGroupConfig
    action_queries: ResolvedOptimizationGroupConfig
    history_qformer: ResolvedOptimizationGroupConfig
    action_head: ResolvedOptimizationGroupConfig

    def named_groups(self) -> tuple[tuple[str, ResolvedOptimizationGroupConfig], ...]:
        return tuple((name, getattr(self, name)) for name in OPTIMIZATION_GROUP_NAMES)


@dataclass(frozen=True)
class ResolvedLoaderConfig:
    global_samples_per_epoch: int
    batch_size_per_rank: int
    num_workers: int
    preprocessing_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool


@dataclass(frozen=True)
class ResolvedDataConfig:
    spec_reference: str
    spec: DataSpec
    root: Path
    anchor_stride: int
    include_tail: bool
    datasets: tuple[ResolvedDatasetConfig, ...]
    normalization: ResolvedNormalizationConfig
    loader: ResolvedLoaderConfig
    train_splits: tuple[str, ...] | None
    eval_splits: tuple[str, ...] | None


@dataclass(frozen=True)
class ResolvedTrainerConfig:
    max_steps: int
    gradient_accumulation_steps: int
    mixed_precision: str
    scheduler: str
    warmup_steps: int
    max_grad_norm: float
    log_interval: int
    save_interval: int


@dataclass(frozen=True)
class TemporalTrainingContract:
    """Temporal values derived only from the resolved architecture."""

    action_horizon: int
    replan_stride: int
    history_capture_offsets: tuple[int, ...]
    history_step_ages: tuple[int, ...]
    num_history_frames: int
    num_ordered_views: int


@dataclass(frozen=True)
class ResolvedTrainConfig:
    source_path: Path
    project_root: Path
    experiment: ResolvedExperimentConfig
    model: ResolvedModelConfig
    data: ResolvedDataConfig
    optimization: ResolvedOptimizationConfig
    trainer: ResolvedTrainerConfig
    temporal: TemporalTrainingContract

    def checkpoint_snapshot(self) -> dict[str, Any]:
        """Return a complete JSON-serializable checkpoint configuration snapshot."""

        return build_checkpoint_snapshot(self)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_train_config(
    path: str | Path,
    *,
    project_root: str | Path,
) -> ResolvedTrainConfig:
    """Load, resolve, and cross-validate one training YAML without hidden defaults."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project_root is not a directory: {root}")
    source_path = _resolve_config_path(path, project_root=root)
    raw = _load_yaml(source_path)

    top = _strict_mapping(
        raw,
        label="training config",
        required={"experiment", "model", "data", "optimization", "trainer"},
    )
    experiment_raw = _strict_mapping(
        top["experiment"],
        label="experiment",
        required={"name", "output_dir", "seed"},
    )
    model_raw = _strict_mapping(
        top["model"],
        label="model",
        required={"architecture_config"},
    )
    data_raw = _strict_mapping(
        top["data"],
        label="data",
        required={
            "spec",
            "root",
            "anchor_stride",
            "include_tail",
            "datasets",
            "normalization",
            "loader",
        },
        optional={"train_splits", "eval_splits"},
    )
    normalization_raw = _strict_mapping(
        data_raw["normalization"],
        label="data.normalization",
        required={"group", "statistics_path"},
    )
    loader_raw = _strict_mapping(
        data_raw["loader"],
        label="data.loader",
        required={
            "global_samples_per_epoch",
            "batch_size_per_rank",
            "num_workers",
            "preprocessing_workers",
            "pin_memory",
            "persistent_workers",
            "drop_last",
        },
    )
    optimization_raw = _strict_mapping(
        top["optimization"],
        label="optimization",
        required={
            "optimizer",
            "beta1",
            "beta2",
            "epsilon",
            "no_decay_rule",
            *OPTIMIZATION_GROUP_NAMES,
        },
    )
    trainer_raw = _strict_mapping(
        top["trainer"],
        label="trainer",
        required={
            "max_steps",
            "gradient_accumulation_steps",
            "mixed_precision",
            "scheduler",
            "warmup_steps",
            "max_grad_norm",
            "log_interval",
            "save_interval",
        },
    )
    dataset_rows = _strict_dataset_rows(data_raw["datasets"])

    experiment = ResolvedExperimentConfig(
        name=_text(experiment_raw["name"], "experiment.name"),
        output_dir=_resolve_declared_path(
            experiment_raw["output_dir"],
            base=root,
            label="experiment.output_dir",
        ),
        seed=_non_negative_int(experiment_raw["seed"], "experiment.seed"),
    )

    architecture_path = _resolve_declared_path(
        model_raw["architecture_config"],
        base=root,
        label="model.architecture_config",
    )
    if not architecture_path.is_file():
        raise FileNotFoundError(f"model.architecture_config is not a file: {architecture_path}")
    architecture = load_architecture_config(architecture_path)
    try:
        architecture.validate_for_policy()
    except ValueError as exc:
        raise ValueError(
            f"model.architecture_config must resolve every accepted policy dimension before training: {exc}"
        ) from exc
    model = ResolvedModelConfig(
        architecture_config=architecture_path,
        architecture=architecture,
        architecture_sha256=canonical_sha256(architecture),
    )

    spec_reference = _text(data_raw["spec"], "data.spec")
    data_spec = _import_data_spec(spec_reference)
    temporal = _derive_and_validate_contract(data_spec, architecture)

    data_root = _resolve_declared_path(
        data_raw["root"],
        base=root,
        label="data.root",
    )
    if not data_root.is_dir():
        raise FileNotFoundError(f"data.root is not a directory: {data_root}")
    datasets = tuple(_resolve_dataset(row, data_root=data_root, index=index) for index, row in enumerate(dataset_rows))
    _validate_dataset_names(datasets)

    train_splits = _optional_string_sequence(
        data_raw.get("train_splits"),
        "data.train_splits",
    )
    eval_splits = _optional_string_sequence(
        data_raw.get("eval_splits"),
        "data.eval_splits",
    )
    group = _text(normalization_raw["group"], "data.normalization.group")
    expected_provenance = _validate_benchmark_data_contract(
        data_spec,
        datasets=datasets,
        group=group,
        train_splits=train_splits,
        eval_splits=eval_splits,
    )

    loader = ResolvedLoaderConfig(
        global_samples_per_epoch=_positive_int(
            loader_raw["global_samples_per_epoch"],
            "data.loader.global_samples_per_epoch",
        ),
        batch_size_per_rank=_positive_int(
            loader_raw["batch_size_per_rank"],
            "data.loader.batch_size_per_rank",
        ),
        num_workers=_non_negative_int(
            loader_raw["num_workers"],
            "data.loader.num_workers",
        ),
        preprocessing_workers=_zero_or_one(
            loader_raw["preprocessing_workers"],
            "data.loader.preprocessing_workers",
        ),
        pin_memory=_boolean(loader_raw["pin_memory"], "data.loader.pin_memory"),
        persistent_workers=_boolean(
            loader_raw["persistent_workers"],
            "data.loader.persistent_workers",
        ),
        drop_last=_boolean(loader_raw["drop_last"], "data.loader.drop_last"),
    )
    if loader.persistent_workers:
        raise ValueError(
            "data.loader.persistent_workers must be false in Phase 1 because "
            "dataset epochs are propagated before fresh worker iterators"
        )

    statistics_path = _resolve_declared_path(
        normalization_raw["statistics_path"],
        base=root,
        label="data.normalization.statistics_path",
    )
    if not statistics_path.is_file():
        raise FileNotFoundError(f"data.normalization.statistics_path is not a file: {statistics_path}")
    dataset_names = tuple(dataset.name for dataset in datasets)
    statistics = load_statistics(
        statistics_path,
        group=group,
        expected_schema_hash=canonical_sha256(data_spec),
        expected_robot_key=data_spec.robot_key,
        expected_datasets=dataset_names,
        expected_provenance=expected_provenance,
    )
    DataSpecNormalizer(
        data_spec=data_spec,
        statistics=statistics,
        statistics_group=group,
    )
    frozen_statistics = _freeze_json(statistics)
    if not isinstance(frozen_statistics, Mapping):
        raise TypeError("loaded statistics artifact must be a mapping")
    normalization = ResolvedNormalizationConfig(
        group=group,
        statistics_path=statistics_path,
        statistics=frozen_statistics,
        content_sha256=str(statistics["content_sha256"]),
    )

    optimization_groups = {
        name: _resolve_optimization_group(
            optimization_raw[name],
            label=f"optimization.{name}",
        )
        for name in OPTIMIZATION_GROUP_NAMES
    }
    if not any(group.trainable for group in optimization_groups.values()):
        raise ValueError("optimization must leave at least one parameter group trainable")
    optimizer_name = _text(optimization_raw["optimizer"], "optimization.optimizer")
    if optimizer_name != "adamw":
        raise ValueError("optimization.optimizer must be exactly 'adamw'")
    no_decay_rule = _text(optimization_raw["no_decay_rule"], "optimization.no_decay_rule")
    if no_decay_rule != "bias_and_low_dimensional":
        raise ValueError("optimization.no_decay_rule must be exactly 'bias_and_low_dimensional'")
    optimization = ResolvedOptimizationConfig(
        optimizer=optimizer_name,
        beta1=_unit_interval(optimization_raw["beta1"], "optimization.beta1"),
        beta2=_unit_interval(optimization_raw["beta2"], "optimization.beta2"),
        epsilon=_positive_float(optimization_raw["epsilon"], "optimization.epsilon"),
        no_decay_rule=no_decay_rule,
        **optimization_groups,
    )

    trainer = ResolvedTrainerConfig(
        max_steps=_positive_int(trainer_raw["max_steps"], "trainer.max_steps"),
        gradient_accumulation_steps=_positive_int(
            trainer_raw["gradient_accumulation_steps"],
            "trainer.gradient_accumulation_steps",
        ),
        mixed_precision=_text(
            trainer_raw["mixed_precision"],
            "trainer.mixed_precision",
        ),
        scheduler=_text(trainer_raw["scheduler"], "trainer.scheduler"),
        warmup_steps=_non_negative_int(
            trainer_raw["warmup_steps"],
            "trainer.warmup_steps",
        ),
        max_grad_norm=_positive_float(
            trainer_raw["max_grad_norm"],
            "trainer.max_grad_norm",
        ),
        log_interval=_positive_int(
            trainer_raw["log_interval"],
            "trainer.log_interval",
        ),
        save_interval=_positive_int(
            trainer_raw["save_interval"],
            "trainer.save_interval",
        ),
    )
    if trainer.warmup_steps > trainer.max_steps:
        raise ValueError(
            f"trainer.warmup_steps must not exceed trainer.max_steps, got {trainer.warmup_steps} > {trainer.max_steps}"
        )
    if trainer.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("trainer.mixed_precision must be one of: no, fp16, bf16")
    if trainer.scheduler != "linear_warmup_decay":
        raise ValueError("trainer.scheduler must be exactly 'linear_warmup_decay'")

    return ResolvedTrainConfig(
        source_path=source_path,
        project_root=root,
        experiment=experiment,
        model=model,
        data=ResolvedDataConfig(
            spec_reference=spec_reference,
            spec=data_spec,
            root=data_root,
            anchor_stride=_positive_int(
                data_raw["anchor_stride"],
                "data.anchor_stride",
            ),
            include_tail=_boolean(
                data_raw["include_tail"],
                "data.include_tail",
            ),
            datasets=datasets,
            normalization=normalization,
            loader=loader,
            train_splits=train_splits,
            eval_splits=eval_splits,
        ),
        optimization=optimization,
        trainer=trainer,
        temporal=temporal,
    )


def build_checkpoint_snapshot(config: ResolvedTrainConfig) -> dict[str, Any]:
    """Build a self-contained, serializable snapshot for checkpoint metadata."""

    if not isinstance(config, ResolvedTrainConfig):
        raise TypeError(f"config must be a ResolvedTrainConfig, got {type(config).__name__}")
    data_spec_payload = asdict(config.data.spec)
    architecture_payload = asdict(config.model.architecture)
    statistics_payload = _thaw_json(config.data.normalization.statistics)
    snapshot: dict[str, Any] = {
        "format": TRAIN_CONFIG_SNAPSHOT_FORMAT,
        "source_config": _snapshot_path(config.source_path, config.project_root),
        "experiment": {
            "name": config.experiment.name,
            "output_dir": _snapshot_path(
                config.experiment.output_dir,
                config.project_root,
            ),
            "seed": config.experiment.seed,
        },
        "model": {
            "architecture_config": _snapshot_path(
                config.model.architecture_config,
                config.project_root,
            ),
            "architecture_sha256": config.model.architecture_sha256,
            "architecture": architecture_payload,
        },
        "data": {
            "spec": config.data.spec_reference,
            "data_spec_sha256": canonical_sha256(config.data.spec),
            "data_spec": data_spec_payload,
            "root": _snapshot_path(config.data.root, config.project_root),
            "anchor_stride": config.data.anchor_stride,
            "include_tail": config.data.include_tail,
            "datasets": [
                {
                    "name": dataset.name,
                    "path": _snapshot_path(dataset.path, config.project_root),
                    "weight": dataset.weight,
                    "splits": None if dataset.splits is None else list(dataset.splits),
                }
                for dataset in config.data.datasets
            ],
            "normalization": {
                "group": config.data.normalization.group,
                "statistics_path": _snapshot_path(
                    config.data.normalization.statistics_path,
                    config.project_root,
                ),
                "content_sha256": config.data.normalization.content_sha256,
                "statistics": statistics_payload,
            },
            "loader": asdict(config.data.loader),
            "train_splits": None if config.data.train_splits is None else list(config.data.train_splits),
            "eval_splits": None if config.data.eval_splits is None else list(config.data.eval_splits),
        },
        "optimization": asdict(config.optimization),
        "trainer": asdict(config.trainer),
        "derived": {
            "temporal_contract": asdict(config.temporal),
            "source": "model.architecture.temporal",
        },
    }
    json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )
    return snapshot


def _strict_dataset_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise TypeError("data.datasets must be a non-empty YAML sequence")
    return tuple(
        _strict_mapping(
            row,
            label=f"data.datasets[{index}]",
            required={"name", "path", "weight"},
            optional={"splits"},
        )
        for index, row in enumerate(value)
    )


def _resolve_dataset(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    index: int,
) -> ResolvedDatasetConfig:
    label = f"data.datasets[{index}]"
    path = _resolve_declared_path(
        row["path"],
        base=data_root,
        label=f"{label}.path",
    )
    try:
        path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"{label}.path must remain inside data.root after resolution: {path}") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"{label}.path is not a directory: {path}")
    return ResolvedDatasetConfig(
        name=_text(row["name"], f"{label}.name"),
        path=path,
        weight=_positive_float(row["weight"], f"{label}.weight"),
        splits=_optional_string_sequence(row.get("splits"), f"{label}.splits"),
    )


def _validate_dataset_names(
    datasets: Sequence[ResolvedDatasetConfig],
) -> None:
    names = [dataset.name for dataset in datasets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"data.datasets names must be unique, duplicates: {duplicates}")


def _resolve_optimization_group(
    value: Any,
    *,
    label: str,
) -> ResolvedOptimizationGroupConfig:
    raw = _strict_mapping(
        value,
        label=label,
        required={"trainable", "learning_rate", "weight_decay"},
    )
    trainable = _boolean(raw["trainable"], f"{label}.trainable")
    learning_rate = _optional_positive_float(raw["learning_rate"], f"{label}.learning_rate")
    weight_decay = _optional_non_negative_float(raw["weight_decay"], f"{label}.weight_decay")
    if trainable and (learning_rate is None or weight_decay is None):
        raise ValueError(f"{label} must set learning_rate and weight_decay when trainable")
    if not trainable and (learning_rate is not None or weight_decay is not None):
        raise ValueError(f"{label} must set learning_rate and weight_decay to null when frozen")
    return ResolvedOptimizationGroupConfig(
        trainable=trainable,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def _validate_benchmark_data_contract(
    data_spec: DataSpec,
    *,
    datasets: tuple[ResolvedDatasetConfig, ...],
    group: str,
    train_splits: tuple[str, ...] | None,
    eval_splits: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    return _validate_shared_benchmark_data_contract(
        benchmark=data_spec.benchmark,
        group=group,
        dataset_names=tuple(dataset.name for dataset in datasets),
        dataset_path_names=tuple(dataset.path.name for dataset in datasets),
        dataset_splits=tuple(dataset.splits for dataset in datasets),
        train_splits=train_splits,
        eval_splits=eval_splits,
    )


def _derive_and_validate_contract(
    data_spec: DataSpec,
    architecture: PrismArchitectureConfig,
) -> TemporalTrainingContract:
    data_spec.validate()
    architecture.validate_for_policy()
    if data_spec.view_names != ORDERED_VIEW_NAMES:
        raise ValueError(
            f"DataSpec must expose exactly two ordered views {ORDERED_VIEW_NAMES!r}, got {data_spec.view_names!r}"
        )
    if data_spec.state_dim != STATE_DIM:
        raise ValueError(f"DataSpec state dimension must be {STATE_DIM}, got {data_spec.state_dim}")
    if data_spec.action_dim != ACTION_DIM:
        raise ValueError(f"DataSpec action dimension must be {ACTION_DIM}, got {data_spec.action_dim}")
    if len(data_spec.action) != ACTION_DIM or any(feature.width != 1 for feature in data_spec.action):
        raise ValueError("DataSpec action must contain seven ordered scalar FeatureSlices")
    for index, feature in enumerate(data_spec.action[:ACTION_MOTION_DIM]):
        if (
            feature.normalization != "q01_q99"
            or feature.temporal_semantics != "delta"
            or feature.source_encoding != "continuous"
        ):
            raise ValueError(
                f"DataSpec action dimension {index} ({feature.name!r}) must use q01_q99 delta continuous semantics"
            )
    gripper = data_spec.action[-1]
    if (
        gripper.name != "action.gripper_open"
        or gripper.normalization != "identity"
        or gripper.temporal_semantics != "absolute"
        or gripper.source_encoding not in GRIPPER_SOURCE_ENCODINGS
    ):
        raise ValueError(
            "DataSpec final action dimension must be action.gripper_open with "
            "identity absolute semantics and a supported gripper source encoding"
        )
    action_head = architecture.action_head
    if action_head.objective != ACTION_OBJECTIVE:
        raise ValueError(f"architecture action objective must be {ACTION_OBJECTIVE!r}, got {action_head.objective!r}")
    if action_head.action_dim != ACTION_DIM or action_head.gripper_index != 6:
        raise ValueError("architecture action head must use action_dim=7 and gripper_index=6")

    history_offsets = tuple(architecture.temporal.history_capture_offsets)
    history_ages = tuple(architecture.temporal.history_step_ages)
    history_count = architecture.history.num_history_frames
    if len(history_offsets) != history_count or len(history_ages) != history_count:
        raise ValueError(
            "architecture history count mismatch: history.num_history_frames="
            f"{history_count}, capture_offsets={history_offsets}, "
            f"step_ages={history_ages}"
        )
    return TemporalTrainingContract(
        action_horizon=architecture.temporal.action_horizon,
        replan_stride=architecture.temporal.replan_stride,
        history_capture_offsets=history_offsets,
        history_step_ages=history_ages,
        num_history_frames=history_count,
        num_ordered_views=len(data_spec.views),
    )


def _import_data_spec(reference: str) -> DataSpec:
    if reference.count(":") != 1:
        raise ValueError(f"data.spec must use the exact module:object form, got {reference!r}")
    module_name, object_name = reference.split(":", 1)
    if (
        not module_name.startswith("experiments.")
        or any(not part.isidentifier() for part in module_name.split("."))
        or not object_name.isidentifier()
    ):
        raise ValueError(
            f"data.spec may import only an exact object from a trusted experiments.* module, got {reference!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(f"failed to import trusted DataSpec module {module_name!r}") from exc
    try:
        value = getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"trusted DataSpec module {module_name!r} has no exact object {object_name!r}") from exc
    if not isinstance(value, DataSpec):
        raise TypeError(f"data.spec {reference!r} must resolve to DataSpec, got {type(value).__name__}")
    value.validate()
    return value


def _resolve_config_path(path: str | Path, *, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"training config is not a file: {resolved}")
    return resolved


def _resolve_declared_path(
    value: Any,
    *,
    base: Path,
    label: str,
) -> Path:
    text = _text(value, label)
    declared = Path(text)
    if declared.is_absolute():
        raise ValueError(f"{label} must be relative to its explicit configuration root, got absolute path {text!r}")
    return (base / declared).resolve()


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid training YAML in {path}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid training YAML in {path}: {exc}") from exc


def _strict_mapping(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    mapping = dict(value)
    invalid_keys = [key for key in mapping if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError(f"{label} keys must be strings, got {invalid_keys!r}")
    allowed = required | (set() if optional is None else optional)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported keys: {unknown}")
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    return mapping


def _optional_string_sequence(
    value: Any,
    label: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a non-empty YAML sequence when present")
    parsed = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicate values: {list(parsed)!r}")
    return parsed


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string, got {value!r}")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def _zero_or_one(value: Any, label: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{label} must be 0 or 1, got {value!r}")
    return value


def _positive_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return parsed


def _non_negative_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")
    return parsed


def _optional_positive_float(value: Any, label: str) -> float | None:
    return None if value is None else _positive_float(value, label)


def _optional_non_negative_float(value: Any, label: str) -> float | None:
    return None if value is None else _non_negative_float(value, label)


def _unit_interval(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if not 0.0 <= parsed < 1.0:
        raise ValueError(f"{label} must be in [0, 1), got {value!r}")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return parsed


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean, got {value!r}")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"statistics content must contain only JSON values, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _snapshot_path(path: Path, project_root: Path) -> str:
    return Path(os.path.relpath(path, project_root)).as_posix()


__all__ = [
    "ACTION_OBJECTIVE",
    "CALVIN_EVAL_SPLITS",
    "CALVIN_STATISTICS_GROUP",
    "CALVIN_TRAIN_SPLITS",
    "LIBERO_DATASET_NAMES",
    "LIBERO_STATISTICS_GROUP",
    "ResolvedDataConfig",
    "ResolvedDatasetConfig",
    "ResolvedExperimentConfig",
    "ResolvedLoaderConfig",
    "ResolvedModelConfig",
    "ResolvedNormalizationConfig",
    "ResolvedOptimizationConfig",
    "ResolvedOptimizationGroupConfig",
    "ResolvedTrainConfig",
    "ResolvedTrainerConfig",
    "TemporalTrainingContract",
    "TRAIN_CONFIG_SNAPSHOT_FORMAT",
    "build_checkpoint_snapshot",
    "load_train_config",
]
