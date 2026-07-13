"""Model-neutral contracts for benchmark data and VLA training samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from prism.schema import PolicyInput


NormalizationMode = Literal["q01_q99", "identity"]
TemporalSemantics = Literal["delta", "absolute"]
SourceEncoding = Literal[
    "continuous",
    "open_01",
    "signed_open_positive",
    "signed_open_negative",
    "constant_zero",
]
LanguageKind = Literal["direct_text", "task_index"]

NORMALIZATION_MODES = frozenset({"q01_q99", "identity"})
TEMPORAL_SEMANTICS = frozenset({"delta", "absolute"})
SOURCE_ENCODINGS = frozenset(
    {
        "continuous",
        "open_01",
        "signed_open_positive",
        "signed_open_negative",
        "constant_zero",
    }
)
LANGUAGE_KINDS = frozenset({"direct_text", "task_index"})
LEROBOT_STORAGE_FORMAT = "lerobot-v2.1"


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_plain_non_negative_int(value: object, *, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True)
class FeatureSlice:
    """Map one canonical feature to a half-open slice of a physical field."""

    name: str
    source_key: str
    start: int
    end: int
    normalization: NormalizationMode
    temporal_semantics: TemporalSemantics
    source_encoding: SourceEncoding

    def __post_init__(self) -> None:
        self.validate()

    @property
    def width(self) -> int:
        return self.end - self.start

    @property
    def source_slice(self) -> slice:
        return slice(self.start, self.end)

    def validate(self) -> None:
        _validate_non_empty_string(self.name, field_name="FeatureSlice.name")
        if "." not in self.name or any(not part for part in self.name.split(".")):
            raise ValueError(f"FeatureSlice.name must be a dotted canonical name, got {self.name!r}")
        _validate_non_empty_string(self.source_key, field_name="FeatureSlice.source_key")
        _validate_plain_non_negative_int(self.start, field_name="FeatureSlice.start")
        if type(self.end) is not int or self.end <= self.start:
            raise ValueError(
                f"FeatureSlice.end must be an integer greater than start, got start={self.start!r}, end={self.end!r}"
            )
        if self.normalization not in NORMALIZATION_MODES:
            raise ValueError(
                f"FeatureSlice.normalization must be one of {sorted(NORMALIZATION_MODES)}, got {self.normalization!r}"
            )
        if self.temporal_semantics not in TEMPORAL_SEMANTICS:
            raise ValueError(
                f"FeatureSlice.temporal_semantics must be one of {sorted(TEMPORAL_SEMANTICS)}, "
                f"got {self.temporal_semantics!r}"
            )
        if self.source_encoding not in SOURCE_ENCODINGS:
            raise ValueError(
                f"FeatureSlice.source_encoding must be one of {sorted(SOURCE_ENCODINGS)}, got {self.source_encoding!r}"
            )
        if self.normalization == "q01_q99" and self.source_encoding != "continuous":
            raise ValueError("q01_q99 normalization is only valid for continuous source values")
        if self.source_encoding != "continuous":
            if self.width != 1:
                raise ValueError("a non-continuous source encoding must select exactly one value")
            if self.normalization != "identity":
                raise ValueError("a non-continuous source encoding must use identity normalization")
            if self.temporal_semantics != "absolute":
                raise ValueError("a non-continuous source encoding must use absolute temporal semantics")


@dataclass(frozen=True)
class ViewSpec:
    """Map one ordered canonical camera view to its physical source key."""

    name: str
    source_key: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_non_empty_string(self.name, field_name="ViewSpec.name")
        _validate_non_empty_string(self.source_key, field_name="ViewSpec.source_key")


@dataclass(frozen=True)
class LanguageSpec:
    """Describe how an instruction is obtained from a physical sample."""

    source_key: str
    kind: LanguageKind

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_non_empty_string(self.source_key, field_name="LanguageSpec.source_key")
        if self.kind not in LANGUAGE_KINDS:
            raise ValueError(f"LanguageSpec.kind must be one of {sorted(LANGUAGE_KINDS)}, got {self.kind!r}")


@dataclass(frozen=True)
class DataSpec:
    """Static mapping from one benchmark's physical schema to canonical fields."""

    name: str
    benchmark: str
    robot_key: str
    storage_format: str
    views: tuple[ViewSpec, ...]
    state: tuple[FeatureSlice, ...]
    action: tuple[FeatureSlice, ...]
    language: LanguageSpec

    def __post_init__(self) -> None:
        self.validate()

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(view.name for view in self.views)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.state)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.action)

    @property
    def state_dim(self) -> int:
        return sum(feature.width for feature in self.state)

    @property
    def action_dim(self) -> int:
        return sum(feature.width for feature in self.action)

    def validate(self) -> None:
        _validate_non_empty_string(self.name, field_name="DataSpec.name")
        _validate_non_empty_string(self.benchmark, field_name="DataSpec.benchmark")
        _validate_non_empty_string(self.robot_key, field_name="DataSpec.robot_key")
        if self.storage_format != LEROBOT_STORAGE_FORMAT:
            raise ValueError(f"DataSpec.storage_format must be {LEROBOT_STORAGE_FORMAT!r}, got {self.storage_format!r}")
        self._validate_tuple(self.views, field_name="views", item_type=ViewSpec)
        self._validate_tuple(self.state, field_name="state", item_type=FeatureSlice)
        self._validate_tuple(self.action, field_name="action", item_type=FeatureSlice)
        if not isinstance(self.language, LanguageSpec):
            raise TypeError(f"DataSpec.language must be LanguageSpec, got {type(self.language).__name__}")

        view_names = [view.name for view in self.views]
        self._validate_unique(view_names, field_name="view canonical names")
        self._validate_unique(
            [view.source_key for view in self.views],
            field_name="view source keys",
        )

        canonical_names = [feature.name for feature in (*self.state, *self.action)]
        self._validate_unique(canonical_names, field_name="feature canonical names")
        self._validate_feature_group(self.state, prefix="state.", field_name="state")
        self._validate_feature_group(self.action, prefix="action.", field_name="action")

    @staticmethod
    def _validate_tuple(value: object, *, field_name: str, item_type: type[object]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"DataSpec.{field_name} must be an explicit tuple")
        if not value:
            raise ValueError(f"DataSpec.{field_name} must contain at least one item")
        invalid = [type(item).__name__ for item in value if not isinstance(item, item_type)]
        if invalid:
            raise TypeError(
                f"DataSpec.{field_name} entries must be {item_type.__name__}, got invalid entries {invalid}"
            )

    @staticmethod
    def _validate_unique(values: list[str], *, field_name: str) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        if duplicates:
            raise ValueError(f"DataSpec {field_name} must be unique, duplicates: {duplicates}")

    @staticmethod
    def _validate_feature_group(
        features: tuple[FeatureSlice, ...],
        *,
        prefix: str,
        field_name: str,
    ) -> None:
        ranges_by_source: dict[str, list[tuple[int, int, str]]] = {}
        for feature in features:
            if not feature.name.startswith(prefix):
                raise ValueError(
                    f"DataSpec.{field_name} canonical name must start with {prefix!r}, got {feature.name!r}"
                )
            source_ranges = ranges_by_source.setdefault(feature.source_key, [])
            for start, end, name in source_ranges:
                if feature.start < end and start < feature.end:
                    raise ValueError(
                        f"DataSpec.{field_name} source slices overlap for {feature.source_key!r}: "
                        f"{name!r} [{start}, {end}) and {feature.name!r} "
                        f"[{feature.start}, {feature.end})"
                    )
            source_ranges.append((feature.start, feature.end, feature.name))


@dataclass(frozen=True)
class VLASample:
    """One normalized model-facing VLA training sample."""

    policy_input: PolicyInput
    dataset_name: str
    statistics_group: str
    episode_index: int
    frame_index: int
    target_actions: np.ndarray
    action_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.policy_input, PolicyInput):
            raise TypeError(f"policy_input must be PolicyInput, got {type(self.policy_input).__name__}")
        _validate_non_empty_string(self.dataset_name, field_name="VLASample.dataset_name")
        _validate_non_empty_string(self.statistics_group, field_name="VLASample.statistics_group")
        _validate_plain_non_negative_int(self.episode_index, field_name="VLASample.episode_index")
        _validate_plain_non_negative_int(self.frame_index, field_name="VLASample.frame_index")
        if not isinstance(self.target_actions, np.ndarray):
            raise TypeError("VLASample.target_actions must be a numpy array")
        if self.target_actions.ndim != 2:
            raise ValueError(f"VLASample.target_actions must have shape [H, A], got {self.target_actions.shape}")
        if not np.issubdtype(self.target_actions.dtype, np.floating):
            raise TypeError("VLASample.target_actions must have a floating dtype")
        if self.target_actions.shape[1] != self.policy_input.action_dim:
            raise ValueError(
                "VLASample.target_actions action dimension must match PolicyInput.action_dim, "
                f"got {self.target_actions.shape[1]} and {self.policy_input.action_dim}"
            )
        if not np.isfinite(self.target_actions).all():
            raise ValueError("VLASample.target_actions must contain only finite values")
        if not isinstance(self.action_valid_mask, np.ndarray):
            raise TypeError("VLASample.action_valid_mask must be a numpy array")
        if self.action_valid_mask.dtype != np.bool_ or self.action_valid_mask.ndim != 1:
            raise ValueError(
                "VLASample.action_valid_mask must be a boolean array with shape [H], "
                f"got dtype={self.action_valid_mask.dtype}, shape={self.action_valid_mask.shape}"
            )
        if self.action_valid_mask.shape[0] != self.target_actions.shape[0]:
            raise ValueError(
                "VLASample.action_valid_mask length must match the action horizon, "
                f"got {self.action_valid_mask.shape[0]} and {self.target_actions.shape[0]}"
            )
        if np.any(self.target_actions[~self.action_valid_mask] != 0):
            raise ValueError("VLASample.target_actions must be zero at invalid padded time steps")


__all__ = [
    "DataSpec",
    "FeatureSlice",
    "LANGUAGE_KINDS",
    "LEROBOT_STORAGE_FORMAT",
    "LanguageKind",
    "LanguageSpec",
    "NORMALIZATION_MODES",
    "NormalizationMode",
    "SOURCE_ENCODINGS",
    "SourceEncoding",
    "TEMPORAL_SEMANTICS",
    "TemporalSemantics",
    "VLASample",
    "ViewSpec",
]
