"""Versioned, model-neutral normalization statistics and transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

import numpy as np

from prism.data.schema import DataSpec, FeatureSlice


STATISTICS_FORMAT = "prism-normalization-v1"
NORMALIZED_MIN = -1.0
NORMALIZED_MAX = 1.0
Q01 = 0.01
Q99 = 0.99
NORMALIZATION_EPS = 1.0e-8

ACTION_DIM = 7
ACTION_CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5)
ACTION_GRIPPER_INDEX = 6
GRIPPER_SOURCE_ENCODINGS = frozenset({"open_01", "signed_open_positive", "signed_open_negative"})

NormalizationStatistics = Dict[str, Any]


@dataclass(frozen=True)
class DataSpecNormalizer:
    """Pickle-safe adapter from assembled DataSpec vectors to normalized values."""

    data_spec: DataSpec
    statistics: NormalizationStatistics
    statistics_group: str
    _group_statistics: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.data_spec, DataSpec):
            raise TypeError(f"data_spec must be DataSpec, got {type(self.data_spec).__name__}")
        self.data_spec.validate()
        artifact = _json_mapping(self.statistics, "statistics")
        group_name = _text(self.statistics_group, "statistics_group")
        validate_statistics(
            artifact,
            group=group_name,
            expected_schema_hash=canonical_sha256(self.data_spec),
            expected_robot_key=self.data_spec.robot_key,
        )
        group = artifact["groups"][group_name]
        if group["state"]["dim"] != self.data_spec.state_dim:
            raise ValueError(
                "statistics state dimension does not match DataSpec: "
                f"{group['state']['dim']} != {self.data_spec.state_dim}"
            )
        if group["action"]["dim"] != self.data_spec.action_dim:
            raise ValueError(
                "statistics action dimension does not match DataSpec: "
                f"{group['action']['dim']} != {self.data_spec.action_dim}"
            )
        expected_state_continuous = _normalization_indices(self.data_spec.state)
        expected_action_continuous = _normalization_indices(self.data_spec.action)
        if tuple(group["state"]["continuous_indices"]) != expected_state_continuous:
            raise ValueError("statistics state normalization indices do not match DataSpec")
        if tuple(group["action"]["continuous_indices"]) != expected_action_continuous:
            raise ValueError("statistics action normalization indices do not match DataSpec")
        gripper = self.data_spec.action[-1]
        if (
            gripper.name != "action.gripper_open"
            or gripper.width != 1
            or gripper.normalization != "identity"
            or gripper.temporal_semantics != "absolute"
            or gripper.source_encoding not in GRIPPER_SOURCE_ENCODINGS
        ):
            raise ValueError("DataSpec action must end with an absolute identity gripper_open")
        object.__setattr__(self, "statistics", artifact)
        object.__setattr__(self, "statistics_group", group_name)
        object.__setattr__(self, "_group_statistics", group)

    def normalize_state(self, raw_assembled: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Canonicalize an already assembled state vector, then normalize it."""

        canonical = canonicalize_assembled_features(raw_assembled, self.data_spec.state, valid_mask=valid_mask)
        return _normalize_state_validated(canonical, self._group_statistics, valid_mask=valid_mask)

    def normalize_action(self, raw_assembled: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Canonicalize assembled action rows, then normalize valid rows."""

        canonical = canonicalize_assembled_features(raw_assembled, self.data_spec.action, valid_mask=valid_mask)
        return _normalize_action_validated(canonical, self._group_statistics, valid_mask=valid_mask)


def canonical_sha256(value: Any) -> str:
    """Hash a value using deterministic canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize independently of mapping insertion order and whitespace."""

    return json.dumps(
        _to_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def statistics_content_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash artifact content without its self-referential content hash."""

    if not isinstance(artifact, Mapping):
        raise TypeError("statistics artifact must be a mapping")
    return canonical_sha256({key: value for key, value in artifact.items() if key != "content_sha256"})


def canonicalize_features(
    source_values: Mapping[str, np.ndarray],
    features: Sequence[FeatureSlice],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble ordered canonical features from physical source vectors.

    Continuous values are copied, constant_zero values are verified, and all
    supported gripper encodings are converted to open_01. Invalid padded rows
    are returned as zeros and are not inspected.
    """

    if not isinstance(source_values, Mapping):
        raise TypeError("source_values must be a mapping")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise ValueError("features must be a non-empty ordered sequence")

    leading_shape: tuple[int, ...] | None = None
    pieces: list[np.ndarray] = []
    row_mask: np.ndarray | None = None
    for feature in features:
        if feature.source_key not in source_values:
            raise KeyError(f"canonical feature {feature.name!r} requires missing source {feature.source_key!r}")
        source = _numeric_array(source_values[feature.source_key], f"source {feature.source_key!r}")
        if source.ndim < 1 or source.shape[-1] < feature.end:
            raise ValueError(
                f"source {feature.source_key!r} cannot provide slice "
                f"[{feature.start}, {feature.end}) from shape {source.shape}"
            )
        if leading_shape is None:
            leading_shape = source.shape[:-1]
            row_mask = _mask(valid_mask, leading_shape, "valid_mask")
        elif source.shape[:-1] != leading_shape:
            raise ValueError("all physical feature sources must share leading dimensions")

        raw = source[..., feature.start : feature.end]
        assert row_mask is not None
        pieces.append(_canonicalize_feature_piece(raw, feature, row_mask))
    return np.ascontiguousarray(np.concatenate(pieces, axis=-1), dtype=np.float32)


def canonicalize_assembled_features(
    raw_assembled: np.ndarray,
    features: Sequence[FeatureSlice],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Canonicalize a vector already concatenated in FeatureSlice order.

    Physical start/end offsets are deliberately not used here: storage has
    already applied them before assembling the vector.
    """

    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise ValueError("features must be a non-empty ordered sequence")
    dimension = sum(feature.width for feature in features)
    values, mask, shape = _transform_input(
        raw_assembled,
        valid_mask,
        expected_dim=dimension,
        label="raw_assembled",
    )
    output = np.zeros(values.shape, dtype=np.float32)
    cursor = 0
    for feature in features:
        end = cursor + feature.width
        output[:, cursor:end] = _canonicalize_feature_piece(values[:, cursor:end], feature, mask)
        cursor = end
    return np.ascontiguousarray(output.reshape(shape), dtype=np.float32)


def _canonicalize_feature_piece(
    raw: np.ndarray,
    feature: FeatureSlice,
    valid_mask: np.ndarray,
) -> np.ndarray:
    output = np.zeros(raw.shape, dtype=np.float32)
    selected = raw[valid_mask]
    if feature.source_encoding == "continuous":
        if not np.isfinite(selected).all():
            raise ValueError(f"valid values for {feature.name!r} must be finite")
        output[valid_mask] = selected.astype(np.float32)
    elif feature.source_encoding == "constant_zero":
        _strict_values(
            selected,
            allowed=(0.0,),
            label=f"{feature.name} constant_zero",
        )
    elif feature.source_encoding in GRIPPER_SOURCE_ENCODINGS:
        if feature.width != 1:
            raise ValueError(f"gripper feature {feature.name!r} must have width one")
        output[..., 0] = canonicalize_gripper(raw[..., 0], feature.source_encoding, valid_mask=valid_mask)
    else:
        raise ValueError(f"unsupported source encoding {feature.source_encoding!r} for {feature.name!r}")
    return output


def compute_statistics(
    state_values: np.ndarray,
    action_values: np.ndarray,
    *,
    group: str,
    robot_key: str,
    datasets: Sequence[str],
    schema_hash: str,
    provenance: Mapping[str, Any] | None = None,
    state_continuous_indices: Sequence[int] | None = None,
    state_valid_mask: np.ndarray | None = None,
    action_valid_mask: np.ndarray | None = None,
) -> NormalizationStatistics:
    """Compute one artifact group from canonical, unnormalized samples.

    Only rows selected by the validity masks enter statistics. Selected rows
    must be finite. Actions must already have the canonical seven-dimensional
    schema with absolute open_01 in dimension seven.
    """

    group_name = _text(group, "group")
    robot_name = _text(robot_key, "robot_key")
    dataset_names = _datasets(datasets)
    _sha(schema_hash, "schema_hash")
    provenance_value = _json_mapping({} if provenance is None else provenance, "provenance")

    states, state_dim = _valid_rows(state_values, state_valid_mask, "state_values")
    actions, action_dim = _valid_rows(action_values, action_valid_mask, "action_values")
    if action_dim != ACTION_DIM:
        raise ValueError(f"action_values must have dimension {ACTION_DIM}, got {action_dim}")
    _strict_values(
        actions[:, ACTION_GRIPPER_INDEX],
        allowed=(0.0, 1.0),
        label="canonical action gripper",
    )

    state_continuous = (
        tuple(range(state_dim))
        if state_continuous_indices is None
        else _indices(
            state_continuous_indices,
            dimension=state_dim,
            label="state_continuous_indices",
        )
    )
    state_identity = tuple(index for index in range(state_dim) if index not in state_continuous)
    group_statistics = {
        "robot_key": robot_name,
        "datasets": list(dataset_names),
        "provenance": provenance_value,
        "state": {
            "dim": state_dim,
            "method": "q01_q99",
            "clip": [NORMALIZED_MIN, NORMALIZED_MAX],
            "continuous_indices": list(state_continuous),
            "identity_indices": list(state_identity),
            **_quantiles(states, state_continuous),
            "count": int(states.shape[0]),
        },
        "action": {
            "dim": ACTION_DIM,
            "continuous_indices": list(ACTION_CONTINUOUS_INDICES),
            "continuous_method": "q01_q99",
            "continuous_clip": [NORMALIZED_MIN, NORMALIZED_MAX],
            **_quantiles(actions, ACTION_CONTINUOUS_INDICES),
            "gripper_index": ACTION_GRIPPER_INDEX,
            "gripper_method": "identity",
            "gripper_semantic": "open_01",
            "gripper_threshold": 0.5,
            "count": int(actions.shape[0]),
        },
        "schema_hash": schema_hash,
    }
    return build_statistics_artifact({group_name: group_statistics})


def build_statistics_artifact(
    groups: Mapping[str, Mapping[str, Any]],
) -> NormalizationStatistics:
    """Create a content-addressed artifact from precomputed groups."""

    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a non-empty mapping")
    normalized: dict[str, Any] = {}
    for name, statistics in groups.items():
        group_name = _text(name, "statistics group name")
        normalized[group_name] = _json_mapping(statistics, f"statistics group {group_name!r}")
    artifact: NormalizationStatistics = {
        "format": STATISTICS_FORMAT,
        "groups": normalized,
    }
    artifact["content_sha256"] = statistics_content_sha256(artifact)
    validate_statistics(artifact)
    return artifact


def validate_statistics(
    artifact: Mapping[str, Any],
    *,
    group: str | None = None,
    expected_schema_hash: str | None = None,
    expected_robot_key: str | None = None,
    expected_datasets: Sequence[str] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Validate integrity plus optional schema and provenance constraints."""

    root = _json_mapping(artifact, "statistics artifact")
    _keys(root, {"format", "groups", "content_sha256"}, "statistics artifact")
    if root["format"] != STATISTICS_FORMAT:
        raise ValueError(f"unsupported statistics format {root['format']!r}; expected {STATISTICS_FORMAT!r}")
    if not isinstance(root["groups"], dict) or not root["groups"]:
        raise ValueError("statistics groups must be a non-empty mapping")
    _sha(root["content_sha256"], "content_sha256")
    actual_hash = statistics_content_sha256(root)
    if root["content_sha256"] != actual_hash:
        raise ValueError(f"statistics content hash mismatch: stored {root['content_sha256']}, computed {actual_hash}")
    for name, statistics in root["groups"].items():
        _text(name, "statistics group name")
        _validate_group(name, statistics)

    expected = (
        expected_schema_hash,
        expected_robot_key,
        expected_datasets,
        expected_provenance,
    )
    if group is None:
        if any(value is not None for value in expected):
            raise ValueError("group is required with expected statistics metadata")
        return
    name = _text(group, "group")
    if name not in root["groups"]:
        raise KeyError(f"statistics group {name!r} is missing")
    selected = root["groups"][name]
    if expected_schema_hash is not None:
        _sha(expected_schema_hash, "expected_schema_hash")
        if selected["schema_hash"] != expected_schema_hash:
            raise ValueError(
                f"statistics schema hash mismatch for {name!r}: "
                f"expected {expected_schema_hash}, got {selected['schema_hash']}"
            )
    if expected_robot_key is not None and selected["robot_key"] != expected_robot_key:
        raise ValueError(
            f"statistics robot_key mismatch for {name!r}: "
            f"expected {expected_robot_key!r}, got {selected['robot_key']!r}"
        )
    if expected_datasets is not None:
        names = list(_datasets(expected_datasets))
        if selected["datasets"] != names:
            raise ValueError(
                f"statistics datasets mismatch for {name!r}: expected {names!r}, got {selected['datasets']!r}"
            )
    if expected_provenance is not None:
        provenance = _json_mapping(expected_provenance, "expected_provenance")
        if selected["provenance"] != provenance:
            raise ValueError(
                f"statistics provenance mismatch for {name!r}: expected {provenance!r}, got {selected['provenance']!r}"
            )


def save_statistics(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically save a validated artifact."""

    validate_statistics(artifact)
    root = _json_mapping(artifact, "statistics artifact")
    payload = (
        json.dumps(
            root,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_statistics(
    path: str | Path,
    *,
    group: str | None = None,
    expected_schema_hash: str | None = None,
    expected_robot_key: str | None = None,
    expected_datasets: Sequence[str] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> NormalizationStatistics:
    """Load an artifact and reject corruption or metadata drift."""

    source = Path(path).expanduser()
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid statistics JSON in {source}: {exc}") from exc
    validate_statistics(
        loaded,
        group=group,
        expected_schema_hash=expected_schema_hash,
        expected_robot_key=expected_robot_key,
        expected_datasets=expected_datasets,
        expected_provenance=expected_provenance,
    )
    return _json_mapping(loaded, "statistics artifact")


def normalize_state(
    state: np.ndarray,
    group_statistics: Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize configured state dimensions in float32."""

    group = _runtime_group(group_statistics)
    return _normalize_state_validated(state, group, valid_mask=valid_mask)


def _normalize_state_validated(
    state: np.ndarray,
    group: Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    section = group["state"]
    values, mask, shape = _transform_input(state, valid_mask, expected_dim=section["dim"], label="state")
    output = np.zeros(values.shape, dtype=np.float32)
    if mask.any():
        selected = values[mask].astype(np.float32, copy=True)
        continuous = np.asarray(section["continuous_indices"], dtype=np.int64)
        if continuous.size:
            selected[:, continuous] = _normalize_continuous(selected[:, continuous], section, clip_key="clip")
        output[mask] = selected
    return output.reshape(shape)


def normalize_action(
    actions: np.ndarray,
    group_statistics: Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize motion, preserve open_01, and zero padded actions."""

    group = _runtime_group(group_statistics)
    return _normalize_action_validated(actions, group, valid_mask=valid_mask)


def _normalize_action_validated(
    actions: np.ndarray,
    group: Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    section = group["action"]
    values, mask, shape = _transform_input(actions, valid_mask, expected_dim=section["dim"], label="actions")
    output = np.zeros(values.shape, dtype=np.float32)
    if mask.any():
        selected = values[mask].astype(np.float32, copy=True)
        continuous = np.asarray(section["continuous_indices"], dtype=np.int64)
        selected[:, continuous] = _normalize_continuous(selected[:, continuous], section, clip_key="continuous_clip")
        _strict_values(
            selected[:, section["gripper_index"]],
            allowed=(0.0, 1.0),
            label="canonical action gripper",
        )
        output[mask] = selected
    return output.reshape(shape)


def denormalize_action(
    normalized_actions: np.ndarray,
    group_statistics: Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Invert motion normalization without clipping model predictions."""

    group = _runtime_group(group_statistics)
    section = group["action"]
    values, mask, shape = _transform_input(
        normalized_actions,
        valid_mask,
        expected_dim=section["dim"],
        label="normalized_actions",
    )
    output = np.zeros(values.shape, dtype=np.float32)
    if mask.any():
        selected = values[mask].astype(np.float32, copy=True)
        continuous = np.asarray(section["continuous_indices"], dtype=np.int64)
        q01 = np.asarray(section["q01"], dtype=np.float32)
        q99 = np.asarray(section["q99"], dtype=np.float32)
        selected[:, continuous] = 0.5 * (selected[:, continuous] + 1.0) * (q99 - q01) + q01
        output[mask] = selected
    return output.reshape(shape)


def canonicalize_gripper(
    values: np.ndarray,
    source_encoding: str,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a strict source gripper encoding to canonical open_01."""

    if source_encoding not in GRIPPER_SOURCE_ENCODINGS:
        raise ValueError(
            f"unsupported gripper source encoding {source_encoding!r}; "
            f"expected one of {sorted(GRIPPER_SOURCE_ENCODINGS)}"
        )
    array = _numeric_array(values, "gripper values")
    mask = _mask(valid_mask, array.shape, "valid_mask")
    selected = array[mask]
    output = np.zeros(array.shape, dtype=np.float32)
    if source_encoding == "open_01":
        _strict_values(selected, allowed=(0.0, 1.0), label=source_encoding)
        output[mask] = selected.astype(np.float32)
    elif source_encoding == "signed_open_positive":
        _strict_values(selected, allowed=(-1.0, 1.0), label=source_encoding)
        output[mask] = ((selected + 1.0) * 0.5).astype(np.float32)
    else:
        _strict_values(selected, allowed=(-1.0, 1.0), label=source_encoding)
        output[mask] = ((1.0 - selected) * 0.5).astype(np.float32)
    return output


def decode_gripper_open(predictions: np.ndarray) -> np.ndarray:
    """Decode predictions with the strict rule prediction > 0.5."""

    values = _numeric_array(predictions, "gripper predictions")
    if not np.isfinite(values).all():
        raise ValueError("gripper predictions must contain only finite values")
    return (values > 0.5).astype(np.float32)


def gripper_open_to_environment(gripper_open: np.ndarray, benchmark: str) -> np.ndarray:
    """Map canonical commands to a benchmark's signed convention."""

    canonical = canonicalize_gripper(gripper_open, "open_01")
    benchmark_name = str(benchmark).lower()
    if benchmark_name == "libero":
        return (1.0 - 2.0 * canonical).astype(np.float32)
    if benchmark_name == "calvin":
        return (2.0 * canonical - 1.0).astype(np.float32)
    raise ValueError(f"unsupported benchmark gripper mapping {benchmark!r}")


def decode_gripper_for_environment(predictions: np.ndarray, benchmark: str) -> np.ndarray:
    """Threshold a model prediction and map it to an environment."""

    return gripper_open_to_environment(decode_gripper_open(predictions), benchmark)


def _quantiles(values: np.ndarray, indices: Sequence[int]) -> dict[str, Any]:
    selected = values[:, tuple(indices)]
    if selected.shape[1] == 0:
        return {
            "q01": [],
            "q99": [],
            "constant_mask": [],
            "clip_rate_low": [],
            "clip_rate_high": [],
        }
    low = np.quantile(selected, Q01, axis=0, method="linear").astype(np.float64)
    high = np.quantile(selected, Q99, axis=0, method="linear").astype(np.float64)
    return {
        "q01": low.tolist(),
        "q99": high.tolist(),
        "constant_mask": (high == low).tolist(),
        "clip_rate_low": np.mean(selected < low, axis=0, dtype=np.float64).tolist(),
        "clip_rate_high": np.mean(selected > high, axis=0, dtype=np.float64).tolist(),
    }


def _normalize_continuous(values: np.ndarray, statistics: Mapping[str, Any], *, clip_key: str) -> np.ndarray:
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    constant = np.asarray(statistics["constant_mask"], dtype=np.bool_)
    normalized = 2.0 * (values - low) / (high - low + np.float32(NORMALIZATION_EPS)) - 1.0
    clip_low, clip_high = statistics[clip_key]
    normalized = np.clip(normalized, np.float32(clip_low), np.float32(clip_high))
    normalized[:, constant] = 0.0
    return normalized.astype(np.float32, copy=False)


def _valid_rows(values: np.ndarray, valid_mask: np.ndarray | None, label: str) -> tuple[np.ndarray, int]:
    array = _numeric_array(values, label)
    if array.ndim < 2 or array.shape[-1] <= 0:
        raise ValueError(f"{label} must have shape [..., dim] with samples")
    mask = _mask(valid_mask, array.shape[:-1], f"{label} valid_mask")
    selected = array.reshape(-1, array.shape[-1])[mask.reshape(-1)].astype(np.float64, copy=False)
    if not selected.shape[0]:
        raise ValueError(f"{label} has no valid non-padding rows")
    if not np.isfinite(selected).all():
        raise ValueError(f"valid {label} rows must contain only finite values")
    return selected, int(array.shape[-1])


def _transform_input(
    values: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    expected_dim: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    array = _numeric_array(values, label)
    if array.ndim < 1 or array.shape[-1] != expected_dim:
        raise ValueError(f"{label} must end with dimension {expected_dim}, got {array.shape}")
    flat = array.reshape(-1, expected_dim)
    mask = _mask(valid_mask, array.shape[:-1], f"{label} valid_mask").reshape(-1)
    if not np.isfinite(flat[mask]).all():
        raise ValueError(f"valid {label} rows must contain only finite values")
    return flat, mask, tuple(array.shape)


def _validate_group(name: str, group: Any) -> None:
    if not isinstance(group, dict):
        raise TypeError(f"statistics group {name!r} must be a mapping")
    _keys(
        group,
        {"robot_key", "datasets", "provenance", "state", "action", "schema_hash"},
        f"statistics group {name!r}",
    )
    _text(group["robot_key"], f"{name}.robot_key")
    _datasets(group["datasets"])
    _json_mapping(group["provenance"], f"{name}.provenance")
    _sha(group["schema_hash"], f"{name}.schema_hash")
    _validate_state(group["state"], name)
    _validate_action(group["action"], name)


def _validate_state(section: Any, name: str) -> None:
    label = f"{name}.state"
    if not isinstance(section, dict):
        raise TypeError(f"{label} must be a mapping")
    _keys(
        section,
        {
            "dim",
            "method",
            "clip",
            "continuous_indices",
            "identity_indices",
            "q01",
            "q99",
            "constant_mask",
            "clip_rate_low",
            "clip_rate_high",
            "count",
        },
        label,
    )
    dimension = _positive_int(section["dim"], f"{label}.dim")
    continuous = _indices(
        section["continuous_indices"],
        dimension=dimension,
        label=f"{label}.continuous_indices",
    )
    identity = _indices(
        section["identity_indices"],
        dimension=dimension,
        label=f"{label}.identity_indices",
    )
    if set(continuous) & set(identity) or tuple(sorted((*continuous, *identity))) != tuple(range(dimension)):
        raise ValueError(f"{label} continuous and identity indices must partition dimensions")
    if section["method"] != "q01_q99" or section["clip"] != [NORMALIZED_MIN, NORMALIZED_MAX]:
        raise ValueError(f"{label} must use q01_q99 with hard clip [-1, 1]")
    _validate_quantiles(section, len(continuous), label)
    _positive_int(section["count"], f"{label}.count")


def _validate_action(section: Any, name: str) -> None:
    label = f"{name}.action"
    if not isinstance(section, dict):
        raise TypeError(f"{label} must be a mapping")
    _keys(
        section,
        {
            "dim",
            "continuous_indices",
            "continuous_method",
            "continuous_clip",
            "q01",
            "q99",
            "constant_mask",
            "clip_rate_low",
            "clip_rate_high",
            "gripper_index",
            "gripper_method",
            "gripper_semantic",
            "gripper_threshold",
            "count",
        },
        label,
    )
    if section["dim"] != ACTION_DIM:
        raise ValueError(f"{label}.dim must equal {ACTION_DIM}")
    if tuple(section["continuous_indices"]) != ACTION_CONTINUOUS_INDICES:
        raise ValueError(f"{label} continuous indices must be the first six")
    if section["continuous_method"] != "q01_q99" or section["continuous_clip"] != [NORMALIZED_MIN, NORMALIZED_MAX]:
        raise ValueError(f"{label} motion must use q01_q99 and hard clip [-1, 1]")
    if section["gripper_index"] != ACTION_GRIPPER_INDEX:
        raise ValueError(f"{label}.gripper_index must equal {ACTION_GRIPPER_INDEX}")
    if (
        section["gripper_method"] != "identity"
        or section["gripper_semantic"] != "open_01"
        or section["gripper_threshold"] != 0.5
    ):
        raise ValueError(f"{label} gripper must use identity open_01 at 0.5")
    _validate_quantiles(section, len(ACTION_CONTINUOUS_INDICES), label)
    _positive_int(section["count"], f"{label}.count")


def _validate_quantiles(section: Mapping[str, Any], size: int, label: str) -> None:
    low = _number_list(section["q01"], size, f"{label}.q01")
    high = _number_list(section["q99"], size, f"{label}.q99")
    constant = section["constant_mask"]
    if not isinstance(constant, list) or len(constant) != size or any(type(value) is not bool for value in constant):
        raise ValueError(f"{label}.constant_mask must be a boolean list of length {size}")
    if any(right < left for left, right in zip(low, high)):
        raise ValueError(f"{label}.q99 must be >= q01")
    if constant != [right == left for left, right in zip(low, high)]:
        raise ValueError(f"{label}.constant_mask does not match q01/q99")
    for key in ("clip_rate_low", "clip_rate_high"):
        rates = _number_list(section[key], size, f"{label}.{key}")
        if any(value < 0.0 or value > 1.0 for value in rates):
            raise ValueError(f"{label}.{key} values must be in [0, 1]")


def _runtime_group(group: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_mapping(group, "group_statistics")
    _validate_group("runtime", normalized)
    return normalized


def _normalization_indices(features: Sequence[FeatureSlice]) -> tuple[int, ...]:
    output: list[int] = []
    cursor = 0
    for feature in features:
        if feature.normalization == "q01_q99":
            output.extend(range(cursor, cursor + feature.width))
        cursor += feature.width
    return tuple(output)


def _indices(values: Sequence[int], *, dimension: int, label: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    output: list[int] = []
    for index in values:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"{label} must contain integers")
        item = int(index)
        if item < 0 or item >= dimension:
            raise ValueError(f"{label} index {item} is outside dimension {dimension}")
        output.append(item)
    if output != sorted(set(output)):
        raise ValueError(f"{label} must be unique and sorted")
    return tuple(output)


def _datasets(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("datasets must be a non-empty sequence")
    output = tuple(_text(value, "dataset name") for value in values)
    if len(set(output)) != len(output):
        raise ValueError("datasets must not contain duplicates")
    return output


def _mask(value: np.ndarray | None, shape: tuple[int, ...], label: str) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=np.bool_)
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{label} must be boolean with shape {shape}, got dtype={array.dtype}, shape={array.shape}")
    return array


def _strict_values(values: np.ndarray, *, allowed: Sequence[float], label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    invalid = values[~np.isin(values, np.asarray(allowed))]
    if invalid.size:
        raise ValueError(
            f"{label} values must belong to {list(allowed)}, got invalid values {np.unique(invalid).tolist()}"
        )


def _numeric_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    numeric = np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    if not numeric or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{label} must contain real numeric values")
    return array


def _number_list(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must be a list of length {size}")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{label} must contain finite numbers")
        output.append(float(item))
    return output


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA256")


def _keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(f"{label} keys mismatch: missing={missing}, unknown={unknown}")


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    normalized = _to_json(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a mapping")
    return normalized


def _to_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings must use string keys")
            output[key] = _to_json(item)
        return output
    if isinstance(value, np.ndarray):
        return _to_json(value.tolist())
    if isinstance(value, np.generic):
        return _to_json(value.item())
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit non-finite floats")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not canonical-JSON compatible")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ACTION_CONTINUOUS_INDICES",
    "ACTION_DIM",
    "ACTION_GRIPPER_INDEX",
    "GRIPPER_SOURCE_ENCODINGS",
    "NORMALIZED_MAX",
    "NORMALIZED_MIN",
    "DataSpecNormalizer",
    "NormalizationStatistics",
    "STATISTICS_FORMAT",
    "build_statistics_artifact",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonicalize_assembled_features",
    "canonicalize_features",
    "canonicalize_gripper",
    "compute_statistics",
    "decode_gripper_for_environment",
    "decode_gripper_open",
    "denormalize_action",
    "gripper_open_to_environment",
    "load_statistics",
    "normalize_action",
    "normalize_state",
    "save_statistics",
    "statistics_content_sha256",
    "validate_statistics",
]
