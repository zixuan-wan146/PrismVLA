from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from prism.data import normalization
from prism.data.normalization import (
    DataSpecNormalizer,
    build_statistics_artifact,
    canonical_sha256,
    canonicalize_assembled_features,
    canonicalize_features,
    canonicalize_gripper,
    compute_statistics,
    decode_gripper_for_environment,
    decode_gripper_open,
    denormalize_action,
    gripper_open_to_environment,
    load_statistics,
    normalize_action,
    normalize_state,
    save_statistics,
    statistics_content_sha256,
    validate_statistics,
)
from prism.data.schema import DataSpec, FeatureSlice, LanguageSpec, ViewSpec


def _spec(*, gripper_encoding: str = "signed_open_positive") -> DataSpec:
    return DataSpec(
        name="fixture",
        benchmark="fixture",
        robot_key="fixture_robot",
        storage_format="lerobot-v2.1",
        views=(ViewSpec("primary", "image"), ViewSpec("wrist", "wrist_image")),
        state=(
            FeatureSlice("state.x", "state", 4, 5, "q01_q99", "absolute", "continuous"),
            FeatureSlice("state.pad", "state", 9, 10, "identity", "absolute", "constant_zero"),
            FeatureSlice("state.width", "state", 12, 13, "q01_q99", "absolute", "continuous"),
        ),
        action=tuple(
            FeatureSlice(
                f"action.{name}",
                "actions",
                20 + index,
                21 + index,
                "q01_q99",
                "delta",
                "continuous",
            )
            for index, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw"))
        )
        + (
            FeatureSlice(
                "action.gripper_open",
                "actions",
                31,
                32,
                "identity",
                "absolute",
                gripper_encoding,
            ),
        ),
        language=LanguageSpec("task_index", "task_index"),
    )


def _values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.arange(100, dtype=np.float64)
    states = np.stack((base, np.zeros_like(base), np.full_like(base, 7.0)), axis=-1)
    motion = np.stack(
        (
            base,
            base + 10.0,
            base - 5.0,
            base * 2.0,
            base * -0.5,
            np.full_like(base, 3.0),
        ),
        axis=-1,
    )
    gripper = (np.arange(100) % 2).astype(np.float64)[:, None]
    actions = np.concatenate((motion, gripper), axis=-1)
    states = np.concatenate((states, np.full((1, 3), np.nan)), axis=0)
    actions = np.concatenate((actions, np.full((1, 7), np.nan)), axis=0)
    valid = np.concatenate((np.ones(100, dtype=np.bool_), np.zeros(1, dtype=np.bool_)))
    return states, actions, valid


def _artifact(
    *,
    schema_hash: str | None = None,
    gripper_encoding: str = "signed_open_positive",
) -> dict:
    states, actions, valid = _values()
    spec = _spec(gripper_encoding=gripper_encoding)
    return compute_statistics(
        states,
        actions,
        group="fixture_group",
        robot_key=spec.robot_key,
        datasets=("fixture_a", "fixture_b"),
        schema_hash=canonical_sha256(spec) if schema_hash is None else schema_hash,
        provenance={"train_splits": ["A", "B", "C"], "eval_splits": ["D"]},
        state_continuous_indices=(0, 2),
        state_valid_mask=valid,
        action_valid_mask=valid,
    )


def _group(artifact: dict | None = None) -> dict:
    artifact = _artifact() if artifact is None else artifact
    return artifact["groups"]["fixture_group"]


def test_statistics_use_valid_float64_rows_and_record_per_dimension_clip_rates() -> None:
    artifact = _artifact()
    group = _group(artifact)

    assert artifact["format"] == "prism-normalization-v1"
    assert artifact["content_sha256"] == statistics_content_sha256(artifact)
    assert group["state"]["count"] == 100
    assert group["action"]["count"] == 100
    assert group["state"]["continuous_indices"] == [0, 2]
    assert group["state"]["identity_indices"] == [1]
    assert group["state"]["q01"] == pytest.approx([0.99, 7.0])
    assert group["state"]["q99"] == pytest.approx([98.01, 7.0])
    assert group["state"]["constant_mask"] == [False, True]
    assert group["action"]["q01"][0] == pytest.approx(0.99)
    assert group["action"]["q99"][0] == pytest.approx(98.01)
    assert group["action"]["constant_mask"] == [False, False, False, False, False, True]
    assert group["action"]["clip_rate_low"] == pytest.approx([0.01] * 5 + [0.0])
    assert group["action"]["clip_rate_high"] == pytest.approx([0.01] * 5 + [0.0])
    assert group["action"]["gripper_method"] == "identity"
    assert group["action"]["gripper_semantic"] == "open_01"


def test_canonical_hash_is_order_independent_and_rejects_nonfinite_json() -> None:
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256({"a": [1, 3], "b": 2})
    with pytest.raises(ValueError, match="non-finite"):
        canonical_sha256({"bad": np.nan})


def test_state_normalization_is_float32_clipped_and_constant_safe() -> None:
    group = _group()
    raw = np.asarray(
        [
            [-50.0, 42.0, 7.0],
            [49.5, -3.0, 7.0],
            [200.0, 9.0, 7.0],
            [np.nan, np.nan, np.nan],
        ]
    )
    valid = np.asarray([True, True, True, False])

    result = normalize_state(raw, group, valid_mask=valid)

    assert result.dtype == np.float32
    assert result[:, 0].tolist() == pytest.approx([-1.0, 0.0, 1.0, 0.0], abs=1e-6)
    assert result[:, 1].tolist() == [42.0, -3.0, 9.0, 0.0]
    assert result[:, 2].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_action_normalization_round_trip_identity_gripper_and_padding() -> None:
    group = _group()
    low = np.asarray(group["action"]["q01"], dtype=np.float32)
    high = np.asarray(group["action"]["q99"], dtype=np.float32)
    middle = (low + high) * 0.5
    raw = np.zeros((4, 7), dtype=np.float32)
    raw[0, :6], raw[0, 6] = low, 0.0
    raw[1, :6], raw[1, 6] = middle, 1.0
    raw[2, :6], raw[2, 6] = high, 0.0
    raw[3] = np.nan
    valid = np.asarray([True, True, True, False])

    normalized = normalize_action(raw, group, valid_mask=valid)
    restored = denormalize_action(normalized, group, valid_mask=valid)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[:3, 0], [-1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(normalized[:3, 5], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(normalized[:3, 6], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(normalized[3], np.zeros(7, dtype=np.float32))
    np.testing.assert_allclose(restored[:3, :6], raw[:3, :6], atol=1e-5)
    np.testing.assert_array_equal(restored[:3, 6], raw[:3, 6])
    np.testing.assert_array_equal(restored[3], np.zeros(7, dtype=np.float32))


def test_action_normalization_hard_clips_motion_and_rejects_noncanonical_gripper() -> None:
    group = _group()
    raw = np.zeros((2, 7), dtype=np.float32)
    raw[0, :6] = -1.0e6
    raw[1, :6] = 1.0e6
    raw[:, 5] = 3.0
    raw[:, 6] = [0.0, 1.0]

    result = normalize_action(raw, group)

    np.testing.assert_array_equal(result[:, :5], [[-1.0] * 5, [1.0] * 5])
    np.testing.assert_array_equal(result[:, 5], [0.0, 0.0])
    raw[0, 6] = 0.5
    with pytest.raises(ValueError, match="canonical action gripper"):
        normalize_action(raw, group)


@pytest.mark.parametrize(
    ("encoding", "source", "expected"),
    [
        ("open_01", [0.0, 1.0], [0.0, 1.0]),
        ("signed_open_positive", [-1.0, 1.0], [0.0, 1.0]),
        ("signed_open_negative", [-1.0, 1.0], [1.0, 0.0]),
    ],
)
def test_gripper_source_encodings_are_strict(encoding: str, source: list[float], expected: list[float]) -> None:
    result = canonicalize_gripper(np.asarray(source), encoding)
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.float32))

    with pytest.raises(ValueError, match="invalid values"):
        canonicalize_gripper(np.asarray([0.25]), encoding)


def test_gripper_canonicalization_ignores_only_explicit_padding() -> None:
    values = np.asarray([-1.0, 0.25])
    result = canonicalize_gripper(
        values,
        "signed_open_positive",
        valid_mask=np.asarray([True, False]),
    )
    np.testing.assert_array_equal(result, [0.0, 0.0])

    with pytest.raises(ValueError, match="finite"):
        canonicalize_gripper(np.asarray([np.nan]), "open_01")
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_gripper(np.asarray([0.0]), "continuous")


def test_gripper_prediction_boundary_and_environment_signs() -> None:
    predictions = np.asarray([0.49, 0.5, 0.50001, 2.0], dtype=np.float32)
    opened = decode_gripper_open(predictions)

    np.testing.assert_array_equal(opened, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(gripper_open_to_environment(opened, "libero"), [1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(gripper_open_to_environment(opened, "calvin"), [-1.0, -1.0, 1.0, 1.0])
    np.testing.assert_array_equal(
        decode_gripper_for_environment(predictions, "libero"),
        [1.0, 1.0, -1.0, -1.0],
    )
    with pytest.raises(ValueError, match="unsupported benchmark"):
        gripper_open_to_environment(opened, "unknown")


def test_canonicalize_features_uses_physical_slices_and_declared_order() -> None:
    spec = _spec()
    state_source = np.zeros((2, 13), dtype=np.float32)
    state_source[:, 4] = [2.0, 4.0]
    state_source[:, 9] = 0.0
    state_source[:, 12] = [6.0, 8.0]

    result = canonicalize_features({"state": state_source}, spec.state)

    np.testing.assert_array_equal(result, [[2.0, 0.0, 6.0], [4.0, 0.0, 8.0]])


def test_canonicalize_assembled_features_does_not_reapply_physical_offsets() -> None:
    spec = _spec(gripper_encoding="signed_open_positive")
    raw_action = np.zeros((2, 7), dtype=np.float32)
    raw_action[:, :6] = np.arange(6, dtype=np.float32)
    raw_action[:, 6] = [-1.0, 1.0]

    result = canonicalize_assembled_features(raw_action, spec.action)

    np.testing.assert_array_equal(result[:, :6], raw_action[:, :6])
    np.testing.assert_array_equal(result[:, 6], [0.0, 1.0])


def test_dataspec_normalizer_is_pickle_safe_and_composes_canonicalization() -> None:
    spec = _spec(gripper_encoding="signed_open_positive")
    adapter = DataSpecNormalizer(spec, _artifact(), "fixture_group")
    restored = pickle.loads(pickle.dumps(adapter))
    raw_state = np.asarray([49.5, 0.0, 7.0], dtype=np.float32)
    raw_actions = np.zeros((2, 7), dtype=np.float32)
    raw_actions[:, :6] = 49.5
    raw_actions[:, 5] = 3.0
    raw_actions[:, 6] = [-1.0, 1.0]

    state = restored.normalize_state(raw_state)
    actions = restored.normalize_action(raw_actions)

    assert state.shape == (3,)
    assert state.dtype == np.float32
    assert state[0] == pytest.approx(0.0, abs=1e-6)
    assert state[1] == 0.0
    assert state[2] == 0.0
    np.testing.assert_array_equal(actions[:, 5], [0.0, 0.0])
    np.testing.assert_array_equal(actions[:, 6], [0.0, 1.0])
    with pytest.raises(ValueError, match="dimension 7"):
        restored.normalize_action(np.zeros((2, 6), dtype=np.float32))


def test_dataspec_normalizer_validates_once_then_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = DataSpecNormalizer(_spec(), _artifact(), "fixture_group")

    def reject_revalidation(name: str, group: object) -> None:
        raise AssertionError(f"unexpected repeated validation of {name}: {group}")

    monkeypatch.setattr(normalization, "_validate_group", reject_revalidation)
    state = adapter.normalize_state(np.asarray([49.5, 0.0, 7.0]))
    actions = np.zeros((1, 7), dtype=np.float32)
    actions[0, :6] = [49.5, 59.5, 44.5, 99.0, -24.75, 3.0]
    actions[0, 6] = 1.0
    normalized_actions = adapter.normalize_action(actions)

    assert state.dtype == np.float32
    assert normalized_actions.dtype == np.float32
    assert normalized_actions[0, 6] == 1.0


def test_dataspec_normalizer_rejects_schema_and_group_mismatch() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="schema hash mismatch"):
        DataSpecNormalizer(spec, _artifact(schema_hash="0" * 64), "fixture_group")
    with pytest.raises(KeyError, match="missing"):
        DataSpecNormalizer(spec, _artifact(), "other")


def test_save_load_validates_hash_schema_datasets_and_provenance(tmp_path: Path) -> None:
    artifact = _artifact()
    path = save_statistics(artifact, tmp_path / "nested" / "statistics.json")
    loaded = load_statistics(
        path,
        group="fixture_group",
        expected_schema_hash=canonical_sha256(_spec()),
        expected_robot_key="fixture_robot",
        expected_datasets=("fixture_a", "fixture_b"),
        expected_provenance={"train_splits": ["A", "B", "C"], "eval_splits": ["D"]},
    )

    assert loaded == artifact
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_statistics(
            path,
            group="fixture_group",
            expected_provenance={"train_splits": ["A", "B", "C", "D"]},
        )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["groups"]["fixture_group"]["action"]["q01"][0] = -999.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_statistics(path)


def test_atomic_save_keeps_previous_file_and_removes_temp_on_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "statistics.json"
    target.write_text("previous", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(normalization.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        save_statistics(_artifact(), target)

    assert target.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".statistics.json.*.tmp"))


def test_validation_rejects_stale_or_incompatible_artifacts() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="group is required"):
        validate_statistics(artifact, expected_robot_key="fixture_robot")

    bad_group = json.loads(json.dumps(_group(artifact)))
    bad_group["action"]["continuous_indices"] = [0, 1, 2, 3, 4, 6]
    with pytest.raises(ValueError, match="first six"):
        build_statistics_artifact({"bad": bad_group})


def test_statistics_fail_on_no_valid_rows_nonfinite_values_and_bad_gripper() -> None:
    states, actions, valid = _values()
    kwargs = {
        "group": "fixture_group",
        "robot_key": "fixture_robot",
        "datasets": ("fixture",),
        "schema_hash": "a" * 64,
        "state_continuous_indices": (0, 2),
        "state_valid_mask": valid,
        "action_valid_mask": valid,
    }
    with pytest.raises(ValueError, match="no valid"):
        compute_statistics(
            states,
            actions,
            **{
                **kwargs,
                "state_valid_mask": np.zeros_like(valid),
            },
        )

    states[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_statistics(states, actions, **kwargs)
    states[0, 0] = 0.0
    actions[0, 6] = -1.0
    with pytest.raises(ValueError, match="canonical action gripper"):
        compute_statistics(states, actions, **kwargs)
