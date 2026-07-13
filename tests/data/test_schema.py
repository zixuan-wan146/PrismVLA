from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from experiments.calvin.data import CALVIN_DATA_SPEC, CALVIN_EVAL_SPLITS, CALVIN_TRAIN_SPLITS
from experiments.libero.data import LIBERO_DATA_SPEC
from prism.data.schema import DataSpec, FeatureSlice, LanguageSpec, VLASample, ViewSpec, data_spec_from_mapping
from prism.schema import PolicyInput


EXPECTED_COMMON_STATE_NAMES = (
    "state.x",
    "state.y",
    "state.z",
    "state.roll",
    "state.pitch",
    "state.yaw",
)
EXPECTED_LIBERO_STATE_NAMES = EXPECTED_COMMON_STATE_NAMES + ("state.gripper_left", "state.gripper_right")
EXPECTED_CALVIN_STATE_NAMES = EXPECTED_COMMON_STATE_NAMES + ("state.pad", "state.gripper_width")
EXPECTED_ACTION_NAMES = (
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper_open",
)


@pytest.mark.parametrize(
    ("spec", "view_sources", "state_source", "state_names", "action_source", "gripper_encoding"),
    [
        (
            LIBERO_DATA_SPEC,
            ("observation.images.image", "observation.images.wrist_image"),
            "observation.state",
            EXPECTED_LIBERO_STATE_NAMES,
            "action",
            "open_01",
        ),
        (
            CALVIN_DATA_SPEC,
            ("image", "wrist_image"),
            "state",
            EXPECTED_CALVIN_STATE_NAMES,
            "actions",
            "signed_open_positive",
        ),
    ],
)
def test_benchmark_specs_have_one_canonical_contract(
    spec: DataSpec,
    view_sources: tuple[str, str],
    state_source: str,
    state_names: tuple[str, ...],
    action_source: str,
    gripper_encoding: str,
) -> None:
    assert spec.storage_format == "lerobot-v2.1"
    assert spec.view_names == ("primary", "wrist")
    assert tuple(view.source_key for view in spec.views) == view_sources
    assert spec.state_names == state_names
    assert spec.action_names == EXPECTED_ACTION_NAMES
    assert spec.state_dim == 8
    assert spec.action_dim == 7
    assert {feature.source_key for feature in spec.state} == {state_source}
    assert {feature.source_key for feature in spec.action} == {action_source}
    assert spec.action[-1].source_encoding == gripper_encoding
    assert spec.action[-1].normalization == "identity"
    assert spec.action[-1].temporal_semantics == "absolute"
    assert spec.language == LanguageSpec(source_key="task_index", kind="task_index")


def test_calvin_split_contract_keeps_scene_d_out_of_training() -> None:
    assert CALVIN_TRAIN_SPLITS == ("A", "B", "C")
    assert CALVIN_EVAL_SPLITS == ("D",)
    assert set(CALVIN_TRAIN_SPLITS).isdisjoint(CALVIN_EVAL_SPLITS)


def test_calvin_constant_padding_has_explicit_identity_semantics() -> None:
    padding = CALVIN_DATA_SPEC.state[6]
    gripper_width = CALVIN_DATA_SPEC.state[7]

    assert (padding.name, padding.source_slice) == ("state.pad", slice(6, 7))
    assert (padding.normalization, padding.temporal_semantics, padding.source_encoding) == (
        "identity",
        "absolute",
        "constant_zero",
    )
    assert gripper_width.name == "state.gripper_width"
    assert gripper_width.source_encoding == "continuous"
    assert gripper_width.normalization == "q01_q99"


def test_feature_slice_exposes_width_and_source_slice() -> None:
    feature = FeatureSlice(
        name="state.position",
        source_key="raw_state",
        start=2,
        end=5,
        normalization="q01_q99",
        temporal_semantics="absolute",
        source_encoding="continuous",
    )

    assert feature.width == 3
    assert feature.source_slice == slice(2, 5)


@pytest.mark.parametrize(
    "changes",
    [
        {"start": -1},
        {"end": 0},
        {"normalization": "min_max"},
        {"temporal_semantics": "relative"},
        {"source_encoding": "binary"},
    ],
)
def test_feature_slice_rejects_invalid_ranges_and_enums(changes: dict[str, object]) -> None:
    values = {
        "name": "action.x",
        "source_key": "action",
        "start": 0,
        "end": 1,
        "normalization": "q01_q99",
        "temporal_semantics": "delta",
        "source_encoding": "continuous",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        FeatureSlice(**values)


def test_discrete_source_encoding_has_strict_semantics() -> None:
    with pytest.raises(ValueError, match="continuous"):
        FeatureSlice("action.gripper_open", "action", 6, 7, "q01_q99", "absolute", "open_01")
    with pytest.raises(ValueError, match="absolute"):
        FeatureSlice("action.gripper_open", "action", 6, 7, "identity", "delta", "open_01")
    with pytest.raises(ValueError, match="exactly one"):
        FeatureSlice("action.gripper_open", "action", 5, 7, "identity", "absolute", "open_01")


def test_data_spec_requires_explicit_ordered_tuples() -> None:
    with pytest.raises(TypeError, match="explicit tuple"):
        DataSpec(
            name="invalid",
            benchmark="invalid",
            robot_key="invalid",
            storage_format="lerobot-v2.1",
            views=[ViewSpec("primary", "image")],
            state=LIBERO_DATA_SPEC.state,
            action=LIBERO_DATA_SPEC.action,
            language=LanguageSpec("task_index", "task_index"),
        )


def test_data_spec_rejects_wrong_storage_and_duplicate_canonical_names() -> None:
    with pytest.raises(ValueError, match="lerobot-v2.1"):
        replace(LIBERO_DATA_SPEC, storage_format="lerobot-v3.0")
    with pytest.raises(ValueError, match="must be unique"):
        replace(LIBERO_DATA_SPEC, state=(LIBERO_DATA_SPEC.state[0], LIBERO_DATA_SPEC.state[0]))


def test_data_spec_rejects_overlapping_source_slices() -> None:
    overlapping = FeatureSlice(
        "state.overlap",
        "observation.state",
        0,
        2,
        "q01_q99",
        "absolute",
        "continuous",
    )

    with pytest.raises(ValueError, match="source slices overlap"):
        replace(LIBERO_DATA_SPEC, state=(LIBERO_DATA_SPEC.state[0], overlapping))


def test_checkpoint_canonical_data_spec_round_trips_without_experiment_imports() -> None:
    reconstructed = data_spec_from_mapping(asdict(CALVIN_DATA_SPEC))

    assert reconstructed == CALVIN_DATA_SPEC


def test_checkpoint_data_spec_rejects_unknown_schema_fields() -> None:
    payload = asdict(CALVIN_DATA_SPEC)
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unknown"):
        data_spec_from_mapping(payload)


def _policy_input() -> PolicyInput:
    current = {
        "primary": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    history = {
        "primary": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "wrist": np.zeros((2, 4, 4, 3), dtype=np.uint8),
    }
    return PolicyInput(
        benchmark="libero",
        prompt="pick up the object",
        images_by_view=current,
        history_images_by_view=history,
        history_step_ages=np.asarray([6, 3], dtype=np.int64),
        history_valid_mask=np.asarray([True, False], dtype=np.bool_),
        state=np.zeros((8,), dtype=np.float32),
        action_dim=7,
        robot_key="libero",
    )


def _sample() -> VLASample:
    return VLASample(
        policy_input=_policy_input(),
        dataset_name="libero_spatial",
        statistics_group="libero",
        episode_index=3,
        frame_index=7,
        target_actions=np.zeros((8, 7), dtype=np.float32),
        action_valid_mask=np.asarray([True, True, True, False, False, False, False, False]),
    )


def test_vla_sample_is_frozen_and_preserves_policy_input() -> None:
    sample = _sample()

    assert sample.policy_input.action_dim == sample.target_actions.shape[1]
    with pytest.raises(FrozenInstanceError):
        sample.frame_index = 9


def test_vla_sample_validates_action_shape_mask_and_padding() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="action dimension"):
        replace(sample, target_actions=np.zeros((8, 6), dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \[H\]"):
        replace(sample, action_valid_mask=np.ones((8,), dtype=np.float32))

    nonzero_padding = sample.target_actions.copy()
    nonzero_padding[-1, 0] = 1.0
    with pytest.raises(ValueError, match="zero at invalid"):
        replace(sample, target_actions=nonzero_padding)
