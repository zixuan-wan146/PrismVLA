"""Static canonical field mapping and split contract for CALVIN."""

from prism.data.benchmark_contracts import CALVIN_EVAL_SPLITS
from prism.data.benchmark_contracts import CALVIN_TRAIN_SPLITS
from prism.data.schema import DataSpec, FeatureSlice, LanguageSpec, ViewSpec

CALVIN_DATA_SPEC = DataSpec(
    name="calvin",
    benchmark="calvin",
    robot_key="calvin",
    storage_format="lerobot-v2.1",
    views=(
        ViewSpec(name="primary", source_key="image"),
        ViewSpec(name="wrist", source_key="wrist_image"),
    ),
    state=(
        FeatureSlice("state.x", "state", 0, 1, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.y", "state", 1, 2, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.z", "state", 2, 3, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.roll", "state", 3, 4, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.pitch", "state", 4, 5, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.yaw", "state", 5, 6, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.pad", "state", 6, 7, "identity", "absolute", "constant_zero"),
        FeatureSlice("state.gripper_width", "state", 7, 8, "q01_q99", "absolute", "continuous"),
    ),
    action=(
        FeatureSlice("action.x", "actions", 0, 1, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.y", "actions", 1, 2, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.z", "actions", 2, 3, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.roll", "actions", 3, 4, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.pitch", "actions", 4, 5, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.yaw", "actions", 5, 6, "q01_q99", "delta", "continuous"),
        FeatureSlice(
            "action.gripper_open",
            "actions",
            6,
            7,
            "identity",
            "absolute",
            "signed_open_positive",
        ),
    ),
    language=LanguageSpec(source_key="task_index", kind="task_index"),
)


__all__ = ["CALVIN_DATA_SPEC", "CALVIN_EVAL_SPLITS", "CALVIN_TRAIN_SPLITS"]
