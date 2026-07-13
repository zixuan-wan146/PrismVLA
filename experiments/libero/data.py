"""Static canonical field mapping for materialized LIBERO training data."""

from prism.data.schema import DataSpec, FeatureSlice, LanguageSpec, ViewSpec


LIBERO_IMAGE_TRANSFORM = "rotate_180"


LIBERO_DATA_SPEC = DataSpec(
    name="libero",
    benchmark="libero",
    robot_key="libero",
    storage_format="lerobot-v2.1",
    views=(
        ViewSpec(name="primary", source_key="observation.images.image"),
        ViewSpec(name="wrist", source_key="observation.images.wrist_image"),
    ),
    state=(
        FeatureSlice("state.x", "observation.state", 0, 1, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.y", "observation.state", 1, 2, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.z", "observation.state", 2, 3, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.roll", "observation.state", 3, 4, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.pitch", "observation.state", 4, 5, "q01_q99", "absolute", "continuous"),
        FeatureSlice("state.yaw", "observation.state", 5, 6, "q01_q99", "absolute", "continuous"),
        FeatureSlice(
            "state.gripper_left",
            "observation.state",
            6,
            7,
            "q01_q99",
            "absolute",
            "continuous",
        ),
        FeatureSlice(
            "state.gripper_right",
            "observation.state",
            7,
            8,
            "q01_q99",
            "absolute",
            "continuous",
        ),
    ),
    action=(
        FeatureSlice("action.x", "action", 0, 1, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.y", "action", 1, 2, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.z", "action", 2, 3, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.roll", "action", 3, 4, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.pitch", "action", 4, 5, "q01_q99", "delta", "continuous"),
        FeatureSlice("action.yaw", "action", 5, 6, "q01_q99", "delta", "continuous"),
        FeatureSlice(
            "action.gripper_open",
            "action",
            6,
            7,
            "identity",
            "absolute",
            "open_01",
        ),
    ),
    language=LanguageSpec(source_key="task_index", kind="task_index"),
)


__all__ = ["LIBERO_DATA_SPEC", "LIBERO_IMAGE_TRANSFORM"]
