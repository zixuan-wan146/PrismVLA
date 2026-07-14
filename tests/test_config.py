from prism.config import load_config, merge_profile_environment


def test_load_libero_and_calvin_configs():
    libero = load_config("experiments/libero/configs/eval.yaml")
    calvin = load_config("experiments/calvin/configs/eval.yaml")

    assert libero.data.benchmark == "libero"
    assert calvin.data.benchmark == "calvin"


def test_dotted_override_updates_one_profile_environment_value():
    profile = load_config(
        "experiments/libero/configs/eval.yaml",
        overrides=["profile_env.PRISM_LIBERO_CAMERA_RESOLUTION=320"],
    )

    assert profile.raw["profile_env"]["PRISM_LIBERO_CAMERA_RESOLUTION"] == 320
    assert profile.raw["profile_env"]["PRISM_LIBERO_VIDEO_FPS"] == 30


def test_ambient_environment_takes_precedence_over_profile_defaults():
    assert merge_profile_environment(
        {"PRISM_CAMERA_SIZE": "448", "PRISM_VIDEO_FPS": "30"},
        {"PRISM_CAMERA_SIZE": "320", "EXTERNAL_ONLY": "kept"},
    ) == {
        "PRISM_CAMERA_SIZE": "320",
        "PRISM_VIDEO_FPS": "30",
        "EXTERNAL_ONLY": "kept",
    }
