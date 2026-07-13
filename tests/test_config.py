from prism.config import load_config


def test_load_libero_and_calvin_configs():
    libero = load_config("experiments/libero/configs/eval.yaml")
    calvin = load_config("experiments/calvin/configs/eval.yaml")

    assert libero.data.benchmark == "libero"
    assert calvin.data.benchmark == "calvin"
