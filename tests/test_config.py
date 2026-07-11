from prism.config import load_config


def test_load_libero_and_calvin_configs():
    libero = load_config('configs/experiment/libero_stage1.yaml')
    calvin = load_config('configs/experiment/calvin_stage1.yaml')

    assert libero.data.benchmark == 'libero'
    assert calvin.data.benchmark == 'calvin'
