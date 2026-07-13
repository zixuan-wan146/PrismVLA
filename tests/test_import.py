def test_import_prism_and_experiment_modules():
    import experiments.calvin.eval
    import experiments.libero.eval
    import prism
    import prism.config
    import prism.data
    import prism.models
    import prism.serve
    import prism.utils

    assert experiments.calvin.eval.CALVIN_BENCHMARK == "calvin"
    assert experiments.libero.eval.LIBERO_BENCHMARK == "libero"
    assert prism.__version__
