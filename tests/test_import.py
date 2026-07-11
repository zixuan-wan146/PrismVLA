def test_import_prism_subpackages():
    import prism
    import prism.config
    import prism.models
    import prism.data
    import prism.training
    import prism.eval
    import prism.serve
    import prism.utils

    assert prism.__version__
