def test_import():
    import importlib
    m = importlib.import_module("src.helical_cell")
    assert m is not None
