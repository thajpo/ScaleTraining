def test_prepare_data_entrypoint_imports():
    import scaletraining.entrypoints.prepare_data as prepare_data

    assert callable(prepare_data.main)
