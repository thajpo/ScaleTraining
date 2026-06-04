def test_train_entrypoint_imports():
    import scaletraining.entrypoints.train as train

    assert callable(train.main)
