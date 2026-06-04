def test_run_evals_entrypoint_imports():
    import scaletraining.entrypoints.run_evals as run_evals

    assert callable(run_evals.main)
