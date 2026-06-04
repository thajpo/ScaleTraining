def test_run_lm_eval_entrypoint_imports():
    import scaletraining.entrypoints.run_lm_eval as run_lm_eval

    assert callable(run_lm_eval.main)
