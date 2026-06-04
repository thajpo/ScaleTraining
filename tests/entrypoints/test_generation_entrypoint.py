def test_generation_entrypoint_imports():
    import scaletraining.entrypoints.generate_from_pretrained as generate_from_pretrained

    assert callable(generate_from_pretrained.main)
