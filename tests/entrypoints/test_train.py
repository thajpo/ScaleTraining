import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


def test_train_entrypoint_imports():
    import scaletraining.entrypoints.train as train

    assert callable(train.main)


def test_training_error_keeps_original_exception_and_finalizes_evidence(
    tmp_path, monkeypatch
):
    import scaletraining.entrypoints.train as train

    cfg = OmegaConf.create(
        {
            "training": {"seed": 13},
            "device": {"device": "cpu"},
            "model": {},
            "tokenizer": {"tokenizer_name": None},
            "paths": {"output_dir": str(tmp_path / "outputs")},
        }
    )
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    finished = []

    monkeypatch.setattr(train, "load_project_config", lambda value: value)
    monkeypatch.setattr(train, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(train, "configure_rocm_and_sdp", lambda cfg: None)
    monkeypatch.setattr(train, "resolve_device", lambda cfg: "cpu")
    monkeypatch.setattr(train, "clear_cuda_cache", lambda: None)
    monkeypatch.setattr(
        train,
        "TextTokenizer",
        lambda cfg: SimpleNamespace(
            vocab_size=16,
            tok_name="fixture-tokenizer",
            tok=SimpleNamespace(name_or_path="fixture-tokenizer"),
        ),
    )
    monkeypatch.setattr(train, "create_run_dir", lambda *args: run_dir)
    monkeypatch.setattr(
        train,
        "init_wandb",
        lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"provider": "wandb", "state": "disabled"}
        ),
    )

    def save_manifest(cfg, out_dir, extra):
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"status": extra["status"], "tracking": extra["tracking"]})
        )

    monkeypatch.setattr(train, "save_run_manifest", save_manifest)
    monkeypatch.setattr(
        train,
        "build_loaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("loader broke")),
    )
    monkeypatch.setattr(train, "finish_wandb", lambda: finished.append(True))

    with pytest.raises(RuntimeError, match="loader broke"):
        train.run_training(cfg)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    report = json.loads((run_dir / "run_report.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"type": "RuntimeError", "message": "loader broke"}
    assert report["summary"]["status"] == "failed"
    assert finished == [True]
