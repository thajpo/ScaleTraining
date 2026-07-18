import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


def test_train_entrypoint_imports():
    import scaletraining.entrypoints.train as train

    assert callable(train.main)


def test_compile_gate_accepts_an_indexed_cuda_device(monkeypatch):
    import scaletraining.entrypoints.train as train

    cfg = OmegaConf.create(
        {
            "training": {"compile_model": True},
            "device": {"device": "cuda:1"},
            "device_resolved": "cuda:1",
        }
    )

    monkeypatch.setattr(train.torch.version, "hip", None)

    assert train._should_compile(cfg) is True


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
            json.dumps(
                {
                    "status": extra["status"],
                    "tracking": extra["tracking"],
                    "fingerprint": "fixture-fingerprint",
                }
            )
        )
        return run_dir / "run_manifest.json"

    monkeypatch.setattr(train, "save_run_manifest", save_manifest)
    monkeypatch.setattr(
        train,
        "build_loaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("loader broke")),
    )
    monkeypatch.setattr(
        train,
        "finish_wandb",
        lambda *, exit_code: finished.append(exit_code),
    )

    with pytest.raises(RuntimeError, match="loader broke"):
        train.run_training(cfg)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    report = json.loads((run_dir / "run_report.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == {"type": "RuntimeError", "message": "loader broke"}
    assert report["summary"]["status"] == "failed"
    assert finished == [1]


def test_completed_training_finalizes_wandb_with_success(tmp_path, monkeypatch):
    import scaletraining.entrypoints.train as train

    cfg = OmegaConf.create(
        {
            "training": {
                "seed": 13,
                "compile_model": False,
                "batch_size": 1,
                "accum_steps": 1,
                "max_train_tokens": 2,
            },
            "device": {"device": "cpu"},
            "model": {
                "use_rope": False,
                "max_seq_len": 2,
                "n_layer": 1,
                "n_head": 1,
                "n_embed": 2,
            },
            "tokenizer": {"tokenizer_name": None},
            "optimizer": {"primary_optimizer": "adamw", "lr": 0.001},
            "paths": {"output_dir": str(tmp_path / "outputs")},
        }
    )
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    finished = []
    manifest_updates = []
    model = SimpleNamespace(
        token_embedding=SimpleNamespace(num_embeddings=16),
    )

    monkeypatch.chdir(tmp_path)
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
    def save_manifest(*args, **kwargs):
        path = run_dir / "run_manifest.json"
        path.write_text(json.dumps({"fingerprint": "fixture-fingerprint"}))
        return path

    monkeypatch.setattr(train, "save_run_manifest", save_manifest)
    monkeypatch.setattr(train, "build_loaders", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(train, "TransformerNetwork", lambda cfg: model)
    monkeypatch.setattr(train, "count_parameters", lambda model: (16, 16))
    monkeypatch.setattr(train, "log_model_metrics", lambda metrics: None)
    monkeypatch.setattr(
        train,
        "training_run",
        lambda *args, **kwargs: {
            "train_loss": [0.25],
            "tokens_processed": 2,
            "tokens_applied": 2,
            "optimizer_steps": 1,
            "stop_reason": "token_budget_reached",
            "incomplete_accumulation_tokens": 0,
            "incomplete_accumulation_microbatches": 0,
        },
    )
    monkeypatch.setattr(train, "save_model", lambda *args, **kwargs: str(run_dir))
    monkeypatch.setattr(
        train,
        "update_run_manifest",
        lambda *args, **kwargs: manifest_updates.append(kwargs),
    )
    monkeypatch.setattr(
        train,
        "refresh_run_report",
        lambda run_dir: (run_dir / "run_report.json", run_dir / "run_report.md"),
    )
    monkeypatch.setattr(
        train,
        "finish_wandb",
        lambda *, exit_code: finished.append(exit_code),
    )

    assert train.run_training(cfg) == 0.25
    assert finished == [0]
    result = json.loads((run_dir / "train_result.json").read_text())
    assert result["tokens_processed"] == 2
    assert result["tokens_applied"] == 2
    assert result["optimizer_steps"] == 1
    assert result["stop_reason"] == "token_budget_reached"
    assert result["dataset_fingerprint"]
    assert manifest_updates[0]["training_progress"]["tokens_processed"] == 2
