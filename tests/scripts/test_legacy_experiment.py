import importlib.util
from collections import Counter
import json
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


recover = _load_script("recover_legacy_runs.py")
plot = _load_script("plot_legacy_experiment.py")
audit = _load_script("audit_legacy_checkpoints.py")
_FIXTURES = _ROOT / "tests" / "fixtures" / "legacy_runs"


def _write_control_archive(
    root: Path,
    run_id: str,
    lr: float,
    *,
    n_layer: int = 1,
) -> Path:
    run_dir = root / f"run-20250919_000000-{run_id}"
    files = run_dir / "files"
    files.mkdir(parents=True)
    config = {
        "_wandb": {
            "value": {
                "e": {
                    run_id: {
                        "git": {"commit": recover.HISTORICAL_CODE_COMMIT},
                        "startedAt": f"2025-09-19T00:00:0{len(run_id)}Z",
                    }
                }
            }
        },
        "logging": {"value": {"wandb_project_name": "tiny-stories-base"}},
        "model": {
            "value": {
                "UE_bias": False,
                "accum_steps": 1,
                "attn_dropout": 0.0,
                "batch_size": 2,
                "bias": True,
                "lr": lr,
                "lr_schedule": "cosine",
                "max_seq_len": 8,
                "max_train_tokens": 100,
                "muon_lr": 0.02,
                "n_embed": 2,
                "n_head": 1,
                "n_hidden": 4,
                "n_layer": n_layer,
                "primary_optimizer": "muon",
                "resid_dropout": 0.0,
                "use_baseline_adam": False,
                "use_moe": False,
                "vocab_size": 4,
                "warmup_tokens": 10,
            }
        },
        "sweep": {"value": {"name": "muon_lr"}},
        "tokenizer": {
            "value": {
                "dataset_tag": "",
                "hf_dataset_names": "roneneldan/TinyStories",
                "tokenizer_name": "fixture-tokenizer",
            }
        },
    }
    (files / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (files / "wandb-summary.json").write_text(
        json.dumps({"model/total_params": 100}), encoding="utf-8"
    )
    return run_dir


def test_legacy_archive_parser_recovers_effective_optimizer_wiring():
    runs = recover.scan_archives(_FIXTURES)

    assert [run["run_id"] for run in runs] == ["fixture1", "fixture2"]
    hybrid, baseline = runs
    assert hybrid["optimizer"] == {
        "recorded_primary": "muon",
        "effective_wiring": "muon_matrices_plus_adamw_auxiliary",
        "matrix_optimizer": "muon",
        "matrix_lr": 0.02,
        "auxiliary_optimizer": "adamw",
        "auxiliary_lr": 0.03,
        "use_baseline_adam": False,
    }
    assert hybrid["training"]["seed"] is None
    assert hybrid["git_commit"] == "fixture-commit"
    assert hybrid["history_file_present"] is True
    assert "history_present" not in hybrid
    assert baseline["optimizer"]["effective_wiring"] == "adamw_all_parameters"
    assert baseline["optimizer"]["matrix_lr"] == 0.0001
    assert baseline["optimizer"]["auxiliary_lr"] is None


def test_legacy_archive_parser_supports_split_schema_groups(tmp_path):
    run_dir = tmp_path / "run-20260101_000000-split"
    files = run_dir / "files"
    files.mkdir(parents=True)
    config = {
        "transformer": {
            "value": {
                "UE_bias": False,
                "attn_dropout": 0.1,
                "bias": True,
                "n_embed": 896,
                "n_head": 14,
                "n_hidden": 4864,
                "n_layer": 24,
                "resid_dropout": 0.2,
                "vocab_size": 50257,
            }
        },
        "optimizer": {
            "value": {
                "lr": 0.01,
                "lr_schedule": "cosine",
                "muon_lr": 0.02,
                "primary_optimizer": "adamuon",
                "use_baseline_adam": False,
                "warmup_tokens": 1_000_000,
            }
        },
        "training": {
            "value": {
                "accum_steps": 3,
                "batch_size": 24,
                "max_train_tokens": 80_000_000,
                "seed": 13,
            }
        },
        "moe": {"value": {"use_moe": False}},
        "tokenizer": {
            "value": {
                "dataset_names": ["HuggingFaceFW/fineweb"],
                "pretrained_tokenizer_name": "fixture-tokenizer",
            }
        },
    }
    (files / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    parsed = recover.parse_run_archive(run_dir)

    assert parsed["architecture"]["n_layer"] == 24
    assert parsed["architecture"]["n_embed"] == 896
    assert parsed["optimizer"]["effective_wiring"] == (
        "adamuon_matrices_plus_adamw_auxiliary"
    )
    assert parsed["optimizer"]["auxiliary_lr"] == 0.01
    assert parsed["training"]["token_budget"] == 80_000_000
    assert parsed["training"]["seed"] == 13
    assert parsed["tokenizer"] == "fixture-tokenizer"


def test_legacy_archive_parser_supports_flat_schema(tmp_path):
    run_dir = tmp_path / "run-20250925_000000-flat"
    files = run_dir / "files"
    files.mkdir(parents=True)
    config = {
        key: {"value": value}
        for key, value in {
            "UE_bias": False,
            "accum_steps": 3,
            "attn_dropout": 0.2,
            "batch_size": 24,
            "bias": True,
            "hf_dataset_names": "refined-web-mix",
            "lr": 0.02,
            "lr_schedule": "cosine",
            "max_train_tokens": 40_000_000,
            "muon_lr": 0.02,
            "n_embed": 896,
            "n_head": 14,
            "n_hidden": 4864,
            "n_layer": 24,
            "primary_optimizer": "muon",
            "resid_dropout": 0.2,
            "seed": 13,
            "tokenizer_name": "fixture-tokenizer",
            "use_baseline_adam": False,
            "use_moe": False,
            "vocab_size": 50257,
            "wandb_project_name": "fine-web-pretrain",
            "warmup_tokens": 1_000_000,
        }.items()
    }
    (files / "config.yaml").write_text(json.dumps(config), encoding="utf-8")

    parsed = recover.parse_run_archive(run_dir)

    assert parsed["project"] == "fine-web-pretrain"
    assert parsed["architecture"]["n_layer"] == 24
    assert parsed["optimizer"]["effective_wiring"] == (
        "muon_matrices_plus_adamw_auxiliary"
    )
    assert parsed["training"]["token_budget"] == 40_000_000
    assert parsed["training"]["seed"] == 13
    assert parsed["dataset"] == "refined-web-mix"


def test_committed_inventory_reflects_schema_aware_recovery():
    payload = json.loads(
        (_ROOT / "research/data/legacy_run_inventory.json").read_text()
    )
    runs = payload["runs"]

    assert payload["schema_version"] == 2
    assert payload["source"]["history_classification"] == "file_presence_only"
    assert payload["source"]["history_file_count"] == 109
    assert Counter(run["optimizer"]["effective_wiring"] for run in runs) == {
        "muon_matrices_plus_adamw_auxiliary": 90,
        "adamuon_matrices_plus_adamw_auxiliary": 12,
        "adamw_all_parameters": 7,
    }
    assert Counter(run["training"]["token_budget"] for run in runs) == {
        1_000_000: 8,
        10_000_000: 55,
        40_000_000: 42,
        80_000_000: 4,
    }


def test_sweep_control_validation_rejects_changes_outside_model_lr(tmp_path):
    first_dir = _write_control_archive(tmp_path, "first", 0.01)
    second_dir = _write_control_archive(tmp_path, "second", 0.02)
    run_dirs = {"first": first_dir, "second": second_dir}
    runs = [recover.parse_run_archive(run_dirs[run_id]) for run_id in run_dirs]

    fixed = recover._validate_sweep_controls(run_dirs, runs)

    assert fixed["n_layer"] == 1
    assert fixed["muon_matrix_lr"] == 0.02

    changed_dir = tmp_path / "changed"
    changed_dir.mkdir()
    third_dir = _write_control_archive(changed_dir, "second", 0.02, n_layer=2)
    changed_dirs = {"first": first_dir, "second": third_dir}
    changed_runs = [
        recover.parse_run_archive(changed_dirs[run_id]) for run_id in changed_dirs
    ]
    with pytest.raises(ValueError, match="differs outside the allowed model.lr"):
        recover._validate_sweep_controls(changed_dirs, changed_runs)


def test_legacy_history_parser_normalizes_keys_and_deduplicates_tokens():
    legacy = recover.read_loss_history(
        _FIXTURES / "run-20250919_111922-fixture1"
    )
    current = recover.read_loss_history(
        _FIXTURES / "run-20250920_100000-fixture2"
    )

    assert [point["tokens"] for point in legacy] == [1_000_000, 2_000_000, 3_000_000]
    assert legacy[-1]["train_loss_per_token"] == 3.9
    assert current[-1] == {
        "tokens": 3_000_000,
        "train_loss_per_token": 3.5,
        "logged_primary_lr": 0.00008,
    }


def test_svg_plots_are_generated_from_normalized_history(tmp_path):
    histories = [
        recover.read_loss_history(path)
        for path in (
            _FIXTURES / "run-20250919_111922-fixture1",
            _FIXTURES / "run-20250920_100000-fixture2",
        )
    ]
    payload = {
        "runs": [
            {
                "run_id": f"fixture{index}",
                "config": {"auxiliary_adamw_lr": lr},
                "terminal": {
                    "tokens": history[-1]["tokens"],
                    "train_loss_per_token": history[-1]["train_loss_per_token"],
                },
                "history": history,
            }
            for index, (lr, history) in enumerate(
                zip((0.03, 0.01), histories, strict=True), start=1
            )
        ]
    }

    paths = plot.render_plots(payload, tmp_path)

    assert all(path.is_file() for path in paths)
    assert "Hybrid Muon training loss" in paths[0].read_text()
    assert "Comparable loss near 3M tokens" in paths[1].read_text()
    assert "Terminal outcomes across auxiliary LR" in paths[2].read_text()
    assert "AdamW LR 0.03" in paths[0].read_text()


def test_checkpoint_audit_distinguishes_integrity_from_runtime_compatibility(
    tmp_path,
):
    checkpoint = tmp_path / "model.pt"
    fixed = {
        "UE_bias": False,
        "bias": True,
        "n_embed": 2,
        "n_hidden": 4,
        "n_layer": 1,
        "use_moe": False,
        "vocab_size": 4,
    }
    expected_shapes = audit._expected_dense_tensor_shapes(fixed)
    torch.save(
        {"state_dict": {key: torch.zeros(shape) for key, shape in expected_shapes.items()}},
        checkpoint,
    )
    run = {
        "run_id": "fixture",
        "checkpoint_run_dir": "fixture-dir",
        "fixed_conditions": fixed,
    }

    result = audit.inspect_checkpoint(checkpoint, run)

    assert result["architecture_integrity"]["status"] == (
        "complete_legacy_schema_match"
    )
    assert result["architecture_integrity"]["expected_tensor_count"] == 14
    assert result["current_runtime_compatibility"]["status"] == "incompatible"
    assert result["current_runtime_compatibility"]["historical_shared_layer_norm_keys"] == 2
    assert result["evaluation"]["status"] == "not_run"
    assert len(result["sha256"]) == 64


def test_checkpoint_audit_rejects_incomplete_state_dict(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "state_dict": {
                "token_embedding.weight": torch.zeros(5, 2),
                "W_ue.weight": torch.zeros(4, 2),
                "transformer_blocks.0.ln.weight": torch.ones(2),
                "transformer_blocks.0.ln.bias": torch.zeros(2),
            }
        },
        checkpoint,
    )
    run = {
        "run_id": "fixture",
        "checkpoint_run_dir": "fixture-dir",
        "fixed_conditions": {
            "UE_bias": False,
            "bias": True,
            "n_embed": 2,
            "n_hidden": 4,
            "n_layer": 1,
            "use_moe": False,
            "vocab_size": 4,
        },
    }

    result = audit.inspect_checkpoint(checkpoint, run)

    assert result["architecture_integrity"]["status"] == "mismatch"
    assert result["architecture_integrity"]["missing_keys"] == [
        "ln.bias",
        "ln.weight",
        "transformer_blocks.0.attention.kqv_block.bias",
        "transformer_blocks.0.attention.kqv_block.weight",
        "transformer_blocks.0.attention.out_projection.bias",
        "transformer_blocks.0.attention.out_projection.weight",
        "transformer_blocks.0.mlp.We.bias",
        "transformer_blocks.0.mlp.We.weight",
        "transformer_blocks.0.mlp.Wh.bias",
        "transformer_blocks.0.mlp.Wh.weight",
    ]
    assert result["architecture_integrity"]["shape_mismatches"] == {
        "token_embedding.weight": {"actual": [5, 2], "expected": [4, 2]}
    }


def test_committed_checkpoint_audit_records_complete_schema_validation():
    sweep = json.loads(
        (_ROOT / "research/data/tiny_stories_aux_lr_sweep.json").read_text()
    )
    payload = json.loads(
        (_ROOT / "research/data/checkpoint_compatibility.json").read_text()
    )
    expected_shapes = audit._expected_dense_tensor_shapes(sweep["fixed_conditions"])

    assert payload["schema_version"] == 2
    assert len(expected_shapes) == 54
    for checkpoint in payload["checkpoints"]:
        integrity = checkpoint["architecture_integrity"]
        assert integrity == {
            "expected_tensor_count": 54,
            "missing_keys": [],
            "problems": [],
            "shape_mismatches": {},
            "status": "complete_legacy_schema_match",
            "unexpected_keys": [],
        }
