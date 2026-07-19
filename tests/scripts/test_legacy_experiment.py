import importlib.util
import json
from pathlib import Path

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
    assert baseline["optimizer"]["effective_wiring"] == "adamw_all_parameters"
    assert baseline["optimizer"]["matrix_lr"] == 0.0001
    assert baseline["optimizer"]["auxiliary_lr"] is None


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
    torch.save(
        {
            "state_dict": {
                "token_embedding.weight": torch.zeros(4, 2),
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
        "fixed_conditions": {"vocab_size": 4, "n_embed": 2, "n_layer": 1},
    }

    result = audit.inspect_checkpoint(checkpoint, run)

    assert result["architecture_integrity"]["status"] == "compatible_with_recorded_config"
    assert result["current_runtime_compatibility"]["status"] == "incompatible"
    assert result["current_runtime_compatibility"]["historical_shared_layer_norm_keys"] == 2
    assert result["evaluation"]["status"] == "not_run"
    assert len(result["sha256"]) == 64
