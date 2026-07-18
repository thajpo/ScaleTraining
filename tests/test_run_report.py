import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from scaletraining.util.artifacts import build_checkpoint_provenance


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_report.py"
_SPEC = importlib.util.spec_from_file_location("run_report", _SCRIPT_PATH)
run_report = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_report)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_report_handles_complete_and_partial_run_dirs(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    provenance = build_checkpoint_provenance(checkpoint, run_dir)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "checkpoint": provenance,
            "fingerprint": "abc123",
            "status": "completed",
            "tracking": {
                "path": "entity/project/run-id",
                "url": "https://wandb.ai/entity/project/runs/run-id",
            },
            "training": {"seed": 13},
        },
    )
    _write_json(
        run_dir / "train_result.json",
        {
            "final_train_loss": 1.5,
            "model_path": "model.pt",
            "checkpoint": provenance,
            "dataset_fingerprint": "abc123",
            "tokens_processed": 8,
            "tokens_applied": 8,
            "optimizer_steps": 1,
            "stop_reason": "token_budget_reached",
            "incomplete_accumulation_tokens": 0,
            "incomplete_accumulation_microbatches": 0,
        },
    )
    _write_json(
        run_dir / "eval_results.json",
        {
            "checkpoint": provenance,
            "dataset": {"fingerprint": "abc123"},
            "validation": {
                "loss": 1.25,
                "perplexity": 3.49,
                "tokens": 8,
                "batches": 1,
            }
        },
    )

    report = run_report.build_report(run_dir)
    json_path, md_path = run_report.write_reports(report, run_dir)

    assert report["summary"]["dataset_fingerprint"] == "abc123"
    assert report["summary"]["status"] == "completed"
    assert report["summary"]["checkpoint"] == "model.pt"
    assert report["summary"]["final_train_loss"] == 1.5
    assert report["summary"]["training_progress"]["tokens_processed"] == 8
    assert report["summary"]["validation"]["tokens"] == 8
    assert report["artifacts"]["lm_eval_result"]["present"] is False
    assert json_path.exists()
    assert md_path.exists()
    assert "[entity/project/run-id](https://wandb.ai/entity/project/runs/run-id)" in (
        md_path.read_text()
    )

    partial = run_report.build_report(tmp_path / "partial")
    assert partial["summary"]["checkpoint"] is None
    assert partial["artifacts"]["manifest"]["present"] is False


def test_run_report_rejects_checkpoint_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(run_dir / "run_manifest.json", {"fingerprint": "abc123"})
    _write_json(
        run_dir / "eval_results.json",
        {
            "checkpoint": {"path": str(tmp_path / "other" / "model.pt")},
            "dataset": {"fingerprint": "abc123"},
        },
    )

    with pytest.raises(
        ValueError,
        match=r"checkpoint mismatch.*write the sidecars into this run directory",
    ):
        run_report.build_report(run_dir)


def test_run_report_rejects_dataset_fingerprint_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    _write_json(run_dir / "run_manifest.json", {"fingerprint": "training-data"})
    _write_json(
        run_dir / "lm_eval_results.json",
        {
            "checkpoint": {"path": str(checkpoint)},
            "dataset": {"fingerprint": "different-data"},
            "tasks": ["hellaswag"],
        },
    )

    with pytest.raises(
        ValueError,
        match=r"dataset fingerprint mismatch.*training dataset configuration",
    ):
        run_report.build_report(run_dir)


def test_run_report_accepts_moved_run_directory(tmp_path):
    original = tmp_path / "original" / "run"
    checkpoint = original / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"portable checkpoint")
    provenance = build_checkpoint_provenance(checkpoint, original)
    _write_json(
        original / "run_manifest.json",
        {"checkpoint": provenance, "fingerprint": "abc123"},
    )
    _write_json(
        original / "train_result.json",
        {
            "checkpoint": provenance,
            "model_path": "model.pt",
            "run_dir": str(original),
            "dataset_fingerprint": "abc123",
        },
    )
    _write_json(
        original / "eval_results.json",
        {
            "checkpoint": provenance,
            "dataset": {"fingerprint": "abc123"},
        },
    )

    report = run_report.build_report(original)
    run_report.write_reports(report, original)

    moved = tmp_path / "archive" / "run"
    moved.parent.mkdir(parents=True)
    shutil.move(original, moved)

    persisted_report = json.loads((moved / "run_report.json").read_text())
    markdown = (moved / "run_report.md").read_text()

    assert persisted_report["run_dir"] == "."
    assert persisted_report["summary"]["checkpoint"] == "model.pt"
    assert persisted_report["train_result"]["run_dir"] == "."
    assert {
        artifact["path"] for artifact in persisted_report["artifacts"].values()
    } == {
        "run_manifest.json",
        "model.pt",
        "model_config.json",
        "train_result.json",
        "eval_results.json",
        "lm_eval_results.json",
    }
    assert persisted_report["run_manifest"]["checkpoint"]["original_path"] == str(
        checkpoint
    )
    assert "- Run directory: `.`" in markdown
    assert f"at `{original / 'model.pt'}`" not in markdown

    rebuilt_report = run_report.build_report(moved)
    assert rebuilt_report["summary"]["checkpoint"] == "model.pt"


def test_run_report_rejects_checkpoint_digest_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted checkpoint")
    provenance = build_checkpoint_provenance(checkpoint, run_dir)
    _write_json(run_dir / "run_manifest.json", {"checkpoint": provenance})
    checkpoint.write_bytes(b"different checkpoint")

    with pytest.raises(ValueError, match=r"checkpoint digest mismatch"):
        run_report.build_report(run_dir)


@pytest.mark.parametrize("artifact", ["eval_results.json", "lm_eval_results.json"])
@pytest.mark.parametrize(
    ("container", "field", "expected"),
    [
        ("checkpoint", None, "checkpoint.path"),
        ("checkpoint", "path", "checkpoint.path"),
        ("checkpoint", "sha256", "checkpoint.sha256"),
        ("dataset", None, "dataset.fingerprint"),
        ("dataset", "fingerprint", "dataset.fingerprint"),
    ],
)
def test_digest_bearing_run_rejects_sidecar_missing_provenance(
    tmp_path, artifact, container, field, expected
):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    provenance = build_checkpoint_provenance(checkpoint, run_dir)
    _write_json(
        run_dir / "run_manifest.json",
        {"checkpoint": provenance, "fingerprint": "abc123"},
    )
    sidecar = {
        "checkpoint": dict(provenance),
        "dataset": {"fingerprint": "abc123"},
    }
    if field is None:
        del sidecar[container]
    else:
        del sidecar[container][field]
    _write_json(run_dir / artifact, sidecar)

    with pytest.raises(
        ValueError,
        match=rf"{artifact} is missing required provenance:.*{expected}",
    ):
        run_report.build_report(run_dir)


def test_digest_bearing_run_rejects_sidecar_without_canonical_fingerprint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    provenance = build_checkpoint_provenance(checkpoint, run_dir)
    _write_json(run_dir / "run_manifest.json", {"checkpoint": provenance})
    _write_json(
        run_dir / "eval_results.json",
        {
            "checkpoint": provenance,
            "dataset": {"fingerprint": "unbound-fingerprint"},
        },
    )

    with pytest.raises(ValueError, match=r"canonical run evidence does not record one"):
        run_report.build_report(run_dir)
