import importlib.util
import json
from pathlib import Path


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
    _write_json(
        run_dir / "run_manifest.json",
        {"fingerprint": "abc123", "training": {"seed": 13}},
    )
    _write_json(
        run_dir / "train_result.json",
        {
            "final_train_loss": 1.5,
            "model_path": str(run_dir / "model.pt"),
        },
    )
    _write_json(
        run_dir / "eval_results.json",
        {
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
    assert report["summary"]["final_train_loss"] == 1.5
    assert report["summary"]["validation"]["tokens"] == 8
    assert report["artifacts"]["lm_eval_result"]["present"] is False
    assert json_path.exists()
    assert md_path.exists()

    partial = run_report.build_report(tmp_path / "partial")
    assert partial["summary"]["checkpoint"] is None
    assert partial["artifacts"]["manifest"]["present"] is False
