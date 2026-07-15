#!/usr/bin/env python3
"""Build a reviewer-facing evidence bundle for one ScaleTraining run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ARTIFACT_FILES = {
    "manifest": "run_manifest.json",
    "train_result": "train_result.json",
    "eval_result": "eval_results.json",
    "lm_eval_result": "lm_eval_results.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_status(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    return {
        "path": str(path.resolve(strict=False)),
        "present": path.exists(),
    }


def _validation_summary(eval_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not eval_result:
        return None
    validation = eval_result.get("validation", {})
    return {
        "loss": validation.get("loss"),
        "perplexity": validation.get("perplexity"),
        "tokens": validation.get("tokens"),
        "batches": validation.get("batches"),
    }


def build_report(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve(strict=False)
    artifacts = {
        name: _artifact_status(run_path, filename)
        for name, filename in ARTIFACT_FILES.items()
    }
    manifest = _read_json(run_path / ARTIFACT_FILES["manifest"])
    train_result = _read_json(run_path / ARTIFACT_FILES["train_result"])
    eval_result = _read_json(run_path / ARTIFACT_FILES["eval_result"])
    lm_eval_result = _read_json(run_path / ARTIFACT_FILES["lm_eval_result"])

    checkpoint = None
    if train_result:
        checkpoint = train_result.get("model_path")
    if checkpoint is None:
        candidate = run_path / "model.pt"
        checkpoint = str(candidate.resolve(strict=False)) if candidate.exists() else None

    dataset_fingerprint = None
    if manifest:
        dataset_fingerprint = manifest.get("fingerprint")
    if dataset_fingerprint is None and eval_result:
        dataset_fingerprint = eval_result.get("dataset", {}).get("fingerprint")

    summary = {
        "checkpoint": checkpoint,
        "dataset_fingerprint": dataset_fingerprint,
        "final_train_loss": (
            train_result.get("final_train_loss") if train_result else None
        ),
        "validation": _validation_summary(eval_result),
        "lm_eval_tasks": (
            lm_eval_result.get("tasks") if lm_eval_result else None
        ),
    }

    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_dir": str(run_path),
        "artifacts": artifacts,
        "summary": summary,
        "run_manifest": manifest,
        "train_result": train_result,
        "eval_result": eval_result,
        "lm_eval_result": lm_eval_result,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    validation = summary.get("validation") or {}
    lines = [
        "# ScaleTraining Run Report",
        "",
        f"- Run directory: `{report['run_dir']}`",
        f"- Checkpoint: `{summary.get('checkpoint') or 'not found'}`",
        f"- Dataset fingerprint: `{summary.get('dataset_fingerprint') or 'not recorded'}`",
        f"- Final train loss: `{summary.get('final_train_loss') if summary.get('final_train_loss') is not None else 'not recorded'}`",
        f"- Validation loss: `{validation.get('loss') if validation else 'not recorded'}`",
        f"- Validation perplexity: `{validation.get('perplexity') if validation else 'not recorded'}`",
        f"- lm-eval tasks: `{', '.join(summary.get('lm_eval_tasks') or []) if summary.get('lm_eval_tasks') else 'not recorded'}`",
        "",
        "## Artifact Status",
    ]
    for name, status in report["artifacts"].items():
        state = "present" if status["present"] else "missing"
        lines.append(f"- `{name}`: {state} at `{status['path']}`")
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], run_dir: str | Path) -> tuple[Path, Path]:
    run_path = Path(run_dir).expanduser().resolve(strict=False)
    run_path.mkdir(parents=True, exist_ok=True)
    json_path = run_path / "run_report.json"
    md_path = run_path / "run_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.run_dir)
    json_path, md_path = write_reports(report, args.run_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
