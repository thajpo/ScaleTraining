"""Build compact, reviewer-facing evidence bundles for training runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACT_FILES = {
    "manifest": "run_manifest.json",
    "checkpoint": "model.pt",
    "model_config": "model_config.json",
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


def _validation_summary(
    eval_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not eval_result:
        return None
    validation = eval_result.get("validation", {})
    return {
        "loss": validation.get("loss"),
        "perplexity": validation.get("perplexity"),
        "tokens": validation.get("tokens"),
        "batches": validation.get("batches"),
    }


def _checkpoint_value(payload: dict[str, Any] | None, key: str) -> Any:
    if not payload:
        return None
    value = payload.get(key)
    if isinstance(value, dict):
        return value.get("path")
    return value


def _resolved_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve(strict=False)


def _validate_provenance(
    run_path: Path,
    manifest: dict[str, Any] | None,
    train_result: dict[str, Any] | None,
    eval_result: dict[str, Any] | None,
    lm_eval_result: dict[str, Any] | None,
) -> None:
    expected_checkpoint = (run_path / ARTIFACT_FILES["checkpoint"]).resolve(
        strict=False
    )
    checkpoint_sources = {
        "run_manifest.json": _checkpoint_value(manifest, "checkpoint"),
        "train_result.json": _checkpoint_value(train_result, "model_path"),
        "eval_results.json": _checkpoint_value(eval_result, "checkpoint"),
        "lm_eval_results.json": _checkpoint_value(lm_eval_result, "checkpoint"),
    }
    checkpoint_mismatches = {
        source: _resolved_path(value)
        for source, value in checkpoint_sources.items()
        if value is not None and _resolved_path(value) != expected_checkpoint
    }

    fingerprint_sources = {
        "run_manifest.json": manifest.get("fingerprint") if manifest else None,
        "train_result.json": (
            train_result.get("dataset_fingerprint") if train_result else None
        ),
        "eval_results.json": (
            (eval_result.get("dataset") or {}).get("fingerprint")
            if eval_result
            else None
        ),
        "lm_eval_results.json": (
            (lm_eval_result.get("dataset") or {}).get("fingerprint")
            if lm_eval_result
            else None
        ),
    }
    recorded_fingerprints = {
        source: str(value)
        for source, value in fingerprint_sources.items()
        if value is not None
    }
    fingerprint_mismatch = len(set(recorded_fingerprints.values())) > 1

    problems = []
    if checkpoint_mismatches:
        details = ", ".join(
            f"{source} references {path}"
            for source, path in checkpoint_mismatches.items()
        )
        problems.append(
            f"checkpoint mismatch ({details}; expected {expected_checkpoint})"
        )
    if fingerprint_mismatch:
        details = ", ".join(
            f"{source} records {fingerprint}"
            for source, fingerprint in recorded_fingerprints.items()
        )
        problems.append(f"dataset fingerprint mismatch ({details})")
    if problems:
        raise ValueError(
            f"Cannot build canonical report for {run_path}: "
            + "; ".join(problems)
            + ". Re-run evaluation for this checkpoint with its training "
            "dataset configuration and write the sidecars into this run directory."
        )


def build_report(run_dir: str | Path) -> dict[str, Any]:
    """Read the available sidecars for one run without requiring all of them."""

    run_path = Path(run_dir).expanduser().resolve(strict=False)
    artifacts = {
        name: _artifact_status(run_path, filename)
        for name, filename in ARTIFACT_FILES.items()
    }
    manifest = _read_json(run_path / ARTIFACT_FILES["manifest"])
    train_result = _read_json(run_path / ARTIFACT_FILES["train_result"])
    eval_result = _read_json(run_path / ARTIFACT_FILES["eval_result"])
    lm_eval_result = _read_json(run_path / ARTIFACT_FILES["lm_eval_result"])
    _validate_provenance(
        run_path,
        manifest,
        train_result,
        eval_result,
        lm_eval_result,
    )

    checkpoint = None
    if train_result:
        checkpoint = train_result.get("model_path")
    if checkpoint is None:
        candidate = run_path / ARTIFACT_FILES["checkpoint"]
        checkpoint = str(candidate.resolve(strict=False)) if candidate.exists() else None

    dataset_fingerprint = None
    if manifest:
        dataset_fingerprint = manifest.get("fingerprint")
    if dataset_fingerprint is None and eval_result:
        dataset_fingerprint = eval_result.get("dataset", {}).get("fingerprint")

    summary = {
        "status": manifest.get("status") if manifest else None,
        "checkpoint": checkpoint,
        "dataset_fingerprint": dataset_fingerprint,
        "final_train_loss": (
            train_result.get("final_train_loss") if train_result else None
        ),
        "training_progress": (
            {
                key: train_result.get(key)
                for key in (
                    "tokens_processed",
                    "tokens_applied",
                    "optimizer_steps",
                    "stop_reason",
                    "incomplete_accumulation_tokens",
                    "incomplete_accumulation_microbatches",
                )
                if key in train_result
            }
            if train_result
            else None
        ),
        "validation": _validation_summary(eval_result),
        "lm_eval_tasks": lm_eval_result.get("tasks") if lm_eval_result else None,
        "tracking": manifest.get("tracking") if manifest else None,
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
    tracking = summary.get("tracking") or {}
    tracking_url = tracking.get("url")
    tracking_label = tracking_url or tracking.get("path") or "not available"
    tracking_display = (
        f"[{tracking.get('path') or tracking_url}]({tracking_url})"
        if tracking_url
        else f"`{tracking_label}`"
    )
    final_train_loss = summary.get("final_train_loss")
    training_progress = summary.get("training_progress") or {}
    validation_perplexity = validation.get("perplexity") if validation else None
    lm_eval_tasks = summary.get("lm_eval_tasks") or []
    final_train_loss_display = (
        final_train_loss if final_train_loss is not None else "not recorded"
    )
    validation_perplexity_display = (
        validation_perplexity
        if validation_perplexity is not None
        else "not recorded"
    )
    lines = [
        "# ScaleTraining Run Report",
        "",
        f"- Run directory: `{report['run_dir']}`",
        f"- Status: `{summary.get('status') or 'not recorded'}`",
        f"- Checkpoint: `{summary.get('checkpoint') or 'not found'}`",
        f"- W&B run: {tracking_display}",
        f"- Dataset fingerprint: `{summary.get('dataset_fingerprint') or 'not recorded'}`",
        f"- Final train loss: `{final_train_loss_display}`",
        f"- Tokens processed: `{training_progress.get('tokens_processed', 'not recorded')}`",
        f"- Tokens applied: `{training_progress.get('tokens_applied', 'not recorded')}`",
        f"- Optimizer steps: `{training_progress.get('optimizer_steps', 'not recorded')}`",
        f"- Stop reason: `{training_progress.get('stop_reason') or 'not recorded'}`",
        f"- Incomplete accumulation tokens: `{training_progress.get('incomplete_accumulation_tokens', 'not recorded')}`",
        f"- Validation loss: `{validation.get('loss') if validation else 'not recorded'}`",
        f"- Validation perplexity: `{validation_perplexity_display}`",
        f"- lm-eval tasks: `{', '.join(lm_eval_tasks) if lm_eval_tasks else 'not recorded'}`",
        "",
        "## Artifact Status",
    ]
    for name, status in report["artifacts"].items():
        state = "present" if status["present"] else "missing"
        lines.append(f"- `{name}`: {state} at `{status['path']}`")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any], run_dir: str | Path
) -> tuple[Path, Path]:
    run_path = Path(run_dir).expanduser().resolve(strict=False)
    run_path.mkdir(parents=True, exist_ok=True)
    json_path = run_path / "run_report.json"
    md_path = run_path / "run_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def refresh_run_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Rebuild both report formats from the sidecars currently on disk."""

    report = build_report(run_dir)
    return write_reports(report, run_dir)


__all__ = [
    "ARTIFACT_FILES",
    "build_report",
    "refresh_run_report",
    "render_markdown",
    "write_reports",
]
