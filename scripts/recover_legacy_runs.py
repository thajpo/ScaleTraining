#!/usr/bin/env python3
"""Recover a deterministic inventory and one legacy experiment from W&B archives."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import yaml


HISTORICAL_CODE_COMMIT = "8ce307041f0f3c753c860d546775301106b59b0c"
SWEEP_RUN_IDS = (
    "emdbrww3",
    "zrnf1np4",
    "mxj0pn09",
    "kh1vl1i7",
    "mx3a2a0x",
    "31377z9j",
)
CHECKPOINT_RUN_DIRS = {
    "emdbrww3": "v=004acfd1__20250919T151430Z",
    "zrnf1np4": "v=004acfd1__20250919T151921Z",
    "mxj0pn09": "v=004acfd1__20250919T152521Z",
    "kh1vl1i7": "v=004acfd1__20250919T153155Z",
    "mx3a2a0x": "v=004acfd1__20250919T153818Z",
    "31377z9j": "v=004acfd1__20250919T154432Z",
}
EXPECTED_SWEEP_AUXILIARY_LRS = {0.005, 0.01, 0.025, 0.03, 0.07, 0.1}
EXPECTED_SWEEP_FIXED_CONDITIONS = {
    "dataset": "roneneldan/TinyStories",
    "dataset_tag": "",
    "tokenizer": "EleutherAI/gpt-neo-125M",
    "vocab_size": 50257,
    "n_layer": 5,
    "n_head": 8,
    "n_embed": 512,
    "n_hidden": 2048,
    "max_seq_len": 1000,
    "bias": True,
    "UE_bias": False,
    "use_moe": False,
    "parameter_count_recorded": 41489408,
    "token_budget": 40000000,
    "batch_size": 32,
    "accum_steps": 4,
    "attn_dropout": 0.2,
    "resid_dropout": 0.2,
    "lr_schedule": "cosine",
    "warmup_tokens": 1000000,
    "muon_matrix_lr": 0.02,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _unwrap(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def _mapping_group(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = _unwrap(config, key)
        if isinstance(value, dict):
            return value
    return {}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _started_at(wandb_config: Any) -> str | None:
    if not isinstance(wandb_config, dict):
        return None
    writers = wandb_config.get("e")
    if not isinstance(writers, dict):
        return None
    for writer in writers.values():
        if isinstance(writer, dict) and writer.get("startedAt"):
            return str(writer["startedAt"])
    return None


def _git_commit(wandb_config: Any, metadata: dict[str, Any]) -> str | None:
    if isinstance(wandb_config, dict):
        writers = wandb_config.get("e")
        if isinstance(writers, dict):
            for writer in writers.values():
                if not isinstance(writer, dict):
                    continue
                git = writer.get("git")
                if isinstance(git, dict) and git.get("commit"):
                    return str(git["commit"])
    git = metadata.get("git")
    if isinstance(git, dict) and git.get("commit"):
        return str(git["commit"])
    return None


def _effective_optimizer(model: dict[str, Any]) -> dict[str, Any]:
    primary = model.get("primary_optimizer")
    auxiliary_lr = _finite_number(model.get("lr"))
    matrix_lr = _finite_number(model.get("muon_lr"))
    baseline = model.get("use_baseline_adam")
    baseline_config = model.get("baseline_adam_config")
    if not isinstance(baseline_config, dict):
        baseline_config = {}

    if baseline:
        baseline_lr = _finite_number(baseline_config.get("lr"))
        return {
            "recorded_primary": primary,
            "effective_wiring": "adamw_all_parameters",
            "matrix_optimizer": "adamw",
            "matrix_lr": _first(baseline_lr, auxiliary_lr),
            "auxiliary_optimizer": None,
            "auxiliary_lr": None,
            "use_baseline_adam": True,
        }
    if primary in {"muon", "adamuon"}:
        return {
            "recorded_primary": primary,
            "effective_wiring": f"{primary}_matrices_plus_adamw_auxiliary",
            "matrix_optimizer": primary,
            "matrix_lr": matrix_lr,
            "auxiliary_optimizer": "adamw",
            "auxiliary_lr": auxiliary_lr,
            "use_baseline_adam": False,
        }
    return {
        "recorded_primary": primary,
        "effective_wiring": "adamw_all_parameters" if primary == "adamw" else None,
        "matrix_optimizer": primary,
        "matrix_lr": auxiliary_lr,
        "auxiliary_optimizer": None,
        "auxiliary_lr": None,
        "use_baseline_adam": bool(baseline) if baseline is not None else None,
    }


def parse_run_archive(run_dir: Path, *, entity: str = "thajpo") -> dict[str, Any]:
    """Parse the stable config/summary metadata from one local W&B run."""

    config_path = run_dir / "files" / "config.yaml"
    config = _read_config(config_path)
    summary = _read_json(run_dir / "files" / "wandb-summary.json")
    metadata = _read_json(run_dir / "files" / "wandb-metadata.json")
    flat_config = {key: _unwrap(config, key) for key in config}
    model = _mapping_group(config, "model", "transformer") or flat_config
    optimizer = _mapping_group(config, "optimizer") or model
    training = _mapping_group(config, "training", "train") or model
    moe = _mapping_group(config, "moe") or model
    tokenizer = _mapping_group(config, "tokenizer") or flat_config
    logging = _mapping_group(config, "logging") or flat_config
    sweep = _mapping_group(config, "sweep")
    wandb_config = _unwrap(config, "_wandb") or {}

    run_id = run_dir.name.rsplit("-", 1)[-1]
    project = logging.get("wandb_project_name")
    history_files = sorted(run_dir.glob("run-*.wandb"))
    fixture_history = run_dir / "history.jsonl"
    terminal_loss = _finite_number(
        _first(
            summary.get("train_per_token_loss"),
            summary.get("train/loss_per_token"),
        )
    )
    terminal_tokens = _finite_number(
        _first(summary.get("used tokens"), summary.get("progress/tokens"))
    )
    dataset = _first(
        tokenizer.get("hf_dataset_names"),
        tokenizer.get("dataset_names"),
    )
    parameter_count = _finite_number(
        _first(summary.get("model/total_params"), summary.get("model.total_params"))
    )
    return {
        "run_id": run_id,
        "archive": run_dir.name,
        "project": project,
        "url": (
            f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            if project
            else None
        ),
        "started_at": _started_at(wandb_config),
        "git_commit": _git_commit(wandb_config, metadata),
        "dataset": dataset,
        "dataset_tag": tokenizer.get("dataset_tag"),
        "tokenizer": _first(
            tokenizer.get("tokenizer_name"),
            tokenizer.get("pretrained_tokenizer_name"),
        ),
        "architecture": {
            "vocab_size": model.get("vocab_size"),
            "n_layer": model.get("n_layer"),
            "n_head": model.get("n_head"),
            "n_embed": model.get("n_embed"),
            "n_hidden": model.get("n_hidden"),
            "max_seq_len": model.get("max_seq_len"),
            "bias": model.get("bias"),
            "UE_bias": model.get("UE_bias"),
            "use_moe": _first(model.get("use_moe"), moe.get("use_moe")),
            "parameter_count_recorded": parameter_count,
        },
        "optimizer": _effective_optimizer(optimizer),
        "training": {
            "token_budget": training.get("max_train_tokens"),
            "batch_size": training.get("batch_size"),
            "accum_steps": training.get("accum_steps"),
            "seed": training.get("seed"),
            "attn_dropout": model.get("attn_dropout"),
            "resid_dropout": model.get("resid_dropout"),
            "lr_schedule": optimizer.get("lr_schedule"),
            "warmup_tokens": optimizer.get("warmup_tokens"),
        },
        "sweep_name": sweep.get("name"),
        "terminal": {
            "train_loss_per_token": terminal_loss,
            "tokens": terminal_tokens,
        },
        "history_file_present": bool(history_files or fixture_history.exists()),
        "config_complete": bool(model and tokenizer),
    }


def scan_archives(wandb_dir: Path, *, entity: str = "thajpo") -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        if not (run_dir / "files" / "config.yaml").exists():
            continue
        runs.append(parse_run_archive(run_dir, entity=entity))
    return sorted(runs, key=lambda run: run["run_id"])


def _history_rows_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            yield value


def _history_rows_from_wandb(path: Path) -> Iterable[dict[str, Any]]:
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:  # pragma: no cover - project dependency in normal use
        raise RuntimeError("The wandb package is required to scan .wandb archives") from exc

    store = DataStore()
    store.open_for_scan(str(path))
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.HasField("history"):
            continue
        row: dict[str, Any] = {}
        for item in record.history.item:
            key = ".".join(item.nested_key) if item.nested_key else item.key
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                continue
        yield row


def read_loss_history(run_dir: Path) -> list[dict[str, float | int]]:
    """Read and normalize token/loss history from a fixture or binary archive."""

    fixture = run_dir / "history.jsonl"
    if fixture.exists():
        rows = _history_rows_from_jsonl(fixture)
    else:
        archives = sorted(run_dir.glob("run-*.wandb"))
        if not archives:
            return []
        rows = _history_rows_from_wandb(archives[0])

    by_tokens: dict[int, dict[str, float | int]] = {}
    for row in rows:
        tokens = _finite_number(
            _first(row.get("used tokens"), row.get("progress/tokens"))
        )
        loss = _finite_number(
            _first(row.get("train_per_token_loss"), row.get("train/loss_per_token"))
        )
        if tokens is None or loss is None:
            continue
        normalized: dict[str, float | int] = {
            "tokens": int(tokens),
            "train_loss_per_token": float(loss),
        }
        logged_lr = _finite_number(
            _first(row.get("lr"), row.get("train/learning_rate"))
        )
        if logged_lr is not None:
            normalized["logged_primary_lr"] = float(logged_lr)
        by_tokens[int(tokens)] = normalized
    return [by_tokens[token] for token in sorted(by_tokens)]


def _source_config_without_auxiliary_lr(run_dir: Path) -> tuple[dict[str, Any], float]:
    config = _read_config(run_dir / "files" / "config.yaml")
    source = {
        key: deepcopy(_unwrap(config, key))
        for key in config
        if key != "_wandb"
    }
    model = source.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Selected run {run_dir.name} does not record model.lr")
    lr = _finite_number(model.pop("lr", None))
    if lr is None:
        raise ValueError(f"Selected run {run_dir.name} has no finite model.lr")
    return source, float(lr)


def _fixed_conditions(run: dict[str, Any]) -> dict[str, Any]:
    architecture = run["architecture"]
    optimizer = run["optimizer"]
    training = run["training"]
    return {
        "dataset": run["dataset"],
        "dataset_tag": run["dataset_tag"],
        "tokenizer": run["tokenizer"],
        "vocab_size": architecture["vocab_size"],
        "n_layer": architecture["n_layer"],
        "n_head": architecture["n_head"],
        "n_embed": architecture["n_embed"],
        "n_hidden": architecture["n_hidden"],
        "max_seq_len": architecture["max_seq_len"],
        "bias": architecture["bias"],
        "UE_bias": architecture["UE_bias"],
        "use_moe": architecture["use_moe"],
        "parameter_count_recorded": architecture["parameter_count_recorded"],
        "token_budget": training["token_budget"],
        "batch_size": training["batch_size"],
        "accum_steps": training["accum_steps"],
        "attn_dropout": training["attn_dropout"],
        "resid_dropout": training["resid_dropout"],
        "lr_schedule": training["lr_schedule"],
        "warmup_tokens": training["warmup_tokens"],
        "muon_matrix_lr": optimizer["matrix_lr"],
    }


def _validate_sweep_controls(
    run_dirs: dict[str, Path],
    selected_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_config: dict[str, Any] | None = None
    baseline_fixed: dict[str, Any] | None = None
    auxiliary_lrs: list[float] = []
    for run in selected_runs:
        run_id = run["run_id"]
        source_config, source_lr = _source_config_without_auxiliary_lr(
            run_dirs[run_id]
        )
        fixed = _fixed_conditions(run)
        if baseline_config is None:
            baseline_config = source_config
            baseline_fixed = fixed
        elif source_config != baseline_config:
            raise ValueError(
                f"Selected run {run_id} differs outside the allowed model.lr field"
            )
        elif fixed != baseline_fixed:
            raise ValueError(
                f"Selected run {run_id} has inconsistent recorded fixed conditions"
            )

        if run["git_commit"] != HISTORICAL_CODE_COMMIT:
            raise ValueError(
                f"Selected run {run_id} records git commit {run['git_commit']!r}; "
                f"expected {HISTORICAL_CODE_COMMIT}"
            )
        if run["optimizer"]["effective_wiring"] != (
            "muon_matrices_plus_adamw_auxiliary"
        ):
            raise ValueError(
                f"Selected run {run_id} does not use the expected hybrid Muon wiring"
            )
        if run["optimizer"]["auxiliary_lr"] != source_lr:
            raise ValueError(
                f"Selected run {run_id} does not map model.lr to auxiliary AdamW"
            )
        auxiliary_lrs.append(source_lr)

    if len(set(auxiliary_lrs)) != len(auxiliary_lrs):
        raise ValueError("Selected runs do not have one distinct model.lr per setting")
    if baseline_fixed is None:
        raise ValueError("No selected runs were provided")
    if baseline_fixed["use_moe"] is not False:
        raise ValueError("Selected runs are not the expected dense model family")
    return baseline_fixed


def build_sweep_payload(
    wandb_dir: Path,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {run["run_id"]: run for run in inventory}
    run_dirs = {
        path.name.rsplit("-", 1)[-1]: path
        for path in wandb_dir.glob("run-*")
        if (path / "files" / "config.yaml").exists()
    }
    missing = [run_id for run_id in SWEEP_RUN_IDS if run_id not in by_id]
    if missing:
        raise ValueError(f"Missing selected W&B archives: {', '.join(missing)}")

    selected_runs = [by_id[run_id] for run_id in SWEEP_RUN_IDS]
    fixed_conditions = _validate_sweep_controls(run_dirs, selected_runs)
    if fixed_conditions != EXPECTED_SWEEP_FIXED_CONDITIONS:
        raise ValueError("Selected runs do not match the documented fixed conditions")
    if {
        run["optimizer"]["auxiliary_lr"] for run in selected_runs
    } != EXPECTED_SWEEP_AUXILIARY_LRS:
        raise ValueError("Selected runs do not match the documented model.lr settings")
    selected = []
    for run_id in SWEEP_RUN_IDS:
        run = by_id[run_id]
        history = read_loss_history(run_dirs[run_id])
        if not history:
            raise ValueError(f"Selected run {run_id} has no usable loss history")
        token_budget = run["training"]["token_budget"]
        terminal_tokens = history[-1]["tokens"]
        selected.append(
            {
                "run_id": run_id,
                "project": run["project"],
                "url": run["url"],
                "checkpoint_run_dir": CHECKPOINT_RUN_DIRS[run_id],
                "config": {
                    "auxiliary_adamw_lr": run["optimizer"]["auxiliary_lr"],
                    "muon_matrix_lr": run["optimizer"]["matrix_lr"],
                    "token_budget": token_budget,
                },
                "terminal": {
                    "tokens": terminal_tokens,
                    "train_loss_per_token": history[-1]["train_loss_per_token"],
                    "budget_fraction": (
                        terminal_tokens / token_budget if token_budget else None
                    ),
                    "stop_reason": (
                        "token_budget_reached"
                        if token_budget and terminal_tokens >= token_budget
                        else "not_recorded_before_budget"
                    ),
                },
                "history": history,
            }
        )

    return {
        "schema_version": 2,
        "experiment_id": "tiny_stories_muon_auxiliary_adamw_lr",
        "source": {
            "kind": "local_wandb_binary_history",
            "project": "tiny-stories-base",
            "run_ids": list(SWEEP_RUN_IDS),
            "historical_code_commit": HISTORICAL_CODE_COMMIT,
            "control_validation": {
                "status": "passed",
                "varied_source_field": "model.lr",
            },
        },
        "hypothesis": (
            "The AdamW learning rate for embeddings, output head, biases, and other "
            "non-Muon parameters materially affects hybrid Muon training dynamics."
        ),
        "fixed_conditions": fixed_conditions,
        "comparison_horizon_tokens": 9000000,
        "varied_parameter": {
            "name": "auxiliary_adamw_lr",
            "historical_label": "sweep.name=muon_lr",
            "correction": (
                "The recorded model.lr changed; model.muon_lr remained 0.02. "
                "Historical optimizer wiring applies model.lr only to parameters "
                "excluded from Muon."
            ),
        },
        "limitations": [
            "No seed was recorded, so this is a single-run-per-setting comparison.",
            "No validation or benchmark histories were preserved for these runs.",
            "Terminal observations occur at different token counts because stop reasons were not recorded.",
            "Historical throughput values are excluded because the legacy timer did not measure the full training compute window.",
        ],
        "runs": selected,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("research/data/legacy_run_inventory.json"),
    )
    parser.add_argument(
        "--sweep-output",
        type=Path,
        default=Path("research/data/tiny_stories_aux_lr_sweep.json"),
    )
    parser.add_argument("--entity", default="thajpo")
    args = parser.parse_args()

    inventory = scan_archives(args.wandb_dir, entity=args.entity)
    inventory_payload = {
        "schema_version": 2,
        "source": {
            "kind": "local_wandb_archives",
            "run_count": len(inventory),
            "history_classification": "file_presence_only",
            "history_file_count": sum(
                run["history_file_present"] for run in inventory
            ),
        },
        "runs": inventory,
    }
    sweep_payload = build_sweep_payload(args.wandb_dir, inventory)
    _write_json(args.inventory_output, inventory_payload)
    _write_json(args.sweep_output, sweep_payload)
    print(f"Recovered {len(inventory)} runs into {args.inventory_output}")
    print(f"Recovered {len(sweep_payload['runs'])} sweep histories into {args.sweep_output}")


if __name__ == "__main__":
    main()
