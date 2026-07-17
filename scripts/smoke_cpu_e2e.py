#!/usr/bin/env python3
"""Run an offline CPU end-to-end smoke test in temporary directories."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "smoke_corpus"


def _hydra_string(value: Path | str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _run(cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _common_overrides(tmp_dir: Path, fixture_dir: Path) -> list[str]:
    tokenized_dir = tmp_dir / "data" / "tokenized"
    return [
        "device=cpu",
        "model=tiny",
        "training=smoke",
        "training.compile_model=false",
        "training.seed=13",
        "training.max_train_tokens=64",
        "training.batch_size=1",
        "training.accum_steps=1",
        "training.eval_max_batches=1",
        "training.eval_batch_size=1",
        "training.loader_num_workers=0",
        "logging.debug_memory=false",
        "optimizer.use_baseline_adam=true",
        "optimizer.primary_optimizer=adamw",
        "optimizer.lr=0.001",
        "optimizer.baseline_adam_config.lr=0.001",
        "optimizer.lr_schedule=constant",
        "optimizer.warmup_tokens=0",
        "model.max_seq_len=16",
        f"tokenizer.dataset_names=[{_hydra_string(fixture_dir)}]",
        "tokenizer.dataset_tag=[null]",
        "tokenizer.is_pretrained=false",
        "tokenizer.custom_tokenizer_vocab_size=128",
        "tokenizer.num_proc=1",
        "tokenizer.pack_num_proc=1",
        "tokenizer.pack_map_batch_size=1000",
        "tokenizer.pack_writer_batch_size=64",
        f"paths.tokenized_train_path={_hydra_string(tokenized_dir)}",
        f"paths.tokenized_eval_path={_hydra_string(tokenized_dir)}",
        f"paths.batched_tokenized_path={_hydra_string(tmp_dir / 'data' / 'packed')}",
        f"paths.tokenizer_train_data={_hydra_string(tmp_dir / 'data' / 'raw')}",
        f"paths.output_dir={_hydra_string(tmp_dir / 'outputs')}",
    ]


def run_smoke(tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    fixture_dir = tmp_dir / "corpus"
    shutil.copytree(FIXTURE_DIR, fixture_dir)
    overrides = _common_overrides(tmp_dir, fixture_dir)
    _run(
        [sys.executable, "-m", "scaletraining.entrypoints.prepare_data", *overrides],
        cwd=tmp_dir,
        env=env,
    )
    _run(
        [sys.executable, "-m", "scaletraining.entrypoints.train", *overrides],
        cwd=tmp_dir,
        env=env,
    )

    run_dirs = sorted((tmp_dir / "outputs").glob("*/model.pt"))
    if not run_dirs:
        raise RuntimeError("Smoke training did not produce a model checkpoint")
    model_path = run_dirs[-1]
    run_dir = model_path.parent
    training_expected = [
        "run_manifest.json",
        "model.pt",
        "model_config.json",
        "train_result.json",
        "run_report.json",
        "run_report.md",
    ]
    missing = [name for name in training_expected if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Smoke training did not automatically create artifacts: {missing}"
        )
    initial_report = json.loads((run_dir / "run_report.json").read_text())
    if initial_report["artifacts"]["eval_result"]["present"]:
        raise RuntimeError("Training report unexpectedly included evaluation results")

    tokenizer_paths = sorted((tmp_dir / "tokenizers").glob("*.json"))
    if not tokenizer_paths:
        raise RuntimeError("Smoke data preparation did not produce a tokenizer JSON")
    tokenizer_path = tokenizer_paths[-1]

    eval_overrides = [
        *overrides,
        f"+tokenizer.tokenizer_name={_hydra_string(tokenizer_path)}",
        f"generation.model_path={_hydra_string(model_path)}",
        f"eval.output_dir={_hydra_string(run_dir)}",
    ]
    _run(
        [sys.executable, "-m", "scaletraining.entrypoints.run_evals", *eval_overrides],
        cwd=tmp_dir,
        env=env,
    )
    missing = [name for name in ["eval_results.json"] if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Smoke run missing expected artifacts: {missing}")
    refreshed_report = json.loads((run_dir / "run_report.json").read_text())
    if not refreshed_report["artifacts"]["eval_result"]["present"]:
        raise RuntimeError("Evaluation did not refresh the run report")

    print(f"Smoke run artifacts: {run_dir}")
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary smoke directory for manual inspection.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.keep_temp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="scaletraining-smoke-"))
        run_smoke(tmp_dir)
        print(f"Kept smoke temp directory: {tmp_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="scaletraining-smoke-") as tmp:
            run_smoke(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
