# Reviewer Demo

This demo path proves the software surface without running a long training job.

## Local Checks

Install dependencies:

```bash
uv sync
```

Run fast tests:

```bash
uv run pytest -q
```

Run lint or at least syntax compilation:

```bash
uv run python -m compileall -q src tests scripts
```

Run the offline CPU end-to-end smoke:

```bash
uv run python scripts/smoke_cpu_e2e.py
```

## Inspect The Training Workflow

Show the happy path without executing a long run:

```bash
uv run python -m scaletraining.entrypoints.prepare_data --help
uv run python -m scaletraining.entrypoints.train --help
uv run python -m scaletraining.entrypoints.run_evals --help
uv run python -m scaletraining.entrypoints.generate_from_pretrained --help
```

The smoke command copies `tests/fixtures/smoke_corpus` into a short temporary
path, writes all data and model artifacts to that temporary directory, and
forces `device=cpu`. It proves training automatically creates
`run_manifest.json`, `model.pt`, `model_config.json`, `train_result.json`,
`run_report.json`, and `run_report.md`; validation then adds `eval_results.json`
and refreshes the same reports.

Inspect model size for a config:

```bash
uv run python scripts/model_size.py
```

Explicitly rebuild a completed run evidence bundle if its sidecars changed:

```bash
uv run python scripts/run_report.py --run-dir outputs/<run>
```

## Heavier Work To Skip In A Review

- `scripts/build_pretraining_corpus.py --preset standard` targets a 1B-token
  corpus and is not a laptop demo.
- Real training runs can require GPU time and W&B credentials.
- `lm-eval` benchmark runs can be slow and should be reported as run artifacts,
  not run live for a reviewer.
- Raw checkpoints under `outputs/` are intentionally ignored; commit compact
  evidence summaries instead of model weights.

## What This Demonstrates

- Config-first ML infrastructure.
- Explicit preprocessing and artifact fingerprinting.
- A shared training surface for dense and MoE models.
- Testable entrypoints and model/data-processing contracts.
- Token-indexed W&B tracking linked from compact local run evidence.
- Automatically refreshed eval sidecars and reviewer-readable run reports.
- A hardware-agnostic CPU smoke path that exercises the artifact contract.
