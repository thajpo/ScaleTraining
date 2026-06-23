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

The smoke command uses `tests/fixtures/smoke_corpus`, writes all data and model
artifacts to a temporary directory, forces `device=cpu`, and checks that the run
contains `run_manifest.json`, `train_result.json`, `eval_results.json`,
`run_report.json`, and `run_report.md`.

Inspect model size for a config:

```bash
uv run python scripts/model_size.py
```

Inspect a completed run evidence bundle:

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
- Reproducible eval sidecars and reviewer-readable run reports.
- A hardware-agnostic CPU smoke path that exercises the artifact contract.
