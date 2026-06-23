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

## Inspect The Training Workflow

Show the happy path without executing a long run:

```bash
uv run python -m scaletraining.entrypoints.prepare_data --help
uv run python -m scaletraining.entrypoints.train --help
uv run python -m scaletraining.entrypoints.run_evals --help
uv run python -m scaletraining.entrypoints.generate_from_pretrained --help
```

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

## What This Demonstrates

- Config-first ML infrastructure.
- Explicit preprocessing and artifact fingerprinting.
- A shared training surface for dense and MoE models.
- Testable entrypoints and model/data-processing contracts.
- Reproducible eval sidecars and reviewer-readable run reports.
