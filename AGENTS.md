# ScaleTraining Agent Guide

## Project Thesis

ScaleTraining is a single-GPU language-model training harness. The thesis is
ML reliability infrastructure: explicit preprocessing, fingerprinted artifacts,
token-budgeted training, checkpoint manifests, reproducible evaluation results,
and reviewer-ready run evidence.

## Planning Files

- Use `code_current.md` for implementation contracts.
- Use `research_current.md` for experiment protocols and run outcomes.
- Keep research history append-only once a run has executed.
- Do not implement directly from brainstormed items; promote work to `Specd`
  first.

## Default Validation

Use lightweight checks unless the user explicitly asks for expensive runs:

```bash
uv run pytest -q
uv run python -m scaletraining.entrypoints.train --help
uv run python -m scaletraining.entrypoints.run_evals --help
uv run python -m scaletraining.entrypoints.run_lm_eval --help
uv run python scripts/run_plan.py --model-size tiny --token-budget 4096 -o device=cpu -o training=smoke
```

Avoid live long training, streaming corpus builds, and lm-eval benchmark runs in
review loops unless requested. Treat generated `outputs/<run>/run_report.json`
and `outputs/<run>/run_report.md` as the canonical evidence bundle for completed
runs.
