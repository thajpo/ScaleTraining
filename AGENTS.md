# ScaleTraining Agent Guide

## Project Thesis

ScaleTraining is a single-GPU language-model training harness. The thesis is
ML reliability infrastructure: explicit preprocessing, fingerprinted artifacts,
token-budgeted training, checkpoint manifests, reproducible evaluation results,
and reviewer-ready run evidence.

## Change Workflow

- Discuss complex or ambiguous changes before implementation.
- Use GitHub issues for substantial approved work and experiment protocols.
- Put implementation scope, non-goals, acceptance evidence, and risks in the
  pull request description.
- Keep repository documentation focused on durable architecture and behavior;
  use Git, issues, and pull requests for planning history and closed work.
- Preserve completed experiment evidence in generated run reports. Add a
  curated research document only after real runs produce conclusions worth
  retaining.

## Closeout Boundary

- Prefer one controlled experiment and a clear written conclusion over adding
  more training-platform features.
- Do not add multi-GPU support merely to broaden the feature list.
- Checkpoints are currently weights-only. Do not claim exact interruption
  recovery without optimizer, progress, and RNG state plus an equivalence test.
- Treat historical checkpoints without validation and report sidecars as
  unverified artifacts, not experimental evidence.
- Treat `research/scale_training_closeout.md` and its generated SVGs as the
  durable historical conclusion. Regenerate them through the committed scripts;
  do not hand-edit generated data or strengthen its bounded claims without new
  evidence.

## Default Validation

Use lightweight checks unless the user explicitly asks for expensive runs:

```bash
uv run pytest -q
uv run python -m scaletraining.entrypoints.train --help
uv run python -m scaletraining.entrypoints.run_evals --help
uv run python -m scaletraining.entrypoints.run_lm_eval --help
uv run python scripts/run_plan.py --model-size tiny --token-budget 4096 -o device=cpu -o training=smoke
uv run python scripts/smoke_cpu_e2e.py
```

Avoid live long training, streaming corpus builds, and lm-eval benchmark runs in
review loops unless requested. Treat generated `outputs/<run>/run_report.json`
and `outputs/<run>/run_report.md` as the canonical evidence bundle for completed
runs.
