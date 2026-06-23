# research_current.md

## Institutional Knowledge
- This is the canonical research planning and run-ledger file for Lean Flow ML split mode.
- Keep research history append-style (hypothesis, config delta, run id, outcome, decision).
- Do not implement code directly from this file; code implementation must be planned in `code_current.md` `Specd`.
- Preserve executed run outcomes here for traceability.

## Beliefs
- [2026-02-08] Controlled comparisons matter more than single-run anecdotes.
  - Rationale: architecture claims should be reproducible.
- [2026-02-08] Use fixed token budgets/seeds/checkpoints for fair model comparisons.
  - Rationale: reduces variance and interpretation noise.
- [2026-02-08] Keep research scope tight during sweeps; avoid unrelated refactors.
  - Rationale: cleaner attribution of effects.

## Brainstormed
### Evaluation and Analysis
- [Research] Reproducible ablation protocol (dense vs MoE variants).
- [Research] PPL vs context-length evaluation.
- [Research] Zipf-lens analysis for low-frequency tokens.
- [MoE Analysis] Expert utilization and routing quality report.

### Data and Training Studies
- [Research] Add second math/logical dataset and compare effects.
- [Research] Inspect expert allocation quality during MoE training.
- [Research] Expert ablations for contribution analysis.
- [Research] Optional expert-vs-base interpretability study.
- [Research] Validate loss-over-total-tokens as primary training metric.

## Specd
### [Research] Dense vs MoE fixed-budget comparison
- status: `ready`
- behavior change (experiment/protocol change):
  - Run one dense baseline and one MoE variant using the same dataset fingerprint, token budget, seed, eval commands, and run report schema.
  - Compare final train loss, validation loss/perplexity, lm-eval results when available, MoE routing metrics, parameter count, and estimated FLOPs.
- files to touch (configs/scripts/analysis notebooks if any):
  - `research_current.md` for protocol and run-ledger updates.
  - Run configs or command overrides only; no model code changes during the experiment.
- fail-first tests (or validation checks for pipeline integrity):
  - Before running, `scripts/run_plan.py` must produce both dense and MoE plans with matching dataset fingerprint, token budget, and seed.
  - After running, each run directory must contain `run_manifest.json`, `train_result.json`, `eval_results.json`, and `run_report.json`.
- non-goals:
  - No architecture changes during the comparison.
  - No optimizer changes during the comparison.
  - No claims from a single failed or partial run.
- risks:
  - Small token budgets may be noisy and should be labeled as smoke evidence.
  - GPU nondeterminism may still introduce small differences despite fixed seeds.
  - MoE routing quality needs metric context; final loss alone is insufficient.
- touch points:
  - `scripts/run_plan.py` for planning commands.
  - `scripts/run_report.py` for final evidence bundles.
  - `research_current.md` run ledger for append-only outcome records.
- line anchors (optional):
  - n/a
- expected diff shape:
  - Documentation/run-ledger updates only after executing runs.
- review checks:
  - Dense and MoE run reports exist and use schema version 1.
  - Both reports show the same dataset fingerprint, seed, and token budget.
  - Research ledger records run IDs, commands, outcomes, and the decision.

When promoting a research item to `Specd`, include:
- behavior change (experiment/protocol change)
- files to touch (configs/scripts/analysis notebooks if any)
- fail-first tests (or validation checks for pipeline integrity)
- non-goals
- risks
- touch points
- line anchors (optional)
- expected diff shape
- review checks

Status values allowed in `Specd`: `ready`, `in_progress`, `review`.

## Run Ledger
- (append executed runs here)
