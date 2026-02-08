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
No research execution contracts are `ready` yet.

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
