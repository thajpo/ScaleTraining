# code_current.md

## Institutional Knowledge
- This is the canonical implementation planning file for Lean Flow ML split mode.
- Implement only from `Specd` items with status `ready`, `in_progress`, or `review`.
- `review` means an active PR is under review with the user.
- Do not prune `Specd` items until PR merge.
- Delivery gates for implementation:
  - spec approved in `Specd`
  - isolated feature branch/worktree
  - fail-first test identified, then pass after implementation
  - regression tests for touched surfaces
  - PR includes scope/non-goals, changed files, fail->pass evidence, and risk/rollback notes

## Beliefs
- [2026-02-08] Spec-first implementation reduces churn and improves review quality.
  - Rationale: this repo has many small changes that benefit from explicit contracts.
- [2026-02-08] Prefer explicit modes/contracts over ambiguous defaults.
  - Rationale: easier debugging and predictable behavior.
- [2026-02-08] Prefer fail-fast behavior over compatibility shims unless compatibility is explicitly required.
  - Rationale: keeps maintenance burden low.

## Brainstormed
### Dense Transformer
- [Dense] Add RMSNorm option.
- [Dense] Add SwiGLU option for dense FFN.

### MoE
- [MoE] Respect `moe_n_layers` in model construction.
- [MoE] Add capacity factor with overflow/drop accounting.
- [MoE] Add router stabilization (z-loss / gate controls).
- [MoE] Expand routing correctness/stability tests.

### Evaluation / Infra
- [Eval] Strengthen eval harness (Wikitext PPL + one MC benchmark).
- [Testing] Add missing model/training-loop regression tests.
- [Wrap-up] Add final README training-run evidence series.

## Completed
### [Dense+MoE] Fix FFN residual wiring and split sublayer LayerNorms
- status: `completed`
- outcome:
  - `MLPBlock.forward` no longer applies an internal residual.
  - `TransformerBlock` and `MoEBlock` use separate `ln1` / `ln2` pre-norm sublayers.
  - Regression coverage exists in `tests/model/test_model.py`.

## Specd
### [Wrap-up] Hardware-agnostic reviewer smoke, CI, and docs
- status: `review`
- behavior change:
  - CI has stable `lint`, `test`, and `smoke` jobs.
  - `lint` uses syntax compilation and public entrypoint help checks.
  - `smoke` runs offline CPU prepare/train/eval/report over local fixture text.
  - Local text dataset specs are supported without HuggingFace network access.
  - CPU smoke skips `torch.compile`.
  - Generated `result.json` and `src/scaletraining.egg-info/` are removed from source control and ignored.
- files to touch:
  - `.github/workflows/ci.yml`
  - `src/scaletraining/data_processing/dataset_utils.py`
  - `src/scaletraining/entrypoints/train.py`
  - `scripts/smoke_cpu_e2e.py`
  - `tests/fixtures/smoke_corpus/*`
  - docs, config schema, and tests
- fail-first tests:
  - Local text dataset loading unit test.
  - Config schema test for `training.compile_model`.
  - Offline smoke script verifies run artifact sidecars in a temp directory.
- non-goals:
  - No final training-run series selection yet.
  - No Ruff/style-lint rollout.
  - No raw checkpoint artifacts committed.
- risks:
  - Smoke proves pipeline wiring, not model quality.
  - Local fixture support must not break existing HuggingFace dataset behavior.
- touch points:
  - `dataset_utils.load_hf_dataset` -> local text branch before HF loading.
  - `train.py` -> compile skip/config gate.
  - `.github/workflows/ci.yml` -> stable job names.
- expected diff shape:
  - Add one smoke script and tiny fixture data.
  - Remove tracked generated artifacts.
  - Modify CI, docs, config, dataset loading, and tests.
- review checks:
  - `uv run pytest -q`
  - `uv run python -m compileall -q src scripts tests`
  - public entrypoint `--help` checks
  - `uv run python scripts/smoke_cpu_e2e.py`

### [Infra] Persist reproducible eval and run evidence artifacts
- status: `review`
- behavior change:
  - Keep `evaluate_perplexity(...) -> (loss, perplexity)` compatible while adding a richer stats helper for artifact writing.
  - `run_evals.py` writes `eval_results.json` next to the resolved checkpoint by default.
  - `run_lm_eval.py` writes `lm_eval_results.json` next to the resolved checkpoint by default.
  - `train.py` keeps writing root `result.json` and also writes `train_result.json` into the checkpoint run directory.
  - `scripts/run_report.py --run-dir outputs/<run>` combines manifest, train, eval, and lm-eval artifacts into `run_report.json` and `run_report.md`.
  - `training.seed` controls Torch/runtime seeding and the shuffled training DataLoader generator.
- files to touch:
  - `src/scaletraining/util/eval_utils.py`
  - `src/scaletraining/entrypoints/run_evals.py`
  - `src/scaletraining/entrypoints/run_lm_eval.py`
  - `src/scaletraining/entrypoints/train.py`
  - `src/scaletraining/data_processing/dataloading.py`
  - `src/scaletraining/config/__init__.py`
  - `conf/training/default.yaml`
  - `conf/training/smoke.yaml`
  - `conf/eval/default.yaml`
  - `scripts/run_report.py`
  - docs and focused tests
- fail-first tests:
  - Unit test artifact writing with a temp checkpoint path and no real model checkpoint.
  - Unit test config schema accepts `eval.write_results`, `eval.output_dir`, and `training.seed`.
  - Unit test `evaluate_perplexity` still returns the existing two-value tuple.
  - Unit test `scripts/run_report.py` handles complete and partial run directories.
- non-goals:
  - No expensive training or live lm-eval benchmark run.
  - No new model architecture features.
  - No checkpoint format migration.
- risks:
  - Eval result JSON must remain valid when metrics are non-finite.
  - Sidecar artifact paths should not surprise users who override checkpoint paths.
  - Seed support improves controlled comparisons but does not guarantee full GPU determinism.
- touch points (path + function/class/block):
  - `src/scaletraining/util/eval_utils.py` -> eval stats/result builders and writers
  - `src/scaletraining/entrypoints/run_evals.py` -> validation result persistence
  - `src/scaletraining/entrypoints/run_lm_eval.py` -> benchmark result persistence
  - `src/scaletraining/entrypoints/train.py` -> run-local train result sidecar
  - `src/scaletraining/data_processing/dataloading.py` -> seeded shuffle generator
- line anchors (optional):
  - n/a
- expected diff shape:
  - Add one reporting script and one repo agent guide.
  - Modify eval/train/config/data loading/docs/tests.
  - Roughly +350 to +550 LOC including tests and docs.
- review checks:
  - `uv run pytest -q tests/config/test_config.py tests/util/test_eval_utils.py tests/test_run_report.py`
  - `uv run python -m compileall -q src scripts tests`
  - `uv run pytest -q`
  - `uv run python -m scaletraining.entrypoints.run_evals --help`
  - `uv run python scripts/run_plan.py --model-size tiny --token-budget 4096 -o device=cpu -o training=smoke`

Required contract for each `Specd` item:
- behavior change
- files to touch
- fail-first tests
- non-goals
- risks
- touch points (path + function/class/block)
- line anchors (optional)
- expected diff shape (add/modify/delete + rough LOC)
- review checks

Status values allowed in `Specd`: `ready`, `in_progress`, `review`.
