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

## Specd
### [Dense+MoE] Fix FFN residual wiring and split sublayer LayerNorms
- status: `ready`
- behavior change:
  - Remove internal residual addition from `MLPBlock.forward`; residual is applied once at block level.
  - Split shared block norm into `ln1` and `ln2` in both `TransformerBlock` and `MoEBlock`.
  - Keep pre-norm residual structure: `x + attention(ln1(x))` and `x + ffn_or_moe(ln2(x))`.
- files to touch:
  - `src/scaletraining/model/model.py`
  - `tests/model/test_model.py`
- fail-first tests:
  - Add a unit test where `MLPBlock` projections are zeroed and dropout disabled; expected output is all zeros (fails on current residual-inside-MLP behavior).
  - Add a unit test asserting `TransformerBlock` and `MoEBlock` expose `ln1` and `ln2`.
  - Add a forward/backward finite smoke test for both dense and MoE block paths.
- non-goals:
  - No `moe_n_layers` policy changes.
  - No router/gating/capacity behavior changes.
  - No checkpoint key remap or compatibility shim for old checkpoints.
- risks:
  - Training dynamics will shift due to corrected residual math and norm placement.
  - Existing checkpoints trained with prior wiring are intentionally unsupported (fail-fast policy).
- touch points (path + function/class/block):
  - `src/scaletraining/model/model.py` -> `MLPBlock.forward`
  - `src/scaletraining/model/model.py` -> `TransformerBlock.__init__` / `TransformerBlock.forward`
  - `src/scaletraining/model/model.py` -> `MoEBlock.__init__` / `MoEBlock.forward`
  - `tests/model/test_model.py` -> new block-structure and residual-behavior tests
- line anchors (optional):
  - `src/scaletraining/model/model.py`: around `MLPBlock`, `TransformerBlock`, `MoEBlock`
- expected diff shape:
  - Modify 2 files.
  - Roughly +60 to +140 LOC (mostly new tests, small model edits).
- review checks:
  - `pytest tests/model/test_model.py`
  - Forward/backward finite checks pass for dense and MoE block smoke tests.
  - No unrelated architecture or routing logic changes in diff.

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
