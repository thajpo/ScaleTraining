# Architecture Notes

## Purpose
This document captures the current ScaleTraining architecture and its closeout
boundaries. It is a shared reference for design decisions and review.

## Current Architecture (As-Is)

### High-level flow
1) Hydra config loads and is normalized into a structured config schema.
2) Data pipeline produces tokenized and optionally packed datasets. Packing means concatenating token sequences
   and slicing them into fixed-length blocks (max_seq_len) for efficient causal LM training.
3) Model is constructed from config (dense or MoE Transformer).
4) Training allocates one evidence directory, records its tracking/lifecycle
   manifest, iterates until the token budget, and writes the checkpoint,
   terminal result, and initial reports.
5) Perplexity and lm-eval entrypoints write provenance-checked sidecars beside
   the checkpoint and refresh the reports.
6) Generation can run on a checkpoint for qualitative inspection.

Entrypoints map to steps as follows:
- `prepare_data.py` handles step (2) explicitly (offline prep).
- `train.py` handles steps (1) through (4) and requires preprocessed artifacts.
- `run_evals.py` and `run_lm_eval.py` handle step (5).
- `generate_from_pretrained.py` handles step (6).

### Entrypoints
- `src/scaletraining/entrypoints/train.py`
  - Loads config, prepares tokenizer, creates the run directory and manifest,
    initializes W&B, builds loaders and the model, runs training, then saves the
    checkpoint, terminal progress, and reports.
- `src/scaletraining/entrypoints/prepare_data.py`
  - Tokenizes and packs datasets based on config. Supports multiple datasets.
  - Note: tokenization loads datasets eagerly via `datasets.load_dataset` (not streaming), so very large HF
    datasets are not practical here; streaming is supported in the mixed-corpus builder path.
- `src/scaletraining/entrypoints/run_evals.py`
  - Loads a checkpoint and the shared validation loader, computes per-token loss
    and perplexity, writes `eval_results.json`, and refreshes the run report.
- `src/scaletraining/entrypoints/run_lm_eval.py`
  - Adapts a checkpoint to lm-evaluation-harness, writes
    `lm_eval_results.json`, and refreshes the run report.
- `src/scaletraining/entrypoints/generate_from_pretrained.py`
  - Loads checkpoint + tokenizer and generates text from a prompt.

### Data pipeline
- Dataset specs live in `cfg.tokenizer.dataset_names` + `cfg.tokenizer.dataset_tag`.
  - `dataset_tag` is an optional disambiguator for variants of a dataset (subset/config/mix) and is used in
    artifact fingerprinting and run metadata so different dataset variants do not collide.
- `dataset_utils.py` supports multiple datasets and concatenates shared splits.
- `tokenization.py` tokenizes text into `input_ids` with EOS; writes metadata.
- `batch_packer.py` packs tokenized sequences into fixed-length blocks.
- `dataloading.py` creates DataLoaders and syncs config with saved metadata.

### Model stack
- `TransformerNetwork` with:
  - Token embedding, Transformer blocks (dense or MoE), final LN, and tied output projection.
  - This is a standard decoder-only baseline; it intentionally omits some modern variants (e.g., RMSNorm,
    SwiGLU/GLU blocks, GQA/MQA, attention bias variants) to keep the core simple.
- MoE:
  - Router + top-k gating, per-expert FFNs, optional shared expert. This is a conventional top-k routing
    design with a load-balance auxiliary loss.
  - Router temperature/noise and load-balance loss (aux) used in training.

### Training loop
- `training_loop.py` handles token-based training until max token budget.
- Loss is computed as summed CE, normalized per token (standard for stable token-based logging and scheduling).
- Optimizers: Muon/AdaMuon + AdamW split for matrices vs other params.
- W&B schema v1 uses processed tokens as the common train, validation,
  performance, compute, and MoE axis. Throughput times the synchronized
  accumulation compute window; FLOPs are estimated, and CUDA peak-memory calls
  target the resolved device.
- Terminal progress separates processed tokens from tokens applied in complete
  optimizer windows and records optimizer steps, stop reason, and unfinished
  accumulation.
- `artifacts.py` creates the shared run directory before training and records
  requested/resolved device provenance, lifecycle, W&B identity, and checkpoint
  identity.

### Evaluation
- `eval_utils.py` provides perplexity evaluation and checkpoint loading helpers.
- `run_evals.py` uses `build_loaders(..., for_training=False)` for validation
  perplexity; `run_lm_eval.py` handles standardized benchmark tasks.
- Sidecars carry a run-relative `model.pt` path, SHA-256 digest, original path,
  and dataset fingerprint. A proposed result is checked against the manifest,
  checkpoint bytes, and existing sidecars before atomic replacement.
- Evaluation output defaults to the checkpoint parent. Any explicit
  `eval.output_dir` must be that checkpoint-owning run directory.

### Artifacts
- Tokenized/packed datasets: fingerprinted paths via `path_utils.py`.
- A run directory under `outputs/` contains `run_manifest.json`, `model.pt`,
  `model_config.json`, `train_result.json`, optional evaluation sidecars, and
  automatically refreshed JSON/Markdown reports.
- Manifests transition from `running` to `completed`, or to `failed` with error
  details. Reports use `.` and run-relative artifact paths so a complete bundle
  remains verifiable after moving; absolute `original_path` fields are only
  provenance.

#### Hashing / fingerprinting contracts
- Fingerprints are computed in `path_utils.config_fingerprint` from a subset of config fields
  (`dataset_names`, `dataset_tag`, tokenizer name, `max_seq_len`, and optional dataset config name).
- Purpose: prevent stale preprocessed data (tokenization/packing) from being reused when those inputs change.
- This hash controls tokenized/packed directory names and is used across:
  - Data prep (`tokenization.py`, `batch_packer.py`) when writing artifacts.
  - Data loading (`dataloading.py`) when resolving artifact locations.
  - Run metadata (`artifacts.py`) for reproducibility.
- This fingerprint does not drive checkpoint resume or optimizer state; checkpoint loading is path-based
  and independent of this hash.
- Contract: if any field that affects tokenization/packing changes, the fingerprint should change.
- What can be fingerprinted: explicit config values used to build tokenized/packed artifacts.
- What cannot be reliably fingerprinted today: code changes, dataset revisions, streaming offsets,
  tokenizer training randomness, and any implicit preprocessing behavior not represented in config.

## Current Tradeoffs

The standard prepare → train → evaluate → report path is explicit and tested.
The remaining tradeoffs are:

- Two parallel data paths:
  - Tokenize+pack flow vs mixed-corpus builder scripts.
- Fingerprints cover explicit preprocessing config, not code revisions, dataset
  revisions, or implicit behavior.
- Checkpoints remain weights-only; exact resume would require optimizer,
  progress, and RNG state plus an equivalence test.
- The harness still needs one controlled experiment and a written conclusion;
  additional generic platform work is not the closeout goal.

## Closeout Architecture

### Principles
- One happy path for standard training and evaluation.
- One primary data pipeline for most users.
- W&B owns detailed token-indexed history; compact local reports own reviewer
  evidence and links back to W&B.
- Evaluation uses training data conventions, shared loaders, and validated
  checkpoint/dataset provenance.
- Data preparation is explicit: training never auto-tokenizes or auto-packs.
- One GPU or CPU device per run; multi-GPU is outside the closeout scope.

### Happy path
1) `prepare_data.py` produces tokenized/packed artifacts for the chosen dataset(s).
2) `train.py` uses packed datasets, logs detailed W&B metrics, and automatically
   creates the checkpoint plus compact run evidence.
3) `run_evals.py` adds validation perplexity; `run_lm_eval.py` adds selected
   benchmarks. Each successful sidecar write refreshes the reports.
4) `generate_from_pretrained.py` supports qualitative checks without modifying
   the canonical evidence bundle.

### Implemented boundaries
- `prepare_data.py` is the official data-preparation entrypoint.
- Training requires preprocessed artifacts instead of tokenizing or packing
  implicitly.
- Evaluation uses shared loaders and metrics, with common logic in
  `eval_utils.py` and thin entrypoints.
- The mixed-corpus builder remains a separate advanced path.

### Mixed-corpus builder (advanced path)
- Keep it separate from the happy path but make it dead simple to use.
- Default single-dataset mode uses `PleIAs/SYNTH`.
- Mixed corpus is curated by us (a deliberate mix of math, coding, etc.).
- CLI expects a single flag `--preset {tiny,standard}` and otherwise uses config defaults.
- Presets control token targets, val ratio, and processing scale for quick iteration.
- Preset targets (same for single dataset and mix): `tiny=50M tokens`, `standard=1B tokens`.
- Default text fields for `PleIAs/SYNTH`: concatenate `query` + `synthetic_answer` (and optionally
  `synthetic_reasoning` if you want chain-of-thought style data).

## Remaining Closeout

1) Recover historical runs, but classify checkpoints without matching validation
   and report sidecars as unverified artifacts.
2) Run one controlled, modest experiment with fixed fingerprints, budgets,
   evaluation settings, and appropriate seeds.
3) Preserve its compact reports and write a specific conclusion backed by the
   W&B history.

## MoE Routing Metrics
- Log lightweight routing stats during training when MoE is enabled.
- Metrics include load balance (min/max/std), router entropy, top-k gate stats, and aux loss.
- Per-layer stats are logged for debugging; aggregate means help spot global collapse.

## Non-goals
- No large refactor of the model or training loop in this phase.
- No change to optimizer design or MoE routing logic.
- No overhaul of Hydra config schema.
