# Architecture Notes

## Purpose
This document captures the current architecture of the ScaleTraining repo and the intended target architecture
after cleanup/unification. It is meant to be a shared reference for design decisions and review.

## Current Architecture (As-Is)

### High-level flow
1) Hydra config loads and is normalized into a structured config schema.
2) Data pipeline produces tokenized and optionally packed datasets. Packing means concatenating token sequences
   and slicing them into fixed-length blocks (max_seq_len) for efficient causal LM training.
3) Model is constructed from config (dense or MoE Transformer).
4) Training loop iterates until token budget, logs metrics, and saves artifacts.
5) Evaluation runs on checkpoints (currently in flux).
6) Generation can run on a checkpoint for qualitative inspection.

Entrypoints map to steps as follows:
- `prepare_data.py` handles step (2) explicitly (offline prep).
- `train.py` handles steps (1) through (4) and requires preprocessed artifacts.
- `run_evals.py` is meant to handle step (5).
- `generate_from_pretrained.py` handles step (6).

### Entrypoints
- `src/scaletraining/entrypoints/train.py`
  - Loads config, prepares tokenizer, builds loaders, constructs model, runs training loop, saves model.
- `src/scaletraining/entrypoints/prepare_data.py`
  - Tokenizes and packs datasets based on config. Supports multiple datasets.
  - Note: tokenization loads datasets eagerly via `datasets.load_dataset` (not streaming), so very large HF
    datasets are not practical here; streaming is supported in the mixed-corpus builder path.
- `src/scaletraining/entrypoints/run_evals.py`
  - Intended to run evals on a pretrained checkpoint; currently a GSM8K stub (tokenizes + builds a loader but
    does not compute a real score).
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
- Logs to W&B via `wandb_utils.py`.
- Saves model + run manifest via `artifacts.py`.

### Evaluation
- `eval_utils.py` provides perplexity evaluation and checkpoint loading helpers.
- `run_evals.py` is not aligned with `build_loaders` or benchmark suite goals.

### Artifacts
- Tokenized/packed datasets: fingerprinted paths via `path_utils.py`.
- Model checkpoints: saved under `outputs/` with run manifests.

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

## Current Pain Points
The largest pain point is too many options without a clear default (especially in tokenization/datasets).
We should simplify the default path while keeping advanced options available.
- Two parallel data paths:
  - Tokenize+pack flow vs mixed-corpus builder scripts.
- Evals are drifting:
  - ARC was removed; current GSM8K eval is stubby and not integrated.
- Mixed config access patterns:
  - Most entrypoints use `load_project_config` (structured, schema-checked), but evals flatten configs into a
    plain dict for convenience. This can bypass schema checks and drift from the “single source of truth.”
- Fingerprinting is implicit and under-documented:
  - Multiple modules rely on `config_fingerprint`, but the “what changes the hash” contract is not explicit,
    so it is easy to change tokenization behavior without changing the fingerprint.
- README and docs do not describe a single “happy path” clearly. // true that

## Target Architecture (To-Be)

### Principles
- One “happy path” for standard training and evaluation.
- One primary data pipeline for most users.
- Evaluation harness aligned with training data conventions and shared loaders.
- Documentation emphasizes the intended workflow and hides legacy paths.
- Data preparation is explicit: training never auto-tokenizes or auto-packs.

### Proposed happy path
1) `prepare_data.py` produces tokenized/packed artifacts for the chosen dataset(s).
2) `train.py` runs training using packed datasets and logs results (fails fast if artifacts are missing).
3) `run_evals.py` runs a stable benchmark suite and perplexity evals.
4) `generate_from_pretrained.py` supports qualitative checks.

### Unification changes (high level)
- Treat `prepare_data.py` as the single official data prep entrypoint.
- Remove implicit tokenization/packing from training; require preprocessed artifacts.
- Align evals to use `build_loaders(..., for_training=False)` and shared metrics.
- Consolidate eval logic in `eval_utils.py` and keep `run_evals.py` thin.
- Document mixed-corpus builder as advanced/experimental (or align its outputs with standard artifacts).

### Mixed-corpus builder (advanced path)
- Keep it separate from the happy path but make it dead simple to use.
- Default single-dataset mode uses `PleIAs/SYNTH`.
- Mixed corpus is curated by us (a deliberate mix of math, coding, etc.).
- CLI expects a single flag `--preset {tiny,standard}` and otherwise uses config defaults.
- Presets control token targets, val ratio, and processing scale for quick iteration.
- Preset targets (same for single dataset and mix): `tiny=50M tokens`, `standard=1B tokens`.
- Default text fields for `PleIAs/SYNTH`: concatenate `query` + `synthetic_answer` (and optionally
  `synthetic_reasoning` if you want chain-of-thought style data).

## Migration Plan (small steps)
1) Docs first: document the happy path and current architecture in this file and README.
2) Evals: restore one stable benchmark (ARC or Wikitext PPL) using shared loader.
3) Data: decide whether mixed-corpus is official or advanced, then align docs and artifact naming.
4) Cleanup: remove or demote redundant entrypoint docs once the happy path is stable.

## Additional Simplifications (from full repo read)
- Dependencies in `pyproject.toml` are incomplete (code imports torch/transformers/datasets/tokenizers/wandb).
- Tests are inconsistent: several test files are empty, while optimizer tests are heavy (slow for unit runs).
- Tokenizer config mismatch: `train_tokenizer.py` reads `tokenizer_vocab_size`, but config uses
  `custom_tokenizer_vocab_size`.
- Config correctness: `moe_n_layers: ${n_layer}` likely should be `${model.n_layer}`.
- Generation config pins a specific checkpoint path; default could be `latest` or empty.

## MoE Routing Metrics (planned)
- Log lightweight routing stats during training when MoE is enabled.
- Metrics include load balance (min/max/std), router entropy, top-k gate stats, and aux loss.
- Per-layer stats are logged for debugging; aggregate means help spot global collapse.

## Non-goals
- No large refactor of the model or training loop in this phase.
- No change to optimizer design or MoE routing logic.
- No overhaul of Hydra config schema.
