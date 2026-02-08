# ScaleTraining

A harness for training dense or MoE transformers on a single GPU. This is a personal project for experimenting with LLM training workflows.

## Quick Start (Happy Path)

```bash
# 1. Prepare data (explicit preprocessing required)
python -m scaletraining.entrypoints.prepare_data

# 2. Train
python -m scaletraining.entrypoints.train

# 3. Evaluate (validation perplexity)
python -m scaletraining.entrypoints.run_evals

# 4. Generate from checkpoint
python -m scaletraining.entrypoints.generate_from_pretrained

# 5. Run standardized benchmarks (hellaswag, mmlu, arc_easy, etc.)
LM_EVAL_TASKS=hellaswag python -m scaletraining.entrypoints.run_lm_eval
```

LEAN_FLOW_QUESTION_DRILL: explicit

**Note:** Training and evaluation now fail fast if preprocessed artifacts are missing. Run `prepare_data.py` first.

## Advanced: Mixed Corpus Builder

For larger pretraining runs with streaming:

```bash
# Single dataset (SYNTH default)
python scripts/build_pretraining_corpus.py --preset tiny

# Curated mix (math, code, etc.)
python scripts/build_pretraining_corpus.py --preset standard --corpus mix
```

Presets:
- `tiny`: 50M tokens (quick iteration)
- `standard`: 1B tokens (full run)

## Configuration

Uses Hydra for configuration management. Key config groups:

- `device`: CUDA/CPU settings, flash attention
- `training`: batch size, accumulation, learning rate schedule
- `model`: architecture (layers, heads, embeddings)
- `moe`: Mixture of Experts (experts, top-k, routing)
- `optimizer`: Muon/AdaMuon/AdamW with scheduling
- `tokenizer`: dataset selection, packing options
- `logging`: Weights & Biases integration
- `generation`: sampling parameters for inference

Override any config value:
```bash
python -m scaletraining.entrypoints.train model.n_layer=8 training.batch_size=32
```

## Entrypoints

- **`prepare_data.py`**: Tokenize and pack datasets offline. Required before training.
- **`train.py`**: Train until token budget, with validation evals.
- **`run_evals.py`**: Compute validation perplexity on a checkpoint.
- **`generate_from_pretrained.py`**: Interactive generation from a trained model.
- **`run_lm_eval.py`**: Run standardized benchmarks via lm-evaluation-harness. Set `LM_EVAL_TASKS` env var.

## Key Concepts

**Fingerprinting**: Datasets are fingerprinted by (dataset_names, tokenizer, max_seq_len). This prevents accidental reuse of stale preprocessed data when these change.

**Explicit Preprocessing**: The training pipeline no longer auto-tokenizes. This separates concerns and makes runs reproducible.

**MoE Routing Metrics**: When training MoE models, routing statistics (entropy, load balance, expert usage) are logged to W&B for debugging expert collapse.

## Testing

```bash
# Quick smoke tests
pytest -q

# Include slow tests (optimizer convergence)
pytest -q -m slow
```

## Project Structure

```
conf/                 # Hydra configs
src/scaletraining/
  entrypoints/        # CLI entrypoints
  model/              # Transformer, MoE, optimizers
  data_processing/    # Tokenization, packing, corpus builder
  util/               # Training loop, eval utils, W&B logging
scripts/              # Standalone utilities (corpus builder)
tests/                # Unit and integration tests
notes/                # Architecture docs and design decisions
```
