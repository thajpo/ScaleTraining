# ScaleTraining Architecture

`ScaleTraining` is a config-driven language-model training harness for dense and
Mixture-of-Experts transformer experiments. It is best read as ML
infrastructure: explicit preprocessing, one shared training loop, checkpointed
runs, evaluation hooks, and testable package entrypoints.

## System Flow

```text
Hydra config
-> explicit data preparation
   -> dataset selection
   -> tokenizer choice
   -> tokenization
   -> fixed-length packing
   -> fingerprinted artifacts
-> model construction
   -> dense transformer
   -> optional MoE blocks
-> token-budget training loop
-> W&B/run manifests/checkpoints
-> perplexity and lm-eval style evaluation
-> qualitative generation from checkpoint
```

## Main Boundaries

- `conf/`: Hydra config groups for model, training, tokenizer, optimizer,
  device, MoE, logging, generation, and evals.
- `src/scaletraining/entrypoints/`: package entrypoints for data prep, training,
  generation, and evaluation.
- `src/scaletraining/data_processing/`: tokenization, packing, dataloading, and
  corpus utilities.
- `src/scaletraining/model/`: transformer, MoE, and optimizer/model helpers.
- `src/scaletraining/util/`: training loop, artifacts, eval helpers, and W&B
  integration.
- `tests/`: focused tests across entrypoints, model behavior, data processing,
  training loop, inference, and utility contracts.

## Engineering Claims

- Training fails fast when preprocessed artifacts are missing instead of
  silently tokenizing inside the training loop.
- Dataset artifacts are fingerprinted from config fields that affect
  tokenization/packing.
- Dense and MoE models share the same training surface.
- Evaluation and generation are separated from training for reproducibility.
- The repo has tests for package entrypoints, data processing, model behavior,
  optimizer behavior, and training-loop utilities.

## Current Limits

- Full training and benchmark runs can require a GPU and significant time.
- Some eval paths are still evolving; do not oversell benchmark completeness.
- Large runs should be represented by explicit run reports rather than broad
  scale claims.
- `notes/architecture.md` contains deeper internal design notes; this document
  is the reviewer-facing summary.
