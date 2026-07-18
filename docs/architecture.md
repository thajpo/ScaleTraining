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
-> allocate one run directory and initialize its manifest
-> token-budget training loop with schema-versioned W&B metrics
-> checkpoint, training result, and automatic run report
-> perplexity and lm-eval style evaluation artifacts
-> refreshed run report evidence bundle
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
- `src/scaletraining/reporting.py`: the shared run-report reader and renderer.
- `src/scaletraining/util/`: artifacts, eval helpers, device support, and W&B
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
- W&B is the detailed metric history, with processed tokens as the shared
  train/validation/MoE comparison axis.
- Training throughput covers the synchronized accumulation compute window, not
  loader wait, evaluation, logging, or report generation. Cumulative FLOPs are
  estimated, and peak allocated/reserved memory is read only from the selected
  CUDA device.
- One local run directory links the W&B identity, configuration fingerprint,
  checkpoint, compact results, and automatically generated reports.
- The run directory exists before tracking or training begins. Its manifest
  records the requested and resolved device, an explicit W&B availability
  state, and a `running` to `completed`/`failed` lifecycle; failures retain the
  original error and a partial report when finalization succeeds. Manifest
  updates sync a complete temporary file before atomically replacing the prior
  valid file, then sync the containing directory. New manifests honor the
  process umask, while updates preserve existing permissions.
- Terminal progress distinguishes processed tokens from tokens applied by a
  completed optimizer step, records the stop reason, and exposes any unfinished
  gradient-accumulation window.
- Checkpoint provenance uses a run-relative identity and SHA-256 content digest;
  the recorded original absolute path is informational so archived run bundles
  remain portable.
- Evaluation sidecars refresh the reports next to the checkpoint instead of
  requiring a separate reporting step.
- Evaluation sidecars are validated in memory and atomically replaced only when
  their checkpoint and dataset provenance agrees with the existing run evidence.
- Evaluation output belongs in the checkpoint's run directory. The generated
  report uses run-relative paths, including `.` for nested eval output-directory
  summaries, so a complete evidence directory can be moved and revalidated
  without rewriting its canonical identities. Original absolute paths remain
  informational provenance.
- The run-local `train_result.json` records `run_dir` as `.` and `model_path` as
  `model.pt`, while Hydra's job-level `result.json` exposes absolute values for
  those fields to aggregation consumers.
- The reviewer smoke path is CPU-only and uses local fixture text instead of
  HuggingFace network access.
- The repo has tests for package entrypoints, data processing, model behavior,
  optimizer behavior, and training-loop utilities.

## Current Limits

- Full training and benchmark runs can require a GPU and significant time.
- CPU smoke runs prove wiring and artifact contracts, not model quality.
- Checkpoints do not include optimizer, progress, or RNG state, so exact
  interruption/resume equivalence is not supported.
- Historical checkpoints without validation and report sidecars are not
  evidence for model-quality claims.
- The repository still needs one controlled experiment and written conclusion;
  further platform expansion is not the closeout goal.
- Multi-GPU support is intentionally outside the closeout scope.
- Some eval paths are still evolving; do not oversell benchmark completeness.
- Large runs should be represented by explicit run reports rather than broad
  scale claims.
- `notes/architecture.md` contains deeper internal design notes; this document
  is the reviewer-facing summary.
