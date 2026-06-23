"""
Hydra-powered training entrypoint.

This wraps the existing trainer with Hydra config + Weights & Biases (W&B).
Run from CLI: `python -m scaletraining.entrypoints.train` or via console script.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn

from scaletraining.data_processing import build_loaders
from scaletraining.model import TransformerNetwork
from scaletraining.util import (
    clear_cuda_cache,
    configure_rocm_and_sdp,
    init_wandb,
    resolve_device,
    save_model,
)
from scaletraining.model.training_loop import training_run
from scaletraining.util.model_stats import (
    count_parameters,
    humanize_bytes,
    humanize_params,
)

from scaletraining.data_processing.tokenizer import TextTokenizer
from scaletraining.config import load_project_config
from scaletraining.util.training_utils import set_random_seed


LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Train the model using Hydra config and log to W&B.
    """

    cfg = load_project_config(cfg)
    set_random_seed(int(cfg.training.seed))

    # Resolve device, configure kernels, and free any stale CUDA cache
    configure_rocm_and_sdp(cfg.device)
    resolve_device(cfg)
    clear_cuda_cache()

    tokenizer = TextTokenizer(cfg)
    try:
        with open_dict(cfg.model):
            cfg.model.vocab_size = tokenizer.vocab_size
    except Exception:
        pass
    try:
        with open_dict(cfg.tokenizer):
            cfg.tokenizer.tokenizer_name = tokenizer.tok_name
    except Exception:
        pass

    init_wandb(cfg, tok=tokenizer.tok, tokenizer_vocab_size=tokenizer.vocab_size)

    train_loader, val_loader = build_loaders(cfg, for_training=True)


    # Dataset artifact logging intentionally disabled.

    # Model + loss
    model = TransformerNetwork(cfg)

    total_params, trainable_params = count_parameters(model)
    readable_total = humanize_params(total_params)
    readable_trainable = humanize_params(trainable_params)
    bytes_fp32 = total_params * 4
    bytes_bf16 = total_params * 2

    size_msg = (
        f"Model parameters: {total_params:,} ({readable_total}); "
        f"Trainable: {trainable_params:,} ({readable_trainable}); "
        f"Approx size fp32: {humanize_bytes(bytes_fp32)}, bf16/fp16: {humanize_bytes(bytes_bf16)}"
    )
    print(size_msg)
    LOGGER.info(size_msg)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "model/total_params": total_params,
                    "model/trainable_params": trainable_params,
                    "model/size_bytes_fp32": bytes_fp32,
                    "model/size_bytes_bf16": bytes_bf16,
                },
                step=0,
            )
            wandb.run.summary["model/total_params"] = total_params
            wandb.run.summary["model/trainable_params"] = trainable_params
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - W&B logging is best-effort
        LOGGER.warning("Failed to log model size to W&B: %s", exc)

    # Compile model for massive speedups
    # ROCm Triton support improved in PyTorch 2.8+, so enable compile there too
    def _should_compile():
        if torch.version.hip is None:
            return True  # CUDA - always compile
        # ROCm: check version (2.8+ has better Triton support)
        major, minor = map(int, torch.__version__.split(".")[:2])
        return (major, minor) >= (2, 8)
    
    if _should_compile():
        model = torch.compile(model, mode="max-autotune")
    else:
        LOGGER.info("Skipping torch.compile on ROCm<2.8 (triton compatibility)")
    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # summed CE, normalized per token in loop

    # Sanity check embedding size vs vocab size after metadata auto-set
    assert model.token_embedding.num_embeddings == cfg.model.vocab_size, (
        f"Model vocab ({model.token_embedding.num_embeddings}) != cfg.model.vocab_size ({cfg.model.vocab_size})"
    )

    # Training loop
    stats = training_run(cfg, model, train_loader, loss_fn=loss_fn, val_loader=val_loader)

    # Save model locally only
    run_dir = save_model(model, cfg, cfg.paths.output_dir)
    print(f"Model saved locally to: {run_dir}")


    # Persist a lightweight result.json in the job directory for easy aggregation
    job_result = {
        "final_train_loss": float(stats['train_loss'][-1]) if stats.get('train_loss') else None,
        "primary_optimizer": cfg.optimizer.primary_optimizer,
        "use_rope": bool(cfg.model.use_rope),
        "lr": float(cfg.optimizer.lr),
        "seed": int(cfg.training.seed),
        "batch_size": int(cfg.training.batch_size),
        "accum_steps": int(cfg.training.accum_steps),
        "max_train_tokens": int(cfg.training.max_train_tokens),
        "max_seq_len": int(cfg.model.max_seq_len),
        "n_layer": int(cfg.model.n_layer),
        "n_head": int(cfg.model.n_head),
        "n_embed": int(cfg.model.n_embed),
        "run_dir": str(run_dir),
        "model_path": str(Path(run_dir) / "model.pt"),
    }
    with open(Path.cwd() / "result.json", "w", encoding="utf-8") as f:
        json.dump(job_result, f, indent=2, sort_keys=True)
    with open(Path(run_dir) / "train_result.json", "w", encoding="utf-8") as f:
        json.dump(job_result, f, indent=2, sort_keys=True)
    # Also print a single-line summary that's easy to grep
    print("RESULT:", json.dumps(job_result))

    # Return an objective for Hydra sweepers (e.g., Optuna)
    if stats.get('train_loss'):
        return float(stats['train_loss'][-1])
    return float('inf')

if __name__ == "__main__":
    # Standard Hydra entrypoint; objective value returned for sweepers
    main()
