"""Helper routines for structured Weights & Biases logging."""

from __future__ import annotations

from pathlib import Path as PathLib
import typing as t

from omegaconf import OmegaConf


try:  # Lazy optional dependency
    import wandb as wandb_sdk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - logging simply becomes a no-op
    wandb_sdk = None  # type: ignore


def log_train_metrics(
    *,
    used_tokens: int,
    loss: float,
    lr: float,
    throughput: float,
    flops_used: float,
) -> None:
    """Log core training-loop statistics keyed by total tokens processed."""

    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    metrics = {
        "used tokens": used_tokens,
        "train_per_token_loss": loss,
        "lr": lr,
        "throughput_tokens_per_s": throughput,
        "FLOPs": flops_used,
    }
    wandb_sdk.log(dict(metrics), step=used_tokens)


def log_eval_metrics(
    *,
    used_tokens: int,
    val_loss: float,
    val_perplexity: float,
) -> None:
    """Log validation loss/perplexity keyed by the training token count."""

    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    metrics = {
        "used tokens": used_tokens,
        "valid_per_token_loss": val_loss,
        "valid_ppl": val_perplexity,
    }
    wandb_sdk.log(dict(metrics), step=used_tokens)


def log_moe_metrics(*, used_tokens: int, metrics: dict) -> None:
    """Log MoE routing statistics keyed by the training token count."""

    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    if not metrics:
        return
    payload = dict(metrics)
    payload["used tokens"] = used_tokens
    wandb_sdk.log(payload, step=used_tokens)


def init_wandb(cfg, tokenizer_vocab_size, tok) -> None:
    """Initialise W&B with descriptive names derived from the tokenizer."""

    try:
        tokenizer_name = tok.name_or_path
        sweep_name = cfg.sweep.name
    except Exception as e:
        print(f"Some configuration options for W&B are not set: {e}")

    tokenizer_path = PathLib(tokenizer_name)
    is_local_json_tokenizer = (
        tokenizer_path.is_file() and tokenizer_path.suffix == ".json"
    )

    if is_local_json_tokenizer:
        name_suffix = "custom"
    else:
        name_suffix = tokenizer_name.rsplit("/", 1)[-1]

    config = OmegaConf.to_container(cfg, resolve=True)
    config["tokenizer_vocab_size"] = tokenizer_vocab_size

    try:
        wandb_sdk.init(
            project=cfg.logging.wandb_project_name,
            config=config,
            reinit=True,
            name=f"{sweep_name}_{name_suffix}",
        )
    except Exception as e:
        print(f"W&B Could not initalize: {e}")


__all__ = ["log_train_metrics", "log_eval_metrics", "log_moe_metrics", "init_wandb"]
