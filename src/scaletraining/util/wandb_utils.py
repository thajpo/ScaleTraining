"""Structured, best-effort Weights & Biases tracking."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


LOGGER = logging.getLogger(__name__)
TRACKING_SCHEMA_VERSION = 1
TRACKED_NAMESPACES = ("train", "validation", "performance", "compute", "moe")

try:  # Lazy optional dependency
    import wandb as wandb_sdk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - tracking becomes a no-op
    wandb_sdk = None  # type: ignore


@dataclass(frozen=True)
class WandbRunIdentity:
    """Serializable reference to the W&B run associated with a local run."""

    provider: str = "wandb"
    schema_version: int = TRACKING_SCHEMA_VERSION
    state: str = "unavailable"
    mode: str | None = None
    entity: str | None = None
    project: str | None = None
    run_id: str | None = None
    path: str | None = None
    url: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "/".join(str(part) for part in value)
    return str(value)


def _identity_from_run(run: Any) -> WandbRunIdentity:
    settings = getattr(run, "settings", None)
    mode = getattr(settings, "mode", None) or os.environ.get("WANDB_MODE") or "online"
    if str(mode).lower() == "disabled":
        return WandbRunIdentity(state="disabled", mode="disabled")
    url = getattr(run, "url", None)
    if not url and hasattr(run, "get_url"):
        try:
            url = run.get_url()
        except Exception:
            url = None
    return WandbRunIdentity(
        state="initialized",
        mode=str(mode),
        entity=getattr(run, "entity", None),
        project=getattr(run, "project", None),
        run_id=getattr(run, "id", None),
        path=_run_path(getattr(run, "path", None)),
        url=str(url) if url else None,
    )


def _define_metrics() -> None:
    if wandb_sdk is None:
        return
    wandb_sdk.define_metric("progress/tokens")
    wandb_sdk.define_metric("progress/optimizer_step")
    for namespace in TRACKED_NAMESPACES:
        wandb_sdk.define_metric(
            f"{namespace}/*",
            step_metric="progress/tokens",
        )


def _tracking_payload(
    *, used_tokens: int, optimizer_step: int, metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "progress/tokens": int(used_tokens),
        "progress/optimizer_step": int(optimizer_step),
        **metrics,
    }


def _log(
    *, used_tokens: int, optimizer_step: int, metrics: dict[str, Any]
) -> None:
    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    try:
        wandb_sdk.log(
            _tracking_payload(
                used_tokens=used_tokens,
                optimizer_step=optimizer_step,
                metrics=metrics,
            )
        )
    except Exception as exc:  # pragma: no cover - SDK failures are environment-specific
        LOGGER.warning("W&B metric logging failed: %s", exc)


def log_train_metrics(
    *,
    used_tokens: int,
    optimizer_step: int,
    loss: float,
    lr: float,
    grad_norm_pre_clip: float,
    throughput: float,
    flops_used: float,
    peak_memory_allocated_bytes: int | None = None,
    peak_memory_reserved_bytes: int | None = None,
) -> None:
    """Log one optimizer window using schema-v1 metric names."""

    metrics: dict[str, Any] = {
        "train/loss_per_token": float(loss),
        "train/learning_rate": float(lr),
        "train/grad_norm_pre_clip": float(grad_norm_pre_clip),
        "performance/tokens_per_second": float(throughput),
        "compute/flops_total": float(flops_used),
    }
    if peak_memory_allocated_bytes is not None:
        metrics["compute/peak_memory_allocated_bytes"] = int(
            peak_memory_allocated_bytes
        )
    if peak_memory_reserved_bytes is not None:
        metrics["compute/peak_memory_reserved_bytes"] = int(
            peak_memory_reserved_bytes
        )
    _log(
        used_tokens=used_tokens,
        optimizer_step=optimizer_step,
        metrics=metrics,
    )


def log_eval_metrics(
    *,
    used_tokens: int,
    optimizer_step: int,
    val_loss: float,
    val_perplexity: float,
) -> None:
    """Log validation metrics against the same token axis as training."""

    _log(
        used_tokens=used_tokens,
        optimizer_step=optimizer_step,
        metrics={
            "validation/loss_per_token": float(val_loss),
            "validation/perplexity": float(val_perplexity),
        },
    )


def log_moe_metrics(
    *, used_tokens: int, optimizer_step: int, metrics: dict[str, Any]
) -> None:
    """Log dynamic MoE routing statistics against processed tokens."""

    if not metrics:
        return
    _log(
        used_tokens=used_tokens,
        optimizer_step=optimizer_step,
        metrics=dict(metrics),
    )


def log_model_metrics(metrics: dict[str, Any]) -> None:
    """Record fixed model-size metadata in W&B history and summary."""

    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    try:
        wandb_sdk.log(dict(metrics))
        summary = getattr(wandb_sdk.run, "summary", None)
        if summary is not None:
            summary.update(metrics)
    except Exception as exc:  # pragma: no cover - SDK failures are environment-specific
        LOGGER.warning("W&B model metadata logging failed: %s", exc)


def init_wandb(cfg: Any, tokenizer_vocab_size: int, tok: Any) -> WandbRunIdentity:
    """Initialize W&B and return only the serializable run identity."""

    if wandb_sdk is None:
        return WandbRunIdentity(state="unavailable")

    tokenizer_name = (
        getattr(tok, "name_or_path", None)
        or getattr(cfg.tokenizer, "tokenizer_name", None)
        or getattr(cfg.tokenizer, "pretrained_tokenizer_name", None)
        or "tokenizer"
    )
    sweep = getattr(cfg, "sweep", None)
    sweep_name = getattr(sweep, "name", None) or "run"
    tokenizer_path = Path(str(tokenizer_name))
    name_suffix = (
        "custom"
        if tokenizer_path.is_file() and tokenizer_path.suffix == ".json"
        else str(tokenizer_name).rsplit("/", 1)[-1]
    )

    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        config = {"config": config}
    config["tokenizer_vocab_size"] = int(tokenizer_vocab_size)
    config["tracking_schema_version"] = TRACKING_SCHEMA_VERSION

    try:
        run = wandb_sdk.init(
            project=cfg.logging.wandb_project_name,
            config=config,
            reinit=True,
            name=f"{sweep_name}_{name_suffix}",
            tags=list(getattr(cfg.logging, "experiment_tags", [])),
        )
    except Exception as exc:
        LOGGER.warning("W&B initialization failed: %s", exc)
        return WandbRunIdentity(
            state="initialization_failed",
            mode=os.environ.get("WANDB_MODE"),
            error=str(exc),
        )

    if run is None:
        return WandbRunIdentity(
            state="initialization_failed",
            mode=os.environ.get("WANDB_MODE"),
            error="wandb.init returned no run",
        )

    try:
        _define_metrics()
    except Exception as exc:  # pragma: no cover - depends on SDK implementation
        LOGGER.warning("W&B metric definition failed: %s", exc)
    return _identity_from_run(run)


def finish_wandb(exit_code: int = 0) -> None:
    """Finalize an active W&B run without changing training error semantics."""

    if wandb_sdk is None or getattr(wandb_sdk, "run", None) is None:
        return
    try:
        wandb_sdk.finish(exit_code=int(exit_code))
    except Exception as exc:  # pragma: no cover - SDK failures are environment-specific
        LOGGER.warning("W&B finalization failed: %s", exc)


__all__ = [
    "TRACKING_SCHEMA_VERSION",
    "WandbRunIdentity",
    "finish_wandb",
    "init_wandb",
    "log_eval_metrics",
    "log_model_metrics",
    "log_moe_metrics",
    "log_train_metrics",
]
