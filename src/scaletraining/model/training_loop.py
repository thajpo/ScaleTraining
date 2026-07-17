"""
Functional training loop utilities.

These helpers implement the training loop without an object-oriented trainer.
Each function has a narrow purpose and explicit inputs/outputs.
"""

from __future__ import annotations

import contextlib
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from omegaconf import open_dict

from scaletraining.util.eval_utils import evaluate_perplexity
from scaletraining.util.training_utils import (
    apply_moe_schedules,
    build_optimizers,
    compute_loss_sum,
    compute_lr_scale_tokens,
    compute_progress_t,
    prepare_targets,
    scale_optimizer_lr,
    split_model_matrix_params,
    log_implementation,
    estimate_flops,
)
from scaletraining.util.wandb_utils import (
    log_eval_metrics,
    log_moe_metrics,
    log_train_metrics,
)


def _synchronize_for_timing(device: str) -> None:
    """Wait for queued accelerator work only when measuring a CUDA window."""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def training_run(
    cfg,
    model: nn.Module,
    train_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, list]:
    """Functional training loop until reaching token budget.

    Args:
        cfg: Hydra config with fields used: device, accum_steps, grad_clip_norm,
             logits_chunk_size, max_train_tokens, debug_memory.
        model: nn.Module with `forward_hidden` and `W_ue` attributes.
        train_loader: DataLoader yielding dicts with 'input_ids'.
        loss_fn: nn.CrossEntropyLoss(reduction='sum') for per-token normalization.
    Returns:
        stats: Averaged per-token training losses for each accumulation window.
    """

    matrix_params, other_params = split_model_matrix_params(
        model.named_parameters(), model=model, model_cfg=cfg.model
    )
    if cfg.logging.log_implementation_details:
        log_implementation(matrix_params, other_params)

    opt_primary, opt_secondary = build_optimizers(
        cfg.optimizer, matrix_params, other_params
    )

    primary_base_lr = (
        float(opt_primary.param_groups[0]["lr"]) if opt_primary is not None else 0.0
    )
    secondary_base_lr = (
        float(opt_secondary.param_groups[0]["lr"]) if opt_secondary is not None else 0.0
    )

    device = str(getattr(cfg, "device_resolved", None) or cfg.device.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.train()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stats = {"train_loss": []}
    used_tokens = 0
    best_train_loss = float("inf")
    tokens_at_best_loss = 0
    early_stop_tokens = max(
        0, int(getattr(cfg.training, "early_stop_tokens_without_improvement", 0))
    )
    early_stop_min_delta = float(getattr(cfg.training, "early_stop_min_delta", 0.0))
    step_in_accum = 0
    optimizer_step = 0
    window_started_at: float | None = None
    accum_loss_sum = 0.0
    accum_token_count = 0
    last_eval_tokens = 0

    def build_moe_metrics(model: nn.Module) -> dict:
        if not hasattr(model, "moe_routing_stats"):
            return {}
        layer_stats = model.moe_routing_stats()
        if not layer_stats:
            return {}
        metrics: dict[str, float] = {}
        scalar_sums: dict[str, float] = {}
        scalar_counts: dict[str, int] = {}
        for layer_idx, stats in layer_stats:
            prefix = f"moe/l{layer_idx}"
            for key, val in stats.items():
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        metrics[f"{prefix}/{key}_{i}"] = float(v)
                else:
                    metrics[f"{prefix}/{key}"] = float(val)
                    scalar_sums[key] = scalar_sums.get(key, 0.0) + float(val)
                    scalar_counts[key] = scalar_counts.get(key, 0) + 1
        for key, total in scalar_sums.items():
            metrics[f"moe/{key}_mean"] = total / max(1, scalar_counts[key])
        aux = (
            float(model.moe_aux_loss().item())
            if hasattr(model, "moe_aux_loss")
            else 0.0
        )
        metrics["moe/aux_loss"] = aux
        return metrics

    stop_training = False
    while used_tokens < cfg.training.max_train_tokens and not stop_training:
        for idx, batch in enumerate(train_loader):
            if step_in_accum == 0:
                _synchronize_for_timing(device)
                window_started_at = time.perf_counter()
            input_ids = batch["input_ids"].to(device)

            ctx = (
                autocast(device_type="cuda", dtype=torch.bfloat16)
                if (device == "cuda" and torch.cuda.is_available())
                else contextlib.nullcontext()
            )
            with ctx:
                hidden = model.forward_hidden(input_ids)
                hidden = hidden[:, :-1, :]
                targets, effective = prepare_targets(input_ids)
                loss_sum = compute_loss_sum(
                    model, hidden, targets, cfg.training.logits_chunk_size, loss_fn
                )
                per_token_loss = loss_sum / max(1, effective)

                aux = (
                    model.moe_aux_loss()
                    if hasattr(model, "moe_aux_loss")
                    else hidden.new_tensor(0.0, dtype=torch.float32)
                )
                total_loss = per_token_loss + float(cfg.moe.moe_lb_coef) * aux.to(
                    per_token_loss.dtype
                )

                loss = total_loss / cfg.training.accum_steps

            loss.backward()
            accum_loss_sum += float(loss_sum.item())
            accum_token_count += int(effective)
            step_in_accum += 1

            used_tokens += int(effective)

            if step_in_accum == cfg.training.accum_steps:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.grad_clip_norm
                )
                lr_scale = compute_lr_scale_tokens(
                    used_tokens, cfg.training, cfg.optimizer
                )
                progress_t = compute_progress_t(
                    used_tokens, cfg.training, cfg.optimizer
                )
                new_lb_coef = apply_moe_schedules(model, cfg.moe, progress_t)
                with open_dict(cfg.moe):
                    cfg.moe.moe_lb_coef = new_lb_coef

                scale_optimizer_lr(opt_primary, primary_base_lr, lr_scale)
                scale_optimizer_lr(opt_secondary, secondary_base_lr, lr_scale)

                opt_primary.step()
                if opt_secondary is not None:
                    opt_secondary.step()

                _synchronize_for_timing(device)
                if window_started_at is None:  # pragma: no cover - loop invariant
                    raise RuntimeError("Accumulation timer was not started")
                elapsed = max(
                    1e-6,
                    time.perf_counter() - window_started_at,
                )

                opt_primary.zero_grad(set_to_none=True)
                if opt_secondary is not None:
                    opt_secondary.zero_grad(set_to_none=True)
                step_in_accum = 0
                optimizer_step += 1
                window_started_at = None

                avg_loss = accum_loss_sum / max(1, accum_token_count)
                stats["train_loss"].append(avg_loss)
                current_lr = (
                    opt_primary.param_groups[0]["lr"]
                    if opt_primary is not None
                    else 0.0
                )
                tps = accum_token_count / elapsed if accum_token_count > 0 else 0.0
                peak_memory_allocated = None
                peak_memory_reserved = None
                if device.startswith("cuda") and torch.cuda.is_available():
                    peak_memory_allocated = torch.cuda.max_memory_allocated()
                    peak_memory_reserved = torch.cuda.max_memory_reserved()
                flops_used = estimate_flops(
                    tokens_used=used_tokens,
                    d_model=cfg.model.n_embed,
                    d_hidden=cfg.model.n_hidden,
                    n_heads=cfg.model.n_head,
                    seq_len=cfg.model.max_seq_len,
                    n_layers=cfg.model.n_layer,
                    n_moe_layers=cfg.moe.moe_n_layers,
                    top_k=cfg.moe.moe_top_k,
                    n_experts=cfg.moe.moe_n_experts,
                    using_moe=cfg.moe.use_moe,
                )

                print(
                    f"Tokens: {used_tokens:,}, Loss: {avg_loss:.4f}, "
                    f"LR: {current_lr:.6g}, tok/s: {tps:.0f}"
                )
                log_train_metrics(
                    used_tokens=used_tokens,
                    optimizer_step=optimizer_step,
                    loss=avg_loss,
                    lr=current_lr,
                    grad_norm_pre_clip=float(grad_norm),
                    throughput=tps,
                    flops_used=flops_used,
                    peak_memory_allocated_bytes=peak_memory_allocated,
                    peak_memory_reserved_bytes=peak_memory_reserved,
                )
                if cfg.moe.use_moe:
                    log_moe_metrics(
                        used_tokens=used_tokens,
                        optimizer_step=optimizer_step,
                        metrics=build_moe_metrics(model),
                    )

                if avg_loss + early_stop_min_delta < best_train_loss:
                    best_train_loss = avg_loss
                    tokens_at_best_loss = used_tokens
                elif (
                    early_stop_tokens > 0
                    and (used_tokens - tokens_at_best_loss) >= early_stop_tokens
                ):
                    print(
                        "Early stopping: no train_loss improvement for "
                        f"{(used_tokens - tokens_at_best_loss):,} tokens; "
                        "stopping run."
                    )
                    stop_training = True

                accum_loss_sum = 0.0
                accum_token_count = 0

                if stop_training:
                    break

                eval_interval = cfg.training.eval_interval_tokens
                max_val_batches = cfg.training.eval_max_batches

                if (
                    val_loader is not None
                    and eval_interval > 0
                    and (used_tokens - last_eval_tokens) >= eval_interval
                ):
                    v_loss, v_ppl = evaluate_perplexity(
                        model,
                        val_loader,
                        cfg,
                        loss_fn,
                        max_batches=max_val_batches,
                    )
                    print(
                        f"[eval] tokens={used_tokens:,} val_loss={v_loss:.4f} val_ppl={v_ppl:.3f}"
                    )
                    log_eval_metrics(
                        used_tokens=used_tokens,
                        optimizer_step=optimizer_step,
                        val_loss=v_loss,
                        val_perplexity=v_ppl,
                    )
                    last_eval_tokens = used_tokens

            if (
                cfg.logging.debug_memory
                and torch.cuda.is_available()
                and (idx % 100 == 0)
            ):
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                    peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
                    print(
                        f"peak MB after step: alloc={peak_alloc:.2f}, reserved={peak_reserv:.2f}"
                    )
                except Exception:
                    pass

            if used_tokens >= cfg.training.max_train_tokens:
                stop_training = True
                break

    return stats
