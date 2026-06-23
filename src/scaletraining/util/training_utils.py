"""Utility helpers shared across the training/evaluation loops."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Tuple

import math

import torch
import torch.nn as nn

from scaletraining.model.optimizers import AdaMuon, Muon


ParameterSplit = Tuple[List[nn.Parameter], List[nn.Parameter]]


def set_random_seed(seed: int) -> None:
    """Set core runtime seeds for controlled single-run comparisons."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_model_matrix_params(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    *,
    model: nn.Module | None = None,
    model_cfg: Any | None = None,
) -> ParameterSplit:
    """Split parameters into Muon-eligible hidden matrices versus everything else."""

    excluded_ids = set()
    if model is not None:
        for attr in ("token_embedding", "W_ue"):
            try:
                excluded_ids.add(id(getattr(model, attr).weight))
            except Exception:
                pass

    def excluded_from_muon(name: str, p: nn.Parameter) -> bool:
        """
        Muon and AdaMuon only operate on rank 2 tensors. 
        They also should not be used on embedding or unembedding matrices in first/final layers
        """
        if id(p) in excluded_ids:
            return True
        if name.startswith("token_embedding") or "embedding" in name:
            return True
        if name.startswith("W_ue"):
            return True
        if p.dim() != 2:
            return True
        if max(p.shape) > 10000:
            return True
        try:
            vocab = int(getattr(model_cfg, "vocab_size")) if model_cfg is not None else None
        except Exception:
            vocab = None
        if vocab is not None and (p.shape[0] == vocab or p.shape[1] == vocab):
            return True
        return False

    seen: set[int] = set()
    muon_mats: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, param in named_params:
        if id(param) in seen:
            continue
        seen.add(id(param))
        if excluded_from_muon(name, param):
            other_params.append(param)
        else:
            muon_mats.append(param)
    return muon_mats, other_params


def build_optimizers(
    optimizer_cfg: Any,
    matrix_params: List[nn.Parameter],
    other_params: List[nn.Parameter],
) -> Tuple[torch.optim.Optimizer | None, torch.optim.Optimizer | None]:
    """Construct primary/secondary optimizers as dictated by the config."""

    if optimizer_cfg.use_baseline_adam:
        all_params = list(matrix_params) + list(other_params)
        baseline_cfg: Dict[str, Any] = optimizer_cfg.baseline_adam_config
        lr = baseline_cfg.get("lr", optimizer_cfg.lr)
        weight_decay = baseline_cfg.get("weight_decay", optimizer_cfg.weight_decay)
        betas = tuple(baseline_cfg.get("betas", (optimizer_cfg.beta, optimizer_cfg.beta2)))
        optimizer = torch.optim.AdamW(
            params=all_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=optimizer_cfg.eps,
        )
        return optimizer, None

    name = optimizer_cfg.primary_optimizer.lower()
    if name == "adamw":
        all_params = list(matrix_params) + list(other_params)
        betas = (optimizer_cfg.beta, optimizer_cfg.beta2)
        optimizer = torch.optim.AdamW(
            params=all_params,
            lr=optimizer_cfg.lr,
            betas=betas,
            weight_decay=optimizer_cfg.weight_decay,
            eps=optimizer_cfg.eps,
        )
        return optimizer, None

    betas = (optimizer_cfg.beta, optimizer_cfg.beta2)
    if name == "muon":
        muon_lr = optimizer_cfg.muon_lr
        primary = Muon(
            params=matrix_params,
            lr=muon_lr,
            beta=betas[0],
            beta2=betas[1],
            weight_decay=optimizer_cfg.weight_decay,
            ns_iters=optimizer_cfg.ns_iters,
            eps=optimizer_cfg.eps,
        )
    elif name == "adamuon":
        muon_lr = optimizer_cfg.muon_lr
        primary = AdaMuon(
            params=matrix_params,
            lr=muon_lr,
            beta=betas[0],
            beta2=betas[1],
            weight_decay=optimizer_cfg.weight_decay,
            ns_iters=optimizer_cfg.ns_iters,
            eps=optimizer_cfg.eps,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer_cfg.primary_optimizer}")

    secondary = torch.optim.AdamW(
        params=other_params,
        lr=optimizer_cfg.lr,
        betas=betas,
        weight_decay=optimizer_cfg.weight_decay,
        eps=optimizer_cfg.eps,
    )
    return primary, secondary


def prepare_targets(input_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Create next-token targets and compute effective token count (no masks)."""
    targets = input_ids[:, 1:]
    total_effective = int(targets.numel())
    return targets, total_effective

def log_implementation(matrix_params, other_params):
    """Sanity check for the parameters that Muon/AdaMuon operate on"""
    def _summarize(params):
        return [tuple(p.shape) for p in params][:10]

    print(
        "[opt-wiring] muon-eligible (hidden 2D) count:",
        len(matrix_params),
        "sample shapes:",
        _summarize(matrix_params),
    )
    print(
        "[opt-wiring] adamw params (embeddings, head, biases, etc.) count:",
        len(other_params),
        "sample shapes:",
        _summarize(other_params),
    )
def compute_loss_sum(
    model: nn.Module,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """Compute summed cross-entropy across the sequence, optionally chunked."""

    time = hidden.size(1)
    if chunk_size <= 0 or chunk_size >= time:
        logits = model.W_ue(hidden)
        return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    loss_sum = hidden.new_tensor(0.0, dtype=torch.float32)
    start = 0
    while start < time:
        end = min(start + chunk_size, time)
        logits_chunk = model.W_ue(hidden[:, start:end, :])
        targets_chunk = targets[:, start:end]
        loss_sum = loss_sum + loss_fn(
            logits_chunk.reshape(-1, logits_chunk.size(-1)),
            targets_chunk.reshape(-1),
        ).to(loss_sum.dtype)
        start = end
    return loss_sum


def compute_lr_scale_tokens(used_tokens: int, training_cfg: Any, optimizer_cfg: Any) -> float:
    """Compute LR scale based on token progress with warmup and schedule."""

    schedule = optimizer_cfg.lr_schedule
    warmup_tokens = int(optimizer_cfg.warmup_tokens or 0)
    min_scale = float(optimizer_cfg.min_lr_scale)
    total = int(training_cfg.max_train_tokens or 0)
    if total <= 0:
        total = max(used_tokens, 1)

    if warmup_tokens > 0 and used_tokens < warmup_tokens:
        return max(0.0, min(1.0, used_tokens / max(1, warmup_tokens)))

    post = used_tokens - warmup_tokens
    denom = max(1, total - warmup_tokens)
    t = max(0.0, min(1.0, post / denom))

    if schedule == "cosine":
        return float(min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * t)))
    if schedule == "linear":
        return float(min_scale + (1.0 - min_scale) * (1.0 - t))
    return 1.0


def compute_progress_t(used_tokens: int, training_cfg: Any, optimizer_cfg: Any) -> float:
    """Token-based progress t in [0, 1] after warmup for generic schedules."""

    warmup_tokens = int(optimizer_cfg.warmup_tokens or 0)
    total = int(training_cfg.max_train_tokens or 0)
    if total <= 0:
        total = max(used_tokens, 1)
    post = max(0, used_tokens - warmup_tokens)
    denom = max(1, total - warmup_tokens)
    return max(0.0, min(1.0, post / denom))


def schedule_value(start: float, end: float, t: float, schedule: str) -> float:
    """Interpolate between start->end with optional cosine/linear schedule."""

    if schedule == "linear":
        return float(start + (end - start) * t)
    if schedule == "cosine":
        return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t)))
    return float(start)


def apply_moe_schedules(model: nn.Module, moe_cfg: Any, progress_t: float) -> float:
    """Update MoE router parameters according to annealing schedules.

    Returns the load-balance coefficient to use for the next accumulation window.
    """

    temp = schedule_value(
        float(moe_cfg.moe_router_temp_start),
        float(moe_cfg.moe_router_temp_end),
        progress_t,
        moe_cfg.moe_router_temp_schedule,
    )
    noise = schedule_value(
        float(moe_cfg.moe_router_noise_start),
        float(moe_cfg.moe_router_noise_end),
        progress_t,
        moe_cfg.moe_router_noise_schedule,
    )
    lb_coef = schedule_value(
        float(moe_cfg.moe_lb_coef_start),
        float(moe_cfg.moe_lb_coef_end),
        progress_t,
        moe_cfg.moe_lb_coef_schedule,
    )
    try:
        from scaletraining.model.model import MoELayer  # local import to avoid circularity

        for module in model.modules():
            if isinstance(module, MoELayer):
                module.router_temp = float(temp)
                module.router_noise = float(noise)
    except Exception:
        pass
    return float(lb_coef)


def scale_optimizer_lr(
    optimizer: torch.optim.Optimizer | None, base_lr: float, lr_scale: float
) -> None:
    """
    Apply a scaling factor to every parameter group in the optimizer.
    Muon/Adamuon have different optimizer scales than AdamW
    For example:
        If AdamW lr is 5e-4
        Muon lr is 1e-2 approximately
    """

    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = base_lr * lr_scale


def estimate_flops(tokens_used, d_model, d_hidden, n_heads, seq_len,
                   n_layers, n_moe_layers, top_k, n_experts, using_moe, capacity=1.0,
                   optimizer_factor=6.0):
    """
    Estimate the amount of flops the model has used. This metric is used to determine how efficient the model is
    """
    dense_params = (n_layers - n_moe_layers) * (4 * d_model**2 + 2 * d_model * d_hidden)

    if using_moe and n_moe_layers:
        expert_params = n_moe_layers * top_k * (2 * d_model * d_hidden)
        router_params = n_moe_layers * d_model * n_experts
    else:
        expert_params = 0
        router_params = 0

    active_params = dense_params + expert_params + router_params

    flops = tokens_used * (
        6.0 * active_params        # forward + backward + optimizer update
        + 12.0 * n_layers * d_model * seq_len  # attention matmuls (seq_len^2 collapses because tokens_used already counts S tokens per batch)
    )
    return flops


__all__ = [
    "ParameterSplit",
    "split_model_matrix_params",
    "build_optimizers",
    "prepare_targets",
    "compute_loss_sum",
    "compute_lr_scale_tokens",
    "compute_progress_t",
    "schedule_value",
    "apply_moe_schedules",
    "scale_optimizer_lr",
]
