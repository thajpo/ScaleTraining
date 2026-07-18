"""
Functional text generation utilities (no trainer classes).
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor
from torch.amp import autocast

from scaletraining.util.device import uses_cuda


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep top-k logits per row and set others to -inf.

    Args:
        logits: float tensor [B, V], unnormalized token scores.
        k: int, number of tokens to keep per row (k>0).
    Returns:
        Filtered logits with same shape.
    """
    if k <= 0 or k >= logits.size(-1):
        return logits
    topk_vals, _ = torch.topk(logits, k)
    min_topk = topk_vals[:, -1].unsqueeze(-1)
    return torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)


@torch.no_grad()
def generate_autoregressive(
    model,
    tokenizer,
    device: str,
    *,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
) -> str:
    """Sample tokens autoregressively from `model` given a text `prompt`.

    Args:
        model: nn.Module with `forward(input_ids)` -> logits [B, T, V].
        tokenizer: HuggingFace tokenizer used to encode/decode text.
        device: PyTorch device string such as ``cpu``, ``cuda``, or ``cuda:1``.
        prompt: seed text to condition on.
        max_new_tokens: number of tokens to sample.
        temperature: >0; divides logits before softmax.
        top_k: if set and >0, keep only top-k tokens at each step.
    Returns:
        Generated text including the prompt.
    """
    model.eval()

    # Ensure eos/pad are present for clean stopping/decoding
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": ""})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    ctx = (
        autocast(device_type="cuda", dtype=torch.bfloat16)
        if uses_cuda(device)
        else contextlib.nullcontext()
    )

    for _ in range(max_new_tokens):
        with ctx:
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / max(1e-6, float(temperature))
            if top_k is not None and top_k > 0:
                next_token_logits = top_k_filter(next_token_logits, int(top_k))
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
