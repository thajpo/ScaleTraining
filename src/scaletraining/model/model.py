from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


class AttentionBlock(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        assert (
            model_cfg.n_embed % model_cfg.n_head == 0
        )  # Ensure that embedding can be evenly split between heads
        self.kqv_block = nn.Linear(
            model_cfg.n_embed, model_cfg.n_embed * 3, bias=model_cfg.bias
        )
        self.out_projection = nn.Linear(
            model_cfg.n_embed, model_cfg.n_embed, bias=model_cfg.bias
        )

        self.n_head = model_cfg.n_head
        self.n_embed = model_cfg.n_embed
        self.resid_dropout = nn.Dropout(model_cfg.resid_dropout)
        self.attn_dropout = model_cfg.attn_dropout

        self.head_dim = model_cfg.n_embed // model_cfg.n_head
        self.max_seq_len = model_cfg.max_seq_len

        # RoPE configuration (on/off)
        self.use_rope = bool(getattr(model_cfg, "use_rope", True))
        rope_cfg = getattr(model_cfg, "rope_config", {})
        theta = getattr(rope_cfg, "theta", None)
        if theta is None and isinstance(rope_cfg, dict):
            theta = rope_cfg.get("theta", 10000.0)
        self.theta = float(theta if theta is not None else 10000.0)

        if self.use_rope:
            assert self.head_dim % 2 == 0, (
                "Head dimension must be even for RoPE, but got %d" % self.head_dim
            )
            self.create_rope_lookup()

    def create_rope_lookup(self):
        """
        A function for creating lookup table for RoPE
        Takes in a max sequence length, and returns two torch tensors:
        cos_freqs, sin_freqs
        Shape: (max_sequence_length, frequency resolution) -> (1000,10000)
        """
        positions = torch.arange(0, self.max_seq_len, 1)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        frequencies = torch.outer(positions, inv_freq)

        self.register_buffer("cos_freqs", torch.cos(frequencies), persistent=False)
        self.register_buffer("sin_freqs", torch.sin(frequencies), persistent=False)

    def _apply_rope(self, q, k, cos_freqs, sin_freqs):
        """Apply RoPE given precomputed cos/sin lookup tables."""
        B, N, T, H = q.shape
        cos = (
            cos_freqs[:T, :]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=q.device, dtype=q.dtype)
        )
        sin = (
            sin_freqs[:T, :]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=q.device, dtype=q.dtype)
        )

        q_even, q_odd = q[..., ::2], q[..., 1::2]
        k_even, k_odd = k[..., ::2], k[..., 1::2]

        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos

        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos

        q_rot = (
            torch.stack([q_rot_even, q_rot_odd], dim=-1)
            .reshape(B, N, T, H)
            .to(device=q.device, dtype=q.dtype)
        )
        k_rot = (
            torch.stack([k_rot_even, k_rot_odd], dim=-1)
            .reshape(B, N, T, H)
            .to(device=k.device, dtype=k.dtype)
        )
        return q_rot, k_rot

    def forward(self, x):
        B, T, E = x.shape
        # Ensure precomputed RoPE tables cover the current sequence length when enabled
        if self.use_rope:
            assert T <= self.max_seq_len, (
                f"Sequence length {T} exceeds RoPE table size {self.max_seq_len}. "
                "Increase model.max_seq_len or reduce input length."
            )

        # x (BTE) ; W.T (E, 3E)
        # x @ W.T -> (B, T, 3E)
        # Then, we split along the embed dimension to get the matrices
        q, k, v = self.kqv_block(x).split(self.n_embed, dim=2)

        # Reshape kqv to expected flash attention shapes
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Apply RoPE when enabled; otherwise leave q/k unchanged
        if self.use_rope:
            q, k = self._apply_rope(q, k, self.cos_freqs, self.sin_freqs)

        # SDPA takes in tensors, with dropout for attention scores
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        return self.resid_dropout(self.out_projection(y))


def _mlp_activation(name: str):
    name = (name or "relu").lower()
    table = {
        "relu": F.relu,
        "gelu": F.gelu,
        "gelu_tanh": partial(F.gelu, approximate="tanh"),
        "gelu_new": partial(F.gelu, approximate="tanh"),
        "silu": F.silu,
        "swish": F.silu,
    }
    fn = table.get(name)
    if fn is None:
        raise ValueError(
            f"Unsupported MLP activation '{name}'. Update _mlp_activation to add it."
        )
    return fn


class MLPBlock(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.Wh = nn.Linear(model_cfg.n_embed, model_cfg.n_hidden, bias=model_cfg.bias)
        self.We = nn.Linear(model_cfg.n_hidden, model_cfg.n_embed, bias=model_cfg.bias)
        self.dropout = nn.Dropout(model_cfg.resid_dropout)
        self.activation = _mlp_activation(getattr(model_cfg, "activation", "relu"))

    def forward(self, x):
        residual = x
        # x -> (B T E)
        x = self.Wh(x)
        x = self.activation(x)
        x = self.We(x)
        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.ln = nn.LayerNorm(model_cfg.n_embed)
        self.attention = AttentionBlock(model_cfg)
        self.mlp = MLPBlock(model_cfg)

    def forward(self, x):
        x = x + self.attention(self.ln(x))
        x = x + self.mlp(self.ln(x))
        return x


class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_hidden, act="swiGLU", bias=True, device=None):
        super().__init__()
        self.act = act
        self.W1 = nn.Linear(
            d_model, 2 * d_hidden if act == "swiGLU" else d_hidden, bias=bias
        )
        self.W2 = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        if self.act == "swiGLU":
            a, b = self.W1(x).chunk(2, dim=-1)
            h = F.silu(a) * b
        else:
            h = F.relu(self.W1(x))
        return self.W2(h)


class MoELayer(nn.Module):
    def __init__(self, model_cfg, moe_cfg):
        super().__init__()
        self.n_experts = moe_cfg.moe_n_experts
        self.top_k = moe_cfg.moe_top_k
        assert 1 <= self.top_k <= self.n_experts

        self._last_aux_loss = None
        self._last_router_stats = None
        self.router_noise = float(moe_cfg.moe_router_noise)
        self.router_temp = float(moe_cfg.moe_router_temp)
        self.router = nn.Linear(model_cfg.n_embed, self.n_experts, bias=False)

        self.experts = nn.ModuleList(
            [
                ExpertFFN(
                    model_cfg.n_embed,
                    moe_cfg.moe_n_hidden,
                    act=moe_cfg.moe_activation,
                    bias=model_cfg.bias,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.shared = (
            ExpertFFN(
                model_cfg.n_embed,
                moe_cfg.moe_n_hidden,
                act=moe_cfg.moe_activation,
                bias=model_cfg.bias,
            )
            if getattr(moe_cfg, "moe_use_shared", False)
            else None
        )

    def forward(self, x):
        B, T, D = x.shape
        z = self.router(x)

        if self.router_noise > 0:
            z = z + self.router_noise * torch.randn_like(z)
        if self.router_temp != 1.0:
            z = z / self.router_temp

        vals, idx = torch.topk(z, self.top_k, dim=-1)
        # Ensure gates match router/logit dtype (avoids dtype mismatches on ROCm)
        gates = F.softmax(vals, dim=-1).to(dtype=z.dtype)

        # generally dont understand this.
        p = F.softmax(z, dim=-1)  # [B,T,E]
        imp = p.mean(dim=(0, 1))  # importance distribution
        E = z.size(-1)
        mass = z.new_zeros(E)
        mass.scatter_add_(0, idx.reshape(-1), gates.reshape(-1))
        load = mass / mass.sum().clamp_min(1e-12)
        self._last_aux_loss = E * (imp * load).sum()
        self._last_router_stats = self._summarize_routing(p, gates, idx, imp, load)

        N = B * T
        flat_x = x.view(N, D)  # Tokens flattened for vectorization
        flat_idx = idx.view(N, -1)  # [N,k] expert indices per token
        flat_gates = gates.view(N, -1)  # [N,k] gate weights per token

        assign_expert = flat_idx.reshape(-1)
        assign_token = torch.arange(N, device=x.device).repeat_interleave(
            self.top_k
        )  # what
        assign_weight = flat_gates.reshape(-1).unsqueeze(-1)

        order = torch.argsort(assign_expert)  # what

        # After sorting by expert:
        # order = torch.argsort(assign_expert)
        sorted_expert = assign_expert[order]  # [N*k]
        sorted_token = assign_token[order]  # [N*k]
        sorted_weight = assign_weight[order]  # [N*k, 1]

        # Compact run-length encoding of consecutive equal expert ids
        present_experts, counts = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        # Offsets are cumulative starts of each run
        offsets = torch.zeros_like(counts)
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)

        # Accumulation buffer must match the compute dtype under autocast (e.g., bfloat16)
        # Router logits `z` reflect the active compute dtype, so use `z.dtype`.
        flat_out = torch.zeros_like(flat_x, dtype=z.dtype)
        # Iterate only present experts
        for i in range(present_experts.numel()):
            e = int(present_experts[i])
            start = int(offsets[i])
            end = int(offsets[i] + counts[i])

            tok_ids = sorted_token[start:end]  # [c]
            w_e = sorted_weight[start:end]  # [c, 1]
            x_e = flat_x.index_select(0, tok_ids)  # [c, D]

            y_e = self.experts[e](x_e)
            # Ensure source matches destination dtype for index_add_
            flat_out.index_add_(0, tok_ids, (w_e * y_e).to(dtype=flat_out.dtype))

        out = flat_out.view(B, T, D)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def _summarize_routing(self, p, gates, idx, imp, load):
        eps = 1e-12
        with torch.no_grad():
            n_experts = int(self.n_experts)
            router_entropy = -(p * (p + eps).log()).sum(dim=-1).mean()
            router_entropy_norm = router_entropy / max(math.log(n_experts), eps)
            topk_entropy = -(gates * (gates + eps).log()).sum(dim=-1).mean()
            topk_entropy_norm = topk_entropy / max(math.log(max(self.top_k, 1)), eps)
            gate_mean = gates.mean()
            gate_max = gates.max()

            load_min = load.min()
            load_max = load.max()
            load_std = load.std()

            imp_min = imp.min()
            imp_max = imp.max()
            imp_std = imp.std()

            top1 = idx[..., 0].reshape(-1)
            top1_counts = torch.bincount(top1, minlength=n_experts).float()
            top1_frac = top1_counts / top1_counts.sum().clamp_min(1.0)
            top1_coverage = (top1_frac > 0).float().mean()

            return {
                "router_entropy": float(router_entropy.item()),
                "router_entropy_norm": float(router_entropy_norm.item()),
                "topk_entropy": float(topk_entropy.item()),
                "topk_entropy_norm": float(topk_entropy_norm.item()),
                "gate_mean": float(gate_mean.item()),
                "gate_max": float(gate_max.item()),
                "load_min": float(load_min.item()),
                "load_max": float(load_max.item()),
                "load_std": float(load_std.item()),
                "imp_min": float(imp_min.item()),
                "imp_max": float(imp_max.item()),
                "imp_std": float(imp_std.item()),
                "top1_coverage": float(top1_coverage.item()),
                "top1_frac": top1_frac.detach().cpu().tolist(),
                "load": load.detach().cpu().tolist(),
            }

    def routing_stats(self):
        return self._last_router_stats


class MoEBlock(nn.Module):
    def __init__(self, model_cfg, moe_cfg):
        super().__init__()
        self.ln = nn.LayerNorm(model_cfg.n_embed)
        self.attention = AttentionBlock(model_cfg)
        self.moe = MoELayer(model_cfg, moe_cfg)

    def forward(self, x):
        x = x + self.attention(self.ln(x))
        x = x + self.moe(self.ln(x))
        return x


class TransformerNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        model_cfg = cfg.model
        moe_cfg = cfg.moe

        vocab_size = getattr(model_cfg, "vocab_size", None)
        if vocab_size is None:
            raise ValueError(
                "model.vocab_size must be set before constructing TransformerNetwork"
            )

        self.token_embedding = nn.Embedding(vocab_size, model_cfg.n_embed)
        self.W_ue = nn.Linear(model_cfg.n_embed, vocab_size, bias=model_cfg.UE_bias)
        self.W_ue.weight = self.token_embedding.weight

        if moe_cfg.use_moe:
            blocks = [MoEBlock(model_cfg, moe_cfg) for _ in range(model_cfg.n_layer)]
        else:
            blocks = [TransformerBlock(model_cfg) for _ in range(model_cfg.n_layer)]

        self.transformer_blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(model_cfg.n_embed)
        self.use_checkpoint = model_cfg.use_checkpoint

    def forward_hidden(self, x):
        """
        Returns pre-logits hidden states after LayerNorm, without projecting to vocab.
        Shape: (B, T, E)
        """
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            if self.training and self.use_checkpoint:
                x = ckpt(block, x)
            else:
                x = block(x)
        x = self.ln(x)
        return x

    def forward(self, x):
        hidden = self.forward_hidden(x)
        return self.W_ue(hidden)

    def moe_aux_loss(self):
        total = None
        for m in self.modules():
            if (
                isinstance(m, MoELayer)
                and getattr(m, "_last_aux_loss", None) is not None
            ):
                total = m._last_aux_loss if total is None else total + m._last_aux_loss
        if total is None:
            return self.W_ue.weight.new_tensor(0.0, dtype=torch.float32)
        return total

    def moe_routing_stats(self):
        stats = []
        for idx, block in enumerate(self.transformer_blocks):
            moe = getattr(block, "moe", None)
            if isinstance(moe, MoELayer):
                layer_stats = moe.routing_stats()
                if layer_stats:
                    stats.append((idx, layer_stats))
        return stats
