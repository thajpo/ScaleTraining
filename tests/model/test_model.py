from types import SimpleNamespace

import pytest
import torch

from scaletraining.model.model import (
    MLPBlock,
    MoEBlock,
    MoELayer,
    TransformerBlock,
    TransformerNetwork,
)


def _model_cfg(**overrides):
    values = {
        "n_layer": 4,
        "max_seq_len": 8,
        "n_head": 2,
        "n_embed": 8,
        "n_hidden": 16,
        "bias": True,
        "UE_bias": False,
        "activation": "relu",
        "attn_dropout": 0.0,
        "resid_dropout": 0.0,
        "use_checkpoint": False,
        "use_rope": True,
        "rope_config": {"theta": 10000.0},
        "vocab_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe_cfg(**overrides):
    values = {
        "use_moe": False,
        "moe_n_experts": 3,
        "moe_top_k": 2,
        "moe_n_hidden": 12,
        "moe_activation": "swiGLU",
        "moe_use_shared": False,
        "moe_n_layers": 0,
        "moe_router_noise": 0.0,
        "moe_router_temp": 1.0,
        "moe_lb_coef": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cfg(*, model=None, moe=None):
    return SimpleNamespace(model=model or _model_cfg(), moe=moe or _moe_cfg())


def test_mlp_block_does_not_apply_residual_internally():
    block = MLPBlock(_model_cfg())
    block.eval()
    for param in block.parameters():
        torch.nn.init.zeros_(param)

    x = torch.randn(2, 3, 8)

    assert torch.equal(block(x), torch.zeros_like(x))


def test_transformer_and_moe_blocks_use_separate_sublayer_norms():
    model_cfg = _model_cfg()
    moe_cfg = _moe_cfg(use_moe=True, moe_n_layers=1)

    dense = TransformerBlock(model_cfg)
    moe = MoEBlock(model_cfg, moe_cfg)

    assert isinstance(dense.ln1, torch.nn.LayerNorm)
    assert isinstance(dense.ln2, torch.nn.LayerNorm)
    assert dense.ln1 is not dense.ln2
    assert isinstance(moe.ln1, torch.nn.LayerNorm)
    assert isinstance(moe.ln2, torch.nn.LayerNorm)
    assert moe.ln1 is not moe.ln2


def test_moe_n_layers_controls_only_the_last_n_blocks():
    cfg = _cfg(moe=_moe_cfg(use_moe=True, moe_n_layers=2))

    model = TransformerNetwork(cfg)

    assert [type(block) for block in model.transformer_blocks] == [
        TransformerBlock,
        TransformerBlock,
        MoEBlock,
        MoEBlock,
    ]


def test_invalid_moe_n_layers_fails_fast():
    cfg = _cfg(moe=_moe_cfg(use_moe=True, moe_n_layers=5))

    with pytest.raises(ValueError, match="moe_n_layers"):
        TransformerNetwork(cfg)


@pytest.mark.parametrize("use_moe", [False, True])
def test_transformer_forward_backward_is_finite(use_moe):
    moe_cfg = _moe_cfg(use_moe=use_moe, moe_n_layers=1 if use_moe else 0)
    cfg = _cfg(model=_model_cfg(n_layer=2), moe=moe_cfg)
    model = TransformerNetwork(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 5))

    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()

    assert logits.shape == (2, 5, cfg.model.vocab_size)
    assert torch.isfinite(logits).all()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in model.parameters()
    )


def test_moe_routing_stats_exist_after_forward():
    cfg = _cfg(model=_model_cfg(n_layer=1), moe=_moe_cfg(use_moe=True, moe_n_layers=1))
    model = TransformerNetwork(cfg)

    _ = model(torch.randint(0, cfg.model.vocab_size, (2, 4)))

    stats = model.moe_routing_stats()
    assert len(stats) == 1
    assert stats[0][0] == 0
    assert "router_entropy" in stats[0][1]
    assert "load" in stats[0][1]
    assert any(isinstance(module, MoELayer) for module in model.modules())
