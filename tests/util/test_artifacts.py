import json

import torch
from omegaconf import OmegaConf

from scaletraining.util.artifacts import (
    create_run_dir,
    save_model,
    save_run_manifest,
    update_run_manifest,
)


def _cfg(tmp_path):
    return OmegaConf.create(
        {
            "device": {"device": "cpu"},
            "device_resolved": "cpu",
            "training": {
                "seed": 13,
                "batch_size": 1,
                "accum_steps": 2,
                "grad_clip_norm": 1.0,
                "logits_chunk_size": 0,
                "max_train_tokens": 64,
                "max_val_tokens": 16,
                "eval_interval_tokens": 0,
            },
            "optimizer": {
                "primary_optimizer": "adamw",
                "lr": 0.001,
                "beta": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0,
                "ns_iters": 3,
                "eps": 1e-8,
                "use_baseline_adam": True,
                "lr_schedule": "constant",
                "warmup_tokens": 0,
                "min_lr_scale": 1.0,
            },
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "n_embed": 4,
                "n_hidden": 8,
                "max_seq_len": 4,
                "vocab_size": 16,
                "bias": True,
                "UE_bias": False,
                "activation": "relu",
                "attn_dropout": 0.0,
                "resid_dropout": 0.0,
                "use_checkpoint": False,
                "use_rope": True,
                "rope_config": {"theta": 10000.0},
            },
            "tokenizer": {
                "dataset_names": ["fixture"],
                "dataset_tag": [None],
                "tokenizer_name": "fixture-tokenizer",
                "pretrained_tokenizer_name": "fixture-tokenizer",
                "tokenizer_type": "wordpiece",
                "hf_dataset_config_name": None,
            },
            "paths": {"output_dir": str(tmp_path / "outputs")},
            "moe": {
                "use_moe": False,
                "moe_n_layers": 0,
                "moe_n_experts": 2,
                "moe_top_k": 1,
                "moe_n_hidden": 8,
                "moe_lb_coef": 0.01,
                "moe_router_temp": 1.0,
                "moe_router_noise": 0.0,
            },
        }
    )


def test_existing_run_directory_is_reused_for_all_checkpoint_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    run_dir = create_run_dir(cfg)
    tracking = {
        "provider": "wandb",
        "schema_version": 1,
        "state": "initialized",
        "url": "https://wandb.ai/entity/project/runs/abc",
    }
    save_run_manifest(
        cfg,
        str(run_dir),
        extra={"status": "running", "tracking": tracking},
    )

    returned = save_model(torch.nn.Linear(4, 4), cfg, run_dir=run_dir)
    update_run_manifest(run_dir, status="completed", finished_at="now")

    assert returned == str(run_dir)
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "model_config.json").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["run_id"] == run_dir.name
    assert manifest["status"] == "completed"
    assert manifest["tracking"] == tracking
    assert manifest["training"]["max_train_tokens"] == 64
    assert manifest["training"]["device_resolved"] == "cpu"
    assert manifest["transformer"]["max_seq_len"] == 4
    assert manifest["moe"]["enabled"] is False


def test_run_directory_allocation_avoids_same_second_collisions(tmp_path):
    cfg = _cfg(tmp_path)

    first = create_run_dir(cfg)
    second = create_run_dir(cfg)

    assert first != second
    assert first.parent == second.parent
    assert second.name.endswith("__2")
