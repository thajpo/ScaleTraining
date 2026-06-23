from omegaconf import OmegaConf

from scaletraining.config import load_project_config


def test_project_config_schema_imports_and_merges_minimal_config():
    cfg = OmegaConf.create(
        {
            "device": {
                "device": "cpu",
                "use_flash_sdp": False,
                "use_mem_efficient_sdp": False,
                "use_math_sdp": True,
            },
            "training": {
                "seed": 13,
                "batch_size": 1,
                "accum_steps": 1,
                "grad_clip_norm": 1.0,
                "max_train_tokens": 8,
                "max_val_tokens": 8,
                "logits_chunk_size": 0,
                "early_stop_tokens_without_improvement": 0,
                "early_stop_min_delta": 0.0,
                "eval_interval_tokens": 0,
                "eval_max_batches": 0,
                "eval_batch_size": 1,
                "loader_num_workers": 0,
                "loader_pin_memory": False,
                "loader_persistent_workers": False,
                "loader_prefetch_factor": 0,
            },
            "model": {
                "n_layer": 1,
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
                "rope_config": {"theta": 10000.0, "use_complex": True},
            },
            "optimizer": {
                "lr": 0.001,
                "muon_lr": 0.001,
                "beta": 0.9,
                "beta2": 0.999,
                "weight_decay": 0.0,
                "ns_iters": 3,
                "eps": 1e-8,
                "primary_optimizer": "adamw",
                "use_baseline_adam": True,
                "baseline_adam_config": {
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                },
                "lr_schedule": "constant",
                "warmup_tokens": 0,
                "min_lr_scale": 1.0,
            },
            "tokenizer": {
                "dataset_names": ["roneneldan/TinyStories"],
                "dataset_tag": [None],
                "strict_dataset_compat": False,
                "hf_token": "",
                "pretrained_tokenizer_name": "bigcode/starcoder2-tokenizer",
                "is_pretrained": True,
                "num_proc": 1,
                "do_packing": False,
                "pack_num_proc": 1,
                "pack_map_batch_size": 1,
                "pack_writer_batch_size": 1,
                "custom_tokenizer_vocab_size": 128,
            },
            "logging": {
                "wandb_project_name": "test",
                "debug_memory": False,
                "experiment_tags": [],
                "log_implementation_details": False,
                "log_dataset_artifacts": False,
                "log_model_artifacts": False,
            },
            "generation": {
                "prompt": "hello",
                "generation_max_tokens": 8,
                "generation_temperature": 1.0,
                "generation_top_k": 10,
                "generate_after_train": False,
                "model_path": "latest",
            },
            "paths": {
                "tokenized_train_path": "data/train/tokenized/tokenized_base",
                "tokenized_eval_path": "data/test/tokenized/tokenized_base",
                "batched_tokenized_path": "data/train/tokenized/tokenized_batched",
                "tokenizer_save_path": "tokenizers/tokenizer.json",
                "tokenizer_train_data": "data/train/raw",
                "output_dir": "outputs",
            },
            "moe": {
                "use_moe": False,
                "moe_n_experts": 2,
                "moe_top_k": 1,
                "moe_n_hidden": 16,
                "moe_activation": "swiGLU",
                "moe_use_shared": False,
                "moe_n_layers": 0,
                "moe_router_noise": 0.0,
                "moe_router_temp": 1.0,
                "moe_lb_coef": 0.01,
                "moe_router_temp_schedule": "none",
                "moe_router_temp_start": 1.0,
                "moe_router_temp_end": 1.0,
                "moe_router_noise_schedule": "none",
                "moe_router_noise_start": 0.0,
                "moe_router_noise_end": 0.0,
                "moe_lb_coef_schedule": "none",
                "moe_lb_coef_start": 0.01,
                "moe_lb_coef_end": 0.01,
            },
        }
    )

    loaded = load_project_config(cfg)

    assert loaded.eval.tasks == "hellaswag"
    assert loaded.eval.write_results is True
    assert loaded.eval.output_dir is None
    assert loaded.training.seed == 13
    assert loaded.sweep is None
