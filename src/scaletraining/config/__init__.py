"""Structured configuration schema and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class DeviceConfig:
    device: str = field(default=MISSING)
    use_flash_sdp: bool = field(default=MISSING)
    use_mem_efficient_sdp: bool = field(default=MISSING)
    use_math_sdp: bool = field(default=MISSING)


@dataclass
class LoggingConfig:
    wandb_project_name: str = field(default=MISSING)
    debug_memory: bool = field(default=MISSING)
    experiment_tags: List[str] = field(default=MISSING)
    log_implementation_details: bool = field(default=MISSING)
    log_dataset_artifacts: bool = field(default=MISSING)
    log_model_artifacts: bool = field(default=MISSING)


@dataclass
class SweepConfig:
    name: str = field(default=MISSING)


@dataclass
class GenerationConfig:
    prompt: str = field(default=MISSING)
    generation_max_tokens: int = field(default=MISSING)
    generation_temperature: float = field(default=MISSING)
    generate_after_train: bool = field(default=MISSING)
    generation_top_k: Optional[int] = field(default=None)
    model_path: Optional[str] = field(default=None)


@dataclass
class PathsConfig:
    tokenized_train_path: str = field(default=MISSING)
    tokenized_eval_path: str = field(default=MISSING)
    batched_tokenized_path: str = field(default=MISSING)
    tokenizer_save_path: str = field(default=MISSING)
    tokenizer_train_data: str = field(default=MISSING)
    output_dir: str = field(default=MISSING)


@dataclass
class RopeConfig:
    theta: float = field(default=MISSING)
    use_complex: bool = field(default=MISSING)


@dataclass
class ModelConfig:
    n_layer: int = field(default=MISSING)
    max_seq_len: int = field(default=MISSING)
    n_head: int = field(default=MISSING)
    n_embed: int = field(default=MISSING)
    n_hidden: int = field(default=MISSING)
    bias: bool = field(default=MISSING)
    UE_bias: bool = field(default=MISSING)
    activation: str = field(default=MISSING)
    attn_dropout: float = field(default=MISSING)
    resid_dropout: float = field(default=MISSING)
    use_checkpoint: bool = field(default=MISSING)
    use_rope: bool = field(default=MISSING)
    rope_config: RopeConfig = field(default=MISSING)
    vocab_size: Optional[int] = field(default=None)


@dataclass
class BaselineAdamConfig:
    lr: float = field(default=MISSING)
    weight_decay: float = field(default=MISSING)
    betas: List[float] = field(default=MISSING)


@dataclass
class OptimizerConfig:
    lr: float = field(default=MISSING)
    muon_lr: float = field(default=MISSING)
    beta: float = field(default=MISSING)
    beta2: float = field(default=MISSING)
    weight_decay: float = field(default=MISSING)
    ns_iters: int = field(default=MISSING)
    eps: float = field(default=MISSING)
    primary_optimizer: str = field(default=MISSING)
    use_baseline_adam: bool = field(default=MISSING)
    baseline_adam_config: BaselineAdamConfig = field(default=MISSING)
    lr_schedule: str = field(default=MISSING)
    warmup_tokens: int = field(default=MISSING)
    min_lr_scale: float = field(default=MISSING)


@dataclass
class TokenizerConfig:
    dataset_names: List[str] = field(default=MISSING)
    dataset_tag: List[Optional[str]] = field(default=MISSING)
    strict_dataset_compat: bool = field(default=MISSING)
    hf_token: str = field(default=MISSING)
    pretrained_tokenizer_name: str = field(default=MISSING)
    is_pretrained: bool = field(default=MISSING)
    num_proc: int = field(default=MISSING)
    do_packing: bool = field(default=MISSING)
    pack_num_proc: int = field(default=MISSING)
    pack_map_batch_size: int = field(default=MISSING)
    pack_writer_batch_size: int = field(default=MISSING)
    custom_tokenizer_vocab_size: int = field(default=MISSING)
    tokenizer_name: Optional[str] = field(default=None)
    tokenizer_type: Optional[str] = field(default=None)
    hf_dataset_config_name: Optional[str] = field(default=None)


@dataclass
class MoeConfig:
    use_moe: bool = field(default=MISSING)
    moe_n_experts: int = field(default=MISSING)
    moe_top_k: int = field(default=MISSING)
    moe_n_hidden: int = field(default=MISSING)
    moe_activation: str = field(default=MISSING)
    moe_use_shared: bool = field(default=MISSING)
    moe_n_layers: int = field(default=MISSING)
    moe_router_noise: float = field(default=MISSING)
    moe_router_temp: float = field(default=MISSING)
    moe_lb_coef: float = field(default=MISSING)
    moe_router_temp_schedule: str = field(default=MISSING)
    moe_router_temp_start: float = field(default=MISSING)
    moe_router_temp_end: float = field(default=MISSING)
    moe_router_noise_schedule: str = field(default=MISSING)
    moe_router_noise_start: float = field(default=MISSING)
    moe_router_noise_end: float = field(default=MISSING)
    moe_lb_coef_schedule: str = field(default=MISSING)
    moe_lb_coef_start: float = field(default=MISSING)
    moe_lb_coef_end: float = field(default=MISSING)


@dataclass
class TrainingConfig:
    batch_size: int = field(default=MISSING)
    accum_steps: int = field(default=MISSING)
    grad_clip_norm: float = field(default=MISSING)
    max_train_tokens: int = field(default=MISSING)
    max_val_tokens: int = field(default=MISSING)
    logits_chunk_size: int = field(default=MISSING)
    early_stop_tokens_without_improvement: int = field(default=MISSING)
    early_stop_min_delta: float = field(default=MISSING)
    eval_interval_tokens: int = field(default=MISSING)
    eval_max_batches: int = field(default=MISSING)
    eval_batch_size: int = field(default=MISSING)
    loader_num_workers: int = field(default=MISSING)
    loader_pin_memory: bool = field(default=MISSING)
    loader_persistent_workers: bool = field(default=MISSING)
    loader_prefetch_factor: int = field(default=MISSING)


@dataclass
class EvalConfig:
    tasks: str = field(default="hellaswag")


@dataclass
class ProjectConfig:
    device: DeviceConfig = field(default=MISSING)
    training: TrainingConfig = field(default=MISSING)
    model: ModelConfig = field(default=MISSING)
    optimizer: OptimizerConfig = field(default=MISSING)
    tokenizer: TokenizerConfig = field(default=MISSING)
    logging: LoggingConfig = field(default=MISSING)
    generation: GenerationConfig = field(default=MISSING)
    paths: PathsConfig = field(default=MISSING)
    moe: MoeConfig = field(default=MISSING)
    eval: EvalConfig = field(default_factory=EvalConfig)
    sweep: SweepConfig = field(default=MISSING)
    device_resolved: Optional[str] = field(default=None)


def load_project_config(cfg: DictConfig) -> DictConfig:
    """Merge runtime config with the structured schema and enforce struct mode."""

    schema = OmegaConf.structured(ProjectConfig)
    merged = OmegaConf.merge(schema, cfg)
    OmegaConf.set_struct(merged, True)
    return merged


__all__ = [
    "ProjectConfig",
    "TrainingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TokenizerConfig",
    "LoggingConfig",
    "GenerationConfig",
    "PathsConfig",
    "MoeConfig",
    "EvalConfig",
    "DeviceConfig",
    "load_project_config",
]
