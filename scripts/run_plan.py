#!/usr/bin/env python3
"""Plan a ScaleTraining run without training.

The report is intended for employer-facing run records: it captures model size,
token budget, target loss, estimated compute, and the exact commands needed to
produce and evaluate a checkpoint later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from scaletraining.config import load_project_config
from scaletraining.model import TransformerNetwork
from scaletraining.util.model_stats import count_parameters, humanize_bytes, humanize_params
from scaletraining.util.path_utils import config_fingerprint, get_packed_directory, get_tokenized_directory
from scaletraining.util.training_utils import estimate_flops


SIZE_PRESETS = {
    "tiny": {
        "model.n_layer": "2",
        "model.n_head": "2",
        "model.n_embed": "64",
        "model.n_hidden": "256",
        "model.max_seq_len": "64",
    },
    "small": {
        "model.n_layer": "6",
        "model.n_head": "4",
        "model.n_embed": "256",
        "model.n_hidden": "1024",
        "model.max_seq_len": "512",
    },
    "medium": {
        "model.n_layer": "12",
        "model.n_head": "8",
        "model.n_embed": "512",
        "model.n_hidden": "2048",
        "model.max_seq_len": "1024",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=REPO_ROOT / "conf", type=Path)
    parser.add_argument("--config-name", default="config")
    parser.add_argument(
        "--model-size",
        choices=sorted(SIZE_PRESETS),
        help="Apply a named architecture preset before explicit overrides.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        help="Override training.max_train_tokens in the planned run.",
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        help="Loss threshold to record as the success criterion for the run.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Hydra-style override, e.g. -o model.n_layer=8.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a Markdown report.",
    )
    return parser.parse_args(argv)


def _preset_overrides(model_size: str | None) -> list[str]:
    if not model_size:
        return []
    return [f"{key}={value}" for key, value in SIZE_PRESETS[model_size].items()]


def load_cfg(config_path: Path, config_name: str, overrides: Iterable[str]):
    config_dir = config_path.expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def _format_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e6 or abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:,.3f}"


def build_report(args: argparse.Namespace) -> dict:
    overrides = _preset_overrides(args.model_size)
    if args.token_budget is not None:
        overrides.append(f"training.max_train_tokens={args.token_budget}")
    overrides.extend(args.override)

    cfg = load_project_config(load_cfg(args.config_path, args.config_name, overrides))
    if getattr(cfg.model, "vocab_size", None) is None:
        with open_dict(cfg.model):
            cfg.model.vocab_size = 49152

    model = TransformerNetwork(cfg)
    total_params, trainable_params = count_parameters(model)
    tokens = int(cfg.training.max_train_tokens)
    n_moe_layers = int(cfg.moe.moe_n_layers) if bool(cfg.moe.use_moe) else 0
    flops = estimate_flops(
        tokens_used=tokens,
        d_model=int(cfg.model.n_embed),
        d_hidden=int(cfg.model.n_hidden),
        n_heads=int(cfg.model.n_head),
        seq_len=int(cfg.model.max_seq_len),
        n_layers=int(cfg.model.n_layer),
        n_moe_layers=n_moe_layers,
        top_k=int(cfg.moe.moe_top_k),
        n_experts=int(cfg.moe.moe_n_experts),
        using_moe=bool(cfg.moe.use_moe),
    )

    override_text = " ".join(overrides)
    train_command = f"uv run python -m scaletraining.entrypoints.train {override_text}".strip()
    eval_command = f"uv run python -m scaletraining.entrypoints.run_evals {override_text}".strip()
    lm_eval_command = (
        f"LM_EVAL_TASKS={cfg.eval.tasks} uv run python -m scaletraining.entrypoints.run_lm_eval {override_text}"
    ).strip()

    return {
        "model_size_preset": args.model_size,
        "target_loss": args.target_loss,
        "success_criterion": (
            f"final_train_loss <= {args.target_loss}"
            if args.target_loss is not None
            else "record final_train_loss, validation loss, and benchmark scores"
        ),
        "token_budget": tokens,
        "dataset_fingerprint": config_fingerprint(cfg)[:8],
        "tokenized_train_dir": get_tokenized_directory(cfg, for_training=True),
        "packed_train_dir": get_packed_directory(cfg),
        "model": {
            "n_layer": int(cfg.model.n_layer),
            "n_head": int(cfg.model.n_head),
            "n_embed": int(cfg.model.n_embed),
            "n_hidden": int(cfg.model.n_hidden),
            "max_seq_len": int(cfg.model.max_seq_len),
            "vocab_size_for_estimate": int(cfg.model.vocab_size),
            "use_moe": bool(cfg.moe.use_moe),
            "moe_n_layers": n_moe_layers,
        },
        "training": {
            "batch_size": int(cfg.training.batch_size),
            "accum_steps": int(cfg.training.accum_steps),
            "effective_batch_size": int(cfg.training.batch_size) * int(cfg.training.accum_steps),
            "optimizer": str(cfg.optimizer.primary_optimizer),
            "lr": float(cfg.optimizer.lr),
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "total_human": humanize_params(total_params),
            "trainable_human": humanize_params(trainable_params),
            "fp32_size": humanize_bytes(total_params * 4),
            "bf16_size": humanize_bytes(total_params * 2),
        },
        "estimated_compute": {
            "flops": flops,
            "flops_human": _format_float(float(flops)),
        },
        "commands": {
            "prepare_data": f"uv run python -m scaletraining.entrypoints.prepare_data {override_text}".strip(),
            "train": train_command,
            "validation_perplexity": eval_command,
            "lm_eval": lm_eval_command,
        },
        "overrides": overrides,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }


def print_markdown(report: dict) -> None:
    print("# ScaleTraining Run Plan")
    print()
    print(f"- Model preset: `{report['model_size_preset'] or 'custom/default'}`")
    print(f"- Token budget: `{report['token_budget']:,}`")
    print(f"- Success criterion: `{report['success_criterion']}`")
    print(f"- Parameters: `{report['parameters']['total_human']}` total, `{report['parameters']['trainable_human']}` trainable")
    print(f"- Estimated compute: `{report['estimated_compute']['flops_human']}` FLOPs")
    print(f"- Dataset fingerprint: `{report['dataset_fingerprint']}`")
    print()
    print("## Commands")
    for name, command in report["commands"].items():
        print(f"- `{name}`: `{command}`")
    print()
    print("## What To Record After Running")
    print("- `result.json` final_train_loss and core hyperparameters")
    print("- `outputs/<run>/run_manifest.json` for reproducibility")
    print("- validation loss/perplexity from `run_evals`")
    print("- lm-eval table for the selected benchmark tasks")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
