"""Helpers for persisting training artifacts and metadata."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .path_utils import _sanitize, config_fingerprint, get_cfg_subset


_REPO_ROOT = Path(__file__).resolve().parents[3]


def write_metadata(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"Warning: could not write metadata to {path}: {exc}")


def read_metadata(path: str) -> Dict[str, Any]:
    """Used across codebase for validating the similarity of run config to existing data, tokenizers, etc"""
    try:
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Warning: could not read metadata, returning empty dictionary: {exc}")
        return {}


def save_run_manifest(cfg: Any, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Used for saving the model configuration"""
    os.makedirs(out_dir, exist_ok=True)
    training_cfg = cfg.training
    optimizer_cfg = cfg.optimizer
    model_cfg = cfg.model
    tokenizer_cfg = cfg.tokenizer
    paths_cfg = cfg.paths

    manifest = {
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": get_cfg_subset(cfg),
        "optimizer": {
            "primary": optimizer_cfg.primary_optimizer,
            "lr": optimizer_cfg.lr,
            "beta": optimizer_cfg.beta,
            "beta2": optimizer_cfg.beta2,
            "weight_decay": optimizer_cfg.weight_decay,
            "ns_iters": optimizer_cfg.ns_iters,
            "eps": optimizer_cfg.eps,
        },
        "training": {
            "seed": training_cfg.seed,
            "batch_size": training_cfg.batch_size,
            "accum_steps": training_cfg.accum_steps,
            "effective_batch_size": training_cfg.batch_size * training_cfg.accum_steps,
            "grad_clip_norm": training_cfg.grad_clip_norm,
            "logits_chunk_size": training_cfg.logits_chunk_size,
            "device": cfg.device.device,
        },
        "transformer": {
            "n_layer": model_cfg.n_layer,
            "n_head": model_cfg.n_head,
            "n_embed": model_cfg.n_embed,
            "n_hidden": model_cfg.n_hidden,
            "activation": getattr(model_cfg, "activation", "relu"),
            "vocab_size": model_cfg.vocab_size,
            "UE_bias": model_cfg.UE_bias,
            "use_checkpoint": model_cfg.use_checkpoint,
        },
        "tokenizer": {
            "tokenizer_name": tokenizer_cfg.tokenizer_name,
            "tokenizer_type": tokenizer_cfg.tokenizer_type,
        },
        "dataset_tag": _first_non_empty(tokenizer_cfg.dataset_tag),
        "fingerprint": config_fingerprint(cfg),
        "implementation": {
            "optimizer": "baseline_adam" if optimizer_cfg.use_baseline_adam else optimizer_cfg.primary_optimizer,
            "rope": {
                "enabled": bool(getattr(model_cfg, "use_rope", True)),
                "theta": getattr(getattr(model_cfg, "rope_config", {}), "theta", 10000),
            },
        },
    }
    if extra:
        manifest.update(extra)
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def _first_non_empty(values):
    for value in values:
        if value not in (None, "", "null"):
            return value
    return None


def save_model(model: torch.nn.Module, cfg: Any, out_root: Optional[str] = None) -> str:
    paths_cfg = cfg.paths
    tokenizer_cfg = cfg.tokenizer

    out_root = out_root or paths_cfg.output_dir
    if out_root and not os.path.isabs(out_root):
        # Prefer cwd resolution if already absolute, otherwise anchor to repo root
        cwd_candidate = Path.cwd() / out_root
        if cwd_candidate.exists():
            out_root = str(cwd_candidate.resolve(strict=False))
        else:
            out_root = str((_REPO_ROOT / out_root).resolve(strict=False))
    tag = _sanitize(_first_non_empty(tokenizer_cfg.dataset_tag) or "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = config_fingerprint(cfg)[:8]
    run_dir_name = "__".join(filter(None, [tag, f"v={fingerprint}", timestamp]))
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.pt")
    base_mod = getattr(model, "_orig_mod", model)
    state = base_mod.state_dict()
    torch.save({"state_dict": state}, model_path)

    # Save model config for easy checkpoint loading without manual overrides
    model_cfg = cfg.model
    model_config = {
        "n_layer": int(model_cfg.n_layer),
        "n_head": int(model_cfg.n_head),
        "n_embed": int(model_cfg.n_embed),
        "n_hidden": int(model_cfg.n_hidden),
        "max_seq_len": int(model_cfg.max_seq_len),
        "vocab_size": int(model_cfg.vocab_size),
        "bias": bool(model_cfg.bias),
        "UE_bias": bool(model_cfg.UE_bias),
        "activation": str(getattr(model_cfg, "activation", "relu")),
        "attn_dropout": float(model_cfg.attn_dropout),
        "resid_dropout": float(model_cfg.resid_dropout),
        "use_rope": bool(getattr(model_cfg, "use_rope", True)),
        "use_checkpoint": bool(model_cfg.use_checkpoint),
    }
    model_config_path = os.path.join(run_dir, "model_config.json")
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    save_run_manifest(cfg, run_dir)
    return run_dir


def find_latest_model_path(output_root: str) -> Optional[str]:
    """
    Return path to the newest \"model.pt\" under `output_root`, if present.
    Used for model generation
    """
    try:
        root = Path(output_root)
        if not root.is_absolute() and not root.exists():
            repo_candidate = (_REPO_ROOT / root).resolve(strict=False)
            if repo_candidate.exists():
                root = repo_candidate
        if not root.exists():
            return None

        latest_link = root / "latest"
        if latest_link.exists():
            link_target = latest_link.resolve()
            candidate = link_target / "model.pt"
            if candidate.exists():
                return str(candidate)

        candidates = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "model.pt"
            if candidate.exists():
                try:
                    mtime = candidate.stat().st_mtime
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, candidate))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return str(candidates[0][1])
    except Exception:
        return None


__all__ = [
    "write_metadata",
    "read_metadata",
    "save_run_manifest",
    "save_model",
    "find_latest_model_path",
]
