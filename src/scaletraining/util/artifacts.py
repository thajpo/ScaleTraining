"""Helpers for persisting training artifacts and metadata."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .path_utils import _sanitize, config_fingerprint, get_cfg_subset


_REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_FILENAME = "model.pt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def checkpoint_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for a checkpoint file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_provenance(
    checkpoint_path: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """Build a portable identity for the checkpoint owned by a run directory."""

    run_path = Path(run_dir).expanduser().resolve(strict=False)
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    expected = (run_path / CHECKPOINT_FILENAME).resolve(strict=False)
    if checkpoint != expected:
        raise ValueError(
            f"Checkpoint {checkpoint} does not belong to run directory {run_path}; "
            f"expected {expected}."
        )
    return {
        "path": CHECKPOINT_FILENAME,
        "sha256": checkpoint_sha256(checkpoint),
        "original_path": str(checkpoint),
    }


def write_metadata(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"Warning: could not write metadata to {path}: {exc}")


def read_metadata(path: str) -> Dict[str, Any]:
    """Read metadata used to compare configs with existing data artifacts."""
    try:
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Warning: could not read metadata, returning empty dictionary: {exc}")
        return {}


def save_run_manifest(cfg: Any, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write the stable configuration and identity for one training run."""
    os.makedirs(out_dir, exist_ok=True)
    training_cfg = cfg.training
    optimizer_cfg = cfg.optimizer
    model_cfg = cfg.model
    tokenizer_cfg = cfg.tokenizer

    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_id": Path(out_dir).name,
        "status": "created",
        "dataset": get_cfg_subset(cfg),
        "optimizer": {
            "primary": optimizer_cfg.primary_optimizer,
            "lr": optimizer_cfg.lr,
            "beta": optimizer_cfg.beta,
            "beta2": optimizer_cfg.beta2,
            "weight_decay": optimizer_cfg.weight_decay,
            "ns_iters": optimizer_cfg.ns_iters,
            "eps": optimizer_cfg.eps,
            "lr_schedule": optimizer_cfg.lr_schedule,
            "warmup_tokens": optimizer_cfg.warmup_tokens,
            "min_lr_scale": optimizer_cfg.min_lr_scale,
        },
        "training": {
            "seed": training_cfg.seed,
            "batch_size": training_cfg.batch_size,
            "accum_steps": training_cfg.accum_steps,
            "effective_batch_size": training_cfg.batch_size * training_cfg.accum_steps,
            "grad_clip_norm": training_cfg.grad_clip_norm,
            "logits_chunk_size": training_cfg.logits_chunk_size,
            "max_train_tokens": training_cfg.max_train_tokens,
            "max_val_tokens": training_cfg.max_val_tokens,
            "eval_interval_tokens": training_cfg.eval_interval_tokens,
            "device_requested": (
                getattr(cfg, "device_requested", None) or cfg.device.device
            ),
            "device_resolved": getattr(cfg, "device_resolved", None),
        },
        "transformer": {
            "n_layer": model_cfg.n_layer,
            "n_head": model_cfg.n_head,
            "n_embed": model_cfg.n_embed,
            "n_hidden": model_cfg.n_hidden,
            "max_seq_len": model_cfg.max_seq_len,
            "activation": getattr(model_cfg, "activation", "relu"),
            "vocab_size": model_cfg.vocab_size,
            "UE_bias": model_cfg.UE_bias,
            "use_checkpoint": model_cfg.use_checkpoint,
        },
        "moe": {
            "enabled": bool(cfg.moe.use_moe),
            "n_layers": int(cfg.moe.moe_n_layers),
            "n_experts": int(cfg.moe.moe_n_experts),
            "top_k": int(cfg.moe.moe_top_k),
            "n_hidden": int(cfg.moe.moe_n_hidden),
            "load_balance_coefficient": float(cfg.moe.moe_lb_coef),
            "router_temperature": float(cfg.moe.moe_router_temp),
            "router_noise": float(cfg.moe.moe_router_noise),
        },
        "tokenizer": {
            "tokenizer_name": tokenizer_cfg.tokenizer_name,
            "tokenizer_type": tokenizer_cfg.tokenizer_type,
        },
        "dataset_tag": _first_non_empty(tokenizer_cfg.dataset_tag),
        "fingerprint": config_fingerprint(cfg),
        "implementation": {
            "optimizer": (
                "baseline_adam"
                if optimizer_cfg.use_baseline_adam
                else optimizer_cfg.primary_optimizer
            ),
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


def update_run_manifest(out_dir: str | Path, **updates: Any) -> Path:
    """Update lifecycle fields without rebuilding or discarding the manifest."""

    manifest_path = Path(out_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.update(updates)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def _first_non_empty(values):
    for value in values:
        if value not in (None, "", "null"):
            return value
    return None


def _resolve_output_root(cfg: Any, out_root: Optional[str] = None) -> Path:
    value = out_root or cfg.paths.output_dir
    root = Path(str(value)).expanduser()
    if not root.is_absolute():
        cwd_candidate = Path.cwd() / root
        root = (
            cwd_candidate.resolve(strict=False)
            if cwd_candidate.exists()
            else (_REPO_ROOT / root).resolve(strict=False)
        )
    return root


def create_run_dir(cfg: Any, out_root: Optional[str] = None) -> Path:
    """Allocate the directory that all artifacts for one run will share."""

    tokenizer_cfg = cfg.tokenizer
    output_root = _resolve_output_root(cfg, out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tag = _sanitize(_first_non_empty(tokenizer_cfg.dataset_tag) or "")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = config_fingerprint(cfg)[:8]
    run_dir_name = "__".join(filter(None, [tag, f"v={fingerprint}", timestamp]))
    suffix = 1
    while True:
        run_dir = (
            output_root / run_dir_name
            if suffix == 1
            else output_root / f"{run_dir_name}__{suffix}"
        )
        try:
            run_dir.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return run_dir


def save_model(
    model: torch.nn.Module,
    cfg: Any,
    out_root: Optional[str] = None,
    *,
    run_dir: str | Path | None = None,
) -> str:
    """Save a checkpoint into an existing run, or allocate a run for callers."""

    allocated_here = run_dir is None
    run_path = create_run_dir(cfg, out_root) if run_dir is None else Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    model_path = run_path / CHECKPOINT_FILENAME
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
    model_config_path = run_path / "model_config.json"
    with model_config_path.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)

    if allocated_here:
        save_run_manifest(cfg, str(run_path), extra={"status": "completed"})
    return str(run_path)


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
    "CHECKPOINT_FILENAME",
    "build_checkpoint_provenance",
    "checkpoint_sha256",
    "write_metadata",
    "read_metadata",
    "create_run_dir",
    "save_run_manifest",
    "update_run_manifest",
    "save_model",
    "find_latest_model_path",
]
