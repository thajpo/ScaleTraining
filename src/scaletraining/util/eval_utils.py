from __future__ import annotations
import contextlib
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Tuple

import torch
from omegaconf import DictConfig, open_dict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scaletraining.model.model import TransformerNetwork
from scaletraining.reporting import validate_evidence_payload
from scaletraining.util import find_latest_model_path
from scaletraining.util.artifacts import build_checkpoint_provenance
from scaletraining.util.device import resolve_device, uses_cuda
from scaletraining.util.path_utils import config_fingerprint, get_cfg_subset


import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from scaletraining.util.training_utils import compute_loss_sum, prepare_targets


_REPO_ROOT = Path(__file__).resolve().parents[3]


@torch.inference_mode()
def evaluate_perplexity_stats(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: Any,
    loss_fn: nn.Module,
    *,
    max_batches: int = 0,
) -> dict[str, Any]:
    """Evaluate per-token loss/perplexity and return artifact-friendly stats."""

    was_training = model.training
    model.eval()
    device = resolve_device(cfg)
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        context = (
            autocast(device_type="cuda", dtype=torch.bfloat16)
            if uses_cuda(device)
            else contextlib.nullcontext()
        )
        with context:
            hidden = model.forward_hidden(input_ids)[:, :-1, :]
            targets, effective = prepare_targets(input_ids)
            loss_sum = compute_loss_sum(
                model,
                hidden,
                targets,
                getattr(cfg.training, "logits_chunk_size", 0),
                loss_fn,
            )
        total_loss += float(loss_sum.item())
        total_tokens += int(effective)
        batches_seen += 1
        if max_batches and batches_seen >= max_batches:
            break
    avg = (total_loss / max(1, total_tokens)) if total_tokens > 0 else float("inf")
    ppl = math.exp(min(50.0, max(-50.0, avg))) if avg != float("inf") else float("inf")
    if was_training:
        model.train()
    return {
        "loss": avg,
        "perplexity": ppl,
        "total_loss": total_loss,
        "tokens": total_tokens,
        "batches": batches_seen,
        "max_batches": int(max_batches),
    }


@torch.inference_mode()
def evaluate_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: Any,
    loss_fn: nn.Module,
    *,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Evaluate average per-token loss and perplexity on a data loader."""

    stats = evaluate_perplexity_stats(
        model,
        data_loader,
        cfg,
        loss_fn,
        max_batches=max_batches,
    )
    avg = float(stats["loss"])
    ppl = float(stats["perplexity"])
    return avg, ppl


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _checkpoint_path_from_cfg(
    cfg: DictConfig,
    checkpoint_path: str | Path | None = None,
) -> Path:
    if checkpoint_path is None:
        checkpoint_path = cfg.generation.model_path
    path = Path(str(checkpoint_path)).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).expanduser()
    return path.resolve(strict=False)


def resolve_eval_output_dir(
    cfg: DictConfig,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Resolve the run directory where checkpoint-bound eval evidence belongs."""

    configured = getattr(cfg.eval, "output_dir", None)
    if configured:
        output_dir = Path(str(configured)).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_REPO_ROOT / output_dir).expanduser()
        return output_dir.resolve(strict=False)
    return _checkpoint_path_from_cfg(cfg, checkpoint_path).parent


def _config_summary(cfg: DictConfig) -> dict[str, Any]:
    return {
        "model": {
            "n_layer": int(cfg.model.n_layer),
            "n_head": int(cfg.model.n_head),
            "n_embed": int(cfg.model.n_embed),
            "n_hidden": int(cfg.model.n_hidden),
            "max_seq_len": int(cfg.model.max_seq_len),
            "use_moe": bool(cfg.moe.use_moe),
            "moe_n_layers": int(cfg.moe.moe_n_layers),
        },
        "training": {
            "seed": int(cfg.training.seed),
            "batch_size": int(cfg.training.batch_size),
            "accum_steps": int(cfg.training.accum_steps),
            "max_train_tokens": int(cfg.training.max_train_tokens),
            "eval_batch_size": int(cfg.training.eval_batch_size),
            "eval_max_batches": int(cfg.training.eval_max_batches),
        },
        "optimizer": {
            "primary": str(cfg.optimizer.primary_optimizer),
            "lr": float(cfg.optimizer.lr),
        },
        "eval": {
            "tasks": str(getattr(cfg.eval, "tasks", "")),
            "write_results": bool(getattr(cfg.eval, "write_results", True)),
            "output_dir": getattr(cfg.eval, "output_dir", None),
        },
    }


def _dataset_summary(cfg: DictConfig) -> dict[str, Any]:
    fingerprint = config_fingerprint(cfg)
    return {
        "fingerprint": fingerprint,
        "fingerprint_short": fingerprint[:8],
        "config": get_cfg_subset(cfg),
    }


def build_eval_result(
    cfg: DictConfig,
    validation: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a validation sidecar bound to the checkpoint-owning run."""

    checkpoint = _checkpoint_path_from_cfg(cfg, checkpoint_path)
    evidence_dir = checkpoint.parent if run_dir is None else Path(run_dir)
    return _jsonable(
        {
            "schema_version": 1,
            "created_at": _utc_now(),
            "checkpoint": build_checkpoint_provenance(checkpoint, evidence_dir),
            "dataset": _dataset_summary(cfg),
            "validation": validation,
            "config_summary": _config_summary(cfg),
        }
    )


def build_lm_eval_result(
    cfg: DictConfig,
    tasks: list[str],
    results: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build an lm-eval sidecar bound to the checkpoint-owning run."""

    checkpoint = _checkpoint_path_from_cfg(cfg, checkpoint_path)
    evidence_dir = checkpoint.parent if run_dir is None else Path(run_dir)
    return _jsonable(
        {
            "schema_version": 1,
            "created_at": _utc_now(),
            "checkpoint": build_checkpoint_provenance(checkpoint, evidence_dir),
            "dataset": _dataset_summary(cfg),
            "tasks": tasks,
            "results": results,
            "config_summary": _config_summary(cfg),
        }
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def write_eval_result(
    cfg: DictConfig,
    validation: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Validate and atomically replace the run's validation sidecar."""

    output_dir = resolve_eval_output_dir(cfg, checkpoint_path)
    payload = build_eval_result(
        cfg,
        validation,
        checkpoint_path=checkpoint_path,
        run_dir=output_dir,
    )
    validate_evidence_payload(output_dir, "eval_result", payload)
    return _write_json(output_dir / "eval_results.json", payload)


def write_lm_eval_result(
    cfg: DictConfig,
    tasks: list[str],
    results: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Validate and atomically replace the run's lm-eval sidecar."""

    output_dir = resolve_eval_output_dir(cfg, checkpoint_path)
    payload = build_lm_eval_result(
        cfg,
        tasks,
        results,
        checkpoint_path=checkpoint_path,
        run_dir=output_dir,
    )
    validate_evidence_payload(output_dir, "lm_eval_result", payload)
    return _write_json(output_dir / "lm_eval_results.json", payload)


def _normalize_output_dir(cfg: DictConfig) -> Path:
    output_root_value = cfg.paths.output_dir
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).expanduser()
    try:
        with open_dict(cfg.paths):
            cfg.paths.output_dir = str(output_root)
    except Exception:
        pass
    return output_root


def _resolve_model_path(cfg: DictConfig, output_root: Path) -> Path:
    model_path_cfg = cfg.generation.model_path
    if not model_path_cfg or str(model_path_cfg).lower() == "latest":
        # Auto-discover latest model under outputs
        auto_path = find_latest_model_path(str(output_root))
        if not auto_path:
            raise RuntimeError(
                "No model_path provided and no latest model found under outputs/. Pass model_path=... or create outputs/<run>/model.pt."
            )
        print(f"[generate] Using latest model: {auto_path}")
        model_path = Path(auto_path)
    else:
        model_path = Path(model_path_cfg).expanduser()
        if not model_path.is_absolute():
            model_path = (_REPO_ROOT / model_path).expanduser()

    try:
        with open_dict(cfg.generation):
            cfg.generation.model_path = str(model_path)
    except Exception:
        pass
    return model_path


def load_pretrained_model_and_tokenizer(cfg: DictConfig):
    device = resolve_device(cfg)
    output_root = _normalize_output_dir(cfg)
    model_path = _resolve_model_path(cfg, output_root)

    # Load tokenizer, supporting local JSON (dataset-specific) via PreTrainedTokenizerFast
    tok_path = cfg.tokenizer.tokenizer_name or cfg.tokenizer.pretrained_tokenizer_name
    if tok_path and not cfg.tokenizer.tokenizer_name:
        try:
            with open_dict(cfg.tokenizer):
                cfg.tokenizer.tokenizer_name = tok_path
        except Exception:
            pass
    if not tok_path:
        raise ValueError(
            "Config must define tokenizer_name or pretrained_tokenizer_name to load tokenizer."
        )
    if isinstance(tok_path, str) and Path(tok_path).exists() and tok_path.endswith('.json'):
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

    if getattr(cfg.model, "vocab_size", None) is None:
        try:
            with open_dict(cfg.model):
                cfg.model.vocab_size = len(tok)
        except Exception:
            pass

    # Build model from config and load weights
    model = TransformerNetwork(cfg).to(device)
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path_obj}. Provide model_path=/absolute/path/to/model.pt or place checkpoints under {cfg.paths.output_dir}."
        )
    ckpt = torch.load(str(model_path_obj), map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Normalize keys from compiled/DataParallel checkpoints if present
    def _strip_prefix(sd, prefix: str):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd
    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict)
    model.eval()

    return model, tok
