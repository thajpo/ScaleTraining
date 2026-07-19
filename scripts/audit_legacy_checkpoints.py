#!/usr/bin/env python3
"""Audit selected legacy checkpoints without claiming retrospective evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


AUDIT_RUN_IDS = ("zrnf1np4", "kh1vl1i7", "mx3a2a0x")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_dense_tensor_shapes(
    fixed: dict[str, Any],
) -> dict[str, tuple[int, ...]]:
    if fixed.get("use_moe") is not False:
        raise ValueError("The legacy checkpoint audit requires a dense model config")
    vocab_size = int(fixed["vocab_size"])
    n_embed = int(fixed["n_embed"])
    n_hidden = int(fixed["n_hidden"])
    n_layer = int(fixed["n_layer"])
    use_bias = bool(fixed["bias"])
    use_output_bias = bool(fixed["UE_bias"])
    shapes = {
        "token_embedding.weight": (vocab_size, n_embed),
        "W_ue.weight": (vocab_size, n_embed),
        "ln.weight": (n_embed,),
        "ln.bias": (n_embed,),
    }
    if use_output_bias:
        shapes["W_ue.bias"] = (vocab_size,)
    for index in range(n_layer):
        prefix = f"transformer_blocks.{index}"
        shapes.update(
            {
                f"{prefix}.ln.weight": (n_embed,),
                f"{prefix}.ln.bias": (n_embed,),
                f"{prefix}.attention.kqv_block.weight": (3 * n_embed, n_embed),
                f"{prefix}.attention.out_projection.weight": (n_embed, n_embed),
                f"{prefix}.mlp.Wh.weight": (n_hidden, n_embed),
                f"{prefix}.mlp.We.weight": (n_embed, n_hidden),
            }
        )
        if use_bias:
            shapes.update(
                {
                    f"{prefix}.attention.kqv_block.bias": (3 * n_embed,),
                    f"{prefix}.attention.out_projection.bias": (n_embed,),
                    f"{prefix}.mlp.Wh.bias": (n_hidden,),
                    f"{prefix}.mlp.We.bias": (n_embed,),
                }
            )
    return shapes


def inspect_checkpoint(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a weights-only state_dict checkpoint")
    tensor_state = {key: value for key, value in state.items() if torch.is_tensor(value)}
    fixed = run["fixed_conditions"]
    expected_shapes = _expected_dense_tensor_shapes(fixed)
    shared_norm_keys = sorted(key for key in tensor_state if ".ln." in key)
    current_dual_norm_keys = sorted(
        key for key in tensor_state if ".ln1." in key or ".ln2." in key
    )
    missing_keys = sorted(set(expected_shapes) - set(tensor_state))
    unexpected_keys = sorted(set(state) - set(expected_shapes))
    shape_mismatches = {
        key: {
            "actual": list(tensor_state[key].shape),
            "expected": list(expected_shape),
        }
        for key, expected_shape in expected_shapes.items()
        if key in tensor_state and tuple(tensor_state[key].shape) != expected_shape
    }
    integrity_problems = []
    if missing_keys:
        integrity_problems.append(f"missing {len(missing_keys)} expected tensor keys")
    if unexpected_keys:
        integrity_problems.append(f"found {len(unexpected_keys)} unexpected state keys")
    if shape_mismatches:
        integrity_problems.append(
            f"found {len(shape_mismatches)} tensors with unexpected shapes"
        )
    return {
        "run_id": run["run_id"],
        "checkpoint_run_dir": run["checkpoint_run_dir"],
        "checkpoint_file": "model.pt",
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "payload_keys": sorted(payload) if isinstance(payload, dict) else [],
        "tensor_count": len(tensor_state),
        "state_dict_numel_including_tied_aliases": sum(
            tensor.numel() for tensor in tensor_state.values()
        ),
        "architecture_integrity": {
            "status": "complete_legacy_schema_match" if not integrity_problems else "mismatch",
            "problems": integrity_problems,
            "expected_tensor_count": len(expected_shapes),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "shape_mismatches": shape_mismatches,
        },
        "current_runtime_compatibility": {
            "status": "incompatible",
            "historical_shared_layer_norm_keys": len(shared_norm_keys),
            "current_dual_layer_norm_keys": len(current_dual_norm_keys),
            "reason": (
                "The 2025 checkpoint has one shared LayerNorm per transformer block "
                "(.ln); the current model has distinct attention and MLP LayerNorms "
                "(.ln1 and .ln2). Loading it into the current runtime would require "
                "a scientifically consequential migration rule."
            ),
        },
        "evaluation": {
            "status": "not_run",
            "reason": (
                "The exact historical validation artifact and dataset revision are "
                "not preserved, the seed is missing, and the current runtime is not "
                "architecture-compatible. A new evaluation would not be comparable "
                "enough to strengthen the recovered claim."
            ),
        },
    }


def build_audit(sweep: dict[str, Any], outputs_root: Path) -> dict[str, Any]:
    fixed_conditions = dict(sweep["fixed_conditions"])
    runs = {run["run_id"]: run for run in sweep["runs"]}
    audits = []
    for run_id in AUDIT_RUN_IDS:
        run = dict(runs[run_id])
        run["fixed_conditions"] = fixed_conditions
        path = outputs_root / run["checkpoint_run_dir"] / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {path}")
        audits.append(inspect_checkpoint(path, run))
    all_match = all(
        checkpoint["architecture_integrity"]["status"]
        == "complete_legacy_schema_match"
        for checkpoint in audits
    )
    conclusion = (
        "All three weights-only payloads match the complete expected legacy dense "
        "tensor key/shape schema, but none is directly compatible with the current "
        "dual-LayerNorm runtime. Retrospective validation was not run."
        if all_match
        else "One or more selected weights-only payloads do not match the complete "
        "expected legacy dense tensor key/shape schema. Retrospective validation "
        "was not run."
    )
    return {
        "schema_version": 2,
        "audit_kind": "legacy_checkpoint_compatibility",
        "selected_run_ids": list(AUDIT_RUN_IDS),
        "evaluation_count": 0,
        "conclusion": conclusion,
        "checkpoints": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-data",
        type=Path,
        default=Path("research/data/tiny_stories_aux_lr_sweep.json"),
    )
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/data/checkpoint_compatibility.json"),
    )
    args = parser.parse_args()
    sweep = json.loads(args.sweep_data.read_text(encoding="utf-8"))
    audit = build_audit(sweep, args.outputs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Audited {len(audit['checkpoints'])} checkpoints into {args.output}")


if __name__ == "__main__":
    main()
