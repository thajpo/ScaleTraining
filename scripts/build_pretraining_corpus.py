"""Thin wrapper around `build_mixed_corpus` for ad-hoc runs from the repo root."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hydra import compose, initialize
from omegaconf import open_dict

from scaletraining.config import load_project_config
from scaletraining.data_processing.corpus_builder import (
    DEFAULT_SYNTH_DATASET,
    PRESET_TARGET_TOKENS,
    build_mixed_corpus,
    build_sources,
)


def _first_non_empty(values):
    for value in values:
        if value not in (None, "", "null"):
            return value
    return None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Assemble and cache a mixed pretraining corpus.")
    parser.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESET_TARGET_TOKENS.keys()),
        help="Corpus size preset (controls total token target).",
    )
    parser.add_argument(
        "--corpus",
        choices=("synth", "mix"),
        default="synth",
        help="Corpus type: single SYNTH dataset or curated mix (default: synth).",
    )
    parser.add_argument(
        "--dataset-id", help="Override dataset id (defaults based on corpus type)."
    )
    parser.add_argument(
        "--tokenizer", help="Tokenizer name/path (defaults to config tokenizer_name)."
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Packing sequence length (defaults to config max_seq_len).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/pretrain/mixed"))
    parser.add_argument("--dataset-tag", help="Optional dataset_tag override.")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--summaries-path",
        type=Path,
        help="Optional JSON file capturing per-source stats.",
    )
    parser.add_argument(
        "--hf-token", help="Hugging Face access token for gated models/datasets."
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Skip re-streaming if raw jsonl exists.",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include synthetic_reasoning in SYNTH text fields.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
    cfg = load_project_config(cfg)

    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    dataset_id = args.dataset_id
    cfg_dataset = list(cfg.tokenizer.dataset_names)
    if dataset_id is None:
        dataset_id = DEFAULT_SYNTH_DATASET if args.corpus == "synth" else "mixed"
    if not dataset_id:
        parser.error("Unable to infer dataset id; pass --dataset-id.")

    tokenizer_name = (
        args.tokenizer
        or cfg.tokenizer.tokenizer_name
        or cfg.tokenizer.pretrained_tokenizer_name
    )
    if not tokenizer_name:
        parser.error("Tokenizer not provided and config has no tokenizer_name.")

    max_seq_len = args.max_seq_len or cfg.model.max_seq_len
    if not max_seq_len:
        parser.error("Max seq len missing; supply --max-seq-len or set in config.")
    max_seq_len = int(max_seq_len)

    with open_dict(cfg.tokenizer):
        cfg.tokenizer.dataset_names = [dataset_id]
        tag_override = (
            args.dataset_tag
            or _first_non_empty(cfg.tokenizer.dataset_tag)
            or dataset_id
        )
        cfg.tokenizer.dataset_tag = [tag_override]
        cfg.tokenizer.tokenizer_name = tokenizer_name
    with open_dict(cfg.model):
        cfg.model.max_seq_len = max_seq_len

    sources = build_sources(
        args.corpus, args.preset, include_reasoning=args.include_reasoning
    )
    tok_dir, pk_dir, summaries = build_mixed_corpus(
        cfg=cfg,
        dataset_id=dataset_id,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        output_root=args.output_root,
        val_ratio=args.val_ratio,
        num_proc=args.num_proc,
        seed=args.seed,
        reuse_raw=args.reuse_raw,
        sources=sources,
    )

    if args.summaries_path:
        args.summaries_path.parent.mkdir(parents=True, exist_ok=True)
        args.summaries_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))

    print("\nDone!")
    print(f"Tokenized shards: {tok_dir}")
    print(f"Packed shards:    {pk_dir}")
    print(f"Use tokenizer.dataset_names: {dataset_id}")
    print(f"Preset: {args.preset} | Corpus: {args.corpus}")


if __name__ == "__main__":
    main()
