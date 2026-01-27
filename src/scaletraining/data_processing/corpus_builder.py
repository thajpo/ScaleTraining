"""Utilities for assembling mixed pretraining corpora."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None  # type: ignore

from datasets import IterableDataset, load_dataset
from transformers import AutoTokenizer

from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.util import (
    get_cfg_subset,
    get_packed_directory,
    get_tokenized_directory,
    read_metadata,
    write_metadata,
)


TOKENS_PER_GB = 250_000_000
DEFAULT_SYNTH_DATASET = "PleIAs/SYNTH"
PRESET_TARGET_TOKENS = {"tiny": 50_000_000, "standard": 1_000_000_000}

FilterFn = Callable[[dict], bool]
CleanerFn = Callable[[str], str]


def _default_clean(value: str) -> str:
    return value.strip()


def _concat_fields(example: dict, fields: Iterable[str], separator: str) -> str:
    parts: list[str] = []
    for field in fields:
        value = example.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for key in ("title", "body", "text"):
                item = value.get(key)
                if isinstance(item, str):
                    parts.append(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for key in ("text", "body", "content", "answer"):
                        maybe = item.get(key)
                        if isinstance(maybe, str):
                            parts.append(maybe)
    return separator.join(part for part in parts if part)


@dataclass
class SourceSpec:
    """Configuration describing a single source dataset."""

    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    text_fields: tuple[str, ...] = ("text",)
    separator: str = "\n"
    target_gb: float = 1.0
    tokens_per_gb: int = TOKENS_PER_GB
    min_chars: int = 200
    shuffle_buffer: int = 10_000
    filter_fn: Optional[FilterFn] = None
    cleaner: CleanerFn = _default_clean
    description: str = ""
    keep_probability: float = 1.0
    use_tokenizer_for_count: bool = False
    tokens_per_char: float = 0.25

    def target_tokens(self) -> int:
        return int(self.target_gb * self.tokens_per_gb)

    def extract_text(self, example: dict) -> str:
        if self.text_fields == ("__concat__",):
            return _concat_fields(example, example.keys(), self.separator)
        return _concat_fields(example, self.text_fields, self.separator)


SOURCES: list[SourceSpec] = [
    SourceSpec(
        dataset="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        text_fields=("text",),
        target_gb=1.0,
        description="FineWeb EDU sample",
    ),
    SourceSpec(
        dataset="wikimedia/wikipedia",
        subset="20231101.en",
        text_fields=("text",),
        target_gb=1.0,
        description="English Wikipedia",
    ),
    SourceSpec(
        dataset="openai/gsm8k",
        subset="main",
        split="train",
        text_fields=("question", "answer"),
        separator="\n\n",
        target_gb=0.5,
        description="GSM8K math",
    ),
    SourceSpec(
        dataset="codeparrot/github-code",
        subset="python",
        split="train",
        text_fields=("code",),
        target_gb=1.0,
        description="GitHub code (python)",
        min_chars=40,
    ),
]

SYNTH_SOURCES: list[SourceSpec] = [
    SourceSpec(
        dataset=DEFAULT_SYNTH_DATASET,
        split="train",
        text_fields=("query", "synthetic_answer"),
        separator="\n\n",
        target_gb=1.0,
        description="PleIAs SYNTH",
        min_chars=1,
    )
]


def _scale_sources(sources: list[SourceSpec], total_tokens: int) -> list[SourceSpec]:
    total_weight = sum(max(spec.target_gb, 0.0) for spec in sources) or 1.0
    scaled: list[SourceSpec] = []
    for spec in sources:
        weight = max(spec.target_gb, 0.0) / total_weight
        target_tokens = int(total_tokens * weight)
        target_gb = target_tokens / float(spec.tokens_per_gb)
        scaled.append(replace(spec, target_gb=target_gb))
    return scaled


def build_sources(
    corpus: str, preset: str, include_reasoning: bool = False
) -> list[SourceSpec]:
    preset = preset.lower()
    if preset not in PRESET_TARGET_TOKENS:
        raise ValueError(
            f"Unsupported preset '{preset}'. Choose from {sorted(PRESET_TARGET_TOKENS)}"
        )
    total_tokens = PRESET_TARGET_TOKENS[preset]

    corpus = corpus.lower()
    if corpus == "synth":
        synth_fields = ("query", "synthetic_answer")
        if include_reasoning:
            synth_fields = ("query", "synthetic_reasoning", "synthetic_answer")
        base = [replace(SYNTH_SOURCES[0], text_fields=synth_fields)]
    elif corpus == "mix":
        base = list(SOURCES)
    else:
        raise ValueError("Unsupported corpus. Use 'synth' or 'mix'.")
    return _scale_sources(base, total_tokens)


class JsonlTokenWriter:
    """Helper for accumulating raw text snippets."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.examples = 0
        self.tokens = 0

    def write(self, text: str, token_count: int) -> None:
        self._handle.write(json.dumps({"text": text}) + "\n")
        self.examples += 1
        self.tokens += token_count

    def close(self) -> None:
        self._handle.close()


def stream_source(
    spec: SourceSpec,
    tokenizer,
    train_writer: JsonlTokenWriter,
    val_writer: JsonlTokenWriter,
    val_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    total_tokens = 0
    target_tokens = spec.target_tokens()

    dataset_kwargs = {"streaming": True, "split": spec.split}
    if spec.subset is not None:
        dataset_kwargs["name"] = spec.subset

    stream = load_dataset(spec.dataset, **dataset_kwargs)
    if isinstance(stream, IterableDataset) and spec.shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=spec.shuffle_buffer)

    train_before = train_writer.examples
    val_before = val_writer.examples

    iterator = stream
    progress_bar = None
    if tqdm is not None:
        desc = f"Streaming {spec.description or spec.dataset}"
        progress_bar = tqdm(iterator, desc=desc, unit="ex", dynamic_ncols=True)
        iterator = progress_bar

    for example in iterator:
        if spec.filter_fn and not spec.filter_fn(example):
            continue
        if rng.random() > spec.keep_probability:
            continue
        text = spec.cleaner(spec.extract_text(example))
        if not text or len(text) < spec.min_chars:
            continue
        if spec.use_tokenizer_for_count:
            encoded = tokenizer(text, add_special_tokens=False)
            ids = encoded.get("input_ids")
            if not ids:
                continue
            count = len(ids)
        else:
            count = max(1, int(len(text) * spec.tokens_per_char))
        total_tokens += count
        if rng.random() < val_ratio:
            val_writer.write(text, count)
        else:
            train_writer.write(text, count)
        if progress_bar is not None:
            progress_bar.set_postfix(tokens=f"{total_tokens / 1e6:.1f}M", refresh=False)
        if total_tokens >= target_tokens:
            break

    if progress_bar is not None:
        progress_bar.close()

    return {
        "dataset": spec.dataset,
        "subset": spec.subset,
        "target_tokens": target_tokens,
        "collected_tokens": total_tokens,
        "train_examples": train_writer.examples - train_before,
        "val_examples": val_writer.examples - val_before,
        "token_count_method": "tokenizer"
        if spec.use_tokenizer_for_count
        else "estimate",
    }


def _load_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": ""})
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, int(tokenizer.eos_token_id)


def tokenize_and_pack(
    raw_dir: Path, cfg, tokenizer_name: str, max_seq_len: int, num_proc: int
) -> tuple[str, str]:
    data_files = {"train": str(raw_dir / "train.jsonl")}
    val_path = raw_dir / "val.jsonl"
    if val_path.exists() and val_path.stat().st_size > 0:
        data_files["validation"] = str(val_path)

    dataset = load_dataset("json", data_files=data_files)
    tokenizer, eos_id = _load_tokenizer(tokenizer_name)

    tok_dir = Path(get_tokenized_directory(cfg, for_training=True))
    pk_dir = Path(get_packed_directory(cfg))
    tok_dir.mkdir(parents=True, exist_ok=True)
    pk_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_batch(batch: dict) -> dict:
        output = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len - 1,
            padding=False,
        )
        input_ids = [ids + [eos_id] for ids in output["input_ids"]]
        return {"input_ids": input_ids}

    dataset["train"].map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing train",
    ).save_to_disk(str(tok_dir / "train"))

    if "validation" in dataset:
        dataset["validation"].map(
            tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
            desc="Tokenizing val",
        ).save_to_disk(str(tok_dir / "val"))

    write_metadata(
        str(tok_dir), {"config": get_cfg_subset(cfg), "tokenizer_name": tokenizer_name}
    )

    pack_and_save(
        tokenized_path=str(tok_dir),
        packed_path=str(pk_dir),
        block_size=int(cfg.model.max_seq_len),
        num_proc=int(cfg.tokenizer.pack_num_proc),
        map_batch_size=int(cfg.tokenizer.pack_map_batch_size),
        writer_batch_size=int(cfg.tokenizer.pack_writer_batch_size),
        metadata={"config": get_cfg_subset(cfg), "tokenizer_name": tokenizer_name},
    )

    return str(tok_dir), str(pk_dir)


def build_mixed_corpus(
    cfg,
    dataset_id: str,
    tokenizer_name: str,
    max_seq_len: int,
    output_root: Path,
    val_ratio: float,
    num_proc: int,
    seed: int,
    reuse_raw: bool,
    sources: Optional[list[SourceSpec]] = None,
) -> tuple[str, str, list[dict]]:
    raw_dir = output_root / dataset_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    if not reuse_raw:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        tokenizer.model_max_length = 1_000_000_000
        try:
            tokenizer.init_kwargs["model_max_length"] = 1_000_000_000
        except Exception:
            pass

        train_writer = JsonlTokenWriter(raw_dir / "train.jsonl")
        val_writer = JsonlTokenWriter(raw_dir / "val.jsonl")

        sources = sources or SOURCES
        for spec in sources:
            print(
                f"Streaming {spec.description or spec.dataset} -> ~{spec.target_tokens():,} tokens"
            )
            summary = stream_source(
                spec,
                tokenizer,
                train_writer,
                val_writer,
                val_ratio=val_ratio,
                seed=seed,
            )
            summaries.append(summary)
            print(
                f"  collected {summary['collected_tokens']:,} tokens"
                f" across {summary['train_examples']} train / {summary['val_examples']} val examples"
            )

        train_writer.close()
        val_writer.close()
    else:
        print("Reusing existing raw jsonl files; skipping streaming.")

    tok_dir, pk_dir = tokenize_and_pack(
        raw_dir=raw_dir,
        cfg=cfg,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        num_proc=num_proc,
    )

    meta = read_metadata(pk_dir) or {}
    meta.update(
        {
            "config": get_cfg_subset(cfg),
            "sources": summaries,
            "tokenizer_name": tokenizer_name,
            "max_seq_len": max_seq_len,
        }
    )
    write_metadata(pk_dir, meta)

    return tok_dir, pk_dir, summaries


__all__ = [
    "SourceSpec",
    "TOKENS_PER_GB",
    "SOURCES",
    "SYNTH_SOURCES",
    "DEFAULT_SYNTH_DATASET",
    "PRESET_TARGET_TOKENS",
    "build_sources",
    "build_mixed_corpus",
]
