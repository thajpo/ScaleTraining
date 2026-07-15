"""Tokenizer wrapper that normalises pretrained/custom handling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from omegaconf import open_dict

from scaletraining.data_processing.dataset_utils import dataset_safe_name
from scaletraining.data_processing.train_tokenizer import train_tokenizer_from_cfg
from scaletraining.util.path_utils import get_tokenized_directory


class TextTokenizer:
    """Unifies access to a tokenizer regardless of how it is sourced."""

    def __init__(self, cfg):
        self._cfg = cfg
        self.dataset_names: list[str] | str = cfg.tokenizer.dataset_names
        self.dataset_configs: Any = getattr(cfg.tokenizer, "dataset_tag", None)
        self.custom_tokenizer_vocab_size: int = cfg.tokenizer.custom_tokenizer_vocab_size

        if cfg.tokenizer.is_pretrained:
            self.tok_name = cfg.tokenizer.pretrained_tokenizer_name
            self.tok, self.vocab_size, self.eos_id, self.pad_token_id = self._load_pretrained(
                self.tok_name
            )
        else:
            (
                self.tok,
                self.vocab_size,
                self.eos_id,
                self.pad_token_id,
                self.tok_name,
            ) = self._load_custom()

        # Expose commonly-used identifiers on the config for downstream helpers.
        try:
            with open_dict(self._cfg.tokenizer):
                self._cfg.tokenizer.tokenizer_name = self.tok_name
                self._cfg.tokenizer.tokenizer_type = "pretrained" if cfg.tokenizer.is_pretrained else "custom"
        except Exception:
            pass

    @staticmethod
    def _load_pretrained(tok_name: str):
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if tok.eos_token_id is None:
            print("Warning, eos token does not exist, using '' as eos token")
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        vocab_size = len(tok) if hasattr(tok, "__len__") else tok.vocab_size
        return tok, vocab_size, tok.eos_token_id, tok.pad_token_id

    @staticmethod
    def _build_tokenizer_path(
        dataset_specs: list[str] | str,
        vocab_size: int | None = None,
        dataset_configs: Any = None,
    ) -> str:
        specs = dataset_specs if isinstance(dataset_specs, list) else [dataset_specs]
        configs = dataset_configs if isinstance(dataset_configs, list) else [dataset_configs] if dataset_configs else [None] * len(specs)
        if len(configs) == 1 and len(specs) > 1:
            configs = configs * len(specs)

        safe_name = dataset_safe_name(specs, configs)
        base_dir = Path.cwd() / "tokenizers"
        base_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_v{int(vocab_size)}" if vocab_size is not None else ""
        return str(base_dir / f"tokenizer_{safe_name}{suffix}.json")

    def _load_custom(self):
        tokenizer_file_path = self._build_tokenizer_path(
            self.dataset_names,
            self.custom_tokenizer_vocab_size,
            self.dataset_configs,
        )

        tokenizer_path = Path(tokenizer_file_path)
        if not tokenizer_path.exists():
            tokenizer_file_path = train_tokenizer_from_cfg(self._cfg)
            tokenizer_path = Path(tokenizer_file_path)

        print(f"Loading local tokenizer file: {tokenizer_path}")

        raw_tokenizer = RawTokenizer.from_file(str(tokenizer_path))
        tok = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": "[EOS]"})
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        vocab_size = len(tok) if hasattr(tok, "__len__") else tok.vocab_size

        return tok, vocab_size, tok.eos_token_id, tok.pad_token_id, str(tokenizer_path)

    def __call__(self, *args, **kwargs):
        return self.tok(*args, **kwargs)

    def tokenized_directory(self, cfg, *, for_training: bool = True) -> str:
        return get_tokenized_directory(cfg, for_training=for_training)
