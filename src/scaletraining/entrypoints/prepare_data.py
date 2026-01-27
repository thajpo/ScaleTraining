"""
Hydra entrypoint to fully prepare data: tokenize and pack.

Supports a single dataset or a list of datasets in cfg.tokenizer.dataset_names.
Each dataset is processed into its own fingerprinted directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from scaletraining.data_processing.tokenization import tokenize_dataset
from scaletraining.data_processing.tokenizer import TextTokenizer
from scaletraining.data_processing.batch_packer import pack_and_save
from scaletraining.util import (
    get_packed_directory,
    get_tokenized_directory,
    write_metadata,
)
from scaletraining.util.path_utils import get_cfg_subset
from scaletraining.config import load_project_config


def _as_list(x: Any) -> list:
    from omegaconf import ListConfig

    if isinstance(x, (list, ListConfig)):
        return list(x)
    if x is None:
        return []
    return [x]


def _dataset_specs(cfg: DictConfig) -> list[Any]:
    """Return dataset specifications from the config (legacy or nested)."""

    return _as_list(cfg.tokenizer.dataset_names)


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    """Create a mutable clone of the provided config."""

    return OmegaConf.create(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent.parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """
    Prepare datasets by tokenizing and packing them.

    Behavior:
      - If cfg.tokenizer.dataset_names is a single spec, prepares that dataset.
      - If it's a list, prepares each dataset independently.
    """
    cfg = load_project_config(cfg)

    specs: Iterable[Any] = _dataset_specs(cfg)

    for spec in specs:
        sub = _clone_cfg(cfg)
        with open_dict(sub.tokenizer):
            sub.tokenizer.dataset_names = [spec] if not isinstance(spec, list) else spec

        # Tokenize
        tokenizer = TextTokenizer(sub)
        tokenize_dataset(sub, tokenizer)

        # Pack
        tok_dir = tokenizer.tokenized_directory(sub)
        pk_dir = get_packed_directory(sub)
        pack_and_save(
            tokenized_path=tok_dir,
            packed_path=pk_dir,
            block_size=int(sub.model.max_seq_len),
            num_proc=int(sub.tokenizer.pack_num_proc),
            map_batch_size=int(sub.tokenizer.pack_map_batch_size),
            writer_batch_size=int(sub.tokenizer.pack_writer_batch_size),
            metadata={"config": get_cfg_subset(sub)},
        )

        # Also write a top-level metadata file for quick inspection (redundant but handy)
        try:
            write_metadata(pk_dir, {"config": get_cfg_subset(sub)})
        except Exception:
            pass


if __name__ == "__main__":
    main()
