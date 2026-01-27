import os
from typing import Any

from datasets import load_from_disk
from torch.utils.data import DataLoader
from omegaconf import open_dict

from scaletraining.data_processing.tokenizer import TextTokenizer
from scaletraining.util.artifacts import read_metadata
from scaletraining.util.path_utils import (
    get_packed_directory,
    get_tokenized_directory,
    get_cfg_subset,
)


def check_tokenizer_metadata_match(cfg, dataset_root, tok_dir, pk_dir):
    meta = (
        read_metadata(dataset_root) or read_metadata(tok_dir) or read_metadata(pk_dir)
    )
    if meta:
        if cfg.tokenizer.strict_dataset_compat:
            current = get_cfg_subset(cfg)
            saved = meta.get("config", {})
            if any(saved.get(k) != current.get(k) for k in current.keys()):
                raise RuntimeError(
                    f"Dataset/tokenizer mismatch. Saved={saved} vs Current={current}"
                )
        saved_vocab = meta.get("tokenizer_vocab_size")
        if saved_vocab is not None:
            try:
                with open_dict(cfg.model):
                    cfg.model.vocab_size = int(saved_vocab)
            except Exception:
                pass
        saved_tok = meta.get("tokenizer_name")
        if saved_tok:
            try:
                with open_dict(cfg.tokenizer):
                    cfg.tokenizer.tokenizer_name = saved_tok
            except Exception:
                pass


def path_exists(path):
    return os.path.isdir(path)


def get_loader_kwargs(training_cfg):
    num_workers = int(getattr(training_cfg, "loader_num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(training_cfg, "loader_pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            getattr(training_cfg, "loader_persistent_workers", False)
        )
        prefetch = getattr(training_cfg, "loader_prefetch_factor", None)
        if prefetch:
            loader_kwargs["prefetch_factor"] = int(prefetch)
    return loader_kwargs


def build_loaders(cfg, for_training: bool = True):
    """Build PyTorch DataLoaders from dataset artifacts.

    When `for_training` is True (default) we operate on packed, fixed-length
    shards and create shuffled loaders suitable for training. When False we
    work directly from the tokenized split directories so evaluation code can
    reuse variable-length text without repacking.
    """
    tokenizer = TextTokenizer(cfg)
    tok_dir = tokenizer.tokenized_directory(
        cfg, for_training=for_training
    )  # Uses fingerprint
    tokenized_train_dir = os.path.join(tok_dir, "train")
    if not path_exists(tokenized_train_dir):
        raise RuntimeError(
            "Tokenized dataset not found. Run `scaletraining-prepare-data` or"
            " `python -m scaletraining.entrypoints.prepare_data` before training/evals."
        )

    pk_dir = get_packed_directory(
        cfg
    )  # expected packed dataset location for this config
    if for_training:
        dataset_root = pk_dir
        packed_data_dir = os.path.join(pk_dir, "train")
        if not path_exists(packed_data_dir):
            raise RuntimeError(
                "Packed dataset not found. Run `scaletraining-prepare-data` or"
                " `python -m scaletraining.entrypoints.prepare_data` before training."
            )
    else:
        dataset_root = tok_dir
    # Compatibility/metadata sync with persisted artifacts.
    check_tokenizer_metadata_match(cfg, dataset_root, tok_dir, pk_dir)

    train = load_from_disk(f"{dataset_root}/train").with_format(
        "torch", columns=["input_ids"]
    )
    loader_kwargs = get_loader_kwargs(cfg.training)

    eval_bsz = getattr(cfg.training, "eval_batch_size", cfg.training.batch_size)
    bsz = int(cfg.training.batch_size if for_training else eval_bsz)

    train_loader = DataLoader(
        train,
        batch_size=bsz,
        shuffle=bool(for_training),
        drop_last=bool(for_training),
        **loader_kwargs,
    )

    val_loader = None
    try:
        val = load_from_disk(f"{dataset_root}/val").with_format(
            "torch", columns=["input_ids"]
        )
        val_loader = DataLoader(
            val,
            batch_size=bsz,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    except Exception:
        pass
    return train_loader, val_loader
