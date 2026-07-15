from typing import Dict, Any, List

import hydra
from omegaconf import DictConfig

from scaletraining.data_processing.dataset_utils import load_hf_dataset
from scaletraining.data_processing.tokenizer import TextTokenizer
from scaletraining.util.artifacts import write_metadata
from scaletraining.util.path_utils import get_cfg_subset, get_tokenized_directory


def tokenize_dataset(cfg, tok: TextTokenizer) -> None:
    """Tokenize text -> input_ids and save to disk.

    Appends a single EOS to each sequence to enable concatenation+packing cleanly.

    Args:
        cfg: Hydra DictConfig with structured sections (tokenizer, model, paths).
    """
    save_path = get_tokenized_directory(cfg, for_training=True)
    max_len = int(cfg.model.max_seq_len)
    if tok.eos_id is None:
        raise ValueError("Tokenizer must define an EOS token id before tokenization.")

    try:
        ds = load_hf_dataset(
            cfg.tokenizer.dataset_names,
            getattr(cfg.tokenizer, "hf_dataset_config_name", None),
        )
    except Exception as e:
        raise RuntimeError(f"Could not load dataset: {e}")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        out = tok(
            examples["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 1,
            padding=False,
        )
        input_ids = out["input_ids"]
        input_ids = [ids + [tok.eos_id] for ids in input_ids]
        return {"input_ids": input_ids}

    train_split = "train" if "train" in ds else list(ds.keys())[0]
    val_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)

    tokenized_train = ds[train_split].map(
        tokenize_function,
        remove_columns=ds[train_split].column_names,
        batched=True,
        num_proc=cfg.tokenizer.num_proc,
        load_from_cache_file=True,
        desc="Tokenizing train",
    )
    tokenized_train.save_to_disk(f"{save_path}/train")

    if val_split:
        tokenized_val = ds[val_split].map(
            tokenize_function,
            remove_columns=ds[val_split].column_names,
            batched=True,
        num_proc=cfg.tokenizer.num_proc,
            load_from_cache_file=True,
            desc="Tokenizing val",
        )
        tokenized_val.save_to_disk(f"{save_path}/val")

    write_metadata(save_path, {
        "config": get_cfg_subset(cfg),
        "tokenizer_name": tok.tok_name,
        "eos_token_id": tok.eos_id,
        "tokenizer_vocab_size": tok.vocab_size,
    })


@hydra.main(version_base=None, config_path='../../../conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for tokenization."""
    tok = TextTokenizer(cfg)
    tokenize_dataset(cfg, tok)
