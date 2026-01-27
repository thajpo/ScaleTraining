from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from omegaconf import DictConfig
import hydra
from pathlib import Path
from scaletraining.config import load_project_config
from scaletraining.data_processing.dataset_utils import (
    dataset_safe_name,
    get_dataset_text_files,
)


def train_tokenizer_from_cfg(cfg: DictConfig) -> str:
    """Train a dataset-specific tokenizer in-process and return its save path.

    This function avoids spawning a subprocess and prevents Hydra re-initialization.
    """
    files = get_dataset_text_files(cfg)
    if not files:
        raise ValueError("No text files found or created for training")

    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=cfg.tokenizer.custom_tokenizer_vocab_size,
        show_progress=True,
        max_token_length=10,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )
    tokenizer.pre_tokenizer = Whitespace()

    print(f"Training tokenizer on {len(files)} file(s): {files}")
    tokenizer.train(files, trainer)

    # Generate dataset-based save path
    dataset_name = dataset_safe_name(
        names=cfg.tokenizer.dataset_names, configs=cfg.tokenizer.dataset_tag
    )
    # Save under project-local tokenizers/
    out_dir = Path.cwd() / "tokenizers"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = cfg.tokenizer.custom_tokenizer_vocab_size
    suffix = f"_v{vocab_size}"
    save_path = out_dir / f"tokenizer_{dataset_name}{suffix}.json"
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved to: {save_path}")
    return str(save_path)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra console script entrypoint for tokenization."""
    cfg = load_project_config(cfg)
    train_tokenizer_from_cfg(cfg)


if __name__ == "__main__":
    main()
