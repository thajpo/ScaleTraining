"""Entrypoint for evaluation runs (validation perplexity only)."""

from __future__ import annotations

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch.nn as nn

from scaletraining.config import load_project_config
from scaletraining.util import resolve_device
from scaletraining.util.eval_utils import (
    evaluate_perplexity_stats,
    write_eval_result,
)
from scaletraining.data_processing import build_loaders
from scaletraining.util.eval_utils import load_pretrained_model_and_tokenizer


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent.parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    cfg = load_project_config(cfg)
    resolve_device(cfg)

    model, _ = load_pretrained_model_and_tokenizer(cfg)
    _, val_loader = build_loaders(cfg, for_training=False)
    if val_loader is None:
        raise RuntimeError(
            "Validation split not found; ensure tokenized val data exists."
        )

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    metrics = evaluate_perplexity_stats(
        model,
        val_loader,
        cfg,
        loss_fn,
        max_batches=int(getattr(cfg.training, "eval_max_batches", 0)),
    )
    v_loss = float(metrics["loss"])
    v_ppl = float(metrics["perplexity"])
    print(f"Validation loss: {v_loss:.4f} | perplexity: {v_ppl:.3f}")
    if bool(getattr(cfg.eval, "write_results", True)):
        result_path = write_eval_result(cfg, metrics)
        print(f"Evaluation results written to: {result_path}")


if __name__ == "__main__":
    main()
