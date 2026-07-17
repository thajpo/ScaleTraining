"""Hydra-powered training entrypoint with W&B and local run evidence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict

from scaletraining.config import load_project_config
from scaletraining.data_processing import build_loaders
from scaletraining.data_processing.tokenizer import TextTokenizer
from scaletraining.model import TransformerNetwork
from scaletraining.model.training_loop import training_run
from scaletraining.reporting import refresh_run_report
from scaletraining.util import (
    clear_cuda_cache,
    configure_rocm_and_sdp,
    create_run_dir,
    finish_wandb,
    init_wandb,
    log_model_metrics,
    resolve_device,
    save_model,
    save_run_manifest,
    update_run_manifest,
)
from scaletraining.util.model_stats import (
    count_parameters,
    humanize_bytes,
    humanize_params,
)
from scaletraining.util.training_utils import set_random_seed


LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _should_compile(cfg: DictConfig) -> bool:
    if not bool(getattr(cfg.training, "compile_model", True)):
        return False
    device = str(getattr(cfg, "device_resolved", None) or cfg.device.device)
    if device != "cuda":
        return False
    if torch.version.hip is None:
        return True
    major, minor = map(int, torch.__version__.split(".")[:2])
    return (major, minor) >= (2, 8)


def _record_failed_run(run_dir: Path, exc: BaseException) -> None:
    """Leave an honest partial report without masking the training exception."""

    try:
        update_run_manifest(
            run_dir,
            status="failed",
            finished_at=_utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        refresh_run_report(run_dir)
    except Exception as artifact_exc:  # pragma: no cover - original error wins
        LOGGER.warning("Could not finalize failed-run evidence: %s", artifact_exc)


def run_training(cfg: DictConfig) -> float:
    """Execute one configured run and return its sweep objective."""

    cfg = load_project_config(cfg)
    set_random_seed(int(cfg.training.seed))

    configure_rocm_and_sdp(cfg.device)
    resolve_device(cfg)
    clear_cuda_cache()

    tokenizer = TextTokenizer(cfg)
    with open_dict(cfg.model):
        cfg.model.vocab_size = tokenizer.vocab_size
    with open_dict(cfg.tokenizer):
        cfg.tokenizer.tokenizer_name = tokenizer.tok_name

    run_dir = create_run_dir(cfg, cfg.paths.output_dir)
    tracking = init_wandb(
        cfg,
        tok=tokenizer.tok,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )

    try:
        save_run_manifest(
            cfg,
            str(run_dir),
            extra={"status": "running", "tracking": tracking.to_dict()},
        )
        train_loader, val_loader = build_loaders(cfg, for_training=True)

        model = TransformerNetwork(cfg)
        total_params, trainable_params = count_parameters(model)
        readable_total = humanize_params(total_params)
        readable_trainable = humanize_params(trainable_params)
        bytes_fp32 = total_params * 4
        bytes_bf16 = total_params * 2

        size_msg = (
            f"Model parameters: {total_params:,} ({readable_total}); "
            f"Trainable: {trainable_params:,} ({readable_trainable}); "
            f"Approx size fp32: {humanize_bytes(bytes_fp32)}, "
            f"bf16/fp16: {humanize_bytes(bytes_bf16)}"
        )
        print(size_msg)
        LOGGER.info(size_msg)
        log_model_metrics(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/size_bytes_fp32": bytes_fp32,
                "model/size_bytes_bf16": bytes_bf16,
            }
        )

        if _should_compile(cfg):
            model = torch.compile(model, mode="max-autotune")
        else:
            LOGGER.info("Skipping torch.compile for this runtime/configuration")
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        assert model.token_embedding.num_embeddings == cfg.model.vocab_size, (
            f"Model vocab ({model.token_embedding.num_embeddings}) "
            f"!= cfg.model.vocab_size ({cfg.model.vocab_size})"
        )

        stats = training_run(
            cfg,
            model,
            train_loader,
            loss_fn=loss_fn,
            val_loader=val_loader,
        )

        run_dir = Path(save_model(model, cfg, run_dir=run_dir))
        print(f"Model saved locally to: {run_dir}")

        final_train_loss = (
            float(stats["train_loss"][-1]) if stats.get("train_loss") else None
        )
        job_result = {
            "final_train_loss": final_train_loss,
            "primary_optimizer": cfg.optimizer.primary_optimizer,
            "use_rope": bool(cfg.model.use_rope),
            "lr": float(cfg.optimizer.lr),
            "seed": int(cfg.training.seed),
            "batch_size": int(cfg.training.batch_size),
            "accum_steps": int(cfg.training.accum_steps),
            "max_train_tokens": int(cfg.training.max_train_tokens),
            "max_seq_len": int(cfg.model.max_seq_len),
            "n_layer": int(cfg.model.n_layer),
            "n_head": int(cfg.model.n_head),
            "n_embed": int(cfg.model.n_embed),
            "run_dir": str(run_dir),
            "model_path": str(run_dir / "model.pt"),
        }
        with (Path.cwd() / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(job_result, handle, indent=2, sort_keys=True)
        with (run_dir / "train_result.json").open("w", encoding="utf-8") as handle:
            json.dump(job_result, handle, indent=2, sort_keys=True)

        update_run_manifest(run_dir, status="completed", finished_at=_utc_now())
        json_report, markdown_report = refresh_run_report(run_dir)
        print(f"Run evidence written to: {json_report} and {markdown_report}")
        print("RESULT:", json.dumps(job_result))
        return final_train_loss if final_train_loss is not None else float("inf")
    except BaseException as exc:
        _record_failed_run(run_dir, exc)
        raise
    finally:
        finish_wandb()


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent.parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> float:
    return run_training(cfg)


if __name__ == "__main__":
    main()
