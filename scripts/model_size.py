#!/usr/bin/env python3
"""Report the model parameter count derived from the active Hydra config."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from scaletraining.config import load_project_config
from scaletraining.model import TransformerNetwork
from scaletraining.util.model_stats import (
    count_parameters,
    humanize_bytes,
    humanize_params,
)


LOGGER = logging.getLogger("scaletraining.scripts.model_size")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default="conf",
        type=Path,
        help="Directory that stores Hydra configuration files (default: ./conf).",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Base Hydra config name to compose (default: config).",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Hydra-style overrides, e.g. --override model.n_layer=12",
    )
    return parser.parse_args(argv)


def load_cfg(config_path: Path, config_name: str, overrides: Iterable[str]) -> object:
    """Compose the Hydra config for the provided path/name."""
    config_dir = config_path.expanduser().resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    overrides_list = list(overrides)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides_list)
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)

    cfg = load_cfg(args.config_path, args.config_name, args.override)
    cfg = load_project_config(cfg)
    if getattr(cfg.model, "vocab_size", None) is None:
        with open_dict(cfg.model):
            cfg.model.vocab_size = 49152
    model = TransformerNetwork(cfg)

    total_params, trainable_params = count_parameters(model)
    readable_total = humanize_params(total_params)
    readable_trainable = humanize_params(trainable_params)

    # Estimate memory footprint for common dtypes
    bytes_fp32 = total_params * 4
    bytes_bf16 = total_params * 2

    msg = (
        f"Model parameters: {total_params:,} ({readable_total}); "
        f"Trainable: {trainable_params:,} ({readable_trainable}); "
        f"Approx size fp32: {humanize_bytes(bytes_fp32)}, bf16/fp16: {humanize_bytes(bytes_bf16)}"
    )
    print(msg)
    LOGGER.info(msg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
