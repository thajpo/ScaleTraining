#!/usr/bin/env python3
"""Refresh the reviewer-facing evidence bundle for one ScaleTraining run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scaletraining.reporting import (
    build_report,
    refresh_run_report,
    render_markdown,
    write_reports,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_path, md_path = refresh_run_report(args.run_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
