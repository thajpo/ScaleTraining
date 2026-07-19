#!/usr/bin/env python3
"""Render dependency-free SVG plots from committed legacy experiment data."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from typing import Any


COLORS = ("#d1495b", "#edae49", "#00798c", "#30638e", "#6a4c93", "#2a9d8f")
BACKGROUND = "#fbfaf7"
INK = "#20242b"
GRID = "#d8d6d0"


def _moving_average(points: list[dict[str, Any]], window: int = 5) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        start = max(0, index - window + 1)
        sample = points[start : index + 1]
        loss = sum(float(item["train_loss_per_token"]) for item in sample) / len(sample)
        values.append((float(point["tokens"]), loss))
    return values


def _svg_text(x: float, y: float, value: str, **attrs: Any) -> str:
    attributes = " ".join(
        f'{("class" if key == "class_" else key.replace("_", "-"))}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{escape(value)}</text>'


def _write_svg(path: Path, body: list[str], width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#20242b}.axis{font-size:13px}.label{font-size:14px;font-weight:600}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#565d68}.legend{font-size:13px}</style>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(document) + "\n", encoding="utf-8")
    return path


def render_loss_curves(payload: dict[str, Any], output: Path) -> Path:
    width, height = 960, 560
    left, right, top, bottom = 78, 240, 92, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_max, y_min, y_max = 40_000_000.0, 2.5, 12.0

    def sx(value: float) -> float:
        return left + value / x_max * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    body = [f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>']
    body.append(_svg_text(left, 38, "Hybrid Muon training loss", class_="title"))
    body.append(
        _svg_text(
            left,
            64,
            "Only the AdamW-managed auxiliary parameter LR varies; 5-point trailing mean",
            class_="subtitle",
        )
    )
    for token_millions in range(0, 41, 5):
        x = sx(token_millions * 1_000_000)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="{GRID}" stroke-width="1"/>')
        body.append(_svg_text(x, top + plot_height + 24, str(token_millions), class_="axis", text_anchor="middle"))
    for loss in (3, 4, 5, 6, 8, 10, 12):
        y = sy(loss)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(_svg_text(left - 12, y + 4, str(loss), class_="axis", text_anchor="end"))
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{INK}" stroke-width="1.5"/>')

    runs = sorted(payload["runs"], key=lambda run: run["config"]["auxiliary_adamw_lr"], reverse=True)
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        points = [
            (tokens, loss)
            for tokens, loss in _moving_average(run["history"])
            if y_min <= loss <= y_max
        ]
        coordinates = " ".join(f"{sx(tokens):.1f},{sy(loss):.1f}" for tokens, loss in points)
        body.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        terminal = run["terminal"]
        terminal_loss = min(y_max, max(y_min, float(terminal["train_loss_per_token"])))
        body.append(f'<circle cx="{sx(float(terminal["tokens"])):.1f}" cy="{sy(terminal_loss):.1f}" r="4" fill="{color}"/>')
        legend_y = top + 22 + index * 45
        body.append(f'<line x1="{left + plot_width + 25}" y1="{legend_y - 5}" x2="{left + plot_width + 57}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>')
        lr = run["config"]["auxiliary_adamw_lr"]
        body.append(_svg_text(left + plot_width + 67, legend_y, f"AdamW LR {lr:g}", class_="label"))
        body.append(_svg_text(left + plot_width + 67, legend_y + 18, f"{terminal['train_loss_per_token']:.3f} at {terminal['tokens'] / 1_000_000:.1f}M", class_="legend"))

    body.append(_svg_text(left + plot_width / 2, height - 19, "Training tokens (millions)", class_="label", text_anchor="middle"))
    body.append(f'<text x="20" y="{top + plot_height / 2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2:.1f})">Train loss / token</text>')
    body.append(_svg_text(left + plot_width + 25, top + 315, "Muon matrix LR: 0.02", class_="subtitle"))
    body.append(_svg_text(left + plot_width + 25, top + 337, "Single run per setting", class_="subtitle"))
    return _write_svg(output, body, width, height)


def render_terminal_outcomes(payload: dict[str, Any], output: Path) -> Path:
    width, height = 860, 500
    left, right, top, bottom = 82, 55, 92, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = math.log10(0.005), math.log10(0.1)
    y_min, y_max = 2.5, 7.5

    def sx(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    body = [f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>']
    body.append(_svg_text(left, 38, "Terminal outcomes across auxiliary LR", class_="title"))
    body.append(_svg_text(left, 64, "Raw final train loss; labels show observed tokens before termination", class_="subtitle"))
    for lr in (0.005, 0.01, 0.025, 0.03, 0.07, 0.1):
        x = sx(lr)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="{GRID}" stroke-width="1"/>')
        body.append(_svg_text(x, top + plot_height + 24, f"{lr:g}", class_="axis", text_anchor="middle"))
    for loss in (3, 4, 5, 6, 7):
        y = sy(loss)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(_svg_text(left - 12, y + 4, str(loss), class_="axis", text_anchor="end"))
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{INK}" stroke-width="1.5"/>')

    runs = sorted(payload["runs"], key=lambda run: run["config"]["auxiliary_adamw_lr"])
    coordinates = " ".join(
        f"{sx(float(run['config']['auxiliary_adamw_lr'])):.1f},{sy(float(run['terminal']['train_loss_per_token'])):.1f}"
        for run in runs
    )
    body.append(f'<polyline points="{coordinates}" fill="none" stroke="#707782" stroke-width="2" stroke-dasharray="5 5"/>')
    for index, run in enumerate(runs):
        lr = float(run["config"]["auxiliary_adamw_lr"])
        terminal = run["terminal"]
        loss = float(terminal["train_loss_per_token"])
        x, y = sx(lr), sy(loss)
        color = COLORS[len(runs) - index - 1]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
        anchor = "end" if lr == 0.025 or lr >= 0.07 else "start"
        offset = -11 if anchor == "end" else 11
        body.append(_svg_text(x + offset, y - 12, f"{loss:.3f}", class_="label", text_anchor=anchor))
        body.append(_svg_text(x + offset, y + 8, f"{terminal['tokens'] / 1_000_000:.1f}M tokens", class_="legend", text_anchor=anchor))

    body.append(_svg_text(left + plot_width / 2, height - 24, "AdamW-managed auxiliary parameter LR (log scale)", class_="label", text_anchor="middle"))
    body.append(f'<text x="22" y="{top + plot_height / 2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">Terminal train loss / token</text>')
    return _write_svg(output, body, width, height)


def _nearest_history_point(run: dict[str, Any], target_tokens: int) -> dict[str, Any]:
    return min(
        run["history"],
        key=lambda point: abs(int(point["tokens"]) - target_tokens),
    )


def render_common_horizon(payload: dict[str, Any], output: Path) -> Path:
    width, height = 860, 500
    left, right, top, bottom = 82, 55, 92, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    target_tokens = int(
        payload.get("comparison_horizon_tokens")
        or min(run["terminal"]["tokens"] for run in payload["runs"])
    )
    runs = sorted(
        payload["runs"],
        key=lambda run: run["config"]["auxiliary_adamw_lr"],
    )
    observations = [
        (run, _nearest_history_point(run, target_tokens)) for run in runs
    ]
    losses = [float(point["train_loss_per_token"]) for _, point in observations]
    x_min, x_max = math.log10(0.005), math.log10(0.1)
    y_min = math.floor(min(losses) * 2) / 2 - 0.25
    y_max = math.ceil(max(losses) * 2) / 2 + 0.25

    def sx(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    body = [f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>']
    body.append(
        _svg_text(
            left,
            38,
            f"Comparable loss near {target_tokens / 1_000_000:g}M tokens",
            class_="title",
        )
    )
    body.append(
        _svg_text(
            left,
            64,
            "Nearest recorded raw train loss for every run at one shared horizon",
            class_="subtitle",
        )
    )
    for lr in (0.005, 0.01, 0.025, 0.03, 0.07, 0.1):
        x = sx(lr)
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="{GRID}" stroke-width="1"/>')
        body.append(
            _svg_text(
                x,
                top + plot_height + 24,
                f"{lr:g}",
                class_="axis",
                text_anchor="middle",
            )
        )
    first_tick = math.ceil(y_min)
    last_tick = math.floor(y_max)
    for loss in range(first_tick, last_tick + 1):
        y = sy(loss)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(
            _svg_text(
                left - 12,
                y + 4,
                str(loss),
                class_="axis",
                text_anchor="end",
            )
        )
    body.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    coordinates = " ".join(
        f"{sx(float(run['config']['auxiliary_adamw_lr'])):.1f},{sy(float(point['train_loss_per_token'])):.1f}"
        for run, point in observations
    )
    body.append(f'<polyline points="{coordinates}" fill="none" stroke="#707782" stroke-width="2" stroke-dasharray="5 5"/>')
    for index, (run, point) in enumerate(observations):
        lr = float(run["config"]["auxiliary_adamw_lr"])
        loss = float(point["train_loss_per_token"])
        x, y = sx(lr), sy(loss)
        color = COLORS[len(runs) - index - 1]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
        anchor = "end" if lr == 0.025 or lr >= 0.07 else "start"
        offset = -11 if anchor == "end" else 11
        body.append(
            _svg_text(
                x + offset,
                y - 12,
                f"{loss:.3f}",
                class_="label",
                text_anchor=anchor,
            )
        )
        body.append(
            _svg_text(
                x + offset,
                y + 8,
                f"{int(point['tokens']) / 1_000_000:.2f}M tokens",
                class_="legend",
                text_anchor=anchor,
            )
        )
    body.append(
        _svg_text(
            left + plot_width / 2,
            height - 24,
            "AdamW-managed auxiliary parameter LR (log scale)",
            class_="label",
            text_anchor="middle",
        )
    )
    body.append(f'<text x="22" y="{top + plot_height / 2:.1f}" class="label" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})">Train loss / token</text>')
    return _write_svg(output, body, width, height)


def render_plots(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = render_loss_curves(payload, output_dir / "auxiliary-lr-loss-curves.svg")
    common = render_common_horizon(
        payload, output_dir / "auxiliary-lr-common-horizon.svg"
    )
    outcomes = render_terminal_outcomes(payload, output_dir / "auxiliary-lr-terminal-outcomes.svg")
    return curves, common, outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/data/tiny_stories_aux_lr_sweep.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/figures"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    paths = render_plots(payload, args.output_dir)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
