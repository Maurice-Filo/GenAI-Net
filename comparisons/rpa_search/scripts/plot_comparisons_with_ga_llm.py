#!/usr/bin/env python3
"""Rebuild comparisons_with_GA.pdf with the 20-seed GenAI-Net-LLM results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.plotting import METHOD_COLORS


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
METHODS = (
    "rl4crn",
    "genai_net_llm",
    "reaction_network_evolution_jl",
    "reaction_network_evolution_jl_constrained_bounded",
)
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
SHORT_LABELS = {
    "rl4crn": "GenAI-\nNet",
    "genai_net_llm": "GenAI-Net-\nLLM",
    "reaction_network_evolution_jl": "RNE",
    "reaction_network_evolution_jl_constrained_bounded": "RNE\n($\\leq$5 rxns)",
}
LEGEND_LABELS = {
    "rl4crn": "GenAI-Net",
    "genai_net_llm": "GenAI-Net-LLM",
    "reaction_network_evolution_jl": "RNE",
    "reaction_network_evolution_jl_constrained_bounded": "RNE ($\\leq$5 reactions)",
}


def run_id(task: str, method: str, seed: int) -> str:
    if method == "rl4crn":
        return f"{task}_full102400_seed{seed}_cvode"
    if method == "genai_net_llm":
        return f"{task}_full102400_seed{seed}_cvode_llm"
    if method == "reaction_network_evolution_jl_constrained_bounded":
        return f"{task}_full102400_seed{seed}_constrained_bounded"
    return f"{task}_full102400_seed{seed}"


def run_dir(task: str, method: str, seed: int) -> Path:
    return RAW_ROOT / method / run_id(task, method, seed)


def read_progress(task: str, method: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir(task, method, seed) / "progress.csv"
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row["candidate_evaluations"]) for row in rows], dtype=float)
    y = np.asarray([float(row["best_so_far_loss"]) for row in rows], dtype=float)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def read_network(task: str, method: str, seed: int) -> dict:
    with (run_dir(task, method, seed) / "best_network.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_complete(n_seeds: int) -> None:
    missing = []
    for task in TASKS:
        for method in METHODS:
            for seed in range(n_seeds):
                directory = run_dir(task, method, seed)
                if not (directory / "progress.csv").exists() or not (directory / "best_network.json").exists():
                    missing.append(f"{method}/{directory.name}")
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:12])
        raise RuntimeError(f"Cannot build the paper figure; {len(missing)} runs are incomplete:\n{preview}")


def step_values(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indexes = np.searchsorted(x, grid, side="right") - 1
    indexes = np.clip(indexes, 0, len(y) - 1)
    return y[indexes]


def fixed_template_count(task: str, method: str) -> int:
    return (2 if task == "rpa" else 4) if method in {"rl4crn", "genai_net_llm"} else 0


def style_boxplot(ax, grouped: list[list[float]], positions: np.ndarray) -> None:
    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.35},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.8},
        capprops={"color": "#4A4A4A", "linewidth": 0.8},
        boxprops={"edgecolor": "#4A4A4A", "linewidth": 0.8},
    )
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.42)


def style_axis(ax, *, y_grid_minor: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55)
    if y_grid_minor:
        ax.grid(axis="y", which="minor", color="#ECECEC", linewidth=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.tick_params(direction="out", length=3, width=0.7, color="#333333")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def write_pairwise_summary(n_seeds: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["task", "seed", "genai_net_loss", "genai_net_llm_loss", "llm_minus_vanilla"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            for seed in range(n_seeds):
                vanilla = read_progress(task, "rl4crn", seed)[1][-1]
                llm = read_progress(task, "genai_net_llm", seed)[1][-1]
                writer.writerow(
                    {
                        "task": task,
                        "seed": seed,
                        "genai_net_loss": vanilla,
                        "genai_net_llm_loss": llm,
                        "llm_minus_vanilla": llm - vanilla,
                    }
                )


def plot(n_seeds: int, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.85), constrained_layout=True)
    grid_spec = fig.add_gridspec(3, 2, height_ratios=(1.18, 0.86, 0.9), hspace=0.14, wspace=0.12)
    rng = np.random.default_rng(0)
    evaluation_grid = np.linspace(1023, 102400, 180)

    for column, task in enumerate(TASKS):
        ax = fig.add_subplot(grid_spec[0, column])
        for method in METHODS:
            curves = []
            for seed in range(n_seeds):
                x, y = read_progress(task, method, seed)
                curves.append(step_values(x, y, evaluation_grid))
            lower, median, upper = np.percentile(np.asarray(curves), [25, 50, 75], axis=0)
            ax.fill_between(
                evaluation_grid,
                lower,
                upper,
                step="post",
                color=METHOD_COLORS[method],
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                evaluation_grid,
                median,
                color=METHOD_COLORS[method],
                linewidth=1.65,
                drawstyle="steps-post",
            )
        ax.set_title(TASK_LABELS[task], fontweight="semibold", pad=5)
        ax.set_xlabel("Candidate evaluations")
        if column == 0:
            ax.set_ylabel("Best-so-far loss")
        ax.set_xlim(0, 102400)
        ax.set_ylim(0, 0.15)
        ax.xaxis.set_major_locator(MultipleLocator(25000))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k")
        )
        style_axis(ax)
        panel_label(ax, chr(ord("A") + column))

    for column, task in enumerate(TASKS):
        ax = fig.add_subplot(grid_spec[1, column])
        grouped = []
        for method in METHODS:
            grouped.append(
                [
                    len(read_network(task, method, seed).get("reactions", []))
                    - fixed_template_count(task, method)
                    for seed in range(n_seeds)
                ]
            )
        positions = np.arange(1, len(METHODS) + 1)
        style_boxplot(ax, grouped, positions)
        for position, method, values in zip(positions, METHODS, grouped):
            ax.scatter(
                np.full(len(values), position) + rng.normal(0, 0.045, len(values)),
                values,
                s=11,
                color=METHOD_COLORS[method],
                alpha=0.62,
                linewidths=0,
            )
        ax.set_xticks(positions, [SHORT_LABELS[method] for method in METHODS])
        if column == 0:
            ax.set_ylabel("Added reactions")
        ax.yaxis.set_major_locator(MultipleLocator(2))
        style_axis(ax)
        panel_label(ax, chr(ord("C") + column))

    for column, task in enumerate(TASKS):
        ax = fig.add_subplot(grid_spec[2, column])
        grouped = []
        for method in METHODS:
            values = []
            skip = fixed_template_count(task, method)
            for seed in range(n_seeds):
                rates = [float(value) for value in read_network(task, method, seed).get("rate_constants", [])]
                values.extend(value for value in rates[skip:] if value > 0)
            grouped.append(values)
        positions = np.arange(1, len(METHODS) + 1)
        style_boxplot(ax, grouped, positions)
        for position, method, values in zip(positions, METHODS, grouped):
            ax.scatter(
                np.full(len(values), position) + rng.normal(0, 0.05, len(values)),
                values,
                s=7,
                color=METHOD_COLORS[method],
                alpha=0.34,
                linewidths=0,
            )
        ax.set_xticks(positions, [SHORT_LABELS[method] for method in METHODS])
        ax.set_yscale("log")
        if column == 0:
            ax.set_ylabel("Added-reaction rate constant")
        style_axis(ax)
        panel_label(ax, chr(ord("E") + column))

    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2.0, label=LEGEND_LABELS[method])
        for method in METHODS
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.35,
        title=f"Median with interquartile range (n = {n_seeds} independent runs)",
        title_fontsize=6.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "GenAI-Net and GenAI-Net-LLM benchmark comparison"}
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "comparisons_with_GA.pdf")
    args = parser.parse_args()
    verify_complete(args.n_seeds)
    plot(args.n_seeds, args.output.resolve())
    summary = ROOT / "comparisons/rpa_search/figures/genai_net_llm_pairwise_20seed.csv"
    write_pairwise_summary(args.n_seeds, summary)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
