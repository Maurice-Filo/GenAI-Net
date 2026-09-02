#!/usr/bin/env python3
"""Compare Flash-assisted and legacy vanilla GenAI-Net at a matched budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
BUDGET = 102400
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
METHODS = {
    "vanilla": {
        "label": "GenAI-Net (vanilla)",
        "color": "#0072B2",
        "directory": "rl4crn",
        "run": "{task}_full102400_seed{seed}_cvode",
    },
    "llm": {
        "label": "GenAI-Net-LLM (Flash)",
        "color": "#009E73",
        "directory": "genai_net_llm_flash_long300",
        "run": "{task}_full307200_seed{seed}_cvode_llm_flash_long300",
    },
}


def read_progress(task: str, method: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    config = METHODS[method]
    path = (
        RAW_ROOT
        / config["directory"]
        / config["run"].format(task=task, seed=seed)
        / "progress.csv"
    )
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row["candidate_evaluations"]) for row in rows])
    y = np.asarray([float(row["best_so_far_loss"]) for row in rows])
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def previous_value(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(x, grid, side="right") - 1
    indices = np.clip(indices, 0, len(y) - 1)
    return y[indices]


def collect(n_seeds: int) -> tuple[np.ndarray, dict, list[dict[str, object]]]:
    grid = np.linspace(1023, BUDGET, 240)
    curves = {}
    rows = []
    for task in TASKS:
        for method in METHODS:
            runs = []
            for seed in range(n_seeds):
                x, y = read_progress(task, method, seed)
                if x[-1] < BUDGET:
                    raise RuntimeError(f"{task} {method} seed {seed} ends at {x[-1]} < {BUDGET}")
                sampled = previous_value(x, y, grid)
                runs.append(sampled)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "method": method,
                        "matched_budget": BUDGET,
                        "best_loss": float(sampled[-1]),
                    }
                )
            curves[(task, method)] = np.asarray(runs)
    return grid, curves, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/flash_long300_vs_vanilla_matched_budget.pdf",
    )
    args = parser.parse_args()
    grid, curves, rows = collect(args.n_seeds)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
    for column, (ax, task) in enumerate(zip(axes, TASKS)):
        for method, config in METHODS.items():
            values = curves[(task, method)]
            median = np.median(values, axis=0)
            q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
            ax.fill_between(grid / 1000, q25, q75, color=config["color"], alpha=0.15)
            ax.plot(grid / 1000, median, color=config["color"], linewidth=1.8, label=config["label"])
            ax.scatter([BUDGET / 1000], [median[-1]], color=config["color"], s=22, zorder=3)
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        ax.set_xlabel("Candidate evaluations ($10^3$)")
        ax.set_ylabel("Best-so-far loss")
        ax.set_xlim(0, BUDGET / 1000)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(color="#DDDDDD", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.03, chr(ord("A") + column), transform=ax.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        f"Matched-budget comparison (n = {args.n_seeds}; median and IQR)",
        fontsize=9,
        fontweight="semibold",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(args.output.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
