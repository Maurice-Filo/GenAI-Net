#!/usr/bin/env python3
"""Plot matched best-so-far trajectories for every primary paper task."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "comparisons/rpa_search/data/raw"
TASKS = (
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
)
LABELS = {
    "rpa": "Robust perfect adaptation",
    "logic": "Logic circuit",
    "classifier": "Autonomous classifier",
    "dose_hill": "Hill response",
    "dose_ultrasensitive": "Ultrasensitive response",
    "dose_biphasic": "Biphasic response",
    "oscillator_mean": "Oscillator mean",
    "oscillator_frequency": "Oscillator frequency",
}
COLORS = {"rl": "#0072B2", "hybrid": "#009E73"}
METHOD_LABELS = {"rl": "RL only", "hybrid": "Full duplex"}
HYBRID = {
    "rpa": (
        "genai_net_llm_flash_rpa_context_free100",
        "rpa_full307200_seed{seed}_cvode_flash_rpa_context_free100",
    ),
    "logic": (
        "genai_net_llm_flash_logic_initial_context_free100",
        "logic_full102400_seed{seed}_cvode_llm_flash_logic_initial_context_free100",
    ),
}
BREADTH_METHOD = "genai_net_llm_flash_breadth_initial_context_free20"
BREADTH_SUFFIX = "cvode_llm_flash_breadth_initial_context_free20"


def progress_path(task: str, seed: int, method: str) -> Path:
    if method == "hybrid":
        if task in HYBRID:
            directory, pattern = HYBRID[task]
            return RAW / directory / pattern.format(seed=seed) / "progress.csv"
        return (
            RAW
            / BREADTH_METHOD
            / f"{task}_full102400_seed{seed}_{BREADTH_SUFFIX}"
            / "progress.csv"
        )
    if task in HYBRID:
        return RAW / "rl4crn" / f"{task}_full102400_seed{seed}_cvode" / "progress.csv"
    return (
        RAW
        / "rl4crn_breadth"
        / f"{task}_full102400_seed{seed}_cvode_rl_only_breadth"
        / "progress.csv"
    )


def read_progress(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    evaluations = np.asarray([float(row["candidate_evaluations"]) for row in rows])
    losses = np.asarray([float(row["best_so_far_loss"]) for row in rows])
    if not len(losses) or np.any(~np.isfinite(losses)) or np.any(losses <= 0):
        raise ValueError(f"Expected finite positive losses in {path}")
    order = np.argsort(evaluations, kind="stable")
    return evaluations[order], losses[order]


def sample_previous(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(x, grid, side="right") - 1
    indices = np.clip(indices, 0, len(y) - 1)
    return y[indices]


def collect(n_seeds: int, grid: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    curves: dict[tuple[str, str], np.ndarray] = {}
    for task in TASKS:
        for method in ("rl", "hybrid"):
            sampled = []
            for seed in range(n_seeds):
                x, y = read_progress(progress_path(task, seed, method))
                sampled.append(sample_previous(x, y, grid))
            curves[(task, method)] = np.asarray(sampled)
    return curves


def write_csv(
    path: Path, grid: np.ndarray, curves: dict[tuple[str, str], np.ndarray]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task", "method", "candidate_evaluations", "median", "q25", "q75"]
        )
        for task in TASKS:
            for method in ("rl", "hybrid"):
                values = curves[(task, method)]
                q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
                for evaluation, lower, center, upper in zip(grid, q25, median, q75):
                    writer.writerow([task, method, evaluation, center, lower, upper])


def plot(
    output: Path, grid: np.ndarray, curves: dict[tuple[str, str], np.ndarray], n_seeds: int
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.4), sharex=True, constrained_layout=True)
    for panel, (ax, task) in enumerate(zip(axes.flat, TASKS)):
        for method in ("rl", "hybrid"):
            values = curves[(task, method)]
            q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
            color = COLORS[method]
            ax.fill_between(grid / 1000, q25, q75, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                grid / 1000,
                median,
                color=color,
                linewidth=1.7,
                drawstyle="steps-post",
                label=METHOD_LABELS[method],
            )
        rl_final = np.median(curves[(task, "rl")][:, -1])
        hybrid_final = np.median(curves[(task, "hybrid")][:, -1])
        ax.text(
            0.98,
            0.95,
            f"final median\nRL {rl_final:.3g}  |  full {hybrid_final:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#333333",
        )
        ax.set_yscale("log")
        ax.set_xlim(grid[0] / 1000, grid[-1] / 1000)
        ax.set_title(LABELS[task], fontweight="semibold")
        ax.set_ylabel("Best loss")
        ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.12,
            1.03,
            chr(ord("A") + panel),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
        )
    for ax in axes[-1]:
        ax.set_xlabel("Evaluated candidates (thousands)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=400 if suffix == ".png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "figures/all_tasks_best_loss",
    )
    args = parser.parse_args()
    grid = np.linspace(1024, 102400, 101)
    curves = collect(args.n_seeds, grid)
    output = args.output.resolve()
    plot(output, grid, curves, args.n_seeds)
    write_csv(output.with_suffix(".csv"), grid, curves)
    print(f"Wrote {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
