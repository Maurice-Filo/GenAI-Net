#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.plotting import METHOD_COLORS, METHOD_LABELS


RAW_ROOT = Path("comparisons/rpa_search/data/raw")
FIGURE_DIR = Path("comparisons/rpa_search/figures")
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
METHODS = (
    "rl4crn",
    "reaction_network_evolution_jl",
    "reaction_network_evolution_jl_constrained_bounded",
)


def _run_id(task: str, method: str, seed: int) -> str:
    if method == "rl4crn":
        return f"{task}_full102400_seed{seed}_cvode"
    if method == "reaction_network_evolution_jl_constrained_bounded":
        return f"{task}_full102400_seed{seed}_constrained_bounded"
    return f"{task}_full102400_seed{seed}"


def _fixed_template_count(task: str, method: str) -> int:
    if method != "rl4crn":
        return 0
    return 2 if task == "rpa" else 4


def _final_ode_simulations(run_dir: Path) -> float:
    progress_path = run_dir / "progress.csv"
    if not progress_path.exists():
        return 0.0
    with progress_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    return float(rows[-1].get("ode_simulations", 0.0))


def _load_rows(n_seeds: int, min_sims: int) -> list[dict]:
    rows = []
    for task in TASKS:
        for method in METHODS:
            skip = _fixed_template_count(task, method)
            for seed in range(n_seeds):
                run_id = _run_id(task, method, seed)
                run_dir = RAW_ROOT / method / run_id
                network_path = run_dir / "best_network.json"
                if not network_path.exists() or _final_ode_simulations(run_dir) < min_sims:
                    continue
                with network_path.open("r", encoding="utf-8") as f:
                    network = json.load(f)
                rate_constants = [float(x) for x in network.get("rate_constants", [])]
                for param_index, value in enumerate(rate_constants[skip:]):
                    if value <= 0:
                        continue
                    rows.append(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "run_id": run_id,
                            "parameter_index": param_index,
                            "parameter_value": value,
                        }
                    )
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "method", "seed", "run_id", "parameter_index", "parameter_value"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], figure_dir: Path, figure_name: str, paper: bool, yscale: str) -> None:
    if paper:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 7.5,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.75,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figsize = (6.8, 2.55)
    else:
        figsize = (10.0, 4.0)

    fig, axes = plt.subplots(1, len(TASKS), figsize=figsize, sharey=True)
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(METHODS) + 1)
    short_labels = {
        "rl4crn": "GenAI-Net",
        "reaction_network_evolution_jl": "RNE",
        "reaction_network_evolution_jl_constrained_bounded": "RNE\nbounded",
    }

    for ax, task in zip(axes, TASKS):
        grouped = [
            [
                float(row["parameter_value"])
                for row in rows
                if row["task"] == task and row["method"] == method
            ]
            for method in METHODS
        ]
        box = ax.boxplot(
            grouped,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS.get(method, "#777777"))
            patch.set_alpha(0.42)

        for pos, method, values in zip(positions, METHODS, grouped):
            if not values:
                continue
            jitter = rng.normal(0.0, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=9,
                color=METHOD_COLORS.get(method, "#777777"),
                alpha=0.42,
                linewidths=0,
                zorder=3,
            )

        ax.set_title(TASK_LABELS[task], pad=5)
        ax.set_xticks(positions)
        ax.set_xticklabels([short_labels[m] for m in METHODS])
        ax.set_yscale(yscale)
        ax.grid(True, which="major", axis="y", alpha=0.2, linewidth=0.55)
        ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3, pad=2)

    axes[0].set_ylabel("Rate constant")
    fig.tight_layout(w_pad=1.4)

    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = (figure_dir / figure_name).with_suffix("")
    for fmt in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=600 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--min-sims", type=int, default=102400)
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument("--figure-name", default="rpa_logic_genai_julia_parameter_values_boxplot.png")
    parser.add_argument("--yscale", choices=("log", "linear"), default="log")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.n_seeds, args.min_sims)
    if not rows:
        raise ValueError("No parameter values found.")

    figure_dir = Path(args.figure_dir)
    csv_path = figure_dir / "rpa_logic_genai_julia_parameter_values.csv"
    _write_csv(rows, csv_path)
    _plot(rows, figure_dir, args.figure_name, args.paper, args.yscale)
    print(f"Wrote {csv_path}")
    print(f"Wrote {(figure_dir / args.figure_name).with_suffix('.png')}")


if __name__ == "__main__":
    main()
