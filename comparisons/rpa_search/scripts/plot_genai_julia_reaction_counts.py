#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from comparisons.rpa_search.src.common.plotting import METHOD_COLORS, METHOD_LABELS


RAW_ROOT = Path("comparisons/rpa_search/data/raw")
FIGURE_DIR = Path("comparisons/rpa_search/figures")
METHODS = ("rl4crn", "reaction_network_evolution_jl")
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}


def _run_id(task: str, method: str, seed: int) -> str:
    suffix = "_cvode" if method == "rl4crn" else ""
    return f"{task}_full102400_seed{seed}{suffix}"


def _fixed_template_count(task: str, method: str) -> int:
    if method != "rl4crn":
        return 0
    return 2 if task == "rpa" else 4


def _load_rows(n_seeds: int, min_sims: int) -> list[dict]:
    rows = []
    for task in TASKS:
        for method in METHODS:
            for seed in range(n_seeds):
                run_id = _run_id(task, method, seed)
                run_dir = RAW_ROOT / method / run_id
                progress_path = run_dir / "progress.csv"
                network_path = run_dir / "best_network.json"
                if not progress_path.exists() or not network_path.exists():
                    continue
                with progress_path.open("r", encoding="utf-8") as f:
                    progress = list(csv.DictReader(f))
                if not progress:
                    continue
                final = progress[-1]
                if float(final.get("ode_simulations", 0.0)) < float(min_sims):
                    continue
                with network_path.open("r", encoding="utf-8") as f:
                    network = json.load(f)
                total_reactions = len(network.get("reactions", []))
                added_reactions = total_reactions - _fixed_template_count(task, method)
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "run_id": run_id,
                        "best_loss": float(final["best_so_far_loss"]),
                        "total_reactions": total_reactions,
                        "added_reactions": added_reactions,
                    }
                )
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "method",
        "seed",
        "run_id",
        "best_loss",
        "total_reactions",
        "added_reactions",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], out_dir: Path, figure_name: str, paper: bool) -> None:
    if paper:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figsize = (3.35, 2.45)
    else:
        figsize = (6.8, 4.2)

    fig, ax = plt.subplots(figsize=figsize)
    positions = []
    data = []
    colors = []
    xticks = []
    xticklabels = []
    width = 0.34
    rng = np.random.default_rng(0)

    for task_i, task in enumerate(TASKS):
        center = task_i * 1.35 + 1.0
        xticks.append(center)
        xticklabels.append(TASK_LABELS[task])
        for method_i, method in enumerate(METHODS):
            pos = center + (-width / 1.8 if method_i == 0 else width / 1.8)
            vals = [
                int(row["added_reactions"])
                for row in rows
                if row["task"] == task and row["method"] == method
            ]
            positions.append(pos)
            data.append(vals)
            colors.append(METHOD_COLORS[method])
            jitter = rng.normal(0.0, 0.025, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=10 if paper else 18,
                color=METHOD_COLORS[method],
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
        boxprops={"linewidth": 0.8, "color": "#333333"},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=6, color=METHOD_COLORS[m])
        for m in METHODS
    ]
    labels = [METHOD_LABELS[m] for m in METHODS]
    ax.legend(handles, labels, frameon=False, loc="upper left")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Added reactions in best solution")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    stem = out_dir / Path(figure_name).stem
    for fmt in ("png", "pdf", "svg") if paper else ("png",):
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=400 if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--min-sims", type=int, default=102400)
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR))
    parser.add_argument("--figure-name", default="rpa_logic_genai_julia_20seed_added_reactions_boxplot.png")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.n_seeds, args.min_sims)
    if not rows:
        raise ValueError("No completed runs found for reaction-count plotting.")

    csv_path = out_dir / "rpa_logic_genai_julia_20seed_added_reactions.csv"
    _write_csv(rows, csv_path)
    _plot(rows, out_dir, args.figure_name, args.paper)
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / args.figure_name}")


if __name__ == "__main__":
    main()
