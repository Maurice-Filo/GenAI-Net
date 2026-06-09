#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from comparisons.rpa_search.src.common.plotting import (
    plot_side_by_side_seed_summary,
    write_seed_summary,
)


METHODS = [
    "rl4crn",
    "reaction_network_evolution_jl",
    "reaction_network_evolution_jl_constrained_bounded",
]
RAW_ROOT = Path("comparisons/rpa_search/data/raw")


def _method_run_ids(task: str, n_seeds: int) -> dict[str, list[str]]:
    return {
        "rl4crn": [f"{task}_full102400_seed{seed}_cvode" for seed in range(n_seeds)],
        "reaction_network_evolution_jl": [f"{task}_full102400_seed{seed}" for seed in range(n_seeds)],
        "reaction_network_evolution_jl_constrained_bounded": [
            f"{task}_full102400_seed{seed}_constrained_bounded" for seed in range(n_seeds)
        ],
    }


def _panel(task: str, title: str, n_seeds: int) -> dict:
    return {
        "raw_root": "comparisons/rpa_search/data/raw",
        "methods": METHODS,
        "benchmark_names": [f"{task}_102400_full"],
        "tasks": [task],
        "method_run_ids": _method_run_ids(task, n_seeds),
        "min_ode_simulations": 102400,
        "title": title,
        "show_panel_label": False,
    }


def _final_loss(task: str, method: str, seed: int, n_seeds: int) -> float:
    run_id = _method_run_ids(task, n_seeds)[method][seed]
    progress = RAW_ROOT / method / run_id / "progress.csv"
    with progress.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {progress}")
    return float(rows[-1]["best_so_far_loss"])


def _write_pairwise_stats(figure_dir: Path, n_seeds: int) -> Path:
    rng = np.random.default_rng(0)
    out = figure_dir / f"rpa_logic_genai_julia_constrained_bounded_{n_seeds}seed_pairwise_stats.csv"
    fields = [
        "task",
        "comparison",
        "n_runs",
        "first_method_mean",
        "second_method_mean",
        "first_method_median",
        "second_method_median",
        "mean_diff_second_minus_first",
        "median_diff_second_minus_first",
        "paired_win_fraction_first",
        "bootstrap_mean_diff_ci_low",
        "bootstrap_mean_diff_ci_high",
    ]
    comparisons = [
        ("rl4crn", "reaction_network_evolution_jl"),
        ("rl4crn", "reaction_network_evolution_jl_constrained_bounded"),
        ("reaction_network_evolution_jl_constrained_bounded", "reaction_network_evolution_jl"),
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task in ("rpa", "logic"):
            for first, second in comparisons:
                a = np.array([_final_loss(task, first, seed, n_seeds) for seed in range(n_seeds)], dtype=float)
                b = np.array([_final_loss(task, second, seed, n_seeds) for seed in range(n_seeds)], dtype=float)
                diff = b - a
                boot = []
                for _ in range(10000):
                    idx = rng.integers(0, n_seeds, size=n_seeds)
                    boot.append(float(np.mean(diff[idx])))
                ci_low, ci_high = np.quantile(np.asarray(boot), [0.025, 0.975])
                writer.writerow(
                    {
                        "task": task,
                        "comparison": f"{first}_vs_{second}",
                        "n_runs": n_seeds,
                        "first_method_mean": float(np.mean(a)),
                        "second_method_mean": float(np.mean(b)),
                        "first_method_median": float(np.median(a)),
                        "second_method_median": float(np.median(b)),
                        "mean_diff_second_minus_first": float(np.mean(diff)),
                        "median_diff_second_minus_first": float(np.median(diff)),
                        "paired_win_fraction_first": float(np.mean(a < b)),
                        "bootstrap_mean_diff_ci_low": float(ci_low),
                        "bootstrap_mean_diff_ci_high": float(ci_high),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--figure-dir", default="comparisons/rpa_search/figures")
    parser.add_argument(
        "--figure-name",
        default="rpa_logic_genai_julia_constrained_bounded_20seed_side_by_side.png",
    )
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out = plot_side_by_side_seed_summary(
        [
            _panel("rpa", "", args.n_seeds),
            _panel("logic", "", args.n_seeds),
        ],
        figure_dir,
        figure_name=args.figure_name,
        formats=("png", "pdf", "svg") if args.paper else ("png",),
        log_x=False,
        log_y=False,
        y_limit=(0.0, 0.15),
        ylabel="Best-so-far loss",
        paper=args.paper,
    )

    for task in ("rpa", "logic"):
        summary_path = figure_dir / f"{task}_genai_julia_constrained_bounded_{args.n_seeds}seed_summary.csv"
        write_seed_summary(
            "comparisons/rpa_search/data/raw",
            summary_path,
            methods=METHODS,
            benchmark_names=[f"{task}_102400_full"],
            tasks=[task],
            method_run_ids=_method_run_ids(task, args.n_seeds),
            min_ode_simulations=102400,
        )
        print(f"Wrote {summary_path}")

    stats_path = _write_pairwise_stats(figure_dir, args.n_seeds)
    print(f"Wrote {stats_path}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
