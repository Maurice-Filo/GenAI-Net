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


METHODS = ["rl4crn", "reaction_network_evolution_jl"]
RAW_ROOT = Path("comparisons/rpa_search/data/raw")


def _method_run_ids(n_seeds: int) -> dict[str, list[str]]:
    return {
        "rl4crn": [f"rpa_full102400_seed{seed}_cvode" for seed in range(n_seeds)],
        "reaction_network_evolution_jl": [f"rpa_full102400_seed{seed}" for seed in range(n_seeds)],
    }


def _final_loss(method: str, seed: int) -> float:
    run_id = _method_run_ids(seed + 1)[method][seed]
    progress = RAW_ROOT / method / run_id / "progress.csv"
    with progress.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {progress}")
    return float(rows[-1]["best_so_far_loss"])


def _write_pairwise_stats(figure_dir: Path, n_seeds: int) -> Path:
    rng = np.random.default_rng(0)
    out = figure_dir / f"rpa_genai_julia_{n_seeds}seed_pairwise_stats.csv"
    genai = np.array([_final_loss("rl4crn", seed) for seed in range(n_seeds)], dtype=float)
    julia = np.array([_final_loss("reaction_network_evolution_jl", seed) for seed in range(n_seeds)], dtype=float)
    diff = julia - genai
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot.append(float(np.mean(diff[idx])))
    ci_low, ci_high = np.quantile(np.asarray(boot), [0.025, 0.975])

    fields = [
        "task",
        "n_runs",
        "genai_mean",
        "julia_mean",
        "genai_median",
        "julia_median",
        "mean_diff_julia_minus_genai",
        "median_diff_julia_minus_genai",
        "paired_win_fraction_genai",
        "bootstrap_mean_diff_ci_low",
        "bootstrap_mean_diff_ci_high",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "task": "rpa",
                "n_runs": n_seeds,
                "genai_mean": float(np.mean(genai)),
                "julia_mean": float(np.mean(julia)),
                "genai_median": float(np.median(genai)),
                "julia_median": float(np.median(julia)),
                "mean_diff_julia_minus_genai": float(np.mean(diff)),
                "median_diff_julia_minus_genai": float(np.median(diff)),
                "paired_win_fraction_genai": float(np.mean(genai < julia)),
                "bootstrap_mean_diff_ci_low": float(ci_low),
                "bootstrap_mean_diff_ci_high": float(ci_high),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--figure-dir", default="comparisons/rpa_search/figures")
    parser.add_argument("--figure-name", default="rpa_genai_julia_20seed_summary.png")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    out = plot_side_by_side_seed_summary(
        [
            {
                "raw_root": str(RAW_ROOT),
                "methods": METHODS,
                "benchmark_names": ["rpa_102400_full"],
                "tasks": ["rpa"],
                "method_run_ids": _method_run_ids(args.n_seeds),
                "min_ode_simulations": 102400,
                "title": "RPA task",
            }
        ],
        figure_dir,
        figure_name=args.figure_name,
        formats=("png", "pdf", "svg") if args.paper else ("png",),
        log_x=False,
        log_y=True,
        y_limit=(None, 10.0),
        ylabel="Best-so-far RPA loss",
        paper=args.paper,
    )

    summary_path = figure_dir / f"rpa_genai_julia_{args.n_seeds}seed_summary.csv"
    write_seed_summary(
        RAW_ROOT,
        summary_path,
        methods=METHODS,
        benchmark_names=["rpa_102400_full"],
        tasks=["rpa"],
        method_run_ids=_method_run_ids(args.n_seeds),
        min_ode_simulations=102400,
    )
    stats_path = _write_pairwise_stats(figure_dir, args.n_seeds)
    print(f"Wrote {summary_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
