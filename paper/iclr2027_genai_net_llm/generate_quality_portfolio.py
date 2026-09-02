#!/usr/bin/env python3
"""Analyze quality-conditioned topology portfolios for the communication ablation."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.rpa_search.scripts.plot_communication_ablation_over_time import (
    _deduplicate,
    _latest,
    database_path,
    read_hof,
    read_llm_candidates,
)
from paper.iclr2027_genai_net_llm.audit_paper_experiments import (
    CAMPAIGN_BASE,
    control_path,
    hybrid_paths,
)


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
METHODS = ("full_duplex", "independent_pool")
METHOD_LABELS = {
    "full_duplex": "Full duplex",
    "independent_pool": "Independent pool",
}
COLORS = {"full_duplex": "#009E73", "independent_pool": "#4D4D4D"}
NO_COMM_ROOT = CAMPAIGN_BASE / "flash-no-communication-long300-20seed"
NO_COMM_SUFFIX = "cvode_llm_flash_no_communication"
EPOCH = 100
ELITE_SIZE = 30
THRESHOLD_RATIOS = np.geomspace(0.5, 2.0, 61)


def rl_endpoint(task: str, seed: int) -> float:
    with (control_path(task, seed) / "progress.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return float(list(csv.DictReader(handle))[-1]["best_so_far_loss"])


def elite_portfolios(task: str, seed: int) -> dict[str, list[dict]]:
    _, full_database = hybrid_paths(task, seed, CAMPAIGN_BASE)
    full_history = read_hof(full_database)
    full = sorted(
        _deduplicate(_latest(full_history, EPOCH)), key=lambda row: row["loss"]
    )[:ELITE_SIZE]

    isolated_database = database_path(
        NO_COMM_ROOT,
        task,
        seed,
        NO_COMM_SUFFIX,
        307200,
    )
    isolated_history = read_hof(isolated_database)
    if isolated_history:
        isolated_history.pop(max(isolated_history), None)
    available_llm = [
        candidate
        for completed_epoch, candidate in read_llm_candidates(isolated_database)
        if completed_epoch <= EPOCH
    ]
    independent = sorted(
        _deduplicate(_latest(isolated_history, EPOCH) + available_llm),
        key=lambda row: row["loss"],
    )[:ELITE_SIZE]
    if len(full) != ELITE_SIZE or len(independent) != ELITE_SIZE:
        raise RuntimeError(
            f"Expected {ELITE_SIZE} elite candidates for {task}/seed{seed}; "
            f"found {len(full)} and {len(independent)}."
        )
    return {"full_duplex": full, "independent_pool": independent}


def qualifying_metrics(candidates: list[dict], threshold: float) -> tuple[int, int]:
    qualifying = [row for row in candidates if float(row["loss"]) < threshold]
    return len(qualifying), len({str(row["topology"]) for row in qualifying})


def collect() -> tuple[list[dict], dict]:
    rows = []
    summary = {}
    for task in TASKS:
        controls = np.asarray([rl_endpoint(task, seed) for seed in range(20)])
        reference = float(np.median(controls))
        portfolios = [elite_portfolios(task, seed) for seed in range(20)]
        for seed, methods in enumerate(portfolios):
            for ratio in THRESHOLD_RATIOS:
                threshold = float(reference * ratio)
                for method in METHODS:
                    candidates, topologies = qualifying_metrics(
                        methods[method], threshold
                    )
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "method": method,
                            "threshold_ratio": float(ratio),
                            "loss_threshold": threshold,
                            "qualifying_candidates": candidates,
                            "qualifying_topologies": topologies,
                        }
                    )

        task_summary = {
            "rl_endpoint_median": reference,
            "threshold_sensitivity": {},
        }
        for label, threshold in zip(
            ("rl_q25", "rl_median", "rl_q75"),
            np.quantile(controls, (0.25, 0.5, 0.75)),
        ):
            topology_counts = {
                method: [
                    qualifying_metrics(portfolio[method], float(threshold))[1]
                    for portfolio in portfolios
                ]
                for method in METHODS
            }
            candidate_counts = {
                method: [
                    qualifying_metrics(portfolio[method], float(threshold))[0]
                    for portfolio in portfolios
                ]
                for method in METHODS
            }
            statistic = wilcoxon(
                topology_counts["full_duplex"], topology_counts["independent_pool"]
            )
            task_summary["threshold_sensitivity"][label] = {
                "loss_threshold": float(threshold),
                "full_duplex_median_topologies": float(
                    np.median(topology_counts["full_duplex"])
                ),
                "independent_pool_median_topologies": float(
                    np.median(topology_counts["independent_pool"])
                ),
                "full_duplex_median_candidates": float(
                    np.median(candidate_counts["full_duplex"])
                ),
                "independent_pool_median_candidates": float(
                    np.median(candidate_counts["independent_pool"])
                ),
                "full_duplex_wins": int(
                    sum(
                        left > right
                        for left, right in zip(
                            topology_counts["full_duplex"],
                            topology_counts["independent_pool"],
                        )
                    )
                ),
                "independent_pool_wins": int(
                    sum(
                        right > left
                        for left, right in zip(
                            topology_counts["full_duplex"],
                            topology_counts["independent_pool"],
                        )
                    )
                ),
                "ties": int(
                    sum(
                        left == right
                        for left, right in zip(
                            topology_counts["full_duplex"],
                            topology_counts["independent_pool"],
                        )
                    )
                ),
                "wilcoxon_pvalue_exploratory": float(statistic.pvalue),
                "full_duplex_counts": topology_counts["full_duplex"],
                "independent_pool_counts": topology_counts["independent_pool"],
            }
        summary[task] = task_summary
    return rows, summary


def write_outputs(rows: list[dict], summary: dict) -> None:
    figures = PAPER / "figures"
    generated = PAPER / "generated"
    figures.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    csv_path = figures / "quality_conditioned_portfolio.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (generated / "quality_conditioned_portfolio_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot(rows: list[dict], summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), sharey=True)
    for column, task in enumerate(TASKS):
        ax = axes[column]
        for method in METHODS:
            medians = []
            lower = []
            upper = []
            for ratio in THRESHOLD_RATIOS:
                values = [
                    row["qualifying_topologies"]
                    for row in rows
                    if row["task"] == task
                    and row["method"] == method
                    and np.isclose(row["threshold_ratio"], ratio)
                ]
                q25, median_value, q75 = np.quantile(values, (0.25, 0.5, 0.75))
                lower.append(q25)
                medians.append(median_value)
                upper.append(q75)
            ax.plot(
                THRESHOLD_RATIOS,
                medians,
                color=COLORS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
            ax.fill_between(
                THRESHOLD_RATIOS,
                lower,
                upper,
                color=COLORS[method],
                alpha=0.15,
                linewidth=0,
            )
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.5, 2.0)
        ax.set_ylim(-0.5, 30.8)
        ax.set_xticks((0.5, 1.0, 2.0), ("0.5", "1", "2"))
        ax.set_title(TASK_LABELS[task], weight="bold")
        ax.set_xlabel("Threshold / median RL-only endpoint")
        ax.set_ylabel("Distinct qualifying topologies")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        median_summary = summary[task]["threshold_sensitivity"]["rl_median"]
        ax.text(
            0.97,
            0.05,
            f"at 1x: pool {median_summary['independent_pool_median_topologies']:g}; "
            f"duplex {median_summary['full_duplex_median_topologies']:g}\n"
            f"full-duplex wins {median_summary['full_duplex_wins']}/20",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )

    axes[0].legend(frameon=False, loc="upper left", fontsize=7.3)
    fig.tight_layout(w_pad=1.3)
    for extension in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(
            PAPER / "figures" / f"quality_conditioned_portfolio.{extension}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    rows, summary = collect()
    write_outputs(rows, summary)
    plot(rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
