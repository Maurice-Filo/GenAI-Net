#!/usr/bin/env python3
"""Generate the frozen contract-v2 Cohort B analysis and diagnostic figures."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from scipy.stats import wilcoxon


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.run_mmc2_harness_smoke import BUILDERS, CONFIGS
from RL4CRN.utils.results_database import serialize_crn


GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
REGISTRY_PATH = GENERATED / "contract_v2_primary_registry.json"
PLAN_PATH = GENERATED / "analysis_plan_v2.json"
RAW = ROOT / "comparisons/rpa_search/data/raw"
CAMPAIGN_BASE = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
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
COMMUNICATION_TASKS = ("rpa", "logic", "classifier")
LABELS = {
    "rpa": "RPA",
    "logic": "Logic circuit",
    "classifier": "Classifier",
    "dose_hill": "Hill response",
    "dose_ultrasensitive": "Ultrasensitive",
    "dose_biphasic": "Biphasic",
    "oscillator_mean": "Oscillator mean",
    "oscillator_frequency": "Oscillator frequency",
}
COLORS = {"rl": "#2463A5", "full": "#16845B", "none": "#666666", "llm": "#D17B24"}
TRAJECTORY_Y_TICKS = {
    "rpa": (0.001, 0.01, 0.1),
    "logic": (0.03, 0.1, 0.3),
    "classifier": (0.001, 0.01, 0.1),
    "dose_hill": (0.06, 0.1, 0.2),
    "dose_ultrasensitive": (0.2, 0.3),
    "dose_biphasic": (0.2, 0.3, 0.4),
    "oscillator_mean": (0.01, 0.1),
    "oscillator_frequency": (0.01, 0.1, 1.0),
}
TIE_RTOL = 1e-9
TIE_ATOL = 1e-12


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def database_path(registry: dict, arm: str, task: str, seed: int) -> Path:
    campaigns = registry[f"{arm}_campaigns"]
    campaign_id = campaigns[0] if seed == 0 else campaigns[1]
    suffix = registry[f"{arm}_run_suffix"]
    run_id = f"{task}_full{registry['candidate_budget']}_seed{seed}_{suffix}"
    return (
        CAMPAIGN_BASE
        / campaign_id
        / "runs"
        / f"{campaign_id}-{task}-seed{seed}"
        / run_id
        / "results.sqlite"
    )


def completed_path(registry: dict, arm: str, task: str, seed: int) -> Path:
    method = registry[f"{arm}_method"]
    suffix = registry[f"{arm}_run_suffix"]
    run_id = f"{task}_full{registry['candidate_budget']}_seed{seed}_{suffix}"
    return RAW / method / run_id / "completed.json"


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def terminal_hof(path: Path) -> list[dict]:
    with connect(path) as connection:
        rows = connection.execute(
            """SELECT e.rank, e.topology_hash, e.loss, e.emitter,
                      e.provenance_class, c.structure_json
               FROM hof_snapshots h
               JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
               JOIN crns c ON c.topology_hash = e.topology_hash
               WHERE h.boundary = 'terminal_boundary'
               ORDER BY e.rank"""
        ).fetchall()
    if len(rows) != 30:
        raise RuntimeError(f"Expected a terminal top-30 HOF in {path}; found {len(rows)}.")
    return [dict(row) for row in rows]


def snapshot_trajectory(path: Path, *, threshold: float) -> list[dict]:
    with connect(path) as connection:
        rows = connection.execute(
            """SELECT h.epoch, h.boundary, h.cumulative_total_evaluations,
                      MIN(e.loss) AS best_loss,
                      COUNT(DISTINCT CASE WHEN e.loss < ? THEN e.topology_hash END)
                          AS qualified_topologies
               FROM hof_snapshots h
               JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
               GROUP BY h.snapshot_id
               ORDER BY h.cumulative_total_evaluations, h.epoch""",
            (float(threshold),),
        ).fetchall()
    return [dict(row) for row in rows]


def llm_accounting(path: Path, *, training_epoch: int = 100) -> dict:
    with connect(path) as connection:
        runs = [dict(row) for row in connection.execute("SELECT * FROM llm_runs")]
        failures = [dict(row) for row in connection.execute("SELECT * FROM llm_failures")]
    completed = len(runs) + len(failures)
    processes = [
        read_json(process_path)
        for process_path in path.parent.glob("harness-workspaces/*/calls/*/process.json")
    ]
    decider_seconds = sum(
        float(row.get("duration_seconds", 0.0))
        for row in processes
        if row.get("response_kind") == "text"
    )
    writer_seconds = sum(
        float(row.get("duration_seconds", 0.0))
        for row in processes
        if row.get("response_kind") == "json"
    )
    return {
        "scheduled": 5,
        "served": completed,
        "successful": len(runs),
        "failed": len(failures),
        "accepted": sum(int(row["accepted"] or 0) for row in runs),
        "rejected": sum(int(row["rejected"] or 0) for row in runs),
        "provider_calls": sum(int(row["provider_call_count"] or 0) for row in runs),
        "recorded_call_processes": len(processes),
        "provider_queue_seconds": sum(
            float(row.get("queue_seconds", 0.0)) for row in processes
        ),
        "decider_service_seconds": decider_seconds,
        "writer_service_seconds": writer_seconds,
        "before_training_boundary": sum(
            int(row["completed_epoch"]) <= training_epoch for row in runs
        ),
        "terminal_only": sum(int(row["completed_epoch"]) > training_epoch for row in runs),
        "median_latency_seconds": (
            float(np.median([float(row["elapsed_seconds"]) for row in runs]))
            if runs
            else None
        ),
    }


def control_loss(plan: dict, task: str, seed: int) -> float:
    records = plan["control_audit"][task]["controls"]
    record = next(row for row in records if int(row["seed"]) == seed)
    # The approved plan freezes this value together with hashes of the audited
    # control artifacts. Some legacy control directories predate completed.json.
    return float(record["endpoint_loss"])


def control_trajectory(plan: dict, task: str, seed: int) -> list[dict]:
    records = plan["control_audit"][task]["controls"]
    record = next(row for row in records if int(row["seed"]) == seed)
    path = ROOT / record["artifact_path"] / "progress.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "candidate_evaluations": int(float(row["candidate_evaluations"])),
            "best_loss": float(row["best_so_far_loss"]),
        }
        for row in rows
    ]


def structural_tokens(structure_json: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for record in json.loads(structure_json)
        )
    )


@lru_cache(maxsize=None)
def fixed_tokens(task: str) -> tuple[str, ...]:
    config = read_json(CONFIGS[task])
    crn, _library, _task, _cfg = BUILDERS[task](config)
    return structural_tokens(serialize_crn(crn)["structure_json"])


def added_structure(task: str, structure_json: str) -> frozenset[str]:
    remaining = Counter(structural_tokens(structure_json))
    for token in fixed_tokens(task):
        if remaining[token] <= 0:
            raise RuntimeError(f"A {task} HOF topology omits a fixed structural record.")
        remaining[token] -= 1
    return frozenset(token for token, count in remaining.items() if count > 0)


def mean_jaccard(task: str, rows: list[dict]) -> float | None:
    structures = [added_structure(task, row["structure_json"]) for row in rows]
    if len(structures) < 2:
        return None
    values = []
    for left, right in combinations(structures, 2):
        union = left | right
        values.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return float(np.mean(values))


def paired_relation(left: float, right: float) -> str:
    if np.isclose(left, right, rtol=TIE_RTOL, atol=TIE_ATOL):
        return "tie"
    return "win" if left < right else "loss"


def wilcoxon_p(left: list[float], right: list[float]) -> float:
    differences = np.asarray(left) - np.asarray(right)
    if np.allclose(differences, 0, rtol=TIE_RTOL, atol=TIE_ATOL):
        return 1.0
    return float(wilcoxon(left, right, zero_method="pratt", alternative="two-sided").pvalue)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty analysis table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(
    registry: dict, plan: dict
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    endpoints = []
    portfolios = []
    insertions = []
    trajectories = []
    thresholds = plan["quality_thresholds"]
    for task in TASKS:
        threshold = float(thresholds[task])
        for seed in range(20):
            full_completed = read_json(completed_path(registry, "full_duplex", task, seed))
            full_hof = terminal_hof(database_path(registry, "full_duplex", task, seed))
            rl = control_loss(plan, task, seed)
            full = float(full_completed["best_loss"])
            endpoints.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "full_duplex",
                    "loss": full,
                    "rl_only_loss": rl,
                    "rl_over_full": rl / full,
                    "paired_result": paired_relation(full, rl),
                }
            )
            qualified = [row for row in full_hof if float(row["loss"]) < threshold]
            portfolios.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "full_duplex",
                    "threshold": threshold,
                    "qualified_candidates": len(qualified),
                    "qualified_topologies": len({row["topology_hash"] for row in qualified}),
                    "qualified_mean_jaccard": mean_jaccard(task, qualified),
                    "rank_one_loss": full,
                    "direct_llm_hof_members": sum(
                        row["provenance_class"] == "direct_llm" for row in full_hof
                    ),
                    "rl_llm_topology_refinements": sum(
                        row["provenance_class"]
                        == "rl_parameter_refinement_of_llm_topology"
                        for row in full_hof
                    ),
                }
            )
            insertions.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "full_duplex",
                    **llm_accounting(database_path(registry, "full_duplex", task, seed)),
                }
            )
            for row in snapshot_trajectory(
                database_path(registry, "full_duplex", task, seed),
                threshold=threshold,
            ):
                trajectories.append(
                    {
                        "task": task,
                        "seed": seed,
                        "arm": "full_duplex",
                        "candidate_evaluations": int(row["cumulative_total_evaluations"]),
                        "best_loss": float(row["best_loss"]),
                        "qualified_topologies": int(row["qualified_topologies"]),
                        "boundary": row["boundary"],
                    }
                )
            for row in control_trajectory(plan, task, seed):
                trajectories.append(
                    {
                        "task": task,
                        "seed": seed,
                        "arm": "rl_only",
                        "candidate_evaluations": row["candidate_evaluations"],
                        "best_loss": row["best_loss"],
                        "qualified_topologies": None,
                        "boundary": "epoch",
                    }
                )
            if task not in COMMUNICATION_TASKS:
                continue
            none_completed = read_json(completed_path(registry, "no_communication", task, seed))
            none_hof = terminal_hof(database_path(registry, "no_communication", task, seed))
            none = float(none_completed["best_loss"])
            endpoints.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "no_communication",
                    "loss": none,
                    "rl_only_loss": rl,
                    "rl_over_full": None,
                    "paired_result": paired_relation(none, rl),
                }
            )
            qualified_none = [row for row in none_hof if float(row["loss"]) < threshold]
            portfolios.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "no_communication",
                    "threshold": threshold,
                    "qualified_candidates": len(qualified_none),
                    "qualified_topologies": len(
                        {row["topology_hash"] for row in qualified_none}
                    ),
                    "qualified_mean_jaccard": mean_jaccard(task, qualified_none),
                    "rank_one_loss": none,
                    "direct_llm_hof_members": sum(
                        row["provenance_class"] == "direct_llm" for row in none_hof
                    ),
                    "rl_llm_topology_refinements": 0,
                }
            )
            insertions.append(
                {
                    "task": task,
                    "seed": seed,
                    "arm": "no_communication",
                    **llm_accounting(database_path(registry, "no_communication", task, seed)),
                }
            )
    return endpoints, portfolios, insertions, trajectories


def summarize(endpoints: list[dict], portfolios: list[dict]) -> dict:
    endpoint_summary = []
    for task in TASKS:
        rows = [row for row in endpoints if row["task"] == task and row["arm"] == "full_duplex"]
        full = [float(row["loss"]) for row in rows]
        rl = [float(row["rl_only_loss"]) for row in rows]
        relations = Counter(row["paired_result"] for row in rows)
        endpoint_summary.append(
            {
                "task": task,
                "n": len(rows),
                "full_median": float(np.median(full)),
                "full_q25": float(np.quantile(full, 0.25)),
                "full_q75": float(np.quantile(full, 0.75)),
                "rl_median": float(np.median(rl)),
                "median_paired_rl_over_full": float(np.median(np.asarray(rl) / full)),
                "wins": relations["win"],
                "ties": relations["tie"],
                "losses": relations["loss"],
                "wilcoxon_p": wilcoxon_p(full, rl),
            }
        )
    adjusted = holm([row["wilcoxon_p"] for row in endpoint_summary])
    for row, value in zip(endpoint_summary, adjusted):
        row["holm_p"] = value

    communication_summary = []
    for task in COMMUNICATION_TASKS:
        full_rows = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "full_duplex"),
            key=lambda row: row["seed"],
        )
        none_rows = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "no_communication"),
            key=lambda row: row["seed"],
        )
        full = [int(row["qualified_topologies"]) for row in full_rows]
        none = [int(row["qualified_topologies"]) for row in none_rows]
        relation = Counter(
            "win" if left > right else "loss" if left < right else "tie"
            for left, right in zip(full, none)
        )
        communication_summary.append(
            {
                "task": task,
                "n": len(full),
                "full_median_qualified_topologies": float(np.median(full)),
                "none_median_qualified_topologies": float(np.median(none)),
                "wins": relation["win"],
                "ties": relation["tie"],
                "losses": relation["loss"],
                "wilcoxon_p": wilcoxon_p([-value for value in full], [-value for value in none]),
            }
        )
    adjusted = holm([row["wilcoxon_p"] for row in communication_summary])
    for row, value in zip(communication_summary, adjusted):
        row["holm_p"] = value
    return {"endpoint": endpoint_summary, "communication_portfolio": communication_summary}


def plot_endpoint(endpoints: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.45))
    rng = np.random.default_rng(27)
    for index, task in enumerate(TASKS):
        rows = [row for row in endpoints if row["task"] == task and row["arm"] == "full_duplex"]
        ratios = np.asarray([float(row["rl_over_full"]) for row in rows])
        jitter = rng.uniform(-0.12, 0.12, len(ratios))
        ax.scatter(ratios, index + jitter, s=22, color=COLORS["full"], alpha=0.72)
        q25, median, q75 = np.quantile(ratios, (0.25, 0.5, 0.75))
        ax.plot((q25, q75), (index, index), color="#222222", linewidth=2.2)
        ax.scatter(median, index, marker="D", s=34, color="#222222", zorder=4)
    ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(TASKS)), [LABELS[task] for task in TASKS])
    ax.set_ylim(len(TASKS) - 0.5, -0.5)
    ax.set_xlabel("Paired endpoint loss ratio, RL-only / full duplex (log scale)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "contract_v2_endpoint_ratios.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "contract_v2_endpoint_ratios.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio(portfolios: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
    rng = np.random.default_rng(73)
    for ax, task in zip(axes, COMMUNICATION_TASKS):
        full = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "full_duplex"),
            key=lambda row: row["seed"],
        )
        none = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "no_communication"),
            key=lambda row: row["seed"],
        )
        for left, right in zip(full, none):
            y1, y2 = left["qualified_topologies"], right["qualified_topologies"]
            dx = rng.uniform(-0.03, 0.03)
            ax.plot((0 + dx, 1 + dx), (y1, y2), color="#BBBBBB", linewidth=0.7, alpha=0.65)
            ax.scatter(0 + dx, y1, color=COLORS["full"], s=20, zorder=3)
            ax.scatter(1 + dx, y2, color=COLORS["none"], s=20, zorder=3)
        ax.set_xticks((0, 1), ("Full duplex", "Independent"))
        ax.set_title(LABELS[task])
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Qualified distinct topologies in terminal top 30")
    fig.tight_layout()
    fig.savefig(FIGURES / "contract_v2_qualified_portfolio.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "contract_v2_qualified_portfolio.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def plot_performance_diversity(portfolios: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    for ax, task in zip(axes, COMMUNICATION_TASKS):
        for arm, color, label in (
            ("full_duplex", COLORS["full"], "Full duplex"),
            ("no_communication", COLORS["none"], "Independent union"),
        ):
            rows = [row for row in portfolios if row["task"] == task and row["arm"] == arm]
            ax.scatter(
                [row["qualified_topologies"] for row in rows],
                [row["rank_one_loss"] for row in rows],
                color=color,
                s=24,
                alpha=0.72,
                label=label,
            )
        ax.set_yscale("log")
        ax.set_title(LABELS[task])
        ax.set_xlabel("Qualified topologies")
        ax.grid(color="#E5E5E5", linewidth=0.6)
    axes[0].set_ylabel("Rank-one loss (log scale)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "contract_v2_performance_diversity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "contract_v2_performance_diversity.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def _interpolate(rows: list[dict], field: str, grid: np.ndarray) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: row["candidate_evaluations"])
    x = np.asarray([row["candidate_evaluations"] for row in ordered], dtype=float)
    y = np.asarray([row[field] for row in ordered], dtype=float)
    unique, indices = np.unique(x, return_index=True)
    values = y[indices]
    return np.interp(grid, unique, values, left=values[0], right=values[-1])


def plot_trajectories(trajectories: list[dict]) -> None:
    grid = np.linspace(1023, 102400, 120)
    grid_thousands = grid / 1000.0
    request_evaluations = np.asarray((1023, 21483, 41943, 62403, 82863)) / 1000.0
    fig, axes = plt.subplots(2, 4, figsize=(7.8, 3.3), sharex=True)
    for index, (ax, task) in enumerate(zip(axes.flat, TASKS)):
        for arm, color, label in (
            ("rl_only", COLORS["rl"], "RL-only"),
            ("full_duplex", COLORS["full"], "Full duplex"),
        ):
            curves = []
            for seed in range(20):
                rows = [
                    row
                    for row in trajectories
                    if row["task"] == task and row["arm"] == arm and row["seed"] == seed
                ]
                curves.append(_interpolate(rows, "best_loss", grid))
            values = np.asarray(curves)
            median = np.median(values, axis=0)
            q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
            ax.plot(grid_thousands, median, color=color, linewidth=1.45, label=label)
            ax.fill_between(
                grid_thousands, q25, q75, color=color, alpha=0.11, linewidth=0
            )
        for request_evaluation in request_evaluations:
            ax.axvline(
                request_evaluation,
                color="#D2D2D2",
                linewidth=0.45,
                linestyle=":",
                zorder=0,
            )
        rl_final = np.median(
            [
                _interpolate(
                    [
                        row
                        for row in trajectories
                        if row["task"] == task
                        and row["arm"] == "rl_only"
                        and row["seed"] == seed
                    ],
                    "best_loss",
                    grid,
                )[-1]
                for seed in range(20)
            ]
        )
        full_final = np.median(
            [
                _interpolate(
                    [
                        row
                        for row in trajectories
                        if row["task"] == task
                        and row["arm"] == "full_duplex"
                        and row["seed"] == seed
                    ],
                    "best_loss",
                    grid,
                )[-1]
                for seed in range(20)
            ]
        )
        ax.set_yscale("log")
        ticks = TRAJECTORY_Y_TICKS[task]
        ax.set_yticks(ticks, [f"{value:g}" for value in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(
            f"{chr(65 + index)}  {LABELS[task]}",
            fontsize=7.4,
            fontweight="semibold",
            loc="left",
            pad=3,
        )
        endpoint_ratio = rl_final / full_final
        if endpoint_ratio >= 100:
            ratio_label = f"{endpoint_ratio:.0f}x"
        elif endpoint_ratio >= 10:
            ratio_label = f"{endpoint_ratio:.1f}x"
        else:
            ratio_label = f"{endpoint_ratio:.2f}x"
        ax.text(
            0.97,
            0.08,
            f"RL/full {ratio_label}",
            transform=ax.transAxes,
            fontsize=5.7,
            color="#444444",
            ha="right",
            va="bottom",
        )
        ax.set_xlim(0, 103)
        ax.set_xticks((0, 50, 100))
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.2, pad=2)
        ax.tick_params(axis="y", which="major", labelsize=6.0, pad=2)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=6.8,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.supxlabel("Candidate evaluations (thousands)", fontsize=7.1, y=0.012)
    fig.supylabel("Median best-so-far canonical loss", fontsize=7.1, x=0.004)
    fig.tight_layout(rect=(0.018, 0.045, 1, 0.955), h_pad=0.72, w_pad=0.42)
    fig.savefig(FIGURES / "contract_v2_best_loss_trajectories.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "contract_v2_best_loss_trajectories.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.75), sharex=True)
    for ax, task in zip(axes.flat, TASKS):
        curves = []
        for seed in range(20):
            rows = [
                row
                for row in trajectories
                if row["task"] == task
                and row["arm"] == "full_duplex"
                and row["seed"] == seed
            ]
            curves.append(_interpolate(rows, "qualified_topologies", grid))
        values = np.asarray(curves)
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        ax.plot(grid, median, color=COLORS["full"], linewidth=1.7)
        ax.fill_between(grid, q25, q75, color=COLORS["full"], alpha=0.16, linewidth=0)
        ax.set_title(LABELS[task], fontsize=8.5)
        ax.grid(color="#E7E7E7", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Candidate evaluations")
    axes[0, 0].set_ylabel("Qualified topologies")
    axes[1, 0].set_ylabel("Qualified topologies")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "contract_v2_qualified_topology_trajectories.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "contract_v2_qualified_topology_trajectories.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_markdown(summary: dict, insertions: list[dict]) -> None:
    lines = [
        "# Contract-v2 Cohort B analysis",
        "",
        "This report is generated only from the audited contract-v2 registry. Historical and quarantined campaigns are not scanned.",
        "",
        "## Rank-one endpoint",
        "",
        "| Task | Full median | RL median | W/T/L | RL/full | Wilcoxon p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["endpoint"]:
        lines.append(
            f"| {LABELS[row['task']]} | {row['full_median']:.4g} | {row['rl_median']:.4g} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['median_paired_rl_over_full']:.3g} | {row['wilcoxon_p']:.3g} | {row['holm_p']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## Qualified portfolio",
            "",
            "| Task | Full median | No-com median | W/T/L | Wilcoxon p | Holm p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["communication_portfolio"]:
        lines.append(
            f"| {LABELS[row['task']]} | {row['full_median_qualified_topologies']:.3g} | "
            f"{row['none_median_qualified_topologies']:.3g} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['wilcoxon_p']:.3g} | {row['holm_p']:.3g} |"
        )
    served = sum(row["served"] for row in insertions)
    scheduled = sum(row["scheduled"] for row in insertions)
    lines.extend(
        [
            "",
            "## Request accounting",
            "",
            f"Across audited runs, {served}/{scheduled} scheduled LLM rounds reached a recorded terminal state. See `contract_v2_insertion_accounting.csv` for seed-level timing and insertion opportunity data.",
            "",
            "Diversity is reported only at the frozen quality threshold and alongside rank-one loss; it is not interpreted independently of performance.",
        ]
    )
    (PAPER / "CONTRACT_V2_PRIMARY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    registry = read_json(REGISTRY_PATH)
    plan = read_json(PLAN_PATH)
    if registry.get("status") != "audited-complete":
        raise SystemExit("The contract-v2 primary registry is not audited and complete.")
    if plan.get("status") != "frozen":
        raise SystemExit("The analysis plan is not frozen.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    endpoints, portfolios, insertions, trajectories = collect(registry, plan)
    summary = summarize(endpoints, portfolios)
    write_csv(GENERATED / "contract_v2_endpoints.csv", endpoints)
    write_csv(GENERATED / "contract_v2_quality_portfolios.csv", portfolios)
    write_csv(GENERATED / "contract_v2_insertion_accounting.csv", insertions)
    write_csv(GENERATED / "contract_v2_trajectories.csv", trajectories)
    (GENERATED / "contract_v2_primary_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_endpoint(endpoints)
    plot_portfolio(portfolios)
    plot_performance_diversity(portfolios)
    plot_trajectories(trajectories)
    write_markdown(summary, insertions)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
