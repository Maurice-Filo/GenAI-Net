#!/usr/bin/env python3
"""Compare full-duplex and isolated RL/LLM search through time and diversity."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
METHODS = ("communication", "communication_rl", "isolated_rl", "independent_pool")
LABELS = {
    "communication": "Legacy HOF-exposed communication",
    "communication_rl": "HOF-exposed RL-provenance archive",
    "isolated_rl": "No communication: RL stream",
    "independent_pool": "No communication: available RL + LLM pool",
}
COLORS = {
    "communication": "#009E73",
    "communication_rl": "#006B4F",
    "isolated_rl": "#D55E00",
    "independent_pool": "#0072B2",
}
LINESTYLES = {
    "communication": "-",
    "communication_rl": "--",
    "isolated_rl": "--",
    "independent_pool": "-",
}


def database_path(
    campaign_root: Path,
    task: str,
    seed: int,
    run_suffix: str,
    candidate_budget: int,
) -> Path:
    run_id = f"{task}_full{candidate_budget}_seed{seed}_{run_suffix}"
    matches = sorted((campaign_root / "runs").glob(f"*/{run_id}/results.sqlite"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one database for {run_id}, found {len(matches)}")
    return matches[0]


def completed_seeds(status_path: Path) -> dict[str, list[int]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result = {task: [] for task in TASKS}
    for row in status.get("completed", []):
        task = str(row.get("task"))
        if task in result:
            result[task].append(int(row["seed"]))
    return {task: sorted(set(seeds)) for task, seeds in result.items()}


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def _candidate(
    topology_hash: object,
    parameters_json: object,
    loss: object,
    reaction_ids_json: object,
) -> dict:
    return {
        "identifier": (str(topology_hash), str(parameters_json)),
        "topology": str(topology_hash),
        "loss": float(loss),
        "reaction_ids": frozenset(int(value) for value in json.loads(str(reaction_ids_json))),
    }


def read_hof(path: Path, *, exclude_terminal_epoch: bool = False) -> dict[int, list[dict]]:
    with _connect(path) as connection:
        rows = connection.execute(
            """SELECT h.epoch, e.topology_hash, e.parameters_json, e.loss,
                      c.reaction_ids_json
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                 JOIN crns c ON c.topology_hash = e.topology_hash
                WHERE e.loss IS NOT NULL
                ORDER BY h.epoch, e.rank"""
        ).fetchall()
    result: dict[int, list[dict]] = {}
    for epoch, topology_hash, parameters_json, loss, reaction_ids_json in rows:
        result.setdefault(int(epoch), []).append(
            _candidate(topology_hash, parameters_json, loss, reaction_ids_json)
        )
    if exclude_terminal_epoch and result:
        result.pop(max(result), None)
    return result


def read_llm_candidates(path: Path) -> list[tuple[int, dict]]:
    with _connect(path) as connection:
        rows = connection.execute(
            """SELECT r.completed_epoch, c.topology_hash, e.parameters_json,
                      c.loss, n.reaction_ids_json
                 FROM llm_candidates c
                 JOIN llm_runs r ON r.llm_run_id = c.llm_run_id
                 JOIN evaluations e
                   ON e.source = 'llm'
                  AND e.topology_hash = c.topology_hash
                  AND json_extract(e.metadata_json, '$.llm_run_id') = c.llm_run_id
                  AND json_extract(e.metadata_json, '$.candidate_index') = c.candidate_index
                 JOIN crns n ON n.topology_hash = c.topology_hash
                WHERE c.valid = 1 AND c.loss IS NOT NULL
                ORDER BY r.completed_epoch, c.candidate_index"""
        ).fetchall()
    return [
        (int(epoch), _candidate(topology_hash, parameters_json, loss, reaction_ids_json))
        for epoch, topology_hash, parameters_json, loss, reaction_ids_json in rows
    ]


def _deduplicate(candidates: list[dict]) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}
    for candidate in candidates:
        identifier = candidate["identifier"]
        if identifier not in best or candidate["loss"] < best[identifier]["loss"]:
            best[identifier] = candidate
    return list(best.values())


def elite_metrics(candidates: list[dict], *, elite_size: int = 30) -> dict[str, float]:
    elite = sorted(_deduplicate(candidates), key=lambda row: row["loss"])[:elite_size]
    if not elite:
        return {
            "best_loss": float("nan"),
            "unique_topologies": float("nan"),
            "effective_topologies": float("nan"),
            "mean_pairwise_jaccard": float("nan"),
        }
    topologies = [row["topology"] for row in elite]
    counts = Counter(topologies)
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(elite)
    effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    structures = {}
    for row in elite:
        structures.setdefault(row["topology"], row["reaction_ids"])
    distances = []
    for left, right in combinations(structures.values(), 2):
        union = left | right
        distances.append(0.0 if not union else 1.0 - len(left & right) / len(union))
    return {
        "best_loss": float(elite[0]["loss"]),
        "unique_topologies": float(len(counts)),
        "effective_topologies": effective,
        "mean_pairwise_jaccard": float(np.mean(distances)) if distances else 0.0,
    }


def _latest(history: dict[int, list[dict]], epoch: int) -> list[dict]:
    available = [snapshot_epoch for snapshot_epoch in history if snapshot_epoch <= epoch]
    return history[max(available)] if available else []


def _through(history: dict[int, list[dict]], epoch: int) -> list[dict]:
    return [
        candidate
        for snapshot_epoch, snapshot in history.items()
        if snapshot_epoch <= epoch
        for candidate in snapshot
    ]


def assert_no_preterminal_llm_leakage(
    isolated_hof: dict[int, list[dict]], isolated_llm: list[tuple[int, dict]]
) -> None:
    if not isolated_hof:
        return
    terminal_epoch = max(isolated_hof)
    llm_identifiers = {candidate["identifier"] for _, candidate in isolated_llm}
    leaked = {
        candidate["identifier"]
        for epoch, snapshot in isolated_hof.items()
        if epoch < terminal_epoch
        for candidate in snapshot
        if candidate["identifier"] in llm_identifiers
    }
    if leaked:
        raise RuntimeError(
            f"No-communication HOF contains {len(leaked)} exact LLM candidates "
            f"before terminal epoch {terminal_epoch}"
        )


def collect_run(
    communication_database: Path,
    isolated_database: Path,
    *,
    max_epoch: int = 300,
    elite_size: int = 30,
) -> list[dict]:
    communication_hof = read_hof(communication_database)
    isolated_hof_with_terminal = read_hof(isolated_database)
    communication_llm = read_llm_candidates(communication_database)
    isolated_llm = read_llm_candidates(isolated_database)
    assert_no_preterminal_llm_leakage(isolated_hof_with_terminal, isolated_llm)
    isolated_hof = dict(isolated_hof_with_terminal)
    if isolated_hof:
        isolated_hof.pop(max(isolated_hof))
    communication_llm_identifiers = {
        candidate["identifier"] for _, candidate in communication_llm
    }
    rows = []
    for epoch in range(max_epoch + 1):
        communication = _latest(communication_hof, epoch)
        communication_rl = [
            candidate
            for candidate in _through(communication_hof, epoch)
            if candidate["identifier"] not in communication_llm_identifiers
        ]
        isolated = _latest(isolated_hof, epoch)
        available_communication_llm = [
            candidate for completed, candidate in communication_llm if completed <= epoch
        ]
        available_isolated_llm = [
            candidate for completed, candidate in isolated_llm if completed <= epoch
        ]
        candidates = {
            "communication": communication,
            "communication_rl": communication_rl,
            "isolated_rl": isolated,
            "independent_pool": isolated + available_isolated_llm,
        }
        proposal_candidates = {
            "communication": available_communication_llm,
            "communication_rl": [],
            "isolated_rl": [],
            "independent_pool": available_isolated_llm,
        }
        for method in METHODS:
            proposals = (
                elite_metrics(
                    proposal_candidates[method],
                    elite_size=len(proposal_candidates[method]),
                )
                if proposal_candidates[method]
                else {"unique_topologies": 0.0, "effective_topologies": 0.0}
            )
            rows.append(
                {
                    "epoch": epoch,
                    "method": method,
                    **elite_metrics(candidates[method], elite_size=elite_size),
                    "available_llm_candidates": len(proposal_candidates[method]),
                    "available_llm_unique_topologies": proposals["unique_topologies"],
                    "available_llm_effective_topologies": proposals["effective_topologies"],
                }
            )
    return rows


def collect(
    communication_root: Path,
    isolated_root: Path,
    status_path: Path,
    *,
    communication_suffix: str,
    isolated_suffix: str,
    candidate_budget: int,
    max_epoch: int,
    elite_size: int,
) -> list[dict]:
    rows = []
    for task, seeds in completed_seeds(status_path).items():
        for seed in seeds:
            run_rows = collect_run(
                database_path(
                    communication_root, task, seed, communication_suffix, candidate_budget
                ),
                database_path(isolated_root, task, seed, isolated_suffix, candidate_budget),
                max_epoch=max_epoch,
                elite_size=elite_size,
            )
            for row in run_rows:
                rows.append({"task": task, "seed": seed, **row})
    return rows


def _summary(rows: list[dict], task: str, method: str, field: str) -> tuple[np.ndarray, ...]:
    subset = [row for row in rows if row["task"] == task and row["method"] == method]
    seeds = sorted({int(row["seed"]) for row in subset})
    epochs = sorted({int(row["epoch"]) for row in subset})
    lookup = {(int(row["seed"]), int(row["epoch"])): float(row[field]) for row in subset}
    values = np.asarray([[lookup[(seed, epoch)] for epoch in epochs] for seed in seeds])
    return (
        np.asarray(epochs),
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 25, axis=0),
        np.nanpercentile(values, 75, axis=0),
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=400 if suffix == ".png" else None,
            facecolor="white",
            metadata={"Title": title} if suffix == ".pdf" else None,
        )
    plt.close(fig)


def plot_over_time(rows: list[dict], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, constrained_layout=True)
    for row_index, task in enumerate(TASKS):
        n_runs = len({row["seed"] for row in rows if row["task"] == task})
        for column, field in enumerate(("best_loss", "effective_topologies")):
            ax = axes[row_index, column]
            for method in METHODS:
                epoch, median, q25, q75 = _summary(rows, task, method, field)
                ax.fill_between(epoch, q25, q75, color=COLORS[method], alpha=0.13, linewidth=0)
                ax.plot(
                    epoch,
                    median,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=1.55,
                    label=LABELS[method],
                )
            if field == "best_loss":
                ax.set_yscale("log")
                ax.set_ylabel("Best-so-far loss")
                ax.set_title(f"{TASK_LABELS[task]} performance (n = {n_runs})", fontweight="semibold")
            else:
                ax.set_ylabel("Effective topologies in top 30")
                ax.set_title(f"{TASK_LABELS[task]} elite diversity", fontweight="semibold")
            ax.set_xlim(0, 300)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 1:
                ax.set_xlabel("Training epoch")
            ax.text(
                -0.12,
                1.03,
                chr(ord("A") + row_index * 2 + column),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=9,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, "Legacy HOF-exposed communication ablation")


def plot_performance_diversity(rows: list[dict], output: Path, max_epoch: int) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for index, (ax, task) in enumerate(zip(axes, TASKS)):
        for method in METHODS:
            subset = [
                row
                for row in rows
                if row["task"] == task and row["method"] == method and row["epoch"] == max_epoch
            ]
            ax.scatter(
                [row["effective_topologies"] for row in subset],
                [row["best_loss"] for row in subset],
                s=22,
                alpha=0.72,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.35,
                label=LABELS[method],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Effective topologies in top 30")
        ax.set_ylabel("Best loss" if index == 0 else "")
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.01,
            0.98,
            chr(ord("A") + index),
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=9,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, "Communication ablation performance-diversity trade-off")


def plot_proposal_diversity(rows: list[dict], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, constrained_layout=True)
    proposal_methods = ("communication", "independent_pool")
    proposal_labels = {
        "communication": "Full-duplex LLM",
        "independent_pool": "Independent LLM",
    }
    for index, (ax, task) in enumerate(zip(axes, TASKS)):
        n_runs = len({row["seed"] for row in rows if row["task"] == task})
        for method in proposal_methods:
            epoch, median, q25, q75 = _summary(
                rows, task, method, "available_llm_unique_topologies"
            )
            ax.fill_between(epoch, q25, q75, color=COLORS[method], alpha=0.13, linewidth=0)
            ax.plot(
                epoch,
                median,
                color=COLORS[method],
                linewidth=1.6,
                label=proposal_labels[method],
            )
        ax.set_xlim(0, 300)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Cumulative unique LLM topologies" if index == 0 else "")
        ax.set_title(f"{TASK_LABELS[task]} proposals (n = {n_runs})", fontweight="semibold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.01,
            0.98,
            chr(ord("A") + index),
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=9,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, "Communication ablation LLM proposal diversity through time")


def write_csv(rows: list[dict], output: Path) -> None:
    fields = (
        "task", "seed", "epoch", "method", "best_loss", "unique_topologies",
        "effective_topologies", "mean_pairwise_jaccard", "available_llm_candidates",
        "available_llm_unique_topologies", "available_llm_effective_topologies",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--communication-root", type=Path, required=True)
    parser.add_argument("--isolated-root", type=Path, required=True)
    parser.add_argument("--isolated-status", type=Path, required=True)
    parser.add_argument("--communication-suffix", default="cvode_llm_flash_long300")
    parser.add_argument("--isolated-suffix", default="cvode_llm_flash_no_communication")
    parser.add_argument("--candidate-budget", type=int, default=307200)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--elite-size", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(
        args.communication_root.expanduser().resolve(),
        args.isolated_root.expanduser().resolve(),
        args.isolated_status.expanduser().resolve(),
        communication_suffix=args.communication_suffix,
        isolated_suffix=args.isolated_suffix,
        candidate_budget=args.candidate_budget,
        max_epoch=args.max_epoch,
        elite_size=args.elite_size,
    )
    if not rows:
        raise RuntimeError("No completed paired runs were found")
    output = args.output.expanduser().resolve()
    write_csv(rows, output.with_suffix(".csv"))
    plot_over_time(rows, output)
    plot_performance_diversity(
        rows, output.with_name(output.stem + "_performance_diversity"), args.max_epoch
    )
    plot_proposal_diversity(
        rows, output.with_name(output.stem + "_llm_proposal_diversity")
    )
    for task in TASKS:
        n_runs = len({row["seed"] for row in rows if row["task"] == task})
        print(f"{TASK_LABELS[task]}: {n_runs} completed paired runs")


if __name__ == "__main__":
    main()
