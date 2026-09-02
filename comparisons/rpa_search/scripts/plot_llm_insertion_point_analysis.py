#!/usr/bin/env python3
"""Analyze when asynchronous LLM batches enter RL and whether they improve the HoF."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
TASK_COLORS = {"rpa": "#0072B2", "logic": "#D55E00"}


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
        raise RuntimeError(
            f"Expected one results database for {task} seed {seed}, found {len(matches)}"
        )
    return matches[0]


def _minimum_hof_loss(connection: sqlite3.Connection, epoch: int) -> float | None:
    row = connection.execute(
        """SELECT MIN(e.loss) FROM hof_snapshot_entries e
             JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
            WHERE h.epoch = ?""",
        (epoch,),
    ).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def launched_workspace_epochs(run_directory: Path) -> set[int]:
    """Return launch epochs backed by an initialized Harness workspace."""

    epochs = set()
    for workspace in sorted((run_directory / "harness-workspaces").glob("*-crn-generation-*")):
        manifest = workspace / "run_manifest.json"
        state_path = workspace / "CONTEXT/SEARCH_STATE.json"
        if not manifest.is_file() or not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        epochs.add(int(state["rl_epoch_at_snapshot"]))
    return epochs


def collect_database(
    database: Path,
    *,
    task: str,
    seed: int,
    expected_launches: list[int],
    launched_epochs: set[int] | None = None,
) -> list[dict[str, object]]:
    if launched_epochs is None:
        launched_epochs = launched_workspace_epochs(database.parent)
    uri = f"file:{database.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        snapshot_epochs = [
            int(row[0])
            for row in connection.execute(
                "SELECT DISTINCT epoch FROM hof_snapshots ORDER BY epoch"
            )
        ]
        if not snapshot_epochs:
            raise RuntimeError(f"No Hall-of-Fame snapshots in {database}")
        final_epoch = max(snapshot_epochs)
        final_candidates = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                """SELECT e.topology_hash, e.parameters_json FROM hof_snapshot_entries e
                     JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                    WHERE h.epoch = ?""",
                (final_epoch,),
            )
        }
        rounds = {
            int(row[1]): row
            for row in connection.execute(
                """SELECT llm_run_id, launched_epoch, completed_epoch, requested,
                          produced, valid_count, elapsed_seconds
                     FROM llm_runs ORDER BY launched_epoch"""
            )
        }
        rows: list[dict[str, object]] = []
        for launched_epoch in expected_launches:
            llm_row = rounds.get(launched_epoch)
            if llm_row is None:
                launched = int(launched_epoch in launched_epochs)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "launched_epoch": launched_epoch,
                        "completed_epoch": "",
                        "final_epoch": final_epoch,
                        "status": "launched_not_served" if launched else "not_launched",
                        "scheduled": 1,
                        "launched": launched,
                        "served": 0,
                        "epoch_lag": "",
                        "normalized_insertion_point": "",
                        "post_insertion_epochs": "",
                        "terminal_insertion": "",
                        "requested": 10,
                        "produced": 0,
                        "valid_count": 0,
                        "elapsed_seconds": "",
                        "pre_insertion_hof_loss": "",
                        "batch_best_loss": "",
                        "batch_to_pre_hof_ratio": "",
                        "entered_hof_at_insertion": 0,
                        "improved_rank1_at_insertion": 0,
                        "immediate_relative_gain": 0.0,
                        "appeared_in_future_hof": 0,
                        "survived_in_final_hof": 0,
                    }
                )
                continue

            llm_run_id, _, completed_epoch, requested, produced, valid_count, elapsed = llm_row
            completed_epoch = int(completed_epoch)
            previous_epochs = [epoch for epoch in snapshot_epochs if epoch < completed_epoch]
            pre_epoch = max(previous_epochs) if previous_epochs else min(snapshot_epochs)
            pre_loss = _minimum_hof_loss(connection, pre_epoch)
            post_loss = _minimum_hof_loss(connection, completed_epoch)
            candidates = connection.execute(
                """SELECT topology_hash, valid, loss FROM llm_candidates
                    WHERE llm_run_id = ?""",
                (llm_run_id,),
            ).fetchall()
            candidate_losses = [
                float(row[2]) for row in candidates if row[1] and row[2] is not None
            ]
            candidate_identifiers = {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    """SELECT topology_hash, parameters_json FROM evaluations
                        WHERE source = 'llm' AND valid = 1
                          AND json_extract(metadata_json, '$.llm_run_id') = ?""",
                    (llm_run_id,),
                )
            }
            insertion_candidates = {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    """SELECT e.topology_hash, e.parameters_json FROM hof_snapshot_entries e
                         JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                        WHERE h.epoch = ?""",
                    (completed_epoch,),
                )
            }
            insertion_rank1 = connection.execute(
                """SELECT e.topology_hash, e.parameters_json FROM hof_snapshot_entries e
                     JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                    WHERE h.epoch = ? AND e.rank = 0""",
                (completed_epoch,),
            ).fetchone()
            future_candidates = {
                (str(row[0]), str(row[1]))
                for row in connection.execute(
                    """SELECT DISTINCT e.topology_hash, e.parameters_json
                         FROM hof_snapshot_entries e
                         JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                        WHERE h.epoch >= ?""",
                    (completed_epoch,),
                )
            }
            best = min(candidate_losses) if candidate_losses else None
            rank1_candidate = (
                None
                if insertion_rank1 is None
                else (str(insertion_rank1[0]), str(insertion_rank1[1]))
            )
            improved_rank1 = bool(
                rank1_candidate in candidate_identifiers
                and pre_loss is not None
                and post_loss is not None
                and post_loss < pre_loss
            )
            immediate_gain = (
                max(0.0, (pre_loss - post_loss) / pre_loss)
                if improved_rank1 and pre_loss and post_loss is not None
                else 0.0
            )
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "launched_epoch": launched_epoch,
                    "completed_epoch": completed_epoch,
                    "final_epoch": final_epoch,
                    "status": "completed",
                    "scheduled": 1,
                    "launched": int(launched_epoch in launched_epochs),
                    "served": 1,
                    "epoch_lag": completed_epoch - launched_epoch,
                    "normalized_insertion_point": completed_epoch / final_epoch,
                    "post_insertion_epochs": max(0, final_epoch - completed_epoch),
                    "terminal_insertion": int(completed_epoch >= final_epoch),
                    "requested": int(requested or 0),
                    "produced": int(produced),
                    "valid_count": int(valid_count),
                    "elapsed_seconds": float(elapsed or 0.0),
                    "pre_insertion_hof_loss": pre_loss,
                    "batch_best_loss": best,
                    "batch_to_pre_hof_ratio": (
                        best / pre_loss if best is not None and pre_loss else ""
                    ),
                    "entered_hof_at_insertion": int(
                        bool(candidate_identifiers & insertion_candidates)
                    ),
                    "improved_rank1_at_insertion": int(improved_rank1),
                    "immediate_relative_gain": immediate_gain,
                    "appeared_in_future_hof": int(
                        bool(candidate_identifiers & future_candidates)
                    ),
                    "survived_in_final_hof": int(
                        bool(candidate_identifiers & final_candidates)
                    ),
                }
            )
    return rows


def collect(
    campaign_root: Path,
    *,
    n_seeds: int,
    run_suffix: str,
    candidate_budget: int,
    epochs: int,
    llm_every: int,
) -> list[dict[str, object]]:
    expected_launches = list(range(0, epochs, llm_every))
    rows = []
    for task in TASKS:
        for seed in range(n_seeds):
            rows.extend(
                collect_database(
                    database_path(
                        campaign_root, task, seed, run_suffix, candidate_budget
                    ),
                    task=task,
                    seed=seed,
                    expected_launches=expected_launches,
                )
            )
    return rows


def bootstrap_run_mean_ci(
    rows: list[dict[str, object]], field: str, *, seed: int = 260117582
) -> tuple[float, float, float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(float(row[field]))
    run_means = np.asarray([np.mean(values) for values in by_seed.values()], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(
        rng.choice(run_means, size=(10_000, len(run_means)), replace=True), axis=1
    )
    return (
        float(np.mean(run_means)),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for task in TASKS:
        group = [row for row in rows if row["task"] == task]
        completed = [row for row in group if row["status"] == "completed"]
        lags = [float(row["epoch_lag"]) for row in completed]
        remaining = [float(row["post_insertion_epochs"]) for row in completed]
        entry, entry_lo, entry_hi = bootstrap_run_mean_ci(group, "entered_hof_at_insertion")
        rank1, rank1_lo, rank1_hi = bootstrap_run_mean_ci(
            group, "improved_rank1_at_insertion"
        )
        retained, retained_lo, retained_hi = bootstrap_run_mean_ci(
            group, "survived_in_final_hof"
        )
        summary.append(
            {
                "task": task,
                "scheduled": sum(int(row["scheduled"]) for row in group),
                "launched": sum(int(row["launched"]) for row in group),
                "served": sum(int(row["served"]) for row in group),
                "launch_fraction": sum(int(row["launched"]) for row in group) / len(group),
                "service_fraction_of_launched": sum(int(row["served"]) for row in group)
                / sum(int(row["launched"]) for row in group),
                "median_epoch_lag": float(np.median(lags)),
                "q25_epoch_lag": float(np.percentile(lags, 25)),
                "q75_epoch_lag": float(np.percentile(lags, 75)),
                "max_epoch_lag": max(lags),
                "median_post_insertion_epochs": float(np.median(remaining)),
                "terminal_insertions": sum(int(row["terminal_insertion"]) for row in completed),
                "entered_hof_fraction": entry,
                "entered_hof_ci_low": entry_lo,
                "entered_hof_ci_high": entry_hi,
                "improved_rank1_fraction": rank1,
                "improved_rank1_ci_low": rank1_lo,
                "improved_rank1_ci_high": rank1_hi,
                "final_survival_fraction": retained,
                "final_survival_ci_low": retained_lo,
                "final_survival_ci_high": retained_hi,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 5.55), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(2.0, 0.42, 2.65))
    timing_axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    failure_axes = [
        fig.add_subplot(grid[1, column], sharex=timing_axes[column])
        for column in range(2)
    ]
    quality_axes = [fig.add_subplot(grid[2, column]) for column in range(2)]
    for column, task in enumerate(TASKS):
        group = [row for row in rows if row["task"] == task]
        completed = [row for row in group if row["status"] == "completed"]
        by_launch = defaultdict(list)
        for row in completed:
            by_launch[int(row["launched_epoch"])].append(float(row["epoch_lag"]))
        launches = np.asarray(sorted(by_launch), dtype=float)
        medians = np.asarray([np.median(by_launch[int(epoch)]) for epoch in launches])
        low = np.asarray([np.percentile(by_launch[int(epoch)], 25) for epoch in launches])
        high = np.asarray([np.percentile(by_launch[int(epoch)], 75) for epoch in launches])

        ax = timing_axes[column]
        jittered_launches = [
            float(row["launched_epoch"]) + (float(row["seed"]) - 9.5) * 0.34
            for row in completed
        ]
        ax.scatter(
            jittered_launches,
            [float(row["epoch_lag"]) for row in completed],
            s=8,
            color=TASK_COLORS[task],
            alpha=0.24,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            launches, low, high, color=TASK_COLORS[task], alpha=0.18, linewidth=0,
            label="Interquartile range", zorder=2,
        )
        ax.plot(
            launches, medians, color=TASK_COLORS[task], marker="o", markersize=3,
            linewidth=1.2, label="Median lag", zorder=3,
        )
        final_epoch = max(int(row["final_epoch"]) for row in group)
        ax.axhline(0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        scheduled = sum(int(row["scheduled"]) for row in group)
        launched = sum(int(row["launched"]) for row in group)
        served = sum(int(row["served"]) for row in group)
        ax.text(
            0.99,
            0.98,
            f"scheduled  {scheduled}\nlaunched    {launched}\nserved      {served}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#444444",
        )
        if column == 0:
            ax.set_ylabel("Insertion lag (RL epochs)")
        padding = max(2.0, 0.02 * final_epoch)
        ax.set_xlim(-padding, final_epoch - 4)
        ax.set_ylim(-0.4, max(float(row["epoch_lag"]) for row in completed) + 0.8)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.04, chr(ord("A") + column), transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.tick_params(axis="x", labelbottom=False)

        failed_by_launch = defaultdict(int)
        for row in group:
            if row["status"] == "launched_not_served":
                failed_by_launch[int(row["launched_epoch"])] += 1
        failure_ax = failure_axes[column]
        failure_counts = [failed_by_launch[int(epoch)] for epoch in launches]
        failure_ax.bar(launches, failure_counts, width=8, color="#B2182B", alpha=0.82)
        max_failures = max(failure_counts, default=0)
        failure_ax.set_ylim(0, max(1, max_failures) + 0.25)
        failure_ax.set_yticks(sorted({0, max_failures}) if max_failures else [0])
        if column == 0:
            failure_ax.set_ylabel("Not served", color="#8B1A2B")
        failure_ax.set_xlabel("LLM launch epoch")
        failure_ax.grid(axis="y", color="#E8E8E8", linewidth=0.45)
        failure_ax.spines[["top", "right"]].set_visible(False)
        failure_ax.tick_params(axis="y", colors="#8B1A2B")

        ax = quality_axes[column]
        ordinary = [row for row in completed if not int(row["survived_in_final_hof"])]
        retained = [row for row in completed if int(row["survived_in_final_hof"])]
        for subset, face, edge, label, zorder in (
            (ordinary, "#D3D3D3", "#777777", "Not in final HoF", 2),
            (retained, TASK_COLORS[task], "white", "Survived in final HoF", 3),
        ):
            x = [100 * float(row["normalized_insertion_point"]) for row in subset if row["batch_to_pre_hof_ratio"] != ""]
            y = [float(row["batch_to_pre_hof_ratio"]) for row in subset if row["batch_to_pre_hof_ratio"] != ""]
            ax.scatter(x, y, s=17, facecolor=face, edgecolor=edge, linewidth=0.45, alpha=0.82, label=label, zorder=zorder)
        ax.axhline(1.0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Insertion point (% of training)")
        if column == 0:
            ax.set_ylabel("Best batch loss / pre-insertion HoF loss")
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.04, chr(ord("C") + column), transform=ax.transAxes, fontweight="bold", fontsize=9)

    timing_axes[0].legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color=TASK_COLORS["rpa"], markersize=3,
                linewidth=1.2, label="Median lag",
            ),
            Line2D(
                [0], [0], marker="o", color=TASK_COLORS["rpa"], alpha=0.3,
                markersize=3, linewidth=0, label="Served requests",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    quality_axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Asynchronous LLM insertion timing and proposal quality",
        fontsize=10.5,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        target = output.with_suffix(suffix)
        fig.savefig(target, dpi=300 if suffix == ".png" else None, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--run-suffix", default="cvode_llm_flash_long300")
    parser.add_argument("--candidate-budget", type=int, default=307200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--llm-every", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(
        args.campaign_root.expanduser().resolve(),
        n_seeds=args.n_seeds,
        run_suffix=args.run_suffix,
        candidate_budget=args.candidate_budget,
        epochs=args.epochs,
        llm_every=args.llm_every,
    )
    summary = summarize(rows)
    write_csv(args.output.with_name(f"{args.output.stem}_requests.csv"), rows)
    write_csv(args.output.with_name(f"{args.output.stem}_summary.csv"), summary)
    plot(rows, args.output)
    for row in summary:
        print(
            f"{TASK_LABELS[str(row['task'])]}: {row['scheduled']} scheduled, "
            f"{row['launched']} launched, {row['served']} served; "
            f"median lag {row['median_epoch_lag']:.1f} epochs; "
            f"HoF entry {100 * row['entered_hof_fraction']:.1f}%; "
            f"rank-1 improvement {100 * row['improved_rank1_fraction']:.1f}%; "
            f"final survival {100 * row['final_survival_fraction']:.1f}%"
        )


if __name__ == "__main__":
    main()
