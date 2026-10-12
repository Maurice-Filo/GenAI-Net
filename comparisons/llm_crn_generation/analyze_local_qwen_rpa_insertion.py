#!/usr/bin/env python3
"""Plot the local Qwen RPA validation trajectory and LLM insertion events."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


DEFAULT_CAMPAIGN = Path.home() / (
    "ai-workspaces/deepseek-test/crn-runs/paper-campaigns/"
    "qwen38-local-v2-rpa-seed9000-100epoch-20260902-r8"
)
DEFAULT_RUN = DEFAULT_CAMPAIGN / (
    "runs/qwen38-local-v2-rpa-seed9000-100epoch-20260902-r8-rpa-seed9000/"
    "rpa_full102400_seed9000_cvode_llm_qwen38_local_contract_v2_rpa_r8"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated/local_qwen_rpa_r8"
RL_COLOR = "#0072B2"
LLM_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_progress(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "epoch": float(row["step"]) - 1,
            "best": float(row["best_so_far_loss"]),
            "elapsed": float(row["elapsed_seconds"]),
        }
        for row in rows
        if int(row["step"]) <= 100
    ]


def open_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def read_requests(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT r.llm_run_id, r.launched_epoch, r.completed_epoch,
               r.elapsed_seconds, r.requested, r.valid_count, MIN(c.loss)
        FROM llm_runs AS r
        JOIN llm_candidates AS c ON c.llm_run_id = r.llm_run_id
        GROUP BY r.llm_run_id
        ORDER BY r.launched_epoch
        """
    ).fetchall()
    requests: list[dict[str, object]] = []
    for llm_run_id, launched, completed, elapsed, requested, valid, best in rows:
        incumbent = connection.execute(
            """
            SELECT e.loss
            FROM hof_snapshots AS s
            JOIN hof_snapshot_entries AS e ON e.snapshot_id = s.snapshot_id
            WHERE s.epoch = ? AND e.rank = 0
            """,
            (completed,),
        ).fetchone()[0]
        inserted = connection.execute(
            """
            SELECT COUNT(DISTINCT e.topology_hash), MIN(e.rank)
            FROM llm_candidates AS c
            JOIN hof_snapshots AS s ON s.epoch = ?
            JOIN hof_snapshot_entries AS e
              ON e.snapshot_id = s.snapshot_id
             AND e.topology_hash = c.topology_hash
             AND e.emitter = 'LLM'
            WHERE c.llm_run_id = ?
            """,
            (completed, llm_run_id),
        ).fetchone()
        requests.append(
            {
                "launch_epoch": int(launched),
                "merge_epoch": int(completed),
                "latency_seconds": float(elapsed),
                "requested": int(requested),
                "valid": int(valid),
                "batch_best_loss": float(best),
                "incumbent_loss_at_merge": float(incumbent),
                "batch_to_incumbent_ratio": float(best / incumbent),
                "direct_hof_topologies": int(inserted[0]),
                "best_direct_hof_rank": (
                    int(inserted[1]) + 1 if inserted[1] is not None else None
                ),
            }
        )
    return requests


def read_rank_one_changes(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT s.epoch, e.loss, e.emitter, e.provenance_class
        FROM hof_snapshots AS s
        JOIN hof_snapshot_entries AS e
          ON e.snapshot_id = s.snapshot_id AND e.rank = 0
        ORDER BY s.epoch
        """
    ).fetchall()
    changes = []
    previous = None
    for epoch, loss, emitter, provenance in rows:
        key = (loss, emitter, provenance)
        if key != previous:
            changes.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "emitter": emitter,
                    "provenance_class": provenance,
                }
            )
        previous = key
    return changes


def make_plot(
    progress: list[dict[str, float]],
    requests: list[dict[str, object]],
    output_dir: Path,
) -> None:
    epochs = np.array([row["epoch"] for row in progress])
    best = np.array([row["best"] for row in progress])
    improvement = np.r_[True, best[1:] < best[:-1]]

    fig, (trajectory_ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10.2, 7.8),
        gridspec_kw={"height_ratios": [1.75, 1]},
        constrained_layout=True,
    )
    trajectory_ax.step(epochs, best, where="post", color=RL_COLOR, linewidth=2.2)
    trajectory_ax.scatter(
        epochs[improvement], best[improvement], color=RL_COLOR, s=27, zorder=4
    )
    for index, request in enumerate(requests, start=1):
        launch = int(request["launch_epoch"])
        merge = int(request["merge_epoch"])
        trajectory_ax.axvspan(launch, merge, color=LLM_COLOR, alpha=0.055)
        trajectory_ax.axvline(merge, color=LLM_COLOR, alpha=0.45, linewidth=1)
        trajectory_ax.scatter(
            merge,
            request["batch_best_loss"],
            marker="D",
            s=54,
            color=LLM_COLOR,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        trajectory_ax.annotate(
            f"R{index}",
            (merge, request["batch_best_loss"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=LLM_COLOR,
        )
    trajectory_ax.set_yscale("log")
    trajectory_ax.set_xlim(0, 99)
    trajectory_ax.set_ylabel("Best-so-far canonical loss")
    trajectory_ax.set_title("Local Qwen RPA validation: asynchronous LLM requests and RL progress")
    trajectory_ax.grid(True, which="both", alpha=0.22)
    trajectory_ax.legend(
        handles=[
            plt.Line2D([], [], color=RL_COLOR, marker="o", label="RL HOF incumbent"),
            plt.Line2D([], [], color=LLM_COLOR, marker="D", linestyle="", label="Best candidate in LLM batch"),
            Patch(facecolor=LLM_COLOR, alpha=0.12, label="LLM request in flight"),
        ],
        loc="upper right",
        frameon=False,
    )

    labels = [
        f"R{i}\n{row['launch_epoch']}->{row['merge_epoch']}"
        for i, row in enumerate(requests, start=1)
    ]
    ratios = np.array([row["batch_to_incumbent_ratio"] for row in requests])
    x = np.arange(len(requests))
    bars = ratio_ax.bar(x, ratios, color=LLM_COLOR, alpha=0.78, width=0.66)
    for bar, request in zip(bars, requests):
        if request["direct_hof_topologies"]:
            bar.set_hatch("///")
            bar.set_edgecolor("#7A2E00")
        ratio_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{bar.get_height():.2f}x\n{request['latency_seconds']:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ratio_ax.axhline(1, color="#333333", linewidth=1.2, linestyle="--")
    ratio_ax.set_xticks(x, labels)
    ratio_ax.set_ylim(0, max(ratios) * 1.24)
    ratio_ax.set_ylabel("LLM batch minimum / HOF loss")
    ratio_ax.set_xlabel("Request and launch -> merge epoch")
    ratio_ax.grid(True, axis="y", alpha=0.22)
    ratio_ax.text(
        0.01,
        0.96,
        "Lower is better; values above 1 did not replace rank one",
        transform=ratio_ax.transAxes,
        va="top",
        fontsize=9,
    )
    ratio_ax.legend(
        handles=[Patch(facecolor=LLM_COLOR, edgecolor="#7A2E00", hatch="///", label="At least one direct LLM HOF insertion")],
        loc="upper right",
        frameon=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "local_qwen_rpa_insertion_diagnostic.png", dpi=240)
    fig.savefig(output_dir / "local_qwen_rpa_insertion_diagnostic.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    progress = read_progress(args.run_root / "progress.csv")
    with open_database(args.run_root / "results.sqlite") as connection:
        requests = read_requests(connection)
        rank_one_changes = read_rank_one_changes(connection)
    make_plot(progress, requests, args.output_dir)
    summary = {
        "status": "development-only-local-model-validation",
        "run_root": str(args.run_root),
        "final_best_loss": progress[-1]["best"],
        "requests": requests,
        "rank_one_changes": rank_one_changes,
        "direct_llm_rank_one_count": sum(
            change["emitter"] == "LLM" for change in rank_one_changes
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "local_qwen_rpa_insertion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
