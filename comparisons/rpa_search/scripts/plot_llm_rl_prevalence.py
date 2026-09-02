#!/usr/bin/env python3
"""Plot exact-LLM versus other Hall-of-Fame records through training."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from comparisons.rpa_search.src.common.plotting import SOURCE_COLORS


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
RL_COLOR = SOURCE_COLORS["rl"]
LLM_COLOR = SOURCE_COLORS["llm"]


def run_id(task: str, seed: int, run_suffix: str, candidate_budget: int) -> str:
    return f"{task}_full{candidate_budget}_seed{seed}_{run_suffix}"


def database_path(
    campaign_root: Path, task: str, seed: int, run_suffix: str, candidate_budget: int
) -> Path:
    run = run_id(task, seed, run_suffix, candidate_budget)
    matches = sorted((campaign_root / "runs").glob(f"*/{run}/results.sqlite"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one results database for {run}, found {len(matches)}")
    return matches[0]


def source_label(value: object) -> str:
    return "LLM" if str(value or "").strip().upper() == "LLM" else "RL"


def initial_sources(connection: sqlite3.Connection) -> dict[str, str]:
    """Resolve each topology's earliest recorded source, matching the web viewer."""

    events: dict[str, list[tuple[float, str]]] = {}
    for topology_hash, source, created_at in connection.execute(
        "SELECT topology_hash, source, created_at FROM evaluations"
    ):
        events.setdefault(str(topology_hash), []).append(
            (float(created_at), source_label(source))
        )
    for topology_hash, task_info_json, created_at in connection.execute(
        """SELECT e.topology_hash, e.task_info_json, h.created_at
             FROM hof_snapshot_entries e
             JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id"""
    ):
        try:
            task_info = json.loads(task_info_json or "{}")
        except (TypeError, json.JSONDecodeError):
            task_info = {}
        events.setdefault(str(topology_hash), []).append(
            (float(created_at), source_label(task_info.get("source")))
        )
    return {
        topology_hash: min(source_events, key=lambda item: item[0])[1]
        for topology_hash, source_events in events.items()
    }


def read_run_history(path: Path) -> dict[int, tuple[float, float]]:
    # Runs are finalized WAL databases; immutable mode prevents SQLite from
    # trying to create sidecar files beside read-only campaign artifacts.
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        sources = initial_sources(connection)
        rows = connection.execute(
            """SELECT h.epoch, e.rank, e.topology_hash
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                ORDER BY h.epoch, e.rank"""
        ).fetchall()

    grouped: dict[int, list[tuple[int, str]]] = {}
    for epoch, rank, topology_hash in rows:
        grouped.setdefault(int(epoch), []).append((int(rank), str(topology_hash)))

    history = {}
    for epoch, entries in grouped.items():
        ordered = sorted(entries)
        labels = [sources.get(topology_hash, "RL") for _, topology_hash in ordered]
        history[epoch] = (
            float(np.mean(np.asarray(labels) == "LLM")),
            float(labels[0] == "LLM"),
        )
    return history


def collect(
    campaign_root: Path, n_seeds: int, run_suffix: str, candidate_budget: int = 102400
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for task in TASKS:
        histories = [
            read_run_history(
                database_path(campaign_root, task, seed, run_suffix, candidate_budget)
            )
            for seed in range(n_seeds)
        ]
        common_epochs = sorted(set.intersection(*(set(history) for history in histories)))
        hof_share = np.asarray(
            [[history[epoch][0] for epoch in common_epochs] for history in histories],
            dtype=float,
        )
        best_share = np.asarray(
            [[history[epoch][1] for epoch in common_epochs] for history in histories],
            dtype=float,
        )
        result[task] = {
            "epochs": np.asarray(common_epochs, dtype=int),
            "llm_mean": np.mean(hof_share, axis=0),
            "llm_q25": np.percentile(hof_share, 25, axis=0),
            "llm_q75": np.percentile(hof_share, 75, axis=0),
            "best_llm": np.mean(best_share, axis=0),
        }
    return result


def write_summary(data: dict[str, dict[str, np.ndarray]], output: Path, n_seeds: int) -> None:
    path = output.with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "epoch",
                "llm_hof_share_mean",
                "llm_hof_share_q25",
                "llm_hof_share_q75",
                "rl_hof_share_mean",
                "fraction_best_llm",
                "n_runs",
            ),
        )
        writer.writeheader()
        for task, values in data.items():
            for index, epoch in enumerate(values["epochs"]):
                llm_mean = float(values["llm_mean"][index])
                writer.writerow(
                    {
                        "task": task,
                        "epoch": int(epoch),
                        "llm_hof_share_mean": llm_mean,
                        "llm_hof_share_q25": float(values["llm_q25"][index]),
                        "llm_hof_share_q75": float(values["llm_q75"][index]),
                        "rl_hof_share_mean": 1.0 - llm_mean,
                        "fraction_best_llm": float(values["best_llm"][index]),
                        "n_runs": n_seeds,
                    }
                )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path, n_seeds: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True, sharey=True, constrained_layout=True)
    for column, (ax, task) in enumerate(zip(axes, TASKS)):
        values = data[task]
        epoch = values["epochs"]
        llm = values["llm_mean"]
        ax.fill_between(epoch, 0, llm, step="post", color=LLM_COLOR, alpha=0.24)
        ax.fill_between(epoch, llm, 1, step="post", color=RL_COLOR, alpha=0.17)
        ax.fill_between(
            epoch,
            values["llm_q25"],
            values["llm_q75"],
            step="post",
            color=LLM_COLOR,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(epoch, llm, color=LLM_COLOR, linewidth=1.7, drawstyle="steps-post")
        ax.plot(
            epoch,
            values["best_llm"],
            color="#00664B",
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            drawstyle="steps-post",
        )
        ax.axhline(0.5, color="#777777", linewidth=0.55, linestyle=(0, (2, 3)), zorder=0)
        ax.set_title(TASK_LABELS[task], fontweight="semibold", pad=5)
        ax.set_xlabel("Training epoch")
        ax.set_xlim(0, int(epoch[-1]))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#333333")
        ax.tick_params(direction="out", length=3, width=0.7, color="#333333")
        ax.text(
            -0.13,
            1.04,
            chr(ord("A") + column),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel("Prevalence across runs")
    handles = [
        Patch(facecolor=LLM_COLOR, alpha=0.24, label="LLM-provenance record share"),
        Patch(facecolor=RL_COLOR, alpha=0.17, label="RL-provenance record share"),
        Line2D(
            [0],
            [0],
            color="#00664B",
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            label="LLM-provenance rank 1",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        title=f"Emitting-process provenance (n = {n_seeds}; mean share, LLM IQR shaded)",
        title_fontsize=6.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "LLM versus RL Hall-of-Fame prevalence through training"}
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--run-suffix", default="cvode_llm")
    parser.add_argument("--candidate-budget", type=int, default=102400)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/llm_rl_prevalence_20seed.pdf",
    )
    args = parser.parse_args()
    data = collect(
        args.campaign_root.expanduser().resolve(),
        args.n_seeds,
        args.run_suffix,
        args.candidate_budget,
    )
    output = args.output.expanduser().resolve()
    plot(data, output, args.n_seeds)
    write_summary(data, output, args.n_seeds)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
