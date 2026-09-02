#!/usr/bin/env python3
"""Plot exact-LLM and other-record best-so-far loss through training."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from comparisons.rpa_search.src.common.plotting import SOURCE_COLORS


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
COLORS = SOURCE_COLORS


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
        raise RuntimeError(f"Expected one results database for {run_id}, found {len(matches)}")
    return matches[0]


def read_run_history(path: Path) -> dict[int, tuple[float, float]]:
    """Return cumulative RL and LLM minima keyed by snapshot epoch."""

    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        llm_candidates = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                """SELECT topology_hash, parameters_json FROM evaluations
                    WHERE source = 'llm' AND valid = 1"""
            )
        }
        rows = connection.execute(
            """SELECT h.epoch, e.topology_hash, e.parameters_json, e.loss
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE e.loss IS NOT NULL
                ORDER BY h.epoch, e.rank"""
        ).fetchall()

    by_epoch: dict[int, list[tuple[str, float]]] = {}
    for epoch, topology_hash, parameters_json, loss in rows:
        identifier = (str(topology_hash), str(parameters_json))
        source = "llm" if identifier in llm_candidates else "rl"
        by_epoch.setdefault(int(epoch), []).append((source, float(loss)))

    best = {"rl": float("inf"), "llm": float("inf")}
    history = {}
    for epoch in sorted(by_epoch):
        for source, loss in by_epoch[epoch]:
            best[source] = min(best[source], loss)
        history[epoch] = (
            best["rl"] if np.isfinite(best["rl"]) else float("nan"),
            best["llm"] if np.isfinite(best["llm"]) else float("nan"),
        )
    return history


def _nan_summary(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.full(values.shape[1], np.nan)
    q25 = np.full(values.shape[1], np.nan)
    q75 = np.full(values.shape[1], np.nan)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if finite.size:
            median[index] = np.median(finite)
            q25[index] = np.percentile(finite, 25)
            q75[index] = np.percentile(finite, 75)
    return median, q25, q75


def collect(
    campaign_root: Path,
    *,
    n_seeds: int,
    run_suffix: str,
    candidate_budget: int,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for task in TASKS:
        histories = [
            read_run_history(
                database_path(campaign_root, task, seed, run_suffix, candidate_budget)
            )
            for seed in range(n_seeds)
        ]
        epochs = sorted(set.intersection(*(set(history) for history in histories)))
        rl = np.asarray([[history[epoch][0] for epoch in epochs] for history in histories])
        llm = np.asarray([[history[epoch][1] for epoch in epochs] for history in histories])
        rl_median, rl_q25, rl_q75 = _nan_summary(rl)
        llm_median, llm_q25, llm_q75 = _nan_summary(llm)
        both = np.isfinite(rl) & np.isfinite(llm)
        llm_leads = np.divide(
            np.sum((llm < rl) & both, axis=0),
            np.sum(both, axis=0),
            out=np.full(len(epochs), np.nan),
            where=np.sum(both, axis=0) > 0,
        )
        result[task] = {
            "epochs": np.asarray(epochs, dtype=int),
            "rl_median": rl_median,
            "rl_q25": rl_q25,
            "rl_q75": rl_q75,
            "llm_median": llm_median,
            "llm_q25": llm_q25,
            "llm_q75": llm_q75,
            "llm_available": np.mean(np.isfinite(llm), axis=0),
            "llm_leads": llm_leads,
        }
    return result


def write_csv(data: dict[str, dict[str, np.ndarray]], output: Path, n_seeds: int) -> None:
    path = output.with_suffix(".csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "task", "epoch", "rl_best_median", "rl_best_q25", "rl_best_q75",
            "llm_best_median", "llm_best_q25", "llm_best_q75",
            "llm_available_fraction", "llm_leads_fraction", "n_runs",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, values in data.items():
            for index, epoch in enumerate(values["epochs"]):
                writer.writerow(
                    {
                        "task": task,
                        "epoch": int(epoch),
                        "rl_best_median": values["rl_median"][index],
                        "rl_best_q25": values["rl_q25"][index],
                        "rl_best_q75": values["rl_q75"][index],
                        "llm_best_median": values["llm_median"][index],
                        "llm_best_q25": values["llm_q25"][index],
                        "llm_best_q75": values["llm_q75"][index],
                        "llm_available_fraction": values["llm_available"][index],
                        "llm_leads_fraction": values["llm_leads"][index],
                        "n_runs": n_seeds,
                    }
                )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path, n_seeds: int) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharex=True, constrained_layout=True)
    for column, (ax, task) in enumerate(zip(axes, TASKS)):
        values = data[task]
        epoch = values["epochs"]
        for source, label in (("rl", "RL provenance"), ("llm", "LLM provenance")):
            ax.fill_between(
                epoch,
                values[f"{source}_q25"],
                values[f"{source}_q75"],
                color=COLORS[source],
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                epoch,
                values[f"{source}_median"],
                color=COLORS[source],
                linewidth=1.7,
                label=label,
            )
        final_rl = float(values["rl_median"][-1])
        final_llm = float(values["llm_median"][-1])
        final_leads = float(values["llm_leads"][-1])
        ax.text(
            0.98,
            0.96,
            f"final median\nRL {final_rl:.4g}\nLLM {final_llm:.4g}\nLLM leads {100 * final_leads:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#333333",
        )
        ax.set_yscale("log")
        ax.set_xlim(0, int(epoch[-1]))
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        ax.set_xlabel("Training epoch")
        if column == 0:
            ax.set_ylabel("Record-group best-so-far loss")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            -0.12,
            1.03,
            chr(ord("A") + column),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
        )
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        f"Emitting-process best-so-far performance (median and IQR, n = {n_seeds})",
        fontsize=10,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "RL and LLM candidate-origin best-so-far performance"}
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=400 if suffix == ".png" else None,
            facecolor="white",
            metadata=metadata if suffix == ".pdf" else None,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--run-suffix", default="cvode_llm_flash_long300")
    parser.add_argument("--candidate-budget", type=int, default=307200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/llm_rl_source_performance.pdf",
    )
    args = parser.parse_args()
    data = collect(
        args.campaign_root.expanduser().resolve(),
        n_seeds=args.n_seeds,
        run_suffix=args.run_suffix,
        candidate_budget=args.candidate_budget,
    )
    output = args.output.expanduser().resolve()
    plot(data, output, args.n_seeds)
    write_csv(data, output, args.n_seeds)
    for task, values in data.items():
        print(
            f"{TASK_LABELS[task]} final medians: RL={values['rl_median'][-1]:.6g}, "
            f"LLM={values['llm_median'][-1]:.6g}; "
            f"LLM leads in {100 * values['llm_leads'][-1]:.1f}% of runs"
        )


if __name__ == "__main__":
    main()
