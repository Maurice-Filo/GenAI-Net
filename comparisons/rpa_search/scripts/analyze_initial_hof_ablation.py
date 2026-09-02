#!/usr/bin/env python3
"""Analyze whether exposing the random initial HOF changes the first LLM batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.rpa_search.scripts.plot_communication_ablation_over_time import (
    TASK_LABELS,
    TASKS,
    _latest,
    _through,
    database_path,
    elite_metrics,
    read_hof,
    read_llm_candidates,
)


CONDITIONS = ("initial_hof", "context_free")
LABELS = {
    "initial_hof": "Initial random HOF shown",
    "context_free": "Initial HOF withheld",
}
COLORS = {"initial_hof": "#D55E00", "context_free": "#009E73"}
DEFAULT_SI_REGISTRY = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/historical_request0_hof_registry.json"
)


def completed_seeds(status_path: Path) -> dict[str, set[int]]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result = {task: set() for task in TASKS}
    for row in status.get("completed", []):
        task = str(row.get("task"))
        if task in result:
            result[task].add(int(row["seed"]))
    return result


def _source_counts(snapshot: list[dict], llm_identifiers: set[tuple[str, str]]) -> tuple[int, int]:
    exact_llm = sum(candidate["identifier"] in llm_identifiers for candidate in snapshot)
    return exact_llm, len(snapshot) - exact_llm


def collect_run(path: Path, *, max_epoch: int, elite_size: int) -> list[dict]:
    hof = read_hof(path)
    llm = read_llm_candidates(path)
    llm_identifiers = {candidate["identifier"] for _, candidate in llm}
    first_llm_completion = min((completed for completed, _ in llm), default=None)
    first_llm_batch = [
        candidate for completed, candidate in llm if completed == first_llm_completion
    ]
    rows = []
    for epoch in range(max_epoch + 1):
        snapshot = _latest(hof, epoch)
        rl_archive = [
            candidate
            for candidate in _through(hof, epoch)
            if candidate["identifier"] not in llm_identifiers
        ]
        available_llm = [candidate for completed, candidate in llm if completed <= epoch]
        available_first_llm = (
            first_llm_batch
            if first_llm_completion is not None and first_llm_completion <= epoch
            else []
        )
        llm_metrics = elite_metrics(available_llm, elite_size=max(1, len(available_llm)))
        first_llm_metrics = elite_metrics(
            available_first_llm, elite_size=max(1, len(available_first_llm))
        )
        exact_llm, rl_origin = _source_counts(snapshot, llm_identifiers)
        rows.append(
            {
                "epoch": epoch,
                **{f"hof_{key}": value for key, value in elite_metrics(snapshot, elite_size=elite_size).items()},
                **{f"rl_{key}": value for key, value in elite_metrics(rl_archive, elite_size=elite_size).items()},
                **{f"llm_{key}": value for key, value in llm_metrics.items()},
                **{f"first_llm_{key}": value for key, value in first_llm_metrics.items()},
                "available_llm_candidates": len(available_llm),
                "hof_exact_llm_entries": exact_llm,
                "hof_rl_origin_entries": rl_origin,
            }
        )
    return rows


def collect(args: argparse.Namespace) -> list[dict]:
    roots = {
        "initial_hof": args.initial_hof_root.resolve(),
        "context_free": args.context_free_root.resolve(),
    }
    statuses = {
        "initial_hof": completed_seeds(args.initial_hof_status.resolve()),
        "context_free": completed_seeds(args.context_free_status.resolve()),
    }
    suffixes = {
        "initial_hof": args.initial_hof_suffix,
        "context_free": args.context_free_suffix,
    }
    rows = []
    for task in args.tasks:
        seeds = sorted(statuses["initial_hof"][task] & statuses["context_free"][task])
        for seed in seeds:
            for condition in CONDITIONS:
                path = database_path(
                    roots[condition], task, seed, suffixes[condition], args.candidate_budget
                )
                for row in collect_run(path, max_epoch=args.max_epoch, elite_size=args.elite_size):
                    rows.append({"task": task, "seed": seed, "condition": condition, **row})
    return rows


def _summary(rows: list[dict], task: str, condition: str, field: str):
    subset = [
        row for row in rows if row["task"] == task and row["condition"] == condition
    ]
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
            "pdf.fonttype": 42,
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


def plot_performance(
    rows: list[dict], output: Path, max_epoch: int, tasks: tuple[str, ...]
) -> None:
    _style()
    fig, axes = plt.subplots(
        len(tasks),
        3,
        figsize=(7.2, 0.45 + 2.1 * len(tasks)),
        constrained_layout=True,
        squeeze=False,
    )
    rng = np.random.default_rng(7)
    for row_index, task in enumerate(tasks):
        seeds = sorted({int(row["seed"]) for row in rows if row["task"] == task})
        ax = axes[row_index, 0]
        values = {}
        for index, condition in enumerate(CONDITIONS):
            values[condition] = [
                float(next(
                    row["first_llm_best_loss"]
                    for row in rows
                    if row["task"] == task
                    and row["seed"] == seed
                    and row["condition"] == condition
                    and row["epoch"] == max_epoch
                ))
                for seed in seeds
            ]
            jitter = rng.normal(0.0, 0.025, len(seeds))
            ax.scatter(
                np.full(len(seeds), index) + jitter,
                values[condition],
                color=COLORS[condition],
                s=22,
                alpha=0.8,
                zorder=3,
            )
        for left, right in zip(values["initial_hof"], values["context_free"]):
            ax.plot((0, 1), (left, right), color="#BBBBBB", linewidth=0.6, zorder=1)
        ax.set_xticks((0, 1), ("Random HOF\nshown", "Random HOF\nwithheld"))
        ax.set_yscale("log")
        ax.set_ylabel("Best direct LLM loss")
        ax.set_title(f"{TASK_LABELS[task]}: first LLM batch", fontweight="semibold")

        ax = axes[row_index, 1]
        for condition in CONDITIONS:
            epoch, median, q25, q75 = _summary(rows, task, condition, "rl_best_loss")
            ax.fill_between(epoch, q25, q75, color=COLORS[condition], alpha=0.14, linewidth=0)
            ax.plot(epoch, median, color=COLORS[condition], linewidth=1.7, label=LABELS[condition])
        ax.set_yscale("log")
        ax.set_xlim(0, max_epoch)
        ax.set_xlabel("Epoch" if row_index == len(tasks) - 1 else "")
        ax.set_ylabel("Best RL-provenance loss")
        ax.set_title("RL search", fontweight="semibold")

        ax = axes[row_index, 2]
        for condition in CONDITIONS:
            epoch, median, q25, q75 = _summary(rows, task, condition, "hof_best_loss")
            ax.fill_between(epoch, q25, q75, color=COLORS[condition], alpha=0.14, linewidth=0)
            ax.plot(epoch, median, color=COLORS[condition], linewidth=1.7)
            insertion_epochs = []
            for seed in seeds:
                run = [
                    row
                    for row in rows
                    if row["task"] == task
                    and int(row["seed"]) == seed
                    and row["condition"] == condition
                ]
                inserted = next(
                    (
                        int(row["epoch"])
                        for row in run
                        if int(row["available_llm_candidates"]) > 0
                    ),
                    None,
                )
                if inserted is not None:
                    insertion_epochs.append(inserted)
            if insertion_epochs:
                ax.axvline(
                    float(np.median(insertion_epochs)),
                    color=COLORS[condition],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
        ax.set_yscale("log")
        ax.set_xlim(0, max_epoch)
        ax.set_xlabel("Epoch" if row_index == len(tasks) - 1 else "")
        ax.set_ylabel("Best joint-HOF loss")
        ax.set_title("Joint archive", fontweight="semibold")

        for column in range(3):
            panel = axes[row_index, column]
            panel.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            panel.spines[["top", "right"]].set_visible(False)
            panel.text(
                0.01,
                0.98,
                chr(ord("A") + row_index * 3 + column),
                transform=panel.transAxes,
                va="top",
                fontweight="bold",
                fontsize=9,
            )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, "Initial Hall-of-Fame context ablation mechanism")


def plot_rl_dynamics(
    rows: list[dict], output: Path, max_epoch: int, tasks: tuple[str, ...]
) -> None:
    """Show every paired RL trajectory and isolate any short-horizon SIL response."""
    _style()
    fig, axes = plt.subplots(
        len(tasks),
        2,
        figsize=(7.2, 0.45 + 2.1 * len(tasks)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, task in enumerate(tasks):
        seeds = sorted({int(row["seed"]) for row in rows if row["task"] == task})
        lookup = {
            (int(row["seed"]), row["condition"], int(row["epoch"])): float(row["rl_best_loss"])
            for row in rows
            if row["task"] == task
        }

        ax = axes[row_index, 0]
        for condition in CONDITIONS:
            for seed in seeds:
                values = [lookup[(seed, condition, epoch)] for epoch in range(max_epoch + 1)]
                ax.plot(
                    range(max_epoch + 1),
                    values,
                    color=COLORS[condition],
                    linewidth=0.65,
                    alpha=0.3,
                )
            epoch, median, _, _ = _summary(rows, task, condition, "rl_best_loss")
            ax.plot(
                epoch,
                median,
                color=COLORS[condition],
                linewidth=2.0,
                label=LABELS[condition],
            )
        ax.set_yscale("log")
        ax.set_xlim(0, max_epoch)
        ax.set_xlabel("Epoch" if row_index == len(tasks) - 1 else "")
        ax.set_ylabel("Best RL-provenance loss")
        ax.set_title(f"{TASK_LABELS[task]}: paired RL trajectories", fontweight="semibold")

        ratio_ax = axes[row_index, 1]
        ratio_values = []
        for seed in seeds:
            ratio = np.asarray(
                [
                    lookup[(seed, "context_free", epoch)]
                    / lookup[(seed, "initial_hof", epoch)]
                    for epoch in range(max_epoch + 1)
                ]
            )
            ratio_values.append(ratio)
            ratio_ax.plot(
                range(max_epoch + 1),
                ratio,
                color="#777777",
                linewidth=0.7,
                alpha=0.45,
            )
        ratio_values = np.asarray(ratio_values)
        median = np.median(ratio_values, axis=0)
        q25, q75 = np.percentile(ratio_values, (25, 75), axis=0)
        epochs = np.arange(max_epoch + 1)
        ratio_ax.fill_between(epochs, q25, q75, color="#009E73", alpha=0.16, linewidth=0)
        ratio_ax.plot(epochs, median, color="#009E73", linewidth=2.0)
        ratio_ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
        ratio_ax.set_xlim(0, max_epoch)
        ratio_ax.set_xlabel("Epoch" if row_index == len(tasks) - 1 else "")
        ratio_ax.set_ylabel("RL loss ratio: withheld / shown")
        ratio_ax.set_title("Paired RL-stream effect", fontweight="semibold")
        ratio_ax.text(
            0.98,
            0.06,
            "< 1 favors withholding",
            transform=ratio_ax.transAxes,
            ha="right",
            color="#555555",
            fontsize=6.3,
        )

        for column in range(2):
            panel = axes[row_index, column]
            panel.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            panel.spines[["top", "right"]].set_visible(False)
            panel.text(
                0.01,
                0.98,
                chr(ord("A") + row_index * 2 + column),
                transform=panel.transAxes,
                va="top",
                fontweight="bold",
                fontsize=9,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _save(fig, output, "Short-horizon RL dynamics in the initial-HOF ablation")


def plot_diversity(
    rows: list[dict], output: Path, max_epoch: int, tasks: tuple[str, ...]
) -> None:
    _style()
    fig, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.6 * len(tasks), 5.0),
        constrained_layout=True,
        squeeze=False,
    )
    fields = (
        ("llm_effective_topologies", "LLM batch"),
        ("rl_effective_topologies", "RL-provenance archive"),
    )
    offsets = np.linspace(-0.24, 0.24, len(fields))
    for panel_index, task in enumerate(tasks):
        ax = axes[0, panel_index]
        subset = [row for row in rows if row["task"] == task and row["epoch"] == max_epoch]
        for offset, (field, label) in zip(offsets, fields):
            medians = []
            lower = []
            upper = []
            for condition in CONDITIONS:
                values = np.asarray(
                    [float(row[field]) for row in subset if row["condition"] == condition]
                )
                median = float(np.nanmedian(values))
                q25, q75 = np.nanpercentile(values, (25, 75))
                medians.append(median)
                lower.append(median - q25)
                upper.append(q75 - median)
            positions = np.arange(len(CONDITIONS)) + offset
            ax.errorbar(
                positions,
                medians,
                yerr=np.asarray((lower, upper)),
                marker="o",
                markersize=5,
                linewidth=1.3,
                capsize=2.5,
                label=label,
            )
        ax.set_xticks(np.arange(len(CONDITIONS)), ("Random HOF", "HOF withheld"))
        ax.set_ylabel("Effective topologies")
        ax.set_title(f"{TASK_LABELS[task]} source diversity", fontweight="semibold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.01, 0.98, chr(ord("A") + panel_index), transform=ax.transAxes, va="top", fontweight="bold")

        source_ax = axes[1, panel_index]
        exact_llm = []
        rl_origin = []
        for condition in CONDITIONS:
            llm_values = [
                float(row["hof_exact_llm_entries"])
                for row in subset
                if row["condition"] == condition
            ]
            rl_values = [
                float(row["hof_rl_origin_entries"])
                for row in subset
                if row["condition"] == condition
            ]
            exact_llm.append(float(np.median(llm_values)))
            rl_origin.append(float(np.median(rl_values)))
        positions = np.arange(len(CONDITIONS))
        source_ax.bar(positions, exact_llm, color="#0072B2", label="LLM-provenance entries")
        source_ax.bar(
            positions,
            rl_origin,
            bottom=exact_llm,
            color="#E69F00",
            label="RL-provenance entries",
        )
        source_ax.set_xticks(positions, ("Random HOF", "HOF withheld"))
        source_ax.set_ylim(0, 30)
        source_ax.set_ylabel("Median final HOF entries")
        source_ax.set_title(f"{TASK_LABELS[task]} final source composition", fontweight="semibold")
        source_ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        source_ax.spines[["top", "right"]].set_visible(False)
        source_ax.text(
            0.01,
            0.98,
            chr(ord("C") + panel_index),
            transform=source_ax.transAxes,
            va="top",
            fontweight="bold",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    source_handles, source_labels = axes[1, 0].get_legend_handles_labels()
    axes[1, -1].legend(source_handles, source_labels, frameon=False, loc="lower right")
    _save(fig, output, "Initial Hall-of-Fame context diversity")


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_paired_tests(rows: list[dict], max_epoch: int, tasks: tuple[str, ...]) -> None:
    for task in tasks:
        final = [row for row in rows if row["task"] == task and row["epoch"] == max_epoch]
        seeds = sorted({int(row["seed"]) for row in final})
        print(f"{TASK_LABELS[task]}: {len(seeds)} paired runs")
        fields = [
            ("first_llm_best_loss", "first direct LLM batch"),
            ("llm_best_loss", "all direct LLM batches"),
            ("rl_best_loss", "RL provenance"),
        ]
        for field, label in fields:
            values = {
                condition: [
                    float(next(
                        row[field]
                        for row in final
                        if row["condition"] == condition and row["seed"] == seed
                    ))
                    for seed in seeds
                ]
                for condition in CONDITIONS
            }
            pairs = [
                (left, right)
                for left, right in zip(values["initial_hof"], values["context_free"])
                if np.isfinite(left) and np.isfinite(right)
            ]
            pvalue = (
                wilcoxon(
                    [left for left, _ in pairs],
                    [right for _, right in pairs],
                ).pvalue
                if pairs and any(left != right for left, right in pairs)
                else float("nan")
            )
            print(
                f"  {label}: random-HOF median={np.median(values['initial_hof']):.6g}; "
                f"withheld median={np.median(values['context_free']):.6g}; p={pvalue:.6g}"
            )


def protocol_selection_rows(
    rows: list[dict], max_epoch: int, tasks: tuple[str, ...]
) -> list[dict]:
    """Return only the request-0 conditioning endpoint allowed by the SI registry."""

    return [
        {
            "evidence_scope": "protocol-selection-only",
            "task": row["task"],
            "seed": row["seed"],
            "condition": row["condition"],
            "first_request_best_loss": row["first_llm_best_loss"],
        }
        for row in rows
        if row["task"] in tasks and int(row["epoch"]) == int(max_epoch)
    ]


def plot_protocol_selection(
    rows: list[dict], output: Path, tasks: tuple[str, ...]
) -> None:
    """Plot only the first-request paired diagnostic; no method endpoint is shown."""

    _style()
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.1 * len(tasks), 2.4),
        constrained_layout=True,
        squeeze=False,
    )
    rng = np.random.default_rng(7)
    for task_index, task in enumerate(tasks):
        axis = axes[0, task_index]
        seeds = sorted({int(row["seed"]) for row in rows if row["task"] == task})
        values = {}
        for condition_index, condition in enumerate(CONDITIONS):
            values[condition] = [
                float(
                    next(
                        row["first_request_best_loss"]
                        for row in rows
                        if row["task"] == task
                        and int(row["seed"]) == seed
                        and row["condition"] == condition
                    )
                )
                for seed in seeds
            ]
            axis.scatter(
                np.full(len(seeds), condition_index) + rng.normal(0, 0.02, len(seeds)),
                values[condition],
                color=COLORS[condition],
                s=24,
                zorder=3,
            )
        for shown, withheld in zip(values["initial_hof"], values["context_free"]):
            axis.plot((0, 1), (shown, withheld), color="#BBBBBB", linewidth=0.7)
        axis.set_xticks((0, 1), ("Random HOF\nshown", "Random HOF\nwithheld"))
        axis.set_yscale("log")
        axis.set_ylabel("Best loss in request 0")
        axis.set_title(TASK_LABELS[task], fontweight="semibold")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    _save(fig, output, "SI protocol-selection-only request-0 HOF diagnostic")


def print_protocol_selection_tests(rows: list[dict], tasks: tuple[str, ...]) -> None:
    for task in tasks:
        subset = [row for row in rows if row["task"] == task]
        seeds = sorted({int(row["seed"]) for row in subset})
        shown = [
            float(next(row["first_request_best_loss"] for row in subset if row["seed"] == seed and row["condition"] == "initial_hof"))
            for seed in seeds
        ]
        withheld = [
            float(next(row["first_request_best_loss"] for row in subset if row["seed"] == seed and row["condition"] == "context_free"))
            for seed in seeds
        ]
        pvalue = (
            float(wilcoxon(shown, withheld).pvalue)
            if any(left != right for left, right in zip(shown, withheld))
            else float("nan")
        )
        print(
            f"{TASK_LABELS[task]} request-0 protocol diagnostic: n={len(seeds)}; "
            f"shown median={np.median(shown):.6g}; "
            f"withheld median={np.median(withheld):.6g}; p={pvalue:.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_SI_REGISTRY)
    parser.add_argument(
        "--evidence-scope",
        choices=("protocol-selection-only",),
        required=True,
    )
    parser.add_argument("--elite-size", type=int, default=30)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.resolve().read_text(encoding="utf-8"))
    if registry.get("evidence_scope") != args.evidence_scope:
        raise RuntimeError("Historical request-0 registry has the wrong evidence scope.")
    quarantine_root = Path(registry["quarantine_root"]).resolve()
    if "quarantine" not in quarantine_root.parts:
        raise RuntimeError("Historical request-0 evidence must remain under quarantine.")
    initial = registry["conditions"]["initial_hof"]
    context_free = registry["conditions"]["context_free"]
    args.initial_hof_root = quarantine_root / initial["raw_root"]
    args.context_free_root = quarantine_root / context_free["raw_root"]
    args.initial_hof_status = quarantine_root / initial["status"]
    args.context_free_status = quarantine_root / context_free["status"]
    args.initial_hof_suffix = initial["run_suffix"]
    args.context_free_suffix = context_free["run_suffix"]
    args.candidate_budget = int(registry["candidate_budget"])
    args.max_epoch = int(registry["epochs"])
    args.tasks = tuple(args.tasks or registry["tasks"])
    rows = collect(args)
    if not rows:
        raise RuntimeError("No completed paired initial-HOF ablation runs were found")
    output = args.output.resolve()
    tasks = tuple(args.tasks)
    selected = protocol_selection_rows(rows, args.max_epoch, tasks)
    write_csv(selected, output.with_suffix(".csv"))
    plot_protocol_selection(selected, output, tasks)
    output.with_suffix(".scope.json").write_text(
        json.dumps(
            {
                "evidence_scope": args.evidence_scope,
                "registry": str(args.registry.resolve()),
                "mandatory_disclosure": registry["mandatory_disclosure"],
                "permitted_claim": registry["permitted_claim"],
                "forbidden_uses": registry["forbidden_uses"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print_protocol_selection_tests(selected, tasks)


if __name__ == "__main__":
    main()
