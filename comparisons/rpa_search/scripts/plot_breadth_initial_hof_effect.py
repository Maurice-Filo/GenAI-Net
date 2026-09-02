#!/usr/bin/env python3
"""Plot paired endpoint effects of withholding the initial random HOF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
TASKS = (
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "classifier",
    "oscillator_mean",
    "oscillator_frequency",
)
TASK_LABELS = {
    "dose_hill": "Hill response",
    "dose_ultrasensitive": "Ultrasensitive",
    "dose_biphasic": "Biphasic",
    "classifier": "Classifier",
    "oscillator_mean": "Oscillator mean",
    "oscillator_frequency": "Oscillator frequency",
}
ARMS = {
    "withheld": (
        "genai_net_llm_flash_breadth_initial_context_free20",
        "cvode_llm_flash_breadth_initial_context_free20",
    ),
    "shown": ("genai_net_llm_flash_breadth", "cvode_llm_flash_breadth"),
    "rl": ("rl4crn_breadth", "cvode_rl_only_breadth"),
}
WITHHELD_COLOR = "#009E73"
RL_COLOR = "#0072B2"
SHOWN_COLOR = "#D55E00"
LLM_COLOR = "#0072B2"
SHORT_TASKS = ("rpa", "logic")
SHORT_TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}


def endpoint(task: str, seed: int, arm: str, raw_root: Path) -> float:
    method, suffix = ARMS[arm]
    path = (
        raw_root
        / method
        / f"{task}_full102400_seed{seed}_{suffix}"
        / "completed.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing matched endpoint: {path}")
    return float(json.loads(path.read_text(encoding="utf-8"))["best_loss"])


def collect(raw_root: Path) -> list[dict]:
    rows = []
    for task in TASKS:
        for control in ("rl", "shown"):
            for seed in range(20):
                method, suffix = ARMS[control]
                control_path = (
                    raw_root
                    / method
                    / f"{task}_full102400_seed{seed}_{suffix}"
                    / "completed.json"
                )
                withheld_method, withheld_suffix = ARMS["withheld"]
                withheld_path = (
                    raw_root
                    / withheld_method
                    / f"{task}_full102400_seed{seed}_{withheld_suffix}"
                    / "completed.json"
                )
                if not control_path.is_file() or not withheld_path.is_file():
                    continue
                withheld = endpoint(task, seed, "withheld", raw_root)
                control_loss = endpoint(task, seed, control, raw_root)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "control": control,
                        "withheld_loss": withheld,
                        "control_loss": control_loss,
                        "fold_advantage": control_loss / withheld,
                        "withheld_wins": int(withheld < control_loss),
                    }
                )
    return rows


def collect_short_diagnostic(raw_root: Path, summary_csv: Path) -> list[dict]:
    with summary_csv.open(encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    first_batch = {
        (row["task"], int(row["seed"]), row["condition"]): float(
            row.get("first_llm_best_loss", row["llm_best_loss"])
        )
        for row in summary_rows
        if int(row["epoch"]) == 20 and row["task"] in SHORT_TASKS
    }
    rows = []
    for task in SHORT_TASKS:
        for seed in range(5):
            final = {}
            for condition, method, suffix in (
                (
                    "shown",
                    "genai_net_llm_flash_initial_hof",
                    "cvode_flash_initial_hof",
                ),
                (
                    "withheld",
                    "genai_net_llm_flash_initial_context_free",
                    "cvode_flash_initial_context_free",
                ),
            ):
                path = (
                    raw_root
                    / method
                    / f"{task}_full20480_seed{seed}_{suffix}"
                    / "completed.json"
                )
                final[condition] = float(
                    json.loads(path.read_text(encoding="utf-8"))["best_loss"]
                )
            rows.extend(
                (
                    {
                        "task": task,
                        "seed": seed,
                        "metric": "first_llm_batch",
                        "withheld_loss": first_batch[(task, seed, "context_free")],
                        "shown_loss": first_batch[(task, seed, "initial_hof")],
                    },
                    {
                        "task": task,
                        "seed": seed,
                        "metric": "joint_hof",
                        "withheld_loss": final["withheld"],
                        "shown_loss": final["shown"],
                    },
                )
            )
    for row in rows:
        row["fold_advantage"] = row["shown_loss"] / row["withheld_loss"]
        row["withheld_wins"] = int(row["withheld_loss"] < row["shown_loss"])
    return rows


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def fold_label(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}x"
    return f"{value:.2g}x"


def draw_panel(ax, rows: list[dict], control: str, panel: str, title: str) -> None:
    rng = np.random.default_rng(17 if control == "rl" else 29)
    for index, task in enumerate(TASKS):
        subset = [row for row in rows if row["task"] == task and row["control"] == control]
        values = np.asarray([float(row["fold_advantage"]) for row in subset])
        jitter = rng.uniform(-0.115, 0.115, len(values))
        ax.scatter(
            values,
            index + jitter,
            s=22,
            color=WITHHELD_COLOR,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            zorder=3,
        )
        low, median, high = np.quantile(values, (0.25, 0.5, 0.75))
        ax.plot([low, high], [index, index], color="#252525", linewidth=2.1, zorder=4)
        ax.scatter(
            [median],
            [index],
            marker="D",
            s=34,
            color="#252525",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        wins = sum(int(row["withheld_wins"]) for row in subset)
        ax.annotate(
            f"{fold_label(median)}  ({wins}/{len(values)})",
            (median, index),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=6.7,
            color="#303030",
            va="top",
        )

    ax.axvspan(1.0, 400.0, color=WITHHELD_COLOR, alpha=0.055, linewidth=0)
    ax.axvline(1.0, color="#303030", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.025, 400)
    ticks = (0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300)
    ax.set_xticks(ticks, [f"{value:g}x" for value in ticks])
    ax.set_yticks(range(len(TASKS)), [TASK_LABELS[task] for task in TASKS])
    ax.set_ylim(len(TASKS) - 0.45, -0.55)
    ax.grid(axis="x", which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"{panel}   {title}", loc="left", fontweight="bold")
    ax.set_xlabel("Fold advantage of initial-HOF withholding (log scale)")


def draw_short_panel(ax, rows: list[dict]) -> None:
    rng = np.random.default_rng(41)
    metrics = (
        ("first_llm_batch", "First LLM batch", LLM_COLOR, -0.13, "o"),
        ("joint_hof", "Joint HOF, epoch 20", WITHHELD_COLOR, 0.13, "s"),
    )
    for task_index, task in enumerate(SHORT_TASKS):
        for metric, _label, color, offset, marker in metrics:
            subset = [
                row for row in rows if row["task"] == task and row["metric"] == metric
            ]
            values = np.asarray([float(row["fold_advantage"]) for row in subset])
            jitter = rng.uniform(-0.045, 0.045, len(values))
            position = task_index + offset
            ax.scatter(
                values,
                position + jitter,
                s=24,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.8,
                zorder=3,
            )
            low, median, high = np.quantile(values, (0.25, 0.5, 0.75))
            ax.plot([low, high], [position, position], color=color, linewidth=2.3, zorder=4)
            ax.scatter(
                [median],
                [position],
                marker="D",
                s=32,
                color="#252525",
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )
            wins = sum(int(row["withheld_wins"]) for row in subset)
            ax.annotate(
                f"{fold_label(median)} ({wins}/5)",
                (median, position),
                xytext=(5, -9 if offset < 0 else 7),
                textcoords="offset points",
                fontsize=6.6,
                color="#303030",
                va="top" if offset < 0 else "bottom",
            )

    ax.axvspan(1.0, 400.0, color=WITHHELD_COLOR, alpha=0.055, linewidth=0)
    ax.axvline(1.0, color="#303030", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.025, 400)
    ticks = (0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300)
    ax.set_xticks(ticks, [f"{value:g}x" for value in ticks])
    ax.set_yticks(range(len(SHORT_TASKS)), [SHORT_TASK_LABELS[task] for task in SHORT_TASKS])
    ax.set_ylim(len(SHORT_TASKS) - 0.45, -0.55)
    ax.grid(axis="x", which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title("C   Prior 20-epoch diagnostic", loc="left", fontweight="bold")
    ax.set_xlabel("Fold advantage of initial-HOF withholding (log scale)")


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument(
        "--short-diagnostic-csv",
        type=Path,
        default=ROOT
        / "comparisons/rpa_search/figures/initial_hof_ablation_20epoch_5seed.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/breadth_initial_hof_withheld_effect",
    )
    args = parser.parse_args()
    rows = collect(args.raw_root.resolve())
    short_rows = collect_short_diagnostic(
        args.raw_root.resolve(), args.short_diagnostic_csv.resolve()
    )
    style()
    fig = plt.figure(figsize=(15.2, 5.15))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.78), wspace=0.16)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    axes[1].sharey(axes[0])
    short_ax = fig.add_subplot(grid[0, 2])
    draw_panel(axes[0], rows, "rl", "A", "Versus RL-only")
    draw_panel(axes[1], rows, "shown", "B", "Versus HOF-exposed Harness")
    draw_short_panel(short_ax, short_rows)
    fig.suptitle(
        "Magnitude of the initial-HOF-withholding effect",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.925,
        "Endpoint loss ratios on matched seeds; values above 1x favor withholding",
        ha="left",
        fontsize=8.5,
        color="#4A4A4A",
    )
    legend = [
        Line2D([0], [0], marker="o", linestyle="", color=WITHHELD_COLOR,
               markeredgecolor="white", label="Matched seed"),
        Line2D([0], [0], marker="D", linestyle="", color="#252525",
               markeredgecolor="white", label="Median; bar is IQR"),
        Line2D([0], [0], marker="o", linestyle="", color=LLM_COLOR,
               markeredgecolor="white", label="Prior diagnostic: first LLM batch"),
        Line2D([0], [0], marker="s", linestyle="", color=WITHHELD_COLOR,
               markeredgecolor="white", label="Prior diagnostic: joint HOF"),
    ]
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
    )
    fig.text(
        0.505,
        0.015,
        "A: 100 epochs, matched n = 20 per task after control completion. B: available HOF-exposed pairs (n = 5; classifier n = 10). C: prior 20-epoch diagnostic, n = 5. Labels show median and withheld wins. Lower loss is better.",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.14)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(
            args.output.with_suffix(suffix),
            dpi=400 if suffix == ".png" else None,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)
    save_csv(rows, args.output.with_suffix(".csv"))
    save_csv(short_rows, args.output.with_name(args.output.name + "_short_diagnostic").with_suffix(".csv"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
