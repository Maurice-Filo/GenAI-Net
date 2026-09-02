#!/usr/bin/env python3
"""Generate manuscript-native architecture and model-selection figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
RAW = ROOT / "comparisons/rpa_search/data/raw"
FIGURES = PAPER / "figures"
COLORS = {
    "rl": "#0072B2",
    "llm": "#D55E00",
    "eval": "#009E73",
    "state": "#6F6F6F",
    "pro": "#CC79A7",
}

CONTEXT_COLORS = {"initial_hof": "#D55E00", "context_free": "#009E73"}
CONTEXT_LABELS = {"initial_hof": "Random HOF shown", "context_free": "Random HOF withheld"}


def style() -> None:
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


def box(ax, xy, width, height, text, color, *, subtitle=None) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor="white",
        edgecolor=color,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.58, text, ha="center", va="center", weight="bold")
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.25,
            subtitle,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#555555",
        )


def arrow(ax, start, end, color, label=None, *, bend=0.0, dashed=False) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.35,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={bend}",
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + (0.035 if bend >= 0 else -0.035)
        ax.text(x, y, label, ha="center", va="center", fontsize=6.5, color=color)


def architecture() -> None:
    style()
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.63), 0.17, 0.18, "RL policy", COLORS["rl"], subtitle="1,023 candidates / epoch")
    box(ax, (0.03, 0.14), 0.21, 0.20, "Harness workspace", COLORS["llm"], subtitle="Decider -> Writer; 10 CRNs")
    box(ax, (0.35, 0.39), 0.16, 0.18, "Validation", COLORS["eval"], subtitle="schema and CRN checks")
    box(ax, (0.58, 0.39), 0.16, 0.18, "CVODE evaluator", COLORS["eval"], subtitle="canonical task loss")
    box(ax, (0.80, 0.65), 0.17, 0.17, "Shared HOF", COLORS["state"], subtitle="loss and emitter records")
    box(ax, (0.80, 0.14), 0.17, 0.17, "SIL replay", COLORS["rl"], subtitle="Eq. 3 replay loss")

    arrow(ax, (0.20, 0.72), (0.35, 0.51), COLORS["rl"])
    arrow(ax, (0.24, 0.24), (0.35, 0.43), COLORS["llm"])
    arrow(ax, (0.51, 0.48), (0.58, 0.48), COLORS["eval"])
    arrow(ax, (0.74, 0.52), (0.80, 0.69), COLORS["eval"])
    arrow(ax, (0.885, 0.65), (0.885, 0.31), COLORS["rl"], "replay")
    arrow(
        ax,
        (0.80, 0.75),
        (0.24, 0.31),
        COLORS["state"],
        bend=0.22,
        dashed=True,
    )
    arrow(
        ax,
        (0.80, 0.20),
        (0.20, 0.67),
        COLORS["rl"],
        bend=-0.23,
    )
    ax.text(0.255, 0.69, "RL batch", color=COLORS["rl"], fontsize=6.5, ha="center")
    ax.text(0.285, 0.25, "proposal batch", color=COLORS["llm"], fontsize=6.5, ha="center")
    ax.text(0.545, 0.52, "valid CRNs", color=COLORS["eval"], fontsize=6.5, ha="center")
    ax.text(0.765, 0.61, "evaluated", color=COLORS["eval"], fontsize=6.5, ha="center")
    ax.text(0.52, 0.82, "HOF + SIL status", color=COLORS["state"], fontsize=6.5, ha="center")
    ax.text(0.54, 0.10, "policy gradient", color=COLORS["rl"], fontsize=6.5, ha="center")
    ax.text(
        0.50,
        0.03,
        "Request 0 omits the random HOF; later requests exchange evaluated state without blocking RL.",
        ha="center",
        fontsize=7.2,
        color="#333333",
    )
    fig.tight_layout(pad=0.4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "architecture.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def endpoint(task: str, method: str, suffix: str, seed: int) -> float:
    path = RAW / method / f"{task}_full102400_seed{seed}_{suffix}" / "completed.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["best_loss"])


def rl_endpoint(task: str, seed: int) -> float:
    path = RAW / "rl4crn" / f"{task}_full102400_seed{seed}_cvode" / "progress.csv"
    with path.open(encoding="utf-8") as handle:
        return float(list(csv.DictReader(handle))[-1]["best_so_far_loss"])


def model_selection() -> None:
    style()
    rng = np.random.default_rng(31)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    methods = (
        ("RL only", COLORS["rl"]),
        ("V4 Flash", COLORS["llm"]),
        ("V4 Pro", COLORS["pro"]),
    )
    for ax, task, title in zip(axes, ("logic", "rpa"), ("Logic circuit", "RPA")):
        values = {
            "RL only": np.asarray([rl_endpoint(task, seed) for seed in range(20)]),
            "V4 Flash": np.asarray(
                [endpoint(task, "genai_net_llm", "cvode_llm", seed) for seed in range(20)]
            ),
            "V4 Pro": np.asarray(
                [endpoint(task, "genai_net_llm_pro", "cvode_llm_pro", seed) for seed in range(20)]
            ),
        }
        for index, (label, color) in enumerate(methods):
            data = values[label]
            jitter = rng.uniform(-0.10, 0.10, len(data))
            ax.scatter(
                index + jitter,
                data,
                s=14,
                color=color,
                alpha=0.6,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            q25, median, q75 = np.quantile(data, (0.25, 0.5, 0.75))
            ax.plot([index - 0.16, index + 0.16], [median, median], color="#222222", linewidth=2.0)
            ax.plot([index, index], [q25, q75], color="#222222", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xticks(range(3), [label for label, _ in methods])
        ax.set_ylabel("Final loss (log scale)")
        ax.set_title(title, weight="bold")
        ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hosted-model selection at the matched 102,400-candidate budget", fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "model_selection.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "model_selection.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rpa_initial_context() -> None:
    style()
    source = FIGURES / "rpa_context_ablation.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["task"] == "rpa"]
    max_epoch = max(int(row["epoch"]) for row in rows)
    final = [row for row in rows if int(row["epoch"]) == max_epoch]
    seeds = sorted({int(row["seed"]) for row in final})

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), constrained_layout=True)
    values = {}
    for index, condition in enumerate(("initial_hof", "context_free")):
        values[condition] = np.asarray(
            [
                float(
                    next(
                        row["first_llm_best_loss"]
                        for row in final
                        if row["condition"] == condition and int(row["seed"]) == seed
                    )
                )
                for seed in seeds
            ]
        )
        axes[0].scatter(
            np.full(len(seeds), index),
            values[condition],
            color=CONTEXT_COLORS[condition],
            s=18,
            alpha=0.82,
            zorder=3,
        )
        axes[0].plot(
            index,
            np.median(values[condition]),
            marker="_",
            markersize=15,
            markeredgewidth=2.0,
            color="#202020",
            zorder=4,
        )
    for shown, withheld in zip(values["initial_hof"], values["context_free"]):
        axes[0].plot((0, 1), (shown, withheld), color="#BBBBBB", linewidth=0.55, zorder=1)
    axes[0].set_xticks((0, 1), ("HOF shown", "HOF withheld"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best first-batch loss")
    axes[0].set_title("A   First LLM batch", loc="left", weight="bold")

    for ax, field, title in zip(
        axes[1:],
        ("rl_best_loss", "hof_best_loss"),
        ("B   RL-emitter archive", "C   Joint HOF"),
    ):
        for condition in ("initial_hof", "context_free"):
            epochs = sorted({int(row["epoch"]) for row in rows if row["condition"] == condition})
            matrix = np.asarray(
                [
                    [
                        float(
                            next(
                                row[field]
                                for row in rows
                                if row["condition"] == condition
                                and int(row["seed"]) == seed
                                and int(row["epoch"]) == epoch
                            )
                        )
                        for epoch in epochs
                    ]
                    for seed in seeds
                ]
            )
            q25, median, q75 = np.quantile(matrix, (0.25, 0.5, 0.75), axis=0)
            ax.fill_between(epochs, q25, q75, color=CONTEXT_COLORS[condition], alpha=0.14, linewidth=0)
            ax.plot(epochs, median, color=CONTEXT_COLORS[condition], linewidth=1.7, label=CONTEXT_LABELS[condition])
        ax.set_yscale("log")
        ax.set_xlim(0, max_epoch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Best loss")
        ax.set_title(title, loc="left", weight="bold")

    for ax in axes:
        ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=6.8, loc="upper right")
    for extension in ("pdf", "png", "svg"):
        kwargs = {"dpi": 400} if extension == "png" else {}
        fig.savefig(
            FIGURES / f"rpa_context_ablation.{extension}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    model_selection()
    rpa_initial_context()
