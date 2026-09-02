#!/usr/bin/env python3
"""Relate final-policy endpoint quality to reaction-set diversity across tasks."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "comparisons/rpa_search/data/raw"
ENDPOINTS = ROOT / "paper/iclr2027_genai_net_llm/generated/primary_results_endpoints.csv"
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
LABELS = {
    "rpa": "RPA",
    "logic": "Logic",
    "classifier": "Classifier",
    "dose_hill": "Hill",
    "dose_ultrasensitive": "Ultrasensitive",
    "dose_biphasic": "Biphasic",
    "oscillator_mean": "Oscillator mean",
    "oscillator_frequency": "Oscillator frequency",
}
FIXED_REACTIONS = {
    "rpa": 2,
    "logic": 4,
    "classifier": 0,
    "dose_hill": 1,
    "dose_ultrasensitive": 1,
    "dose_biphasic": 1,
    "oscillator_mean": 0,
    "oscillator_frequency": 1,
}
HYBRID_SPECIAL = {
    "rpa": (
        "genai_net_llm_flash_rpa_context_free100",
        "rpa_full307200_seed{seed}_cvode_flash_rpa_context_free100",
    ),
    "logic": (
        "genai_net_llm_flash_logic_initial_context_free100",
        "logic_full102400_seed{seed}_cvode_llm_flash_logic_initial_context_free100",
    ),
}
RL_COLOR = "#0072B2"
HYBRID_COLOR = "#009E73"


def network_path(task: str, method: str, seed: int) -> Path:
    if method == "rl":
        if task in ("rpa", "logic"):
            run = f"{task}_full102400_seed{seed}_cvode"
            return RAW / "rl4crn" / run / "best_network.json"
        run = f"{task}_full102400_seed{seed}_cvode_rl_only_breadth"
        return RAW / "rl4crn_breadth" / run / "best_network.json"
    if task in HYBRID_SPECIAL:
        method_name, run_pattern = HYBRID_SPECIAL[task]
        return RAW / method_name / run_pattern.format(seed=seed) / "best_network.json"
    run = f"{task}_full102400_seed{seed}_cvode_llm_flash_breadth_initial_context_free20"
    return (
        RAW
        / "genai_net_llm_flash_breadth_initial_context_free20"
        / run
        / "best_network.json"
    )


def load_endpoints(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        result[(row["task"], int(row["seed"]))] = {
            "rl": float(row["rl_loss"]),
            "hybrid": float(row["hybrid_loss"]),
            "fold": float(row["rl_over_hybrid"]),
        }
    return result


def topology(task: str, method: str, seed: int) -> frozenset[int]:
    path = network_path(task, method, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(value) for value in data["reaction_ids"]]
    return frozenset(ids[FIXED_REACTIONS[task] :])


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def diversity(topologies: list[frozenset[int]]) -> dict[str, float]:
    distances = [jaccard(left, right) for left, right in combinations(topologies, 2)]
    return {
        "unique": len(set(topologies)),
        "mean_jaccard": float(np.mean(distances)),
        "median_jaccard": float(np.median(distances)),
        "reaction_richness": len(set().union(*topologies)),
    }


def collect(endpoint_path: Path, top_k: int) -> list[dict]:
    endpoints = load_endpoints(endpoint_path)
    rows = []
    for task in TASKS:
        by_method = {}
        for method in ("rl", "hybrid"):
            values = [
                {
                    "seed": seed,
                    "loss": endpoints[(task, seed)][method],
                    "topology": topology(task, method, seed),
                }
                for seed in range(20)
            ]
            by_method[method] = {
                "all": diversity([row["topology"] for row in values]),
                "top": diversity(
                    [row["topology"] for row in sorted(values, key=lambda row: row["loss"])[:top_k]]
                ),
            }
        folds = [endpoints[(task, seed)]["fold"] for seed in range(20)]
        row = {
            "task": task,
            "n": 20,
            "top_k": top_k,
            "median_paired_rl_over_hybrid": float(np.median(folds)),
        }
        for method in ("rl", "hybrid"):
            for scope in ("all", "top"):
                for metric, value in by_method[method][scope].items():
                    row[f"{method}_{scope}_{metric}"] = value
        row["all_mean_jaccard_delta"] = (
            row["hybrid_all_mean_jaccard"] - row["rl_all_mean_jaccard"]
        )
        row["top_mean_jaccard_delta"] = (
            row["hybrid_top_mean_jaccard"] - row["rl_top_mean_jaccard"]
        )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axes[0]
    for row in rows:
        x = np.log10(float(row["median_paired_rl_over_hybrid"]))
        y = float(row["all_mean_jaccard_delta"])
        ax.scatter(x, y, s=38, color=HYBRID_COLOR, edgecolor="white", linewidth=0.5, zorder=3)
        offsets = {
            "rpa": (7, 7),
            "logic": (12, 13),
            "classifier": (-62, 9),
            "dose_hill": (20, 5),
            "dose_ultrasensitive": (20, -22),
            "dose_biphasic": (20, 10),
            "oscillator_mean": (44, 12),
            "oscillator_frequency": (22, -27),
        }
        horizontal_alignment = "right" if row["task"] == "classifier" else "left"
        ax.annotate(
            LABELS[row["task"]],
            (x, y),
            xytext=offsets[row["task"]],
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
            ha=horizontal_alignment,
            arrowprops={"arrowstyle": "-", "color": "#AAAAAA", "linewidth": 0.45},
        )
    ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"Endpoint quality effect, $\log_{10}(L_{RL}/L_{mixed})$")
    ax.set_ylabel("Mixed - RL mean Jaccard distance")
    ax.set_title("A   Quality and structural diversity", loc="left", fontweight="bold")
    ax.grid(color="#E0E0E0", linewidth=0.5)
    ax.set_xlim(-0.12, 2.15)
    ax.set_ylim(-0.22, 0.07)
    ax.text(0.98, 0.97, "better loss / more diverse", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color="#555555")
    ax.text(0.98, 0.03, "better loss / less diverse", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, color="#555555")

    ax = axes[1]
    y_positions = np.arange(len(TASKS))
    for index, row in enumerate(rows):
        rl = float(row["rl_top_mean_jaccard"])
        hybrid = float(row["hybrid_top_mean_jaccard"])
        ax.plot((rl, hybrid), (index, index), color="#A8A8A8", linewidth=1.0, zorder=1)
        ax.scatter(
            rl,
            index,
            s=28,
            color=RL_COLOR,
            edgecolor="white",
            linewidth=0.45,
            label="RL" if index == 0 else None,
            zorder=2,
        )
        ax.scatter(
            hybrid,
            index,
            s=28,
            color=HYBRID_COLOR,
            edgecolor="white",
            linewidth=0.45,
            label="Mixed" if index == 0 else None,
            zorder=2,
        )
        unique_rl = int(row["rl_top_unique"])
        unique_hybrid = int(row["hybrid_top_unique"])
        ax.annotate(
            f"{unique_rl}/{unique_hybrid}",
            (max(rl, hybrid), index),
            xytext=(5, -1),
            textcoords="offset points",
            fontsize=6.2,
            color="#444444",
            va="center",
        )
    ax.set_yticks(y_positions, [LABELS[task] for task in TASKS])
    ax.set_ylim(len(TASKS) - 0.5, -0.5)
    ax.set_xlim(0.15, 0.96)
    ax.set_xlabel("Mean Jaccard distance, best 10 endpoints")
    ax.set_title("B   Quality-conditioned diversity", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    ax.text(
        0.98,
        0.98,
        "labels show unique RL/mixed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color="#555555",
    )
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(
            output.with_suffix(suffix),
            dpi=400 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=Path, default=ENDPOINTS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/final_quality_diversity_20seed",
    )
    args = parser.parse_args()
    if not 2 <= args.top_k <= 20:
        parser.error("--top-k must be between 2 and 20")
    rows = collect(args.endpoints.resolve(), args.top_k)
    write_csv(rows, args.output.with_suffix(".csv"))
    plot(rows, args.output)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
