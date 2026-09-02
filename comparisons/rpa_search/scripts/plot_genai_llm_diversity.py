#!/usr/bin/env python3
"""Compare final-network structural diversity for GenAI-Net and GenAI-Net-LLM."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
METHODS = ("rl4crn", "genai_net_llm")
METHOD_LABELS = {"rl4crn": "GenAI-Net", "genai_net_llm": "GenAI-Net-LLM"}
METHOD_COLORS = {"rl4crn": "#0072B2", "genai_net_llm": "#009E73"}


def run_id(task: str, method: str, seed: int) -> str:
    suffix = "_cvode" if method == "rl4crn" else "_cvode_llm"
    return f"{task}_full102400_seed{seed}{suffix}"


def fixed_reaction_count(task: str) -> int:
    return 2 if task == "rpa" else 4


def load_topologies(n_seeds: int) -> dict[str, dict[str, list[frozenset[int]]]]:
    result: dict[str, dict[str, list[frozenset[int]]]] = {}
    for task in TASKS:
        result[task] = {}
        for method in METHODS:
            topologies = []
            for seed in range(n_seeds):
                path = RAW_ROOT / method / run_id(task, method, seed) / "best_network.json"
                if not path.is_file():
                    raise RuntimeError(f"Missing final network: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                reaction_ids = [int(value) for value in data.get("reaction_ids", [])]
                topologies.append(frozenset(reaction_ids[fixed_reaction_count(task) :]))
            result[task][method] = topologies
    return result


def load_final_losses(n_seeds: int) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for task in TASKS:
        result[task] = {}
        for method in METHODS:
            losses = []
            for seed in range(n_seeds):
                path = RAW_ROOT / method / run_id(task, method, seed) / "progress.csv"
                if not path.is_file():
                    raise RuntimeError(f"Missing progress data: {path}")
                with path.open("r", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows:
                    raise RuntimeError(f"Empty progress data: {path}")
                losses.append(float(rows[-1]["best_so_far_loss"]))
            result[task][method] = losses
    return result


def jaccard_distance(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def accumulation_distribution(
    topologies: list[frozenset[int]], *, permutations: int, rng: np.random.Generator
) -> np.ndarray:
    counts = np.empty((permutations, len(topologies)), dtype=float)
    indexes = np.arange(len(topologies))
    for row in range(permutations):
        discovered: set[frozenset[int]] = set()
        for column, index in enumerate(rng.permutation(indexes)):
            discovered.add(topologies[int(index)])
            counts[row, column] = len(discovered)
    return counts


def summarize(
    topologies: dict[str, dict[str, list[frozenset[int]]]],
    losses: dict[str, dict[str, list[float]]],
    output: Path,
) -> None:
    summary_path = output.with_suffix(".csv")
    pairwise_path = output.with_name(f"{output.stem}_pairwise.csv")
    runs_path = output.with_name(f"{output.stem}_runs.csv")
    summary_rows = []
    pairwise_rows = []
    run_rows = []
    for task in TASKS:
        for method in METHODS:
            values = topologies[task][method]
            distances = [jaccard_distance(a, b) for a, b in combinations(values, 2)]
            novelty = [
                float(
                    np.mean(
                        [
                            jaccard_distance(topology, other)
                            for other_seed, other in enumerate(values)
                            if other_seed != seed
                        ]
                    )
                )
                for seed, topology in enumerate(values)
            ]
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "n_runs": len(values),
                    "unique_topologies": len(set(values)),
                    "unique_topology_fraction": len(set(values)) / len(values),
                    "reaction_richness": len(set().union(*values)),
                    "mean_pairwise_jaccard_distance": float(np.mean(distances)),
                    "median_pairwise_jaccard_distance": float(np.median(distances)),
                    "pairwise_q25": float(np.percentile(distances, 25)),
                    "pairwise_q75": float(np.percentile(distances, 75)),
                    "mean_final_loss": float(np.mean(losses[task][method])),
                    "median_final_loss": float(np.median(losses[task][method])),
                }
            )
            for seed, (loss, mean_distance) in enumerate(zip(losses[task][method], novelty)):
                run_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "final_loss": loss,
                        "mean_jaccard_distance_to_method_peers": mean_distance,
                        "reaction_ids": " ".join(str(value) for value in sorted(values[seed])),
                    }
                )
            for pair_index, ((seed_a, topology_a), (seed_b, topology_b)) in enumerate(
                combinations(enumerate(values), 2)
            ):
                pairwise_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "pair_index": pair_index,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "jaccard_distance": jaccard_distance(topology_a, topology_b),
                    }
                )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    with pairwise_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pairwise_rows[0].keys())
        writer.writeheader()
        writer.writerows(pairwise_rows)
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)


def style_axis(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.tick_params(direction="out", length=3, width=0.7, color="#333333")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot(
    topologies: dict[str, dict[str, list[frozenset[int]]]],
    losses: dict[str, dict[str, list[float]]],
    output: Path,
    *,
    n_seeds: int,
    permutations: int,
) -> None:
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
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 0.88, 0.92)},
    )
    rng = np.random.default_rng(260117582)
    for column, task in enumerate(TASKS):
        accumulation_ax = axes[0, column]
        distance_ax = axes[1, column]
        performance_ax = axes[2, column]
        for method in METHODS:
            values = topologies[task][method]
            distribution = accumulation_distribution(values, permutations=permutations, rng=rng)
            lower, median, upper = np.percentile(distribution, [25, 50, 75], axis=0)
            runs = np.arange(1, n_seeds + 1)
            accumulation_ax.fill_between(
                runs,
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.14,
                linewidth=0,
            )
            accumulation_ax.plot(
                runs,
                median,
                color=METHOD_COLORS[method],
                linewidth=1.7,
                marker="o",
                markersize=2.3,
                markevery=2,
            )
            accumulation_ax.annotate(
                f"{len(set(values))}/{n_seeds}",
                xy=(n_seeds, median[-1]),
                xytext=(-3, 5 if method == "rl4crn" else -8),
                textcoords="offset points",
                color=METHOD_COLORS[method],
                fontsize=6.4,
                fontweight="semibold",
                ha="right",
            )

        grouped = [
            [jaccard_distance(a, b) for a, b in combinations(topologies[task][method], 2)]
            for method in METHODS
        ]
        positions = np.arange(1, len(METHODS) + 1)
        violin = distance_ax.violinplot(
            grouped,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.22,
        )
        for body, method in zip(violin["bodies"], METHODS):
            body.set_facecolor(METHOD_COLORS[method])
            body.set_edgecolor(METHOD_COLORS[method])
            body.set_alpha(0.22)
            body.set_linewidth(0.7)
        box = distance_ax.boxplot(
            grouped,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.3},
            whiskerprops={"color": "#444444", "linewidth": 0.8},
            capprops={"color": "#444444", "linewidth": 0.8},
            boxprops={"edgecolor": "#444444", "linewidth": 0.8},
        )
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.40)
        for position, method, distances in zip(positions, METHODS, grouped):
            distance_ax.text(
                position,
                0.055,
                f"mean {np.mean(distances):.1%}",
                color=METHOD_COLORS[method],
                fontsize=6.2,
                fontweight="semibold",
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
                zorder=4,
            )

        novelty_by_method = {}
        for method in METHODS:
            values = topologies[task][method]
            novelty_by_method[method] = np.asarray(
                [
                    np.mean(
                        [
                            jaccard_distance(topology, other)
                            for other_seed, other in enumerate(values)
                            if other_seed != seed
                        ]
                    )
                    for seed, topology in enumerate(values)
                ],
                dtype=float,
            )
        for seed in range(n_seeds):
            performance_ax.plot(
                [novelty_by_method[method][seed] for method in METHODS],
                [losses[task][method][seed] for method in METHODS],
                color="#A8A8A8",
                linewidth=0.45,
                alpha=0.45,
                zorder=1,
            )
        for method in METHODS:
            novelty = novelty_by_method[method]
            method_losses = np.asarray(losses[task][method], dtype=float)
            performance_ax.scatter(
                novelty,
                method_losses,
                s=17,
                color=METHOD_COLORS[method],
                alpha=0.76,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            performance_ax.scatter(
                [np.median(novelty)],
                [np.median(method_losses)],
                s=46,
                marker="X",
                color=METHOD_COLORS[method],
                edgecolors="#202020",
                linewidths=0.45,
                zorder=3,
            )
            performance_ax.annotate(
                f"median {np.median(method_losses):.3g}",
                xy=(np.median(novelty), np.median(method_losses)),
                xytext=(6, 6 if method == "rl4crn" else -10),
                textcoords="offset points",
                color=METHOD_COLORS[method],
                fontsize=6.1,
                fontweight="semibold",
                ha="left",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
                zorder=4,
            )

        accumulation_ax.set_title(TASK_LABELS[task], fontweight="semibold", pad=5)
        accumulation_ax.set_xlabel("Independent runs sampled")
        accumulation_ax.set_xlim(1, n_seeds)
        accumulation_ax.set_ylim(0, n_seeds + 1)
        accumulation_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        accumulation_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        distance_ax.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
        distance_ax.set_ylim(0, 1.02)
        distance_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        performance_ax.set_xlabel("Mean distance to method peers")
        performance_ax.set_xlim(0, 1.02)
        performance_ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        performance_ax.set_yscale("log")
        if column == 0:
            accumulation_ax.set_ylabel("Distinct final topologies")
            distance_ax.set_ylabel("Pairwise Jaccard distance")
            performance_ax.set_ylabel("Final best loss")
        style_axis(accumulation_ax)
        style_axis(distance_ax)
        style_axis(performance_ax)
        panel_label(accumulation_ax, chr(ord("A") + column))
        panel_label(distance_ax, chr(ord("C") + column))
        panel_label(performance_ax, chr(ord("E") + column))

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#A8A8A8", linewidth=0.8, label="Paired seed"),
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="X",
                linestyle="none",
                markersize=5.5,
                label="Method median",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        title=(
            f"Final best reaction sets (n = {n_seeds} paired seeds; fixed template excluded)"
        ),
        title_fontsize=6.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "GenAI-Net and GenAI-Net-LLM topology diversity"}
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/genai_net_llm_diversity_20seed.pdf",
    )
    args = parser.parse_args()
    topologies = load_topologies(args.n_seeds)
    losses = load_final_losses(args.n_seeds)
    output = args.output.expanduser().resolve()
    plot(
        topologies,
        losses,
        output,
        n_seeds=args.n_seeds,
        permutations=args.permutations,
    )
    summarize(topologies, losses, output)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
