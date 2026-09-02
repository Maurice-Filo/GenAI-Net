#!/usr/bin/env python3
"""Compare the RPA reasoning-Harness campaign with matched RL-only runs."""

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
    read_hof,
    read_llm_candidates,
)


RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
RL_ROOT = RAW_ROOT / "rl4crn"
HARNESS_ROOT = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns/"
    "flash-rpa-initial-hof-withheld-100epoch-20seed/runs"
)
COLORS = {
    "rl_only": "#0072B2",
    "harness": "#009E73",
    "hybrid_rl": "#0072B2",
    "llm": "#D55E00",
}


def _progress(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = np.asarray([float(row["best_so_far_loss"]) for row in rows])
    return values[:100]


def _hybrid_run(root: Path, seed: int) -> dict:
    run_name = f"rpa_full307200_seed{seed}_cvode_flash_rpa_context_free100"
    direct = root / run_name
    matches = [direct] if direct.is_dir() else sorted(root.glob(f"*-rpa-seed{seed}/{run_name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Harness run for seed {seed} under {root}")
    run = matches[0]
    database = run / "results.sqlite"
    hof = read_hof(database)
    llm = read_llm_candidates(database)
    llm_ids = {candidate["identifier"] for _, candidate in llm}
    llm_topologies = {candidate["topology"] for _, candidate in llm}
    by_completion: dict[int, set[tuple[str, str]]] = {}
    for epoch, candidate in llm:
        by_completion.setdefault(epoch, set()).add(candidate["identifier"])

    joint = []
    rl_origin = []
    archive = []
    direct_events = []
    previous = None
    for epoch in range(100):
        snapshot = hof[epoch]
        archive.extend(snapshot)
        best = snapshot[0]
        joint.append(best["loss"])
        eligible = [candidate for candidate in archive if candidate["identifier"] not in llm_ids]
        rl_origin.append(min((candidate["loss"] for candidate in eligible), default=np.nan))
        if (
            previous is not None
            and best["loss"] < previous["loss"]
            and best["identifier"] in by_completion.get(epoch, set())
        ):
            direct_events.append(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "before_loss": previous["loss"],
                    "after_loss": best["loss"],
                    "fold_improvement": previous["loss"] / best["loss"],
                }
            )
        previous = best

    final = hof[max(epoch for epoch in hof if epoch <= 100)][0]
    return {
        "joint": np.asarray(joint),
        "rl_origin": np.asarray(rl_origin),
        "final": float(final["loss"]),
        "final_exact_llm": final["identifier"] in llm_ids,
        "final_llm_topology": final["topology"] in llm_topologies,
        "direct_events": direct_events,
    }


def collect(n_seeds: int, harness_root: Path) -> tuple[dict, list[dict], list[dict]]:
    curves = {"rl_only": [], "harness": [], "hybrid_rl": []}
    endpoints = []
    events = []
    for seed in range(n_seeds):
        rl = _progress(RL_ROOT / f"rpa_full102400_seed{seed}_cvode/progress.csv")
        hybrid = _hybrid_run(harness_root, seed)
        curves["rl_only"].append(rl)
        curves["harness"].append(hybrid["joint"])
        curves["hybrid_rl"].append(hybrid["rl_origin"])
        endpoints.append(
            {
                "seed": seed,
                "rl_only_loss": float(rl[-1]),
                "harness_loss": hybrid["final"],
                "rl_over_harness": float(rl[-1] / hybrid["final"]),
                "final_exact_llm": int(hybrid["final_exact_llm"]),
                "final_llm_topology": int(hybrid["final_llm_topology"]),
            }
        )
        events.extend(hybrid["direct_events"])
    return {key: np.asarray(value) for key, value in curves.items()}, endpoints, events


def summarize(endpoints: list[dict], events: list[dict]) -> dict:
    rl = np.asarray([row["rl_only_loss"] for row in endpoints])
    harness = np.asarray([row["harness_loss"] for row in endpoints])
    test = wilcoxon(harness, rl)
    return {
        "n": len(endpoints),
        "budgets": {"rl_only": 102400, "harness_rl": 102300, "harness_llm": 50},
        "rl_only_median": float(np.median(rl)),
        "harness_median": float(np.median(harness)),
        "median_paired_rl_over_harness": float(np.median(rl / harness)),
        "harness_wins": int(np.sum(harness < rl)),
        "rl_only_wins": int(np.sum(rl < harness)),
        "ties": int(np.sum(rl == harness)),
        "wilcoxon_pvalue": float(test.pvalue),
        "direct_llm_insertion_events": len(events),
        "runs_with_direct_llm_insertion": len({row["seed"] for row in events}),
        "final_exact_llm_candidates": sum(row["final_exact_llm"] for row in endpoints),
        "final_llm_origin_topologies": sum(row["final_llm_topology"] for row in endpoints),
    }


def _band(ax, x: np.ndarray, values: np.ndarray, color: str, label: str, **kwargs) -> None:
    median = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, (25, 75), axis=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, median, color=color, linewidth=1.8, label=label, **kwargs)


def plot(curves: dict, endpoints: list[dict], events: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    epochs = np.arange(1, 101)

    ax = axes[0]
    _band(ax, epochs, curves["rl_only"], COLORS["rl_only"], "RL only")
    _band(ax, epochs, curves["harness"], COLORS["harness"], "Full duplex")
    ax.set(xlabel="Epoch", ylabel="Best-so-far loss", title="Matched search")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    _band(ax, epochs, curves["hybrid_rl"], COLORS["hybrid_rl"], "RL-provenance archive")
    _band(ax, epochs, curves["harness"], COLORS["harness"], "Joint HOF")
    insertion_epochs = sorted({row["epoch"] for row in events})
    for epoch in insertion_epochs:
        ax.axvline(epoch, color=COLORS["llm"], alpha=0.08, linewidth=0.6)
    ax.plot([], [], color=COLORS["llm"], linewidth=1.2, label="LLM-provenance insertion")
    ax.set(xlabel="Epoch", ylabel="Best-so-far loss", title="Emitter trajectories")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    rl = np.asarray([row["rl_only_loss"] for row in endpoints])
    harness = np.asarray([row["harness_loss"] for row in endpoints])
    for left, right in zip(rl, harness):
        ax.plot((0, 1), (left, right), color="#BBBBBB", linewidth=0.65, zorder=1)
    rng = np.random.default_rng(41)
    ax.scatter(rng.normal(0, 0.018, len(rl)), rl, color=COLORS["rl_only"], s=20, zorder=2)
    ax.scatter(1 + rng.normal(0, 0.018, len(harness)), harness,
               color=COLORS["harness"], s=20, zorder=2)
    ax.plot(0, np.median(rl), marker="_", markersize=15, markeredgewidth=2.2,
            color="#111111", zorder=3)
    ax.plot(1, np.median(harness), marker="_", markersize=15, markeredgewidth=2.2,
            color="#111111", zorder=3)
    ax.set_xticks((0, 1), ("RL only", "Full duplex"))
    ax.set(ylabel="Final best loss", title="Paired endpoints")
    ax.set_yscale("log")

    for index, ax in enumerate(axes):
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.14, 1.03, chr(ord("A") + index), transform=ax.transAxes,
                fontweight="bold")
    fig.tight_layout(w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=400 if suffix == ".png" else None,
                    facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--harness-root", type=Path, default=HARNESS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/rpa_reasoning_vs_rl_only",
    )
    args = parser.parse_args()
    curves, endpoints, events = collect(args.n_seeds, args.harness_root.resolve())
    summary = summarize(endpoints, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output.with_name(args.output.name + "_endpoints.csv"), endpoints)
    _write_csv(args.output.with_name(args.output.name + "_insertions.csv"), events)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot(curves, endpoints, events, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
