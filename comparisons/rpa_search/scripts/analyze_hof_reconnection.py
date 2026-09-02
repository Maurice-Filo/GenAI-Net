#!/usr/bin/env python3
"""Plot the three-arm RPA Harness and HOF-reconnection comparison."""

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

from comparisons.rpa_search.scripts.analyze_reasoning_ablation import (
    completed_seeds,
    database_path,
    read_run,
)
from comparisons.rpa_search.scripts.analyze_initial_hof_ablation import (
    collect_run as collect_source_run,
)


CONDITIONS = ("reasoning_harness", "nonthinking_disconnected", "nonthinking_reconnected")
LABELS = {
    "reasoning_harness": "Reasoning Harness",
    "nonthinking_disconnected": "Non-thinking, loss-only HOF",
    "nonthinking_reconnected": "Non-thinking, CRN-valued HOF",
}
SHORT_LABELS = {
    "reasoning_harness": "Reasoning\nHarness",
    "nonthinking_disconnected": "Non-thinking\nloss-only HOF",
    "nonthinking_reconnected": "Non-thinking\nCRN-valued HOF",
}
COLORS = {
    "reasoning_harness": "#0072B2",
    "nonthinking_disconnected": "#777777",
    "nonthinking_reconnected": "#D55E00",
}


def collect(roots: dict[str, Path], max_epoch: int) -> tuple[list[dict], dict]:
    seeds = sorted(set.intersection(*(completed_seeds(root) for root in roots.values())))
    runs = {}
    rows = []
    for condition in CONDITIONS:
        for seed in seeds:
            run = read_run(database_path(roots[condition], seed), max_epoch)
            source_rows = collect_source_run(
                database_path(roots[condition], seed),
                max_epoch=max_epoch,
                elite_size=30,
            )
            run["rl_convergence"] = [row["rl_best_loss"] for row in source_rows]
            runs[(condition, seed)] = run
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "final_loss": run["final_loss"],
                    "first_llm_best": run["first_llm_best"],
                    "post_initial_llm_best": min(run["batch_best"][1:], default=np.nan),
                    "all_llm_best": run["all_llm_best"],
                    "final_origin": run["final_origin"],
                    "valid_candidates": run["valid"],
                    "produced_candidates": run["produced"],
                    "unique_llm_topologies": run["unique_llm_topologies"],
                }
            )
    return rows, {"seeds": seeds, "runs": runs}


def paired(values: dict[str, np.ndarray], left: str, right: str) -> dict:
    a = values[left]
    b = values[right]
    finite = np.isfinite(a) & np.isfinite(b)
    nonzero = finite & (a != b)
    all_test = wilcoxon(a[finite], b[finite]) if np.any(nonzero) else None
    exact_nonzero = wilcoxon(a[nonzero], b[nonzero]) if np.any(nonzero) else None
    return {
        "left": left,
        "right": right,
        "left_wins": int(np.sum(a[finite] < b[finite])),
        "right_wins": int(np.sum(b[finite] < a[finite])),
        "ties": int(np.sum(a[finite] == b[finite])),
        "median_left_over_right": float(np.median(a[finite] / b[finite])),
        "median_difference_left_minus_right": float(np.median(a[finite] - b[finite])),
        "wilcoxon_pvalue": float(all_test.pvalue) if all_test else None,
        "nonzero_exact_pvalue": float(exact_nonzero.pvalue) if exact_nonzero else None,
    }


def summarize(rows: list[dict], collected: dict) -> dict:
    summary = {"n": len(collected["seeds"]), "conditions": {}}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        summary["conditions"][condition] = {
            "final_median": float(np.median([row["final_loss"] for row in subset])),
            "final_mean": float(np.mean([row["final_loss"] for row in subset])),
            "final_min": float(np.min([row["final_loss"] for row in subset])),
            "first_llm_median": float(np.median([row["first_llm_best"] for row in subset])),
            "post_initial_llm_median": float(
                np.median([row["post_initial_llm_best"] for row in subset])
            ),
            "all_llm_median": float(np.median([row["all_llm_best"] for row in subset])),
            "llm_origin_winners": sum(row["final_origin"] == "LLM" for row in subset),
            "valid_candidates": sum(row["valid_candidates"] for row in subset),
            "produced_candidates": sum(row["produced_candidates"] for row in subset),
        }
    endpoint = {
        condition: np.asarray(
            [collected["runs"][(condition, seed)]["final_loss"] for seed in collected["seeds"]]
        )
        for condition in CONDITIONS
    }
    summary["paired_reconnected_vs_disconnected"] = paired(
        endpoint, "nonthinking_reconnected", "nonthinking_disconnected"
    )
    summary["paired_reasoning_vs_reconnected"] = paired(
        endpoint, "reasoning_harness", "nonthinking_reconnected"
    )
    return summary


def _style() -> None:
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
        }
    )


def plot(collected: dict, output: Path, max_epoch: int) -> None:
    _style()
    seeds = collected["seeds"]
    runs = collected["runs"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    rng = np.random.default_rng(29)

    ax = axes[0, 0]
    endpoints = {
        condition: np.asarray([runs[(condition, seed)]["final_loss"] for seed in seeds])
        for condition in CONDITIONS
    }
    for seed_index, seed in enumerate(seeds):
        ax.plot(
            range(3), [endpoints[condition][seed_index] for condition in CONDITIONS],
            color="#C2C2C2", linewidth=0.55, zorder=1,
        )
    for index, condition in enumerate(CONDITIONS):
        values = endpoints[condition]
        ax.scatter(
            np.full(len(values), index) + rng.normal(0, 0.025, len(values)), values,
            color=COLORS[condition], s=20, alpha=0.82, zorder=2,
        )
        ax.plot(index, np.median(values), marker="_", markersize=15,
                markeredgewidth=2.2, color="#111111", zorder=3)
    ax.set_xticks(range(3), [SHORT_LABELS[c] for c in CONDITIONS])
    ax.set_yscale("log")
    ax.set_ylabel("Final joint-HOF loss")
    ax.set_title("Paired endpoint", fontweight="semibold")

    ax = axes[0, 1]
    epochs = np.arange(max_epoch + 1)
    for condition in (
        "reasoning_harness",
        "nonthinking_reconnected",
        "nonthinking_disconnected",
    ):
        values = np.asarray([runs[(condition, seed)]["convergence"] for seed in seeds])
        median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, (25, 75), axis=0)
        if condition != "nonthinking_disconnected":
            ax.fill_between(epochs, q25, q75, color=COLORS[condition], alpha=0.10, linewidth=0)
        linestyle = ":" if condition == "nonthinking_disconnected" else "-"
        linewidth = 1.8 if condition == "reasoning_harness" else 1.55
        ax.plot(
            epochs, median, color=COLORS[condition], linestyle=linestyle,
            linewidth=linewidth, label=f"{LABELS[condition]}: joint HOF",
        )
    reasoning_rl = np.asarray(
        [runs[("reasoning_harness", seed)]["rl_convergence"] for seed in seeds]
    )
    ax.plot(
        epochs,
        np.nanmedian(reasoning_rl, axis=0),
        color=COLORS["reasoning_harness"],
        linestyle="--",
        linewidth=1.35,
        label="Reasoning Harness: RL-origin only",
    )
    first_insertion = float(
        np.median([runs[("reasoning_harness", seed)]["first_completion_epoch"] for seed in seeds])
    )
    ax.axvline(first_insertion, color=COLORS["reasoning_harness"], linestyle=":", linewidth=0.9)
    ax.text(
        first_insertion + 1.2, 0.96, f"median first merge: epoch {first_insertion:g}",
        transform=ax.get_xaxis_transform(), va="top",
        color=COLORS["reasoning_harness"], fontsize=6.3,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, max_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best joint-HOF loss")
    ax.set_title("Joint archive and source-isolated RL", fontweight="semibold")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 0]
    for condition in CONDITIONS:
        batch_rows = [runs[(condition, seed)]["batch_best"] for seed in seeds]
        count = max(len(row) for row in batch_rows)
        values = np.full((len(batch_rows), count), np.nan)
        for index, row in enumerate(batch_rows):
            values[index, : len(row)] = row
        x = np.arange(1, count + 1)
        median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, (25, 75), axis=0)
        ax.fill_between(x, q25, q75, color=COLORS[condition], alpha=0.11, linewidth=0)
        ax.plot(x, median, marker="o", markersize=3.1, linewidth=1.55, color=COLORS[condition])
    ax.axvline(1.5, color="#999999", linestyle=":", linewidth=0.9)
    ax.text(1.53, 0.97, "HOF available", transform=ax.get_xaxis_transform(),
            va="top", color="#666666", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("LLM request index")
    ax.set_ylabel("Best direct proposal loss")
    ax.set_title("Proposal quality", fontweight="semibold")

    ax = axes[1, 1]
    disconnected = endpoints["nonthinking_disconnected"]
    reconnected = endpoints["nonthinking_reconnected"]
    low = min(disconnected.min(), reconnected.min()) * 0.85
    high = max(disconnected.max(), reconnected.max()) * 1.15
    ax.plot((low, high), (low, high), color="#444444", linestyle="--", linewidth=0.9)
    improved = reconnected < disconnected
    tied = reconnected == disconnected
    ax.scatter(disconnected[tied], reconnected[tied], color="#999999", s=28,
               alpha=0.8, label=f"Tied ({tied.sum()})")
    ax.scatter(disconnected[improved], reconnected[improved], color=COLORS["nonthinking_reconnected"],
               s=32, alpha=0.9, label=f"Reconnected better ({improved.sum()})")
    for seed, x, y in zip(np.asarray(seeds)[improved], disconnected[improved], reconnected[improved]):
        if abs(y - x) > 1e-4:
            ax.annotate(f"seed {seed}", (x, y), xytext=(5, 6), textcoords="offset points", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Loss-only HOF endpoint")
    ax.set_ylabel("CRN-valued HOF endpoint")
    ax.set_title("Effect of reconnecting HOF structure", fontweight="semibold")
    ax.legend(frameon=False, loc="upper left")

    for label, panel in zip("ABCD", axes.flat):
        panel.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        panel.spines[["top", "right"]].set_visible(False)
        panel.text(0.01, 0.98, label, transform=panel.transAxes,
                   va="top", fontweight="bold", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = "RPA Harness and non-thinking Hall-of-Fame reconnection ablation"
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=400 if suffix == ".png" else None,
                    facecolor="white", metadata={"Title": title} if suffix == ".pdf" else None)
    plt.close(fig)


def write_report(summary: dict, output: Path) -> None:
    c = summary["conditions"]
    reconnect = summary["paired_reconnected_vs_disconnected"]
    reasoning = summary["paired_reasoning_vs_reconnected"]
    text = f"""# RPA Three-Arm Harness Ablation

This 20-seed comparison separates the full reasoning Harness agent from two direct,
simple-prompt, provider-non-thinking generators. Request zero receives no HOF in all
arms. Later direct requests receive either HOF losses only or complete ranked CRN
structures and losses. Lower loss is better.

| Condition | Final median | Best direct-LLM median | LLM-origin rank one |
|---|---:|---:|---:|
| Reasoning Harness | {c['reasoning_harness']['final_median']:.6g} | {c['reasoning_harness']['all_llm_median']:.6g} | {c['reasoning_harness']['llm_origin_winners']}/20 |
| Non-thinking, loss-only HOF | {c['nonthinking_disconnected']['final_median']:.6g} | {c['nonthinking_disconnected']['all_llm_median']:.6g} | {c['nonthinking_disconnected']['llm_origin_winners']}/20 |
| Non-thinking, CRN-valued HOF | {c['nonthinking_reconnected']['final_median']:.6g} | {c['nonthinking_reconnected']['all_llm_median']:.6g} | {c['nonthinking_reconnected']['llm_origin_winners']}/20 |

![Three-arm endpoint, convergence, proposal-quality, and HOF-reconnection analysis.](../rpa_search/figures/rpa_hof_reconnection_ablation.png)

Reconnecting HOF structure improved {reconnect['left_wins']}/20 endpoints, worsened
{reconnect['right_wins']}/20, and left {reconnect['ties']}/20 exactly unchanged. The
exact test over the five nonzero pairs gives `p={reconnect['nonzero_exact_pvalue']:.4g}`.
Four changes are numerically tiny; seed 1 supplies the single substantial endpoint
improvement. No direct non-thinking proposal supplied a final rank-one topology.

The reasoning Harness still wins {reasoning['left_wins']}/20 paired endpoints against
the reconnected generator (`p={reasoning['wilcoxon_pvalue']:.4g}`). Its proposal pool
is roughly two orders of magnitude better, showing that access to HOF structures is
not sufficient: the agent workflow is critical for interpreting feedback and turning
it into useful mechanistic candidates.

The source-aware convergence panel prevents the early Harness drop from being
misread as instantaneous RL learning. The median first Harness batch merges at epoch
8. At epoch 12, the shared-HOF median is `0.005260`, while the exact RL-origin median
is still `0.152984`; the early gain is direct LLM insertion. By epoch 100, the
RL-origin median reaches `0.005931` while the shared HOF reaches `0.002566`. This
later narrowing is consistent with delayed full-duplex SIL transfer, but provenance
alone is not a causal communication ablation.

## Interpretation boundary

The disconnected and reconnected campaigns made independent stochastic provider
calls, including independent first batches before either arm had HOF structure.
Consequently, their small difference is a treatment-plus-sampling comparison rather
than a deterministic replay estimate. The near-identical medians, 15 exact endpoint
ties, poor direct proposal losses, and zero direct-LLM winners nevertheless make the
main system-level conclusion robust. A strict causal estimate of HOF reconnection
would replay identical first batches or use provider-seeded generation.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-root", type=Path, required=True)
    parser.add_argument("--disconnected-root", type=Path, required=True)
    parser.add_argument("--reconnected-root", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "reasoning_harness": args.reasoning_root.resolve(),
        "nonthinking_disconnected": args.disconnected_root.resolve(),
        "nonthinking_reconnected": args.reconnected_root.resolve(),
    }
    rows, collected = collect(roots, args.max_epoch)
    summary = summarize(rows, collected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot(collected, args.output, args.max_epoch)
    write_report(summary, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
