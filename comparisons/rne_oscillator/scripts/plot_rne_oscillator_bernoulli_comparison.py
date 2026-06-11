#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "comparisons/rne_oscillator/figures"

RUNS = [
    {
        "label": "Our method\nbatch 100",
        "kind": "ours",
        "csv": FIG_DIR
        / "rne_oscillator_analysis_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_n400.csv",
    },
    {
        "label": "Our method\nbatch 200",
        "kind": "ours",
        "csv": FIG_DIR
        / "rne_oscillator_analysis_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_bs200_n400.csv",
    },
    {
        "label": "RNE paper\n22%",
        "kind": "paper",
        "successes": round(0.22 * 700),
        "n": 700,
        "reported_rate": 0.22,
    },
    {
        "label": "RNE paper\n26%",
        "kind": "paper",
        "successes": round(0.26 * 700),
        "n": 700,
        "reported_rate": 0.26,
    },
]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def fisher_p_greater(k_ours: int, n_ours: int, k_ref: int, n_ref: int) -> float:
    result = fisher_exact(
        [[k_ours, n_ours - k_ours], [k_ref, n_ref - k_ref]],
        alternative="greater",
    )
    return float(result.pvalue if hasattr(result, "pvalue") else result[1])


def star_label(p_value: float) -> str:
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "n.s."


def bracket_label(p_value: float) -> str:
    return f"{star_label(p_value)}\np={p_value:.1e}"


def add_bracket(ax, x0: float, x1: float, y: float, text: str, height: float = 0.012) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="0.15", linewidth=0.8, clip_on=False)
    ax.text((x0 + x1) / 2.0, y + height + 0.004, text, ha="center", va="bottom", fontsize=8.5)


def main() -> None:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        if run["kind"] == "ours":
            df = pd.read_csv(run["csv"])
            n = len(df)
            successes = int(df["rne_posthoc_success"].fillna(False).astype(bool).sum())
            reported_rate = np.nan
        else:
            n = int(run["n"])
            successes = int(run["successes"])
            reported_rate = float(run["reported_rate"])
        lo, hi = wilson_interval(successes, n)
        rows.append(
            {
                "label": run["label"].replace("\n", " "),
                "kind": run["kind"],
                "successes": successes,
                "n": n,
                "rate": successes / n,
                "reported_rate": reported_rate,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    summary = pd.DataFrame(rows)
    paper = summary[summary["kind"] == "paper"].reset_index(drop=True)
    ours = summary[summary["kind"] == "ours"].reset_index(drop=True)
    for idx, row in ours.iterrows():
        for _, ref in paper.iterrows():
            col = f"p_vs_{int(round(ref['reported_rate'] * 100))}pct_paper"
            summary.loc[summary["label"] == row["label"], col] = fisher_p_greater(
                int(row["successes"]),
                int(row["n"]),
                int(ref["successes"]),
                int(ref["n"]),
            )
    out_csv = FIG_DIR / "rne_oscillator_bernoulli_success_comparison.csv"
    summary.to_csv(out_csv, index=False)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = [0, 2, 1, 3]
    summary = summary.iloc[order].reset_index(drop=True)
    colors = ["#4C78A8", "#8C8C8C", "#4C78A8", "#8C8C8C"]
    labels = ["ours\nB=100", "RNE\nB=100", "ours\nB=200", "RNE\nB=200"]
    x = np.array([0.0, 0.52, 1.38, 1.90])
    rate = summary["rate"].to_numpy(float)
    low = summary["ci95_low"].to_numpy(float)
    high = summary["ci95_high"].to_numpy(float)
    yerr = np.vstack([rate - low, high - rate])

    fig, ax = plt.subplots(figsize=(4.35, 2.85), constrained_layout=True)
    ax.bar(x, rate, color=colors, edgecolor="0.2", linewidth=0.55, width=0.34)
    ax.errorbar(x, rate, yerr=yerr, fmt="none", ecolor="0.1", elinewidth=0.9, capsize=3, capthick=0.9)
    for xi, row in zip(x, summary.itertuples(index=False)):
        ax.text(
            xi,
            0.10,
            f"{row.successes}/{row.n}",
            ha="center",
            va="center",
            fontsize=6.8,
            color="white",
            fontweight="bold",
        )

    by_label = summary.set_index("label")
    our_bs100 = by_label.loc["Our method batch 100"]
    our_bs200 = by_label.loc["Our method batch 200"]
    paper22 = by_label.loc["RNE paper 22%"]
    paper26 = by_label.loc["RNE paper 26%"]
    p100_22 = fisher_p_greater(int(our_bs100.successes), int(our_bs100.n), int(paper22.successes), int(paper22.n))
    p100_26 = fisher_p_greater(int(our_bs100.successes), int(our_bs100.n), int(paper26.successes), int(paper26.n))
    p200_22 = fisher_p_greater(int(our_bs200.successes), int(our_bs200.n), int(paper22.successes), int(paper22.n))
    p200_26 = fisher_p_greater(int(our_bs200.successes), int(our_bs200.n), int(paper26.successes), int(paper26.n))
    add_bracket(ax, x[0], x[1], 0.405, bracket_label(p100_22), height=0.010)
    add_bracket(ax, x[2], x[3], 0.492, bracket_label(p200_26), height=0.010)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.32, 2.22)
    ax.set_ylim(0, 0.565)
    ax.set_ylabel("Posthoc success rate")
    ax.set_title("RNE-posthoc success rates", pad=6, fontsize=10)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_prefix = FIG_DIR / "rne_oscillator_bernoulli_success_comparison"
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"CSV: {out_csv}")
    print(f"PNG: {out_prefix.with_suffix('.png')}")
    print(f"PDF: {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
