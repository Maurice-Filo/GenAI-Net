"""Create line-only maxRPA plots with Wilson confidence whiskers.

This figure is built from ``maxrpa_ns_sweep.csv`` from scratch.  Exact points
are plotted as filled markers.  Sampled points are plotted as open markers with
their 95% Wilson confidence intervals shown as standard whisker error bars.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "maxrpa_ns_sweep.csv"


def asymmetric_yerr(sampled):
    """Return matplotlib-style asymmetric 95% Wilson errors in percent."""

    low = sampled["ci_low_percent"].to_numpy()
    high = sampled["ci_high_percent"].to_numpy()
    center = sampled["percent"].to_numpy()
    return np.vstack([center - low, high - center])


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["percent"] = 100.0 * df["fraction"]

    species_colors = {
        2: "#1f77b4",
        3: "#ff7f0e",
        4: "#2ca02c",
        5: "#9467bd",
    }

    fig, (ax_species, ax_reactions) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for n, group in df.groupby("n_species"):
        group = group.sort_values("m_reactions")
        exact = group[group["mode"] == "exact"]
        sampled = group[group["mode"] == "sample"]
        color = species_colors[int(n)]

        ax_species.plot(
            group["m_reactions"],
            group["percent"],
            color=color,
            linewidth=1.8,
            linestyle="-",
            label=f"{int(n)} species",
            zorder=3,
        )
        if not sampled.empty:
            ax_species.errorbar(
                sampled["m_reactions"],
                sampled["percent"],
                yerr=asymmetric_yerr(sampled),
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.7,
                markersize=6.2,
                elinewidth=2.0,
                capsize=5.0,
                capthick=1.8,
                linestyle="none",
                zorder=6,
            )
        ax_species.scatter(
            exact["m_reactions"],
            exact["percent"],
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            s=34,
            zorder=5,
        )

    cmap = plt.get_cmap("tab10")
    for i, (m, group) in enumerate(df.groupby("m_reactions")):
        group = group.sort_values("n_species")
        exact = group[group["mode"] == "exact"]
        sampled = group[group["mode"] == "sample"]
        color = cmap(i % 10)

        ax_reactions.plot(
            group["n_species"],
            group["percent"],
            color=color,
            linewidth=1.45,
            linestyle="-",
            label=f"m={int(m)}",
            zorder=3,
        )
        if not sampled.empty:
            ax_reactions.errorbar(
                sampled["n_species"],
                sampled["percent"],
                yerr=asymmetric_yerr(sampled),
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                markersize=5.7,
                elinewidth=1.85,
                capsize=4.8,
                capthick=1.65,
                linestyle="none",
                zorder=6,
            )
        ax_reactions.scatter(
            exact["n_species"],
            exact["percent"],
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            s=28,
            zorder=5,
        )

    for ax in (ax_species, ax_reactions):
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("maxRPA fraction (%)")

    ax_species.set_title("Grouped by species")
    ax_species.set_xlabel("number of reactions m")
    ax_species.set_xticks(range(2, 11))
    ax_species.legend(frameon=False, fontsize=8, loc="upper right")

    ax_reactions.set_title("Grouped by reaction count")
    ax_reactions.set_xlabel("number of species n")
    ax_reactions.set_xticks(range(2, 6))
    ax_reactions.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    fig.text(
        0.995,
        0.01,
        "Whiskers: 95% Wilson CI for sampled estimates; filled markers: exact values.",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.5)
    fig.savefig(ROOT / "maxrpa_ns_sweep_lines_ci.png", dpi=400)
    fig.savefig(ROOT / "maxrpa_ns_sweep_lines_ci.pdf")


if __name__ == "__main__":
    main()
