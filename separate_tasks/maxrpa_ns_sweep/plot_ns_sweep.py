"""Plot the n=2..5, m=2..10 maxRPA sweep."""

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
    sampled_samples = sorted(
        int(value)
        for value in df.loc[df["mode"] == "sample", "samples"].dropna().unique()
        if int(value) > 0
    )
    sample_note = (
        f"{sampled_samples[0]:,} samples per sampled cell"
        if len(sampled_samples) == 1
        else "variable sample counts"
    )

    colors = {
        2: "#4C78A8",
        3: "#F58518",
        4: "#54A24B",
        5: "#B279A2",
    }

    fig = plt.figure(figsize=(11.4, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.30)
    ax_by_m = fig.add_subplot(gs[0, 0])
    ax_by_n = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    for n, group in df.groupby("n_species"):
        group = group.sort_values("m_reactions")
        exact = group[group["mode"] == "exact"]
        sampled = group[group["mode"] == "sample"]

        ax_by_m.plot(
            group["m_reactions"],
            group["percent"],
            color=colors[int(n)],
            linewidth=1.7,
            label=f"{int(n)} species",
            alpha=0.9,
        )
        ax_by_m.scatter(
            exact["m_reactions"],
            exact["percent"],
            color=colors[int(n)],
            edgecolor="white",
            linewidth=0.7,
            s=38,
            zorder=3,
        )
        if not sampled.empty:
            ax_by_m.errorbar(
                sampled["m_reactions"],
                sampled["percent"],
                yerr=asymmetric_yerr(sampled),
                fmt="o",
                color=colors[int(n)],
                markerfacecolor="white",
                markeredgewidth=1.7,
                capsize=5.0,
                capthick=1.8,
                elinewidth=2.0,
                markersize=6.2,
                linestyle="none",
                zorder=6,
            )

    ax_by_m.set_xlabel("number of reactions m")
    ax_by_m.set_ylabel("maxRPA fraction (%)")
    ax_by_m.set_title("Grouped by species")
    ax_by_m.set_xticks(range(2, 11))
    ax_by_m.grid(True, axis="y", alpha=0.25)
    ax_by_m.spines["top"].set_visible(False)
    ax_by_m.spines["right"].set_visible(False)
    ax_by_m.legend(frameon=False, fontsize=8, loc="upper right")

    cmap = plt.get_cmap("tab10")
    for i, (m, group) in enumerate(df.groupby("m_reactions")):
        group = group.sort_values("n_species")
        exact = group[group["mode"] == "exact"]
        sampled = group[group["mode"] == "sample"]
        color = cmap(i % 10)
        ax_by_n.plot(
            group["n_species"],
            group["percent"],
            color=color,
            linewidth=1.35,
            alpha=0.86,
            label=f"m={int(m)}",
        )
        ax_by_n.scatter(
            exact["n_species"],
            exact["percent"],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=28,
            zorder=3,
        )
        if not sampled.empty:
            ax_by_n.errorbar(
                sampled["n_species"],
                sampled["percent"],
                yerr=asymmetric_yerr(sampled),
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.6,
                capsize=4.8,
                capthick=1.65,
                elinewidth=1.85,
                markersize=5.7,
                linestyle="none",
                zorder=6,
            )

    ax_by_n.set_xlabel("number of species n")
    ax_by_n.set_ylabel("maxRPA fraction (%)")
    ax_by_n.set_title("Grouped by reaction count")
    ax_by_n.set_xticks(range(2, 6))
    ax_by_n.set_ylim(top=ax_by_n.get_ylim()[1] * 1.16)
    ax_by_n.grid(True, axis="y", alpha=0.25)
    ax_by_n.spines["top"].set_visible(False)
    ax_by_n.spines["right"].set_visible(False)
    ax_by_n.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    heat = df.pivot(index="n_species", columns="m_reactions", values="percent")
    modes = df.pivot(index="n_species", columns="m_reactions", values="mode")
    im = ax_heat.imshow(heat.values, aspect="auto", cmap="viridis")
    cmap_heat = im.cmap
    norm_heat = im.norm
    ax_heat.set_xticks(np.arange(len(heat.columns)), labels=heat.columns)
    ax_heat.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax_heat.set_xlabel("number of reactions m")
    ax_heat.set_ylabel("number of species n")
    ax_heat.set_title("Fraction heatmap")

    for i, n in enumerate(heat.index):
        for j, m in enumerate(heat.columns):
            value = heat.loc[n, m]
            suffix = "" if modes.loc[n, m] == "exact" else "*"
            r, g, b, _ = cmap_heat(norm_heat(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "black" if luminance > 0.55 else "white"
            ax_heat.text(
                j,
                i,
                f"{value:.1f}{suffix}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.026, pad=0.025)
    cbar.set_label("maxRPA fraction (%)")

    fig.text(
        0.99,
        0.01,
        f"* sampled estimate ({sample_note}); open markers with whiskers show sampled cells with 95% Wilson CI",
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.075, right=0.93, top=0.94, bottom=0.085)
    fig.savefig(ROOT / "maxrpa_ns_sweep.png", dpi=300)
    fig.savefig(ROOT / "maxrpa_ns_sweep.pdf")

    fig_zoom, (ax_zoom_m, ax_zoom_n) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    sampled_all = df[df["mode"] == "sample"].copy()
    for n, group in sampled_all.groupby("n_species"):
        group = group.sort_values("m_reactions")
        ax_zoom_m.fill_between(
            group["m_reactions"],
            group["ci_low_percent"],
            group["ci_high_percent"],
            color=colors[int(n)],
            alpha=0.30,
            linewidth=0,
        )
        ax_zoom_m.plot(
            group["m_reactions"],
            group["percent"],
            marker="o",
            color=colors[int(n)],
            linewidth=1.5,
            markersize=4.5,
            label=f"{int(n)} species",
        )

    cmap = plt.get_cmap("tab10")
    for i, (m, group) in enumerate(sampled_all.groupby("m_reactions")):
        group = group.sort_values("n_species")
        color = cmap(i % 10)
        ax_zoom_n.fill_between(
            group["n_species"],
            group["ci_low_percent"],
            group["ci_high_percent"],
            color=color,
            alpha=0.28,
            linewidth=0,
        )
        ax_zoom_n.plot(
            group["n_species"],
            group["percent"],
            marker="o",
            color=color,
            linewidth=1.3,
            markersize=4.0,
            label=f"m={int(m)}",
        )

    ax_zoom_m.set_xlabel("number of reactions m")
    ax_zoom_m.set_ylabel("maxRPA fraction (%)")
    ax_zoom_m.set_title("Sampled cells by species")
    ax_zoom_m.set_xticks(range(2, 11))
    ax_zoom_m.grid(True, axis="y", alpha=0.25)
    ax_zoom_m.spines["top"].set_visible(False)
    ax_zoom_m.spines["right"].set_visible(False)
    ax_zoom_m.legend(frameon=False, fontsize=8)

    ax_zoom_n.set_xlabel("number of species n")
    ax_zoom_n.set_ylabel("maxRPA fraction (%)")
    ax_zoom_n.set_title("Sampled cells by reaction count")
    ax_zoom_n.set_xticks(range(2, 6))
    ax_zoom_n.grid(True, axis="y", alpha=0.25)
    ax_zoom_n.spines["top"].set_visible(False)
    ax_zoom_n.spines["right"].set_visible(False)
    ax_zoom_n.legend(frameon=False, fontsize=7, ncol=2)

    fig_zoom.tight_layout()
    fig_zoom.savefig(ROOT / "maxrpa_ns_sweep_sampled_ci.png", dpi=300)
    fig_zoom.savefig(ROOT / "maxrpa_ns_sweep_sampled_ci.pdf")

    sampled = df[df["mode"] == "sample"].copy()
    sampled["ci_half_width_percent"] = (
        sampled["ci_high_percent"] - sampled["ci_low_percent"]
    ) / 2.0

    fig_ci, ax_ci = plt.subplots(figsize=(6.4, 3.7))
    for n, group in sampled.groupby("n_species"):
        group = group.sort_values("m_reactions")
        ax_ci.plot(
            group["m_reactions"],
            group["ci_half_width_percent"],
            marker="o",
            linewidth=1.6,
            markersize=4.5,
            color=colors[int(n)],
            label=f"{int(n)} species",
        )
    ax_ci.set_xlabel("number of reactions m")
    ax_ci.set_ylabel("95% Wilson half-width (%)")
    ax_ci.set_title("Uncertainty of sampled maxRPA estimates")
    ax_ci.set_xticks(range(2, 11))
    ax_ci.grid(True, axis="y", alpha=0.25)
    ax_ci.spines["top"].set_visible(False)
    ax_ci.spines["right"].set_visible(False)
    ax_ci.legend(frameon=False, fontsize=8)
    fig_ci.tight_layout()
    fig_ci.savefig(ROOT / "maxrpa_ns_sweep_ci_width.png", dpi=300)
    fig_ci.savefig(ROOT / "maxrpa_ns_sweep_ci_width.pdf")


if __name__ == "__main__":
    main()
