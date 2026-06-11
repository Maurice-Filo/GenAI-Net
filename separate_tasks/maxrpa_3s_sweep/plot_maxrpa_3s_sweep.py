"""Plot exact maxRPA fractions for 3-species CRNs with m=2..6 reactions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "maxrpa_3s_sweep.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["percent"] = 100.0 * df["decimal_portion"]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    color = "#4C78A8"
    ax.plot(
        df["m"],
        df["percent"],
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        color=color,
    )
    ax.fill_between(df["m"], df["percent"], color=color, alpha=0.13)

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['percent']:.2f}%",
            (row["m"], row["percent"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("number of reactions m")
    ax.set_ylabel("maxRPA fraction (%)")
    ax.set_title("Exact deterministic maxRPA fraction for 3-species CRNs")
    ax.set_xticks(df["m"])
    ax.set_ylim(0, max(df["percent"]) * 1.25)
    ax.grid(True, axis="y", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(ROOT / "maxrpa_3s_sweep.png", dpi=300)
    fig.savefig(ROOT / "maxrpa_3s_sweep.pdf")


if __name__ == "__main__":
    main()
