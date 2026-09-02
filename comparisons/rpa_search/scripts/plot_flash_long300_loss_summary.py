#!/usr/bin/env python3
"""Plot final best-loss distributions for the 300-epoch Flash campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw/genai_net_llm_flash_long300"
TASKS = (
    ("logic", "Logic circuit", "#007C6C"),
    ("rpa", "RPA", "#D55E00"),
)


def load_losses(raw_root: Path, task: str) -> list[float]:
    paths = sorted(
        raw_root.glob(f"{task}_full307200_seed*_cvode_llm_flash_long300/completed.json")
    )
    losses = [float(json.loads(path.read_text(encoding="utf-8"))["best_loss"]) for path in paths]
    if len(losses) != 20:
        raise RuntimeError(f"expected 20 completed {task} runs, found {len(losses)}")
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/flash_long300_best_loss_distribution.pdf",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), constrained_layout=True)
    rng = np.random.default_rng(260817582)
    for ax, (task, label, color) in zip(axes, TASKS):
        losses = np.asarray(load_losses(args.raw_root, task))
        jitter = rng.uniform(-0.08, 0.08, size=len(losses))
        ax.boxplot(
            losses,
            positions=[0],
            widths=0.38,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.4},
            boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        ax.scatter(jitter, losses, s=18, color=color, alpha=0.78, edgecolors="white", linewidths=0.35)
        ax.set_title(f"{label} (n = 20)", fontweight="semibold")
        ax.set_xticks([])
        ax.set_ylabel("Best loss")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params(axis="y", direction="out", length=3)
        ax.text(
            0.02,
            0.97,
            f"median {np.median(losses):.6f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(args.output.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
