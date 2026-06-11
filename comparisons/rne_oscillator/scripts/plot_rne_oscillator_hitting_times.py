#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "comparisons/rne_oscillator/data"
RAW_ROOT = DATA_DIR / "raw"
FIG_DIR = ROOT / "comparisons/rne_oscillator/figures"

THRESHOLD = 20.0

RUNS = [
    {
        "label": "B=100",
        "method": "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20",
        "analysis_csv": FIG_DIR
        / "rne_oscillator_analysis_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_n400.csv",
        "color": "#4C78A8",
    },
    {
        "label": "B=200",
        "method": "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_bs200",
        "analysis_csv": FIG_DIR
        / "rne_oscillator_analysis_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_bs200_n400.csv",
        "color": "#F58518",
    },
]

OUT_PREFIX = FIG_DIR / "rne_oscillator_loss20_hitting_time_comparison"


def seed_label(seed) -> str:
    s = str(seed)
    if s.startswith("seed"):
        return s
    return f"seed{int(float(s)):03d}"


def last_segment(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df:
        return df
    epochs = pd.to_numeric(df["epoch"], errors="coerce").to_numpy()
    resets = np.flatnonzero(np.diff(epochs) < 0)
    if len(resets):
        return df.iloc[int(resets[-1]) + 1 :].reset_index(drop=True)
    return df


def first_hit(progress_path: Path, threshold: float) -> tuple[float, float, float] | tuple[None, None, None]:
    try:
        df = pd.read_csv(progress_path)
    except Exception:
        return None, None, None
    df = last_segment(df)
    if df.empty or "saved_best_loss" not in df or "epoch" not in df:
        return None, None, None

    loss = pd.to_numeric(df["saved_best_loss"], errors="coerce")
    hits = df.loc[loss < threshold]
    if hits.empty:
        return None, None, None
    row = hits.iloc[0]
    return float(row["epoch"]), float(row.get("candidate_evaluations", np.nan)), float(row["saved_best_loss"])


def collect_hitting_times() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        analysis = pd.read_csv(run["analysis_csv"])
        analysis["seed_label"] = analysis["seed"].map(seed_label)
        if "loss_success" not in analysis:
            analysis["loss_success"] = analysis["best_loss"] < THRESHOLD
        successful = analysis[analysis["loss_success"].fillna(False).astype(bool)]

        for row in successful.itertuples(index=False):
            progress_path = RAW_ROOT / run["method"] / row.seed_label / "progress.csv"
            hit_epoch, hit_evals, hit_loss = first_hit(progress_path, THRESHOLD)
            if hit_epoch is None:
                continue
            rows.append(
                {
                    "setting": run["label"],
                    "seed_label": row.seed_label,
                    "best_loss": float(row.best_loss),
                    "hit_epoch": hit_epoch,
                    "hit_candidate_evaluations": hit_evals,
                    "hit_loss": hit_loss,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = collect_hitting_times()
    if df.empty:
        raise RuntimeError("No successful hitting times found.")

    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_PREFIX.with_suffix(".csv")
    df.to_csv(out_csv, index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    positions = np.arange(len(RUNS), dtype=float)
    rng = np.random.default_rng(4)

    data = [df.loc[df["setting"] == run["label"], "hit_epoch"].to_numpy(float) for run in RUNS]
    parts = ax.violinplot(data, positions=positions, widths=0.55, showmeans=False, showmedians=False, showextrema=False)
    for body, run in zip(parts["bodies"], RUNS):
        body.set_facecolor(run["color"])
        body.set_edgecolor("none")
        body.set_alpha(0.24)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.25},
        whiskerprops={"color": "#444444", "linewidth": 0.8},
        capprops={"color": "#444444", "linewidth": 0.8},
        boxprops={"edgecolor": "#444444", "linewidth": 0.8},
    )
    for patch, run in zip(box["boxes"], RUNS):
        patch.set_facecolor(run["color"])
        patch.set_alpha(0.45)

    for x0, values, run in zip(positions, data, RUNS):
        jitter = rng.uniform(-0.085, 0.085, size=len(values))
        ax.scatter(
            np.full(len(values), x0) + jitter,
            values,
            s=16,
            color=run["color"],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.78,
            zorder=3,
        )
        med = np.median(values)
        ax.text(
            x0,
            med,
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([run["label"] for run in RUNS])
    ax.set_ylabel(f"First epoch with loss < {THRESHOLD:g}")
    ax.set_title("Loss-threshold hitting times")
    ax.grid(axis="y", alpha=0.22)
    ax.set_xlim(-0.55, len(RUNS) - 0.45)

    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"))
    plt.close(fig)

    summary = df.groupby("setting")["hit_epoch"].agg(["count", "median", "mean", "min", "max"])
    print(summary)
    print(f"CSV: {out_csv}")
    print(f"PNG: {OUT_PREFIX.with_suffix('.png')}")
    print(f"PDF: {OUT_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
