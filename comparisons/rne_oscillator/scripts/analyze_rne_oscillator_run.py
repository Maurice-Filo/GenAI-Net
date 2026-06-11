#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "comparisons/rne_oscillator/data"
RAW_ROOT = DATA_DIR / "raw"
POSTHOC_DIR = DATA_DIR / "posthoc"
FIG_DIR = ROOT / "comparisons/rne_oscillator/figures"
DEFAULT_METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20"
DEFAULT_SUMMARY = DATA_DIR / "rne_oscillator_100_runs_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20.csv"
PLOT_TITLE = "Fixed frequency and amplitude oscillators"
COLORS = {
    "loss_success": "#1f77b4",
    "loss_failure": "#bdbdbd",
    "posthoc": "#ff7f0e",
    "overlap": "#2ca02c",
    "threshold": "#222222",
    "box_fill": "#e8f1fb",
    "failure_fill": "#eeeeee",
}


def seed_label(seed) -> str:
    s = str(seed)
    if s.startswith("seed"):
        return s
    return f"seed{int(float(s)):03d}"


def run_command(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def maybe_run_posthoc(args) -> tuple[Path, Path]:
    antimony_dir = POSTHOC_DIR / f"{args.method}_antimony"
    posthoc_csv = Path(args.posthoc_csv) if args.posthoc_csv else POSTHOC_DIR / f"rne_posthoc_{args.tag}.csv"

    if args.skip_posthoc and posthoc_csv.exists():
        return antimony_dir, posthoc_csv

    export_script = ROOT / "comparisons/rne_oscillator/scripts/export_rl4crn_best_to_antimony.py"
    run_command(
        [
            sys.executable,
            str(export_script),
            "--method",
            args.method,
            "--losses-csv",
            str(args.summary_csv),
            "--raw-root",
            str(args.raw_root),
            "--output-dir",
            str(antimony_dir),
        ]
    )

    evaluator = ROOT / "comparisons/rne_oscillator/scripts/evaluate_rne_posthoc_py.py"
    run_command([sys.executable, str(evaluator), str(antimony_dir), str(posthoc_csv)])
    return antimony_dir, posthoc_csv


def load_progress_traces(raw_root: Path, method: str) -> dict[str, pd.DataFrame]:
    method_dir = raw_root / method
    traces: dict[str, pd.DataFrame] = {}
    for progress_path in sorted(method_dir.glob("seed*/progress.csv")):
        try:
            df = pd.read_csv(progress_path)
        except Exception:
            continue
        if df.empty:
            continue
        if "epoch" in df:
            epochs = pd.to_numeric(df["epoch"], errors="coerce").to_numpy()
            resets = np.flatnonzero(np.diff(epochs) < 0)
            if len(resets):
                df = df.iloc[int(resets[-1]) + 1 :].reset_index(drop=True)
        traces[progress_path.parent.name] = df
    return traces


def load_hyperparameters(raw_root: Path, method: str, df: pd.DataFrame) -> dict:
    seed_labels = [seed_label(seed) for seed in sorted(df["seed"].astype(int))]
    for seed in seed_labels:
        config_path = raw_root / method / seed / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def hyperparameter_text(config: dict, total_runs: int) -> str:
    train = config.get("train", {})
    solver = config.get("solver", {})
    policy = config.get("policy", {})
    agent = config.get("agent", {})
    risk = agent.get("risk_scheduler", {})
    entropy = agent.get("entropy_scheduler", {})
    head_entropy = policy.get("entropy_weights_per_head", {})

    lines = [
        f"runs: {total_runs}",
        f"epochs: {train.get('epochs', '')}",
        f"batch size: {train.get('batch_size', '')}",
        f"batch multiplier: {train.get('batch_multiplier', '')}",
        f"max added reactions: {train.get('max_added_reactions', '')}",
        f"policy: depth {policy.get('depth', '')}, width {policy.get('deep_layer_size', '')}",
        f"learning rate: {agent.get('learning_rate', '')}",
        (
            "risk: "
            f"{risk.get('risk', '')} -> {risk.get('max_risk', '')}, "
            f"+{risk.get('risk_update', '')} / {risk.get('risk_schedule', '')} epochs"
        ),
        f"entropy weight: {entropy.get('entropy_weight', '')}",
        (
            "head entropy: "
            f"structure {head_entropy.get('structure', '')}, "
            f"continuous {head_entropy.get('continuous', '')}"
        ),
        (
            "solver: "
            f"{solver.get('algorithm', '')}, "
            f"rtol {solver.get('rtol', '')}, atol {solver.get('atol', '')}"
        ),
        f"success: loss < {config.get('success_threshold', 20.0)}",
    ]
    return "\n".join(lines)


def merged_results(summary_csv: Path, posthoc_csv: Path | None, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    df["seed_label"] = df["seed"].map(seed_label)
    df["loss_success"] = df["best_loss"] < 20.0

    if posthoc_csv is not None and posthoc_csv.exists():
        ph = pd.read_csv(posthoc_csv)
        ph["seed_label"] = ph["seed"].map(seed_label)
        keep = [
            "seed_label",
            "parse_ok",
            "rne_is_oscillator",
            "rne_is_broken_oscillator",
            "rne_fixed_by_reaction_removal",
            "rne_posthoc_success",
            "error",
        ]
        df = df.merge(ph[keep], on="seed_label", how="left")
    else:
        df["rne_posthoc_success"] = np.nan

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def plot_analysis(
    *,
    df: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    out_prefix: Path,
    method: str,
    threshold: float,
    hparams: dict,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(9.8, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.35, 1.12], width_ratios=[1.0, 1.0, 1.28])
    ax_trace = fig.add_subplot(gs[0, :])
    ax_box = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])
    ax_hparams = fig.add_subplot(gs[1, 2])

    success_seeds = set(df.loc[df["loss_success"], "seed_label"])
    palette = plt.get_cmap("tab20").colors
    for color_idx, (seed, tr) in enumerate(sorted(traces.items())):
        if "epoch" not in tr or "saved_best_loss" not in tr:
            continue
        is_success = seed in success_seeds
        ax_trace.plot(
            tr["epoch"],
            tr["saved_best_loss"],
            color=palette[color_idx % len(palette)],
            alpha=0.76 if is_success else 0.11,
            linewidth=1.28 if is_success else 0.52,
            zorder=2 if is_success else 1,
        )
    ax_trace.axhline(
        threshold,
        color=COLORS["threshold"],
        linestyle=(0, (4, 3)),
        linewidth=1.1,
    )
    ax_trace.set_xlim(0, 801)
    upper = np.nanpercentile(df["best_loss"], 95)
    ax_trace.set_ylim(0, max(upper * 1.12, threshold * 2.0))
    ax_trace.set_xlabel("Epoch")
    ax_trace.set_ylabel("Best saved loss")
    ax_trace.set_title(PLOT_TITLE)
    ax_trace.grid(alpha=0.2)
    ax_trace.text(
        0.985,
        threshold + max(upper * 0.025, 2.0),
        f"success threshold: loss < {threshold:g}",
        ha="right",
        va="bottom",
        fontsize=9,
        transform=ax_trace.get_yaxis_transform(),
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    losses = np.asarray(df["best_loss"], dtype=float)
    ax_box.boxplot(
        [losses],
        vert=True,
        patch_artist=True,
        widths=0.42,
        boxprops={"facecolor": COLORS["box_fill"], "edgecolor": COLORS["loss_success"], "linewidth": 1.0},
        medianprops={"color": COLORS["threshold"], "linewidth": 1.3},
        whiskerprops={"color": COLORS["loss_success"], "linewidth": 0.9},
        capprops={"color": COLORS["loss_success"], "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 2.6, "markerfacecolor": COLORS["loss_failure"], "markeredgecolor": "none", "alpha": 0.45},
    )
    rng = np.random.default_rng(0)
    x = 1.0 + rng.uniform(-0.08, 0.08, size=len(losses))
    colors = np.where(df["loss_success"].to_numpy(), COLORS["loss_success"], COLORS["loss_failure"])
    ax_box.scatter(x, losses, s=12, c=colors, alpha=0.65, linewidths=0)
    ax_box.axhline(threshold, color=COLORS["threshold"], linestyle=(0, (4, 3)), linewidth=1.0)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels([f"{len(df)} runs"])
    ax_box.set_ylabel("Final best loss")
    ax_box.set_title("Loss distribution")
    ax_box.grid(axis="y", alpha=0.2)

    total = len(df)
    loss_success = int(df["loss_success"].sum())
    if "rne_posthoc_success" in df and df["rne_posthoc_success"].notna().any():
        rne_success = int(df["rne_posthoc_success"].fillna(False).astype(bool).sum())
        overlap = int((df["loss_success"] & df["rne_posthoc_success"].fillna(False).astype(bool)).sum())
    else:
        rne_success = 0
        overlap = 0

    labels = ["RL4CRN\nloss", "RNE\nposthoc"]
    successes = np.array([loss_success, rne_success], dtype=int)
    failures = total - successes
    xpos = np.arange(len(labels))
    ax_bar.bar(
        xpos,
        successes,
        color=[COLORS["loss_success"], COLORS["posthoc"]],
        edgecolor="0.2",
        linewidth=0.5,
        label="success",
    )
    ax_bar.bar(xpos, failures, bottom=successes, color=COLORS["failure_fill"], edgecolor="0.2", linewidth=0.5, label="failure")
    for x0, val in zip(xpos, successes):
        ax_bar.text(x0, val + max(total * 0.02, 1), f"{val}/{total}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, total * 1.12)
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Success criteria")
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.text(
        0.98,
        0.96,
        "grey = failure",
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    ax_hparams.axis("off")
    ax_hparams.set_title("Hyperparameters")
    ax_hparams.text(
        0.02,
        0.98,
        hyperparameter_text(hparams, total),
        transform=ax_hparams.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        family="monospace",
        linespacing=1.22,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#cccccc", "linewidth": 0.7},
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--tag", default="8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20")
    parser.add_argument("--posthoc-csv", default=None)
    parser.add_argument("--skip-posthoc", action="store_true", help="Reuse an existing posthoc CSV if present.")
    args = parser.parse_args()

    posthoc_csv: Path | None
    _, posthoc_csv = maybe_run_posthoc(args)

    merged_csv = FIG_DIR / f"rne_oscillator_analysis_{args.tag}.csv"
    df = merged_results(args.summary_csv, posthoc_csv, merged_csv)
    traces = load_progress_traces(args.raw_root, args.method)
    hparams = load_hyperparameters(args.raw_root, args.method, df)
    out_prefix = FIG_DIR / f"rne_oscillator_analysis_{args.tag}"
    plot_analysis(df=df, traces=traces, out_prefix=out_prefix, method=args.method, threshold=args.threshold, hparams=hparams)

    loss_success = int(df["loss_success"].sum())
    posthoc_success = int(df["rne_posthoc_success"].fillna(False).astype(bool).sum()) if "rne_posthoc_success" in df else 0
    overlap = int((df["loss_success"] & df["rne_posthoc_success"].fillna(False).astype(bool)).sum()) if "rne_posthoc_success" in df else 0
    print(f"Runs: {len(df)}")
    print(f"RL4CRN loss successes: {loss_success}/{len(df)}")
    print(f"RNE posthoc successes: {posthoc_success}/{len(df)}")
    print(f"Overlap: {overlap}/{len(df)}")
    print(f"Merged CSV: {merged_csv}")
    print(f"Figure PNG: {out_prefix.with_suffix('.png')}")
    print(f"Figure PDF: {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
