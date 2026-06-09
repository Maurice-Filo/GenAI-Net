#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "comparisons/rne_oscillator/data"
FIG_DIR = ROOT / "comparisons/rne_oscillator/figures"

METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20"
TAG = "8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_n400"
BASE_SUMMARY = DATA_DIR / "rne_oscillator_100_runs_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20.csv"
EXT_SUMMARY = DATA_DIR / "rne_oscillator_300_runs_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_seeds100_399.csv"
COMBINED_SUMMARY = DATA_DIR / "rne_oscillator_400_runs_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20.csv"


def merge_summaries(paths: list[Path], output: Path) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("seed").drop_duplicates("seed", keep="last").reset_index(drop=True)
    if len(df) != 400:
        raise RuntimeError(f"Expected 400 unique seeds, found {len(df)}")
    missing = sorted(set(range(400)) - set(df["seed"].astype(int)))
    if missing:
        raise RuntimeError(f"Missing seeds: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def run_analysis(summary_csv: Path, tag: str, skip_posthoc: bool) -> Path:
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "comparisons/rne_oscillator/scripts/analyze_rne_oscillator_run.py"),
        "--method",
        METHOD,
        "--summary-csv",
        str(summary_csv),
        "--tag",
        tag,
    ]
    if skip_posthoc:
        cmd.append("--skip-posthoc")
    subprocess.run(cmd, cwd=ROOT, check=True)
    return FIG_DIR / f"rne_oscillator_analysis_{tag}.csv"


def summarize(merged_csv: Path, p0: float) -> None:
    df = pd.read_csv(merged_csv)
    loss_success = int(df["loss_success"].fillna(False).astype(bool).sum())
    posthoc_success = int(df["rne_posthoc_success"].fillna(False).astype(bool).sum())
    overlap = int(
        (
            df["loss_success"].fillna(False).astype(bool)
            & df["rne_posthoc_success"].fillna(False).astype(bool)
        ).sum()
    )
    n = len(df)
    p_value = float(binom.sf(posthoc_success - 1, n, p0))
    top_valid = (
        df[df["rne_posthoc_success"].fillna(False).astype(bool)]
        .sort_values("best_loss")
        .head(10)
    )

    print("\nFinal RNE oscillator depth-3 n=400 summary")
    print(f"Runs: {n}")
    print(f"RL4CRN loss successes: {loss_success}/{n} = {loss_success / n:.3f}")
    print(f"RNE posthoc successes: {posthoc_success}/{n} = {posthoc_success / n:.3f}")
    print(f"Overlap: {overlap}/{n} = {overlap / n:.3f}")
    print(f"One-sided binomial p-value vs p0={p0:.3f}: {p_value:.6g}")
    print("\nTop RNE-posthoc-valid CRNs by RL4CRN loss:")
    cols = ["seed", "run_id", "best_loss", "loss_success", "rne_posthoc_success"]
    print(top_valid[cols].to_string(index=False))
    print(f"\nCombined summary: {COMBINED_SUMMARY}")
    print(f"Merged analysis: {merged_csv}")
    print(f"Figure PNG: {FIG_DIR / f'rne_oscillator_analysis_{TAG}.png'}")
    print(f"Figure PDF: {FIG_DIR / f'rne_oscillator_analysis_{TAG}.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-posthoc", action="store_true", help="Reuse existing posthoc CSV if present.")
    parser.add_argument("--p0", type=float, default=0.22, help="RNE paper baseline success probability.")
    args = parser.parse_args()

    merge_summaries([BASE_SUMMARY, EXT_SUMMARY], COMBINED_SUMMARY)
    merged_csv = run_analysis(COMBINED_SUMMARY, TAG, args.skip_posthoc)
    summarize(merged_csv, args.p0)


if __name__ == "__main__":
    main()
