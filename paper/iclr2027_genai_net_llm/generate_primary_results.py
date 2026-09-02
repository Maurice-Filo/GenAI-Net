#!/usr/bin/env python3
"""Generate the frozen eight-task primary comparison from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "comparisons/rpa_search/data/raw"
TASKS = (
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
)
LABELS = {
    "rpa": "RPA",
    "logic": "Logic circuit",
    "classifier": "Classifier",
    "dose_hill": "Hill response",
    "dose_ultrasensitive": "Ultrasensitive",
    "dose_biphasic": "Biphasic",
    "oscillator_mean": "Oscillator mean",
    "oscillator_frequency": "Oscillator frequency",
}
HYBRID = {
    "rpa": (
        "genai_net_llm_flash_rpa_context_free100",
        "rpa_full307200_seed{seed}_cvode_flash_rpa_context_free100",
    ),
    "logic": (
        "genai_net_llm_flash_logic_initial_context_free100",
        "logic_full102400_seed{seed}_cvode_llm_flash_logic_initial_context_free100",
    ),
}
BREADTH_METHOD = "genai_net_llm_flash_breadth_initial_context_free20"
BREADTH_SUFFIX = "cvode_llm_flash_breadth_initial_context_free20"


def completed_loss(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["best_loss"])


def progress_loss(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty progress file: {path}")
    return float(rows[-1]["best_so_far_loss"])


def paths(task: str, seed: int) -> tuple[Path, Path]:
    if task in HYBRID:
        method, pattern = HYBRID[task]
        hybrid = RAW / method / pattern.format(seed=seed) / "completed.json"
        rl = RAW / "rl4crn" / f"{task}_full102400_seed{seed}_cvode" / "progress.csv"
        return hybrid, rl
    hybrid = (
        RAW
        / BREADTH_METHOD
        / f"{task}_full102400_seed{seed}_{BREADTH_SUFFIX}"
        / "completed.json"
    )
    rl = (
        RAW
        / "rl4crn_breadth"
        / f"{task}_full102400_seed{seed}_cvode_rl_only_breadth"
        / "completed.json"
    )
    return hybrid, rl


def collect() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for task in TASKS:
        for seed in range(20):
            hybrid_path, rl_path = paths(task, seed)
            absent = [str(path) for path in (hybrid_path, rl_path) if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            hybrid = completed_loss(hybrid_path)
            rl = progress_loss(rl_path) if rl_path.name == "progress.csv" else completed_loss(rl_path)
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "rl_loss": rl,
                    "hybrid_loss": hybrid,
                    "rl_over_hybrid": rl / hybrid,
                    "hybrid_wins": int(hybrid < rl),
                }
            )
    return rows, missing


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for task in TASKS:
        subset = [row for row in rows if row["task"] == task]
        if not subset:
            continue
        rl = np.asarray([row["rl_loss"] for row in subset])
        hybrid = np.asarray([row["hybrid_loss"] for row in subset])
        pvalue = float(wilcoxon(hybrid, rl, alternative="two-sided").pvalue)
        summaries.append(
            {
                "task": task,
                "n": len(subset),
                "rl_median": float(np.median(rl)),
                "hybrid_median": float(np.median(hybrid)),
                "median_paired_fold": float(np.median(rl / hybrid)),
                "wins": int(np.sum(hybrid < rl)),
                "pvalue": pvalue,
            }
        )
    adjusted = holm_adjust([row["pvalue"] for row in summaries])
    for row, pvalue in zip(summaries, adjusted):
        row["holm_pvalue"] = pvalue
    return summaries


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_loss(value: float) -> str:
    return f"{value:.3g}"


def fmt_p(value: float) -> str:
    return f"{value:.2g}"


def write_tex(summaries: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & $n$ & RL & \method{} & RL/\method{} & Wins \\",
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{LABELS[row['task']]} & {row['n']} & {fmt_loss(row['rl_median'])} & "
            f"{fmt_loss(row['hybrid_median'])} & {row['median_paired_fold']:.2g}$\\times$ & "
            f"{row['wins']}/{row['n']} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], output: Path, *, interim: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    rng = np.random.default_rng(73)
    all_ratios = []
    for index, task in enumerate(TASKS):
        subset = [row for row in rows if row["task"] == task]
        ratios = np.asarray([row["rl_over_hybrid"] for row in subset])
        if not len(ratios):
            continue
        all_ratios.extend(ratios)
        jitter = rng.uniform(-0.12, 0.12, len(ratios))
        ax.scatter(
            ratios,
            index + jitter,
            s=21,
            color="#009E73",
            alpha=0.72,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        q25, median, q75 = np.quantile(ratios, (0.25, 0.5, 0.75))
        ax.plot((q25, q75), (index, index), color="#222222", linewidth=2.2, zorder=4)
        ax.scatter(median, index, marker="D", s=34, color="#222222", zorder=5)
        wins = sum(row["hybrid_wins"] for row in subset)
        ax.annotate(
            f"{median:.2g}x; {wins}/{len(subset)}",
            (median, index),
            xytext=(5, -9),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )
    positive = [value for value in all_ratios if value > 0]
    low = min(0.02, min(positive) / 1.5) if positive else 0.02
    high = max(200.0, max(positive) * 1.5) if positive else 200.0
    ax.axvspan(1, high, color="#009E73", alpha=0.055, linewidth=0)
    ax.axvline(1, color="#333333", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.set_yticks(range(len(TASKS)), [LABELS[task] for task in TASKS])
    ax.set_ylim(len(TASKS) - 0.5, -0.5)
    ax.grid(axis="x", which="major", color="#D7D7D7", linewidth=0.55)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Paired loss ratio, RL / full duplex (log scale)")
    if interim:
        ax.set_title("INCOMPLETE ARTIFACT FREEZE", loc="left", fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--figure-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()

    rows, missing = collect()
    if missing and not args.allow_incomplete:
        preview = "\n".join(missing[:12])
        raise SystemExit(f"Missing {len(missing)} required artifacts:\n{preview}")
    if not rows:
        raise SystemExit("No matched primary results found")
    summaries = summarize(rows)
    stem = "primary_results" if not missing else "primary_results_interim"
    figure_stem = "primary_cross_task" if not missing else "primary_cross_task_interim"
    write_csv(rows, args.output_dir / f"{stem}_endpoints.csv")
    write_csv(summaries, args.output_dir / f"{stem}_summary.csv")
    write_tex(summaries, args.output_dir / f"{stem}_table.tex")
    (args.output_dir / f"{stem}_summary.json").write_text(
        json.dumps({"summary": summaries, "missing": missing}, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(rows, args.figure_dir / figure_stem, interim=bool(missing))
    print(f"Wrote {len(rows)} paired endpoints; missing {len(missing)} artifacts")


if __name__ == "__main__":
    main()
