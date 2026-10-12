"""Post-hoc qualification-threshold curves from saved complete RPA design ledgers."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from comparisons.search_exploration.analyze_rpa_statistical import COLORS, LABELS
from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT = {
    "julia": ROOT / "comparisons/search_exploration/reports/julia-rpa-100epoch-restart-20260915/run",
    "evolution": Path("/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/run"),
    "ga": Path("/local1/rossin/search-exploration/rpa-remaining-controls-20260915/standalone-ga/run"),
}


def topology_losses(path):
    """Count a topology once at its best complete score, regardless of eviction."""
    with sqlite3.connect(f"file:{path / 'exploration.sqlite'}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT json_extract(payload,'$.topology_hash'), "
            "min(json_extract(payload,'$.loss')) FROM designs "
            "WHERE json_extract(payload,'$.simulation_complete')=1 "
            "AND json_extract(payload,'$.topology_hash') IS NOT NULL "
            "AND json_type(payload,'$.loss') IN ('integer','real') "
            "GROUP BY json_extract(payload,'$.topology_hash')").fetchall()
    losses = [float(loss) for _, loss in rows if np.isfinite(loss) and loss >= 0]
    if len(losses) != len(rows):
        raise ValueError(f"Invalid complete topology scores in {path}")
    return np.sort(losses)


def qualifying_counts(losses, thresholds):
    return np.searchsorted(np.sort(losses), thresholds, side="left")


def complete_pairs(plan, progress):
    runtime = {j["name"]: j for j in progress["jobs"]}
    available = {(j["arm"], j["seed"]): j for j in plan["jobs"]
        if runtime.get(j["name"], {}).get("status") == "complete"}
    seeds = [s for s in plan["seeds"] if all((a, s) in available for a in plan["arms"])]
    return seeds, [available[a, s] for s in seeds for a in plan["arms"]]


def render(root):
    root = Path(root).resolve()
    plan = read_status(root / "active_plan.json") or read_status(root / "plan.json")
    progress = read_status(root / plan.get("progress_file", "progress.json"))
    seeds, jobs = complete_pairs(plan, progress)
    frozen = plan["threshold"]
    selected_thresholds = [.001, .002, .005, .01, frozen, .02, .03, .05, .1]
    thresholds = np.unique(np.r_[np.geomspace(.0005, .1, 501), selected_thresholds])
    records = []
    for cohort, entries in (("campaign", [(j["arm"], j["seed"], Path(j.get("output", root / "runs" / j["name"]))) for j in jobs]),
                            ("development", [(a, 0, p) for a, p in DEVELOPMENT.items()])):
        for arm, seed, path in entries:
            manifest = read_status(path / "manifest.json")
            completed = read_status(path / "completed.json")
            if not manifest or not completed or completed.get("status") != "complete":
                raise ValueError(f"Endpoint curve requires complete saved data: {path}")
            if manifest["task"] != "rpa" or manifest["config"]["quality_threshold"] != frozen:
                raise ValueError(f"Unexpected task/threshold: {path}")
            if manifest["config"]["seed"] != seed:
                raise ValueError("Seed mismatch")
            losses = topology_losses(path)
            records.append({"cohort": cohort, "arm": arm, "seed": seed, "path": str(path),
                "best_complete_loss": float(losses[0]) if len(losses) else None,
                "valid_topologies_discovered": len(losses), "evaluations": completed["counters"]["expensive_evaluations"],
                "curve": qualifying_counts(losses, thresholds).tolist(),
                "selected_counts": qualifying_counts(losses, selected_thresholds).tolist()})
    output = root / "analysis/threshold_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", {"matched_complete_seeds": seeds, "frozen_threshold": frozen,
        "thresholds": thresholds.tolist(), "selected_thresholds": selected_thresholds,
        "records": records, "definition": "distinct labelled topology hashes with best complete recorded loss strictly below threshold",
        "post_hoc": True, "changes_primary_threshold": False})
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cohort", "arm", "seed", "threshold", "qualifying_topologies"])
        for record in records:
            writer.writerows([record["cohort"], record["arm"], record["seed"], float(t), n]
                for t, n in zip(thresholds, record["curve"]))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import ScalarFormatter, FixedLocator
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
        upper = 10.
        for ax, cohort, title in zip(axes, ("campaign", "development"),
                (f"A  Current campaign: {len(seeds)} matched seed{'s' if len(seeds)!=1 else ''}",
                 "B  Earlier development runs: seed 0")):
            ax.axvline(frozen, color=".4", linestyle="--", linewidth=1., zorder=0)
            ax.text(frozen, 1.025, "Frozen threshold", transform=ax.get_xaxis_transform(),
                fontsize=8, color=".35", ha="center")
            for arm in plan["arms"]:
                rows = [r for r in records if r["cohort"] == cohort and r["arm"] == arm]
                if not rows:
                    continue
                values = np.asarray([r["curve"] for r in rows])
                mean = values.mean(axis=0)
                upper = max(upper, float(mean.max()))
                ax.plot(thresholds, mean, color=COLORS[arm], lw=2.2, label=LABELS[arm])
                index = int(np.flatnonzero(thresholds == frozen)[0])
                ax.scatter([frozen], [mean[index]], color=COLORS[arm], s=26, zorder=4)
                if len(rows) > 1:
                    sd = values.std(axis=0, ddof=1)
                    upper = max(upper, float((mean+sd).max()))
                    ax.fill_between(thresholds, np.maximum(0, mean-sd), mean+sd, color=COLORS[arm], alpha=.12)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=10, linscale=1)
            ax.set(xlim=(.0005, .1), xlabel="Loss threshold (stricter on the left)")
            ax.set_title(title, loc="left", pad=30)
            ax.xaxis.set_major_locator(FixedLocator([.0005, .001, .002, .005, .01, .02, .05, .1]))
            ax.set_xticklabels(["0.0005", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1"], fontsize=8)
            ax.yaxis.set_major_locator(FixedLocator([0, 5, 10, 50, 100, 500, 1000, 10000, 100000]))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", which="major", color=".9", linewidth=.6)
        axes[0].set_ylabel("Distinct qualifying topologies discovered")
        axes[0].set_ylim(0, upper*1.2)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3, frameon=False)
        fig.text(.5, .015, "Count axis: linear from 0 to 10, logarithmic above 10. Development runs are not pooled with the campaign.",
            ha="center", fontsize=8, color=".35")
        fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.81, wspace=.13)
        fig.savefig(output / "qualifying_topologies_vs_threshold.png", dpi=220)
        fig.savefig(output / "qualifying_topologies_vs_threshold.pdf")
        plt.close(fig)
    lines = ["# Qualifying Topologies Versus Loss Threshold", "",
        f"Current-campaign complete three-arm matched seeds: {seeds}. No unfinished endpoint is pooled.", "",
        "![Threshold curves](qualifying_topologies_vs_threshold.png)", "",
        "Each topology is counted once, at its lowest recorded loss among complete simulations. "
        "Counts come from the full design ledger, not the live archive or its frozen qualified flag. "
        "The inequality is strict: loss < threshold. Invalid/incomplete simulations never qualify. "
        "Intermediate optimizer trials not submitted as candidate designs are not counted.", "",
        "Panel A uses only fully completed matched seed blocks from the current campaign. "
        "Panel B shows the earlier seed-0 development runs separately; their configurations, "
        "admission schedules and compute expenditure are not identical to the current protocol. "
        "No cross-panel pooling or significance claim is made. With multiple matched seeds, "
        "lines show means and shading shows one sample standard deviation, not a confidence interval. "
        "With one seed there is no uncertainty band.", "",
        "The vertical dashed line is the frozen primary threshold. This post-hoc sensitivity analysis "
        "does not change it. Larger counts at permissive thresholds do not imply better RPA performance "
        "or independent mechanisms. Rates, solver and task loss are unchanged; no simulations rerun.", "",
        "## Counts at Selected Thresholds", "",
        "| Cohort / seed / method | " + " | ".join(f"{t:.5g}" for t in selected_thresholds) + " |",
        "|---|" + "---:|" * len(selected_thresholds)]
    for record in records:
        lines.append(f"| {record['cohort']} / {record['seed']} / {LABELS[record['arm']]} | " +
            " | ".join(str(n) for n in record["selected_counts"]) + " |")
    lines += ["", "Refresh without touching the frozen campaign implementation:", "", "```bash",
        ".venv/bin/python -m comparisons.search_exploration.plot_rpa_thresholds " + str(root), "```", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(render(args.root))
