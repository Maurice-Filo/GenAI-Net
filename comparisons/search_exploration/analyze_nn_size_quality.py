"""Analyze saved NN-size pilot data without running evaluators or model calls."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def paired_summary(rows, task):
    complete = {(r["seed"], r["embedding"]): r for r in rows if r["task"] == task and r["status"] == "complete"}
    seeds = sorted(s for s, size in complete if size == 1024 and (s, 10240) in complete)
    pairs = [(complete[s, 1024], complete[s, 10240]) for s in seeds]
    def counts(key, smaller):
        wins = ties = losses = 0
        for a, b in pairs:
            x, y = a[key], b[key]
            if x is None or y is None:
                continue
            tied = math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) if smaller else x == y
            if tied:
                ties += 1
            elif (x < y if smaller else x > y):
                wins += 1
            else:
                losses += 1
        return {"wins": wins, "ties": ties, "losses": losses}
    ratios = [a["loss"]/b["loss"] for a, b in pairs if a["loss"] is not None and b["loss"] is not None and b["loss"] > 0]
    return {"task": task, "matched_complete_seeds": seeds, "n": len(seeds),
            "unscorable_loss_pairs": sum(a["loss"] is None or b["loss"] is None for a, b in pairs),
            "small_vs_large_loss": counts("loss", True),
            "small_vs_large_qualified_topologies": counts("qualified_topologies", False),
            "median_paired_loss_ratio": statistics.median(ratios) if ratios else None}


def analyze(root):
    from RL4CRN.search_exploration.engine import atomic_json
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    rows, histories = [], []
    for job in plan["jobs"]:
        folder = root / "runs" / job["name"]
        metrics = []
        if (folder / "metrics.jsonl").exists():
            # Interrupted final JSONL lines are not evidence of a committed epoch.
            for line in (folder / "metrics.jsonl").read_text().splitlines():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        metrics = list({m["epoch"]: m for m in metrics}.values())
        metrics.sort(key=lambda m: m["epoch"])
        completed = json.loads((folder / "completed.json").read_text()) if (folder / "completed.json").exists() else None
        failed = (folder / "failed.json").exists()
        status = "complete" if completed else "failed" if failed else "running" if folder.exists() else "queued"
        last = completed["metrics"] if completed else metrics[-1] if metrics else {}
        row = {"task": job["task"], "seed": job["seed"], "embedding": job["embedding"], "status": status,
               "epochs": completed["epochs"] if completed else (metrics[-1]["epoch"]+1 if metrics else 0),
               "evaluations": completed["counters"].get("expensive_evaluations", 0) if completed else last.get("Compute/expensive_evaluations", 0),
               "loss": last.get("Archive/best_loss"), "qualified_topologies": last.get("Archive/qualified_topologies"),
               "qualified_regions": last.get("Archive/qualified_regions"),
               "fit_failed_jobs": sum(m.get("Fit/failed_jobs", 0) for m in metrics),
               "elapsed_seconds": completed.get("elapsed_seconds") if completed else None}
        rows.append(row)
        histories.append((row, metrics))
    summaries = [paired_summary(rows, t) for t in plan["tasks"]]
    atomic_json(output / "summary.json", {"endpoints": rows, "paired": summaries,
        "caveat": "Exploratory five-seed pilot. Endpoint comparisons use complete matched pairs only; no significance claim.",
        "budget": plan["budget"]})
    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# NN size: search-quality pilot", "",
            f"Campaign: `{root}`", "", plan["budget"], "",
            "Only intermediate dimension differs within each matched task/seed pair. No LLM is used.",
            "The smaller network is 1024; the reference is 10240. Both use canonical CVODE scoring.", "",
            "| Task | Complete pairs | Loss W/T/L (small vs large) | Qualified-topology W/T/L | Median loss ratio |",
            "| --- | ---: | --- | --- | --- |"]
    for summary in summaries:
        show = lambda c: f"{c['wins']}/{c['ties']}/{c['losses']}"
        ratio = summary["median_paired_loss_ratio"]
        text.append(f"| {summary['task']} | {summary['n']} | {show(summary['small_vs_large_loss'])} | "
                    f"{show(summary['small_vs_large_qualified_topologies'])} | {ratio if ratio is not None else 'pending'} |")
    text += ["", "## Status", "",
             f"Complete: {sum(r['status']=='complete' for r in rows)} / {len(rows)}; "
             f"failed: {sum(r['status']=='failed' for r in rows)}; "
             f"recorded fitting failures: {sum(r['fit_failed_jobs'] for r in rows)}.", "",
             "A shared cap is not a guarantee of identical consumed evaluations: cache reuse, timeouts and the epoch horizon matter.",
             "Use the saved evaluation-axis trajectories and actual endpoint counts alongside the paired endpoint table.",
             "Counts describe the live discovery archive. Descriptor coverage is not mechanistic diversity.",
             "Logic qualification additionally requires all truth-table rows correct; RPA uses its frozen loss threshold.",
             "No automatic retry or silent failure exclusion is performed. Five seeds provide pilot evidence, not an equivalence test.", "",
             f"Complete pairs without a scorable loss: {sum(s['unscorable_loss_pairs'] for s in summaries)}.", "",
             "![Convergence](convergence.png)", "",
             "Each line is one seed; dashed lines are unfinished runs. Values are held constant between observed cohort boundaries.", ""]
    (output / "REPORT.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(2, len(plan["tasks"]), figsize=(12, 7), squeeze=False, constrained_layout=True)
    colors = {1024: "#2878b5", 10240: "#666666"}
    for column, task in enumerate(plan["tasks"]):
        for row, history in histories:
            if row["task"] != task or not history:
                continue
            for index, key in enumerate(("Archive/best_loss", "Archive/qualified_topologies")):
                valid = [m for m in history if m.get(key) is not None and (index != 0 or m[key] > 0)]
                axes[index, column].step([m["Compute/expensive_evaluations"] for m in valid],
                    [m[key] for m in valid], where="post", color=colors[row["embedding"]], alpha=.6,
                    linestyle="-" if row["status"] == "complete" else "--")
        for index, label in enumerate(("Best canonical loss", "Qualifying archived topologies")):
            ax = axes[index, column]
            for size, color in colors.items():
                ax.plot([], [], color=color, label=f"{size:,}")
            ax.set(xlabel="Actual canonical evaluations", ylabel=label, title=task.upper())
            if index == 0:
                ax.set_yscale("log")
            else:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(title="Intermediate dimension", frameon=False)
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))
