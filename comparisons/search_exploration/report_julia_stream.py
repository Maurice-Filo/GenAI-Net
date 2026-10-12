"""Render local Julia-stream pilot artifacts without recomputation or uploads."""

import argparse
import json
from pathlib import Path


def render(root):
    manifest = json.loads((root / "manifest.json").read_text())
    completed = json.loads((root / "completed.json").read_text())
    jobs = json.loads((root / "optimization_results.json").read_text())
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    pruning = json.loads((root / "post_pruning/results.json").read_text()) if (root / "post_pruning/results.json").exists() else None
    counters, metrics = completed["counters"], completed["metrics"]
    joint = counters.get("julia/joint_evaluations", 0)
    total = counters.get("expensive_evaluations", 0)
    raw = [json.loads(path.read_text()) for path in sorted((root / "julia_jobs").glob("*.json")) if path.name != "workers.json"]
    rescued = sum(not j["job"]["candidate"]["simulation_complete"]
        and bool((j.get("outcome", {}).get("best") or {}).get("simulation_complete")) for j in raw)
    statuses = {}
    for job in raw:
        for phase in job["phases"]:
            for trace in phase["traces"]:
                statuses[trace["status"]] = statuses.get(trace["status"], 0) + 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    epoch = [r["epoch"] for r in rows]
    markers = {"marker": "o", "markersize": 3, "markevery": max(1, len(rows)//20)}
    axes[0, 0].plot(epoch, [r["Archive/best_loss"] for r in rows], color="#2878b5", **markers)
    axes[0, 0].set(ylabel="Best canonical loss", ylim=(0, 1.1 * max(r["Archive/best_loss"] for r in rows)))
    losses = [r["Archive/best_loss"] for r in rows]
    if min(losses) > 0 and max(losses) / min(losses) > 20:
        axes[0, 0].set(yscale="log", ylim=(min(losses)*.8, max(losses)*1.1))
    axes[0, 1].plot(epoch, [r["Archive/qualified_topologies"] for r in rows], color="#14866d", **markers)
    axes[0, 1].set(ylabel="Qualifying archive topologies", ylim=(0, max(1, 1.1 * max(r["Archive/qualified_topologies"] for r in rows))))
    axes[0, 1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[1, 0].plot(epoch, [r.get("Compute/julia/joint_evaluations", 0) for r in rows], color="#14866d", label="Joint loss/gradient")
    axes[1, 0].plot(epoch, [r["Compute/expensive_evaluations"] - r.get("Compute/julia/joint_evaluations", 0) for r in rows], color="#2878b5", label="Canonical")
    axes[1, 0].set(ylabel="Evaluation attempts")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(epoch, [100 * r["Julia/measured_fit_share"] for r in rows], color="#a63b59", **markers)
    axes[1, 1].axhline(100 * manifest["julia_recipe"]["compute_share"], color="#555555", linestyle="--", label="Admission target")
    axes[1, 1].set(ylabel="Measured fitting\nworker-time share (%)")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("Logged boundary")
        shown = rows if len(rows) <= 6 else [rows[i] for i in sorted({0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1})]
        ax.set_xticks([r["epoch"] for r in shown], [f"{r['epoch']+1}" if r["epoch"] < completed["epochs"] else "Drain" for r in shown])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    label = "full single-seed evaluation" if completed["epochs"] >= 100 else "engineering pilot"
    fig.suptitle("RPA asynchronous Julia fitting: " + label)
    fig.savefig(root / "julia_summary.png", dpi=200, bbox_inches="tight", pad_inches=.15)
    fig.savefig(root / "julia_summary.pdf", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)
    lines = ["# Julia Stream Evaluation", "", "Single-seed evaluation, not an equal-compute method comparison.", "",
        f"- Status: {completed['status']}; {completed['epochs']} epochs, batch {manifest['config']['batch_size']}.",
        f"- Search / Julia workers: {manifest['search_workers']} / {manifest['optimization_workers']}.",
        f"- Search + drain + optional pruning wall time: {completed['elapsed_seconds']:.1f} seconds.",
        f"- Best canonical loss: {metrics['Archive/best_loss']:.9g}.",
        f"- Archive: {metrics['Archive/topologies']} topologies, {metrics['Archive/qualified_topologies']} qualifying.",
        f"- Fitting jobs merged: {len(jobs)}; improving their inherited candidate: {metrics['Optimization/improved']}.",
        f"- Initially incomplete inherited candidates rescued to canonical completion: {rescued}.",
        f"- Search-stage evaluations: {total} = {total-joint} canonical + {joint} joint loss/gradient attempts.",
        f"- Julia scenario solves: {counters.get('julia/scenario_solves', 0)}; failed attempts remain charged.",
        f"- Terminal pending jobs / reservations: {completed['pending_optimization_jobs']} / {counters.get('julia/reserved', 0)}.",
        f"- Measured fitting worker-time share: {100*metrics['Julia/measured_fit_share']:.1f}% "
        f"(target {100*manifest['julia_recipe']['compute_share']:.0f}%).", "",
        "The share is an admission controller, not a hard real-time partition. It uses completed "
        "search evaluator seconds and fitting-worker elapsed seconds (including preparation and canonical "
        "verification), not OS CPU accounting. Cold fitting/JIT and pending jobs can overshoot a short pilot. "
        "Worker startup is excluded from that share but included in run wall time.", "",
        "Canonical and joint evaluations have different costs. The count budget prevents overbooking; "
        "it does not make them equivalent compute units.", "", "## Fitter Outcomes", "",
        "Counts below are per optimizer call, not independent topologies; screening and continuation are separate calls.", "",
        "| Status | Calls |", "|---|---:|"]
    lines += [f"| {status} | {count} |" for status, count in sorted(statuses.items())]
    lines += ["", "## Pruning", ""]
    if pruning:
        lines += [f"Verified parents: {len(pruning['parents'])}; accepted deletions: "
            f"{sum(p['derived'] is not None for p in pruning['parents'])}; separate evaluations: {pruning['evaluations']}.",
            "Original archive entries are unchanged. `post_pruning/results.json` contains derived provenance and rejected trials."]
    else:
        lines += ["Not requested for this run."]
    lines += ["", "![Pilot summary](julia_summary.png)", "", "## Reproduction", "",
        "`manifest.json` and `source.tar.gz` freeze search settings/source; `julia_jobs/workers.json` "
        "records Julia/helper/manifest fingerprints. Per-job JSON records rate starts, failures, "
        "optimizer checkpoints, budgets, and canonical validation. LLM calls were disabled. "
        f"Comet logging: {'enabled' if (root / 'comet.json').exists() else 'disabled'}.", "",
        "Additional matched seeds and controlled resource comparisons are needed to estimate search benefit. "
        "This single run cannot establish a general diversity or optimizer superiority claim."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
