"""Saved-results-only evaluation of asynchronous versus synchronous exploration."""

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np


def method_label(summary):
    arm = summary["proposal_arm"]
    return {"mutation-random-no-rl": "No RL", "rl-random-no-mutation": "No archive mutation",
        "standalone-ga": "Standalone GA (bounded)"}.get(arm,
        "Hybrid " + summary["method"].rsplit("-", 1)[-1]
        if summary["method"].startswith("search-streams-") else "Synchronous")


def read_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    status = json.loads((path / "status.json").read_text()) if (path / "status.json").exists() else {"status": "initializing"}
    records = {}
    if (path / "metrics.jsonl").exists():
        for line in (path / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            records[row["epoch"]] = row
    rows = [records[k] for k in sorted(records)]
    times = {}
    outcomes = {}
    contributions = []
    first_qualified_epochs = []
    if rows:
        with sqlite3.connect("file:" + str(path / "exploration.sqlite") + "?mode=ro", uri=True) as database:
            database.execute("BEGIN")
            for epoch, timestamp in database.execute("SELECT json_extract(payload, '$.epoch'), MAX(created) "
                    "FROM events WHERE kind='archive_insertion' GROUP BY json_extract(payload, '$.epoch')"):
                times[epoch] = timestamp - (path / "manifest.json").stat().st_mtime
            outcomes["incomplete_simulations"] = database.execute("SELECT count(*) FROM evaluation_summaries WHERE json_extract(payload,'$.simulation_complete')=0").fetchone()[0]
            outcomes["failed_evaluation_events"] = database.execute("SELECT count(*) FROM events WHERE kind='evaluation_failed'").fetchone()[0]
            outcomes["discovered_qualifying_topologies"] = database.execute("SELECT count(DISTINCT json_extract(payload,'$.topology_hash')) FROM designs WHERE json_extract(payload,'$.qualified')=1").fetchone()[0]
            first_qualified_epochs = sorted(row[0] for row in database.execute(
                "SELECT MIN(json_extract(e.payload,'$.epoch')) FROM events e JOIN designs d "
                "ON d.key=json_extract(e.payload,'$.candidate_hash') "
                "WHERE e.kind='archive_insertion' AND json_extract(d.payload,'$.qualified')=1 "
                "GROUP BY json_extract(d.payload,'$.topology_hash')"))
            for stream, source, count, gain in database.execute("SELECT json_extract(payload,'$.stream'), json_extract(payload,'$.source'), count(*), SUM(json_extract(payload,'$.gain')) FROM events WHERE kind='archive_insertion' GROUP BY json_extract(payload,'$.stream'), json_extract(payload,'$.source')"):
                contributions.append({"stream": stream or "synchronous_fitting", "source": source,
                    "records": count, "archive_utility_gain": gain})
    for row in rows:
        row["Discovery/qualified_topologies"] = sum(e <= row["epoch"] for e in first_qualified_epochs)
    metrics = status.get("metrics", rows[-1] if rows else {})
    summary = {"path": str(path), "method": manifest["schema"], "status": status["status"],
        "proposal_arm": manifest.get("proposal_arm", "rl-mutation-random"),
        "task": manifest["task"], "quality_threshold": manifest["config"]["quality_threshold"],
        "epochs": status.get("epochs", status.get("epoch", 0)),
        "loss": metrics.get("Archive/best_loss"), "qualified_topologies": metrics.get("Archive/qualified_topologies"),
        "evaluations": metrics.get("Compute/expensive_evaluations", 0),
        "search_evaluations": metrics.get("Compute/evaluations/search"),
        "optimization_evaluations": metrics.get("Compute/evaluations/optimization"),
        "fit_failures": metrics.get("Fit/failed_jobs_cumulative", sum(r.get("Fit/failed_jobs", 0) for r in rows)),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "optimization_improved": metrics.get("Optimization/improved"),
        "optimization_merged": metrics.get("Optimization/merged"),
        "pending_optimization": status.get("pending_optimization_jobs", metrics.get("Optimization/pending")),
        "sqlite_bytes": sum(f.stat().st_size for f in path.glob("*.sqlite*")),
        "stream_contributions": contributions, **outcomes}
    if status["status"] == "complete" and (manifest["schema"].startswith("search-streams-")
            or manifest["schema"] == "standalone-ga-bounded-v1"):
        from comparisons.search_exploration.audit_streams import audit
        summary["artifact_audit"] = audit(path)
    return summary, rows, times


def analyze(paths, output, discovered=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [read_run(p) for p in paths]
    if len({(r[0]["task"], r[0]["quality_threshold"]) for r in runs}) != 1:
        raise ValueError("Comparison requires the same task and qualification threshold")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    count_field = "Discovery/qualified_topologies" if discovered else "Archive/qualified_topologies"
    summary_count = "discovered_qualifying_topologies" if discovered else "qualified_topologies"
    count_label = "Distinct qualifying topologies discovered" if discovered else "Qualifying archived topologies"
    colors = ("#2673b8", "#d07127", "#228460", "#a94464", "#777777")
    if len(runs) > len(colors):
        raise ValueError("At most five labeled comparison arms supported")
    for (summary, rows, times), color in zip(runs, colors):
        color = {"rl-mutation-random": "#2673b8", "mutation-random-no-rl": "#d07127",
            "rl-random-no-mutation": "#a94464", "standalone-ga": "#228460"}.get(summary["proposal_arm"], color)
        if summary["method"] == "exploration-v1":
            color = "#d07127"
        label = method_label(summary)
        if summary["status"] != "complete":
            label += " (in progress)"
        for column, field in enumerate(("Archive/best_loss", count_field)):
            axes[0, column].step([r["Compute/expensive_evaluations"] for r in rows],
                [r[field] for r in rows], where="post", label=label, color=color)
            timed = [r for r in rows if r["epoch"] in times]
            axes[1, column].step([times[r["epoch"]] / 60 for r in timed],
                [r[field] for r in timed], where="post", label=label, color=color)
    for i in range(2):
        axes[i, 0].set_yscale("log")
        axes[i, 0].axhline(runs[0][0]["quality_threshold"], color="#777777", linestyle=":",
            linewidth=1, label="Qualification threshold")
        max_count = max((r.get(count_field, 0) for _, rows, _ in runs for r in rows), default=0)
        axes[i, 1].set_ylim(-.05, max(1., max_count * 1.08))
        axes[i, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
        for j in range(2):
            ax = axes[i, j]
            if i == 0:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))
            ax.set_xlabel("Candidate evaluations (CVODE)" if i == 0 else "Minutes to archive commit")
            ax.set_ylabel("Best canonical loss" if j == 0 else count_label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    summaries = [r[0] for r in runs]
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = ["# Independent search/optimization streams", "",
        "Saved-artifact analysis only; no evaluator or model calls.", "",
        f"Count metric: {count_label}.", "",
        "| Method | Status | Epochs / GA batches | Actual evaluations | Best loss | Qualifying topologies | Elapsed minutes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        elapsed = round(s["elapsed_seconds"] / 60, 2) if s["elapsed_seconds"] is not None else "pending"
        label = method_label(s)
        lines.append(f"| {label} | {s['status']} | {s['epochs']} | {s['evaluations']} | {s['loss']} | {s.get(summary_count, 0)} | {elapsed} |")
    if all(rows for _, rows, _ in runs):
        limit = min(rows[-1]["Compute/expensive_evaluations"] for _, rows, _ in runs)
        boundaries = []
        for summary, rows, _ in runs:
            eligible = [r for r in rows if r["Compute/expensive_evaluations"] <= limit]
            if eligible:
                r = eligible[-1]
                boundaries.append({"method": method_label(summary), "shared_limit": limit,
                    "recorded_evaluations": r["Compute/expensive_evaluations"],
                    "loss": r["Archive/best_loss"], "qualified_topologies": r[count_field], "count_metric": count_field})
        (output / "shared_budget_boundaries.json").write_text(json.dumps(boundaries, indent=2) + "\n")
        lines += ["", f"## Shared Cost Limit: {limit:,} Evaluations", "",
            "Latest recorded boundary at or below the shared limit; counts are shown",
            "explicitly because checkpoints do not occur at identical evaluation counts.",
            "No interpolation or exact matched-budget claim is made.", "",
            "| Method | Recorded evaluations | Best loss | Qualifying topologies |",
            "| --- | ---: | ---: | ---: |"]
        for b in boundaries:
            lines.append(f"| {b['method']} | {b['recorded_evaluations']} | {b['loss']} | {b['qualified_topologies']} |")
    has_synchronous = any(s["method"] == "exploration-v1" for s in summaries)
    lines += ["", "![Comparison](comparison.png)", "", "## Interpretation", "",
        "This is a single-seed engineering evaluation, not a significance or equivalence test.",
        "One evaluation scores one complete reaction/rate vector over the canonical task scenarios,",
        "not one individual solver trajectory. In-flight reservations are charged conservatively.",
        "Same epoch schedules and evaluation caps do not imply the same consumed budget.",
        *(["Asynchronous search evaluates starting rates immediately and refines a selected subset;",
        "the synchronous reference refines every proposal before updating. Both scheduling and",
        "rate-treatment of on-policy rewards change, so this is not a pure storage-speed ablation.",
        "The synchronous exploration prototype and asynchronous v1 retain the original archive",
        "eviction rule, which can discard high-utility non-representatives before poorer entries",
        "when the archive is full. Their counts describe that executed retention policy."] if has_synchronous else []),
        *(["Version 2 also corrects the inherited non-representative archive eviction utility sign;",
           "the frozen synchronous reference retains the original rule. This additional difference",
           "must not be attributed to asynchronous communication alone."]
          if has_synchronous and any(s["method"] == "search-streams-v2" for s in summaries) else []),
        *(["The no-RL control replaces the RL proposal share with random structures and",
           "skips both policy-gradient and imitation updates. Archive mutation share, corrected",
           "retention, evaluator, rate fitter and queue settings are preserved. Both arms use",
           "asynchronous selected refinement, not the old synchronous implementation.",
           "This tests removal of the learned emitter, not superiority to a tuned standalone GA.",
           "Random draws and job completion schedules are not paired after the intervention.",
           "The same 100-epoch proposal budget and total cap do not fix actual evaluator usage."]
          if any(s["proposal_arm"] == "mutation-random-no-rl" for s in summaries) else []),
        *(["The no-mutation control replaces archive-mutation proposals with random proposals;",
           "the RL share, rate fitter, archive and imitation are unchanged. It is not legacy",
           "RL-only joint structure/rate learning: evolutionary rate fitting remains present."]
          if any(s["proposal_arm"] == "rl-random-no-mutation" for s in summaries) else []),
        *(["Standalone GA reuses the original 45% rate-mutation / 45% reaction-replacement /",
           "10% random-restart operators and a top-30 parent archive, with explicit rate clipping",
           "to the shared bounds and no LLM replay. Its original log-uniform initialization",
           "prior is preserved, not replaced by the hybrid's constant starting rate.",
           "It runs to the reference's actual physical-evaluation count, not 100 RL epochs.",
           "Duplicate evaluations use the same exact-cache machinery; proposal counts can differ.",
           "Parent archive capacities remain algorithm-specific. All-discovered counts avoid",
           "a reporting cap but do not equalize the search effects of those different archives."]
          if any(s["proposal_arm"] == "standalone-ga" for s in summaries) else []),
        *(["Discovered counts deduplicate topology hashes across the full recorded design ledger,",
           "including designs subsequently evicted. First qualification is assigned to its",
           "recorded search/merge boundary, not an inferred worker completion time."] if discovered else []),
        "Compare observed loss and qualifying counts versus actual evaluations; do not extrapolate",
        "the shorter trajectory or treat unmatched endpoints as a controlled superiority result.",
        "Wall-clock curves use recorded archive-commit timestamps relative to manifest creation,",
        "not interpolated epoch durations. Concurrent machine load can affect those times.",
        "All dispatched optimizer jobs must drain before an asynchronous run is called complete.",
        "A zero qualifying count is reported as zero, not hidden behind relative loss improvements.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovered", action="store_true", help="Plot all discovered qualifying topologies, not archive retention")
    args = parser.parse_args()
    analyze(args.runs, args.output, discovered=args.discovered)
