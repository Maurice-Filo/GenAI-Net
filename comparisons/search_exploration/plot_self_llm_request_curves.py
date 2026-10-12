"""Observed direct-LLM request prefixes with no outward RL/HOF conditioning."""

import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
PILOT = Path("/local1/rossin/search-exploration/parity12-flash-llm-only-20round-20260918")
TASKS = ("rpa", "logic", "classifier")
LABELS = {"rpa": "RPA", "logic": "Original Logic", "classifier": "Classifier"}
LLM_COLOR, RL_COLOR = "#a64e85", "#168577"


def read(path):
    return json.loads(path.read_text())


def valid_loss(loss, valid):
    return bool(valid and loss is not None and math.isfinite(loss) and loss >= 0)


def request_prefix(batches):
    """Failures retain an existing best but cannot manufacture a new observation."""
    best = math.inf
    values = []
    for batch in batches:
        losses = [r["loss"] for r in batch if valid_loss(r["loss"], r["valid"])]
        best = min(best, min(losses, default=math.inf))
        values.append(best)
    return values


def self_context(workspace):
    state = read(workspace / "CONTEXT/SEARCH_STATE.json")
    if state["hall_of_fame_size"] or read(workspace / "CONTEXT/HALL_OF_FAME.json")["entries"]:
        raise ValueError(f"Outward HOF exposed: {workspace}")
    if (workspace / "CONTEXT/EXPLORATION_STATE.json").exists():
        raise ValueError(f"Outward exploration state exposed: {workspace}")
    for name in ("SIL_STATUS.md", "EXCLUDED_TOPOLOGIES.md"):
        if "withheld" not in (workspace / "CONTEXT" / name).read_text().lower():
            raise ValueError(f"Outward state not explicitly withheld: {workspace / name}")
    return {"snapshot_epoch": state["rl_epoch_at_snapshot"],
        "has_self_feedback": state["llm_feedback"] != "No previous LLM feedback is available.",
        "has_self_best": state["llm_best"] != "No LLM-generated candidate has been accepted yet."}


def historical():
    primary = read(GENERATED / "contract_v2_primary_summary.json")
    references = {r["task"]: r["rl_median"] for r in primary["endpoint"]}
    rows, inputs = [], []
    for filename in ("contract_v2_none_seed0_audit.json", "contract_v2_none_seeds1to19_audit.json"):
        for run in read(GENERATED / filename)["runs"]:
            if run["status"] != "pass" or run["task"] not in TASKS:
                raise ValueError("Expected audited historical no-communication cohort")
            folder = Path(run["run_root"])
            manifest = read(folder / "run_manifest.json")
            if manifest["communication_mode"] != "none" or manifest["model"] != "deepseek-v4-flash":
                raise ValueError("Not a Flash no-communication run")
            contexts = {}
            for workspace in sorted((folder / "harness-workspaces").iterdir()):
                context = self_context(workspace)
                epoch = context["snapshot_epoch"]
                if epoch in contexts:
                    raise ValueError("Ambiguous repeated request epoch")
                contexts[epoch] = {**context, "workspace": str(workspace)}
            database = folder / "results.sqlite"
            wal = Path(str(database) + "-wal")
            if wal.exists() and wal.stat().st_size:
                raise ValueError("Read-only immutable access requires a finalized database")
            with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                db.row_factory = sqlite3.Row
                starts = db.execute("SELECT created_at FROM training_runs").fetchall()
                if len(starts) != 1:
                    raise ValueError("Ambiguous run time origin")
                started_at = starts[0]["created_at"]
                events = []
                for request in db.execute("SELECT * FROM llm_runs ORDER BY launched_epoch"):
                    candidates = [dict(r) for r in db.execute(
                        "SELECT loss,valid FROM llm_candidates WHERE llm_run_id=?", (request["llm_run_id"],))]
                    events.append({"launched_epoch": request["launched_epoch"],
                        "completed_epoch": request["completed_epoch"], "requested": request["requested"],
                        "arrived_at": request["created_at"],
                        "status": "completed", "candidates": candidates})
                for failure in db.execute("SELECT * FROM llm_failures ORDER BY launched_epoch"):
                    events.append({"launched_epoch": failure["launched_epoch"],
                        "completed_epoch": failure["completed_epoch"], "requested": failure["requested"],
                        "arrived_at": failure["created_at"],
                        "status": "failed", "candidates": []})
            events.sort(key=lambda r: r["launched_epoch"])
            if [r["launched_epoch"] for r in events] != [0, 20, 40, 60, 80]:
                raise ValueError("Unexpected request schedule; do not silently pool unequal horizons")
            prefix = request_prefix([r["candidates"] for r in events])
            for index, (event, best) in enumerate(zip(events, prefix)):
                finite = [r["loss"] for r in event["candidates"] if valid_loss(r["loss"], r["valid"])]
                rows.append({"task": run["task"], "seed": run["seed"], "request": index,
                    "launched_epoch": event["launched_epoch"], "completed_epoch": event["completed_epoch"],
                    "run_started_at": started_at, "arrived_at": event["arrived_at"],
                    "arrival_minutes": (event["arrived_at"] - started_at) / 60,
                    "status": event["status"], "requested": event["requested"], "valid": len(finite),
                    "batch_best_loss": min(finite, default=None),
                    "best_so_far_loss": best if math.isfinite(best) else None,
                    "rl_median_reference": references[run["task"]],
                    "reaches_rl_reference": best <= references[run["task"]],
                    **contexts[event["launched_epoch"]]})
            inputs.append({"task": run["task"], "seed": run["seed"], "run_root": str(folder),
                "database_sha256_from_prior_audit": run["database_sha256"],
                "task_prompt_sha256": manifest["task_prompt_sha256"]})
    return rows, inputs


def aggregate(rows):
    summary = []
    for task in TASKS:
        for request in range(5):
            group = [r for r in rows if r["task"] == task and r["request"] == request]
            if len(group) != 20 or {r["seed"] for r in group} != set(range(20)):
                raise ValueError("Expected all twenty seeds at each request")
            values = np.array([r["best_so_far_loss"] if r["best_so_far_loss"] is not None else np.inf for r in group])
            # All historical prefixes must have results; never discard missing seeds from the band.
            if not np.isfinite(values).all():
                raise ValueError("Incomplete prefix requires explicit censored-summary plotting")
            lo, median, hi = np.quantile(values, [.25, .5, .75])
            summary.append({"task": task, "request": request, "n": 20,
                "q25": float(lo), "median": float(median), "q75": float(hi),
                "reaches_rl_reference": sum(r["reaches_rl_reference"] for r in group),
                "self_feedback_available": sum(r["has_self_feedback"] for r in group),
                "failed_requests": sum(r["status"] == "failed" for r in group),
                "rl_median_reference": group[0]["rl_median_reference"]})
    return summary


def pilot():
    rows, batches = [], []
    seen_correct = set()
    for history in read(PILOT / "history.json"):
        workspace = Path(history["workspace"])
        context = self_context(workspace)
        candidates = read(workspace / "canonical_records.json")
        complete = [r["record"] for r in candidates if r["record"] is not None
                    and r["valid"] and r["record"]["simulation_complete"]]
        batches.append([{"loss": r["loss"], "valid": True} for r in complete])
        correct = [r for r in complete if r["all_rows_correct"]]
        seen_correct.update(r["topology_hash"] for r in correct)
        best = request_prefix(batches)[-1]
        if not math.isclose(best, history["best_loss"], rel_tol=1e-12):
            raise ValueError("Pilot loss prefix does not match saved history")
        rows.append({"request": history["round"], "batch_best_loss": min((r["loss"] for r in complete), default=None),
            "best_so_far_loss": best, "complete_candidates": len(complete),
            "all_rows_correct_proposals_in_batch": len(correct),
            "cumulative_correct_topologies": len(seen_correct), **context})
    failed = read(PILOT / "failed.json") if (PILOT / "failed.json").exists() else None
    return rows, failed


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output, summary, pilot_rows, failed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.5))
    fig.subplots_adjust(top=.81, bottom=.12, left=.08, right=.98, hspace=.42, wspace=.32)
    fig.suptitle("LLM improvement without RL/HOF feedback", y=.98, fontsize=17, weight="bold")
    fig.text(.5, .923, "Historical Flash cohorts | 20 seeds per task | 10 proposals per request", ha="center", color="#555555")
    fig.legend(handles=[Line2D([], [], color=LLM_COLOR, marker="o", label="Direct LLM: median best-so-far"),
        Patch(facecolor=LLM_COLOR, alpha=.18, label="Across-seed interquartile range"),
        Line2D([], [], color=RL_COLOR, linestyle="--", label="RL-only endpoint median")],
        loc="upper center", bbox_to_anchor=(.5, .90), ncol=3, frameon=False, fontsize=10)
    for col, task in enumerate(TASKS):
        data = [r for r in summary if r["task"] == task]
        x = [r["request"] for r in data]
        top, bottom = axes[:, col]
        top.fill_between(x, [r["q25"] for r in data], [r["q75"] for r in data], color=LLM_COLOR, alpha=.18)
        top.plot(x, [r["median"] for r in data], color=LLM_COLOR, marker="o", lw=2)
        top.axhline(data[0]["rl_median_reference"], color=RL_COLOR, linestyle="--", lw=1.5)
        top.set(title=f"{'ABC'[col]}   {LABELS[task]}", yscale="log", ylabel="Best-so-far loss (log scale)")
        counts = [r["reaches_rl_reference"] for r in data]
        bottom.plot(x, np.array(counts) / 20 * 100, marker="o", color=LLM_COLOR, lw=2)
        for xx, count in zip(x, counts):
            bottom.annotate(f"{count}/20", (xx, count * 5), xytext=(0, 9 if count < 18 else -17),
                            textcoords="offset points", ha="center", fontsize=10)
        bottom.set(ylim=(-6, 106), yticks=[0, 25, 50, 75, 100],
                   ylabel="Seeds reaching RL reference (%)", xlabel="LLM request (0 = first batch)")
        for ax in (top, bottom):
            ax.set_xticks(x)
            ax.set_xlim(-.18, 4.18)
            ax.grid(axis="y", alpha=.18)
    fig.text(.08, .025, "Reference attainment is not functional correctness. Request order is not wall time; historical feedback can lag.",
             fontsize=10, color="#555555")
    fig.savefig(output / "SELF_LLM_REQUEST_CURVES.png", dpi=200)
    fig.savefig(output / "SELF_LLM_REQUEST_CURVES.pdf")

    parity, ax = plt.subplots(1, 2, figsize=(11.7, 4.9))
    parity.subplots_adjust(top=.79, bottom=.24, left=.08, right=.98, wspace=.3)
    parity.suptitle("Parity-12: self-feedback-only Flash pilot", fontsize=16, weight="bold", y=.98)
    parity.text(.5, .90, "One sequence, not a replicated cohort | 10 proposals per request | no RL or rate fitting", ha="center", color="#555555", fontsize=11)
    x = [r["request"] for r in pilot_rows]
    ax[0].plot(x, [r["batch_best_loss"] for r in pilot_rows], color="#777777", ls=":", marker="s", label="This batch's best")
    ax[0].plot(x, [r["best_so_far_loss"] for r in pilot_rows], color=LLM_COLOR, marker="o", lw=2, label="Best so far")
    ax[0].set(title="A   Trajectory loss", yscale="log", ylabel="Canonical weighted L1 loss")
    ax[0].legend(frameon=False, fontsize=10)
    counts = [r["all_rows_correct_proposals_in_batch"] for r in pilot_rows]
    ax[1].bar(x, counts, color=LLM_COLOR, width=.58)
    for xx, count in zip(x, counts):
        ax[1].text(xx, count + .22, f"{count}/10", ha="center", fontsize=11)
    ax[1].set(title="B   All 16 rows classified correctly", ylim=(0, 11.5), yticks=[0, 2, 4, 6, 8, 10],
              ylabel="Correct proposals in this batch")
    for axis in ax:
        axis.set_xlabel("LLM request (0 = first batch)")
        axis.set_xticks(x + ([len(x)] if failed else []))
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        if failed:
            axis.axvspan(len(x) - .4, len(x) + .4, color="#eeeeee")
            axis.text(len(x), .64, "Timeout\nno result", transform=axis.get_xaxis_transform(),
                      ha="center", fontsize=10, color="#555555")
        axis.set_xlim(-.5, len(x) + .5 if failed else len(x) - .5)
    parity.text(.08, .06, "Four completed batches; request 4 timed out. No extrapolation to the remaining planned requests.\n"
                "Classification uses final OUT with a 0.5 threshold; trajectory loss also charges transient deviations.",
                fontsize=10, color="#555555")
    parity.savefig(output / "PARITY12_SELF_LLM_REQUESTS.png", dpi=200)
    parity.savefig(output / "PARITY12_SELF_LLM_REQUESTS.pdf")
    with PdfPages(output / "SELF_LLM_ITERATION_REPORT.pdf") as pdf:
        pdf.savefig(fig)
        pdf.savefig(parity)
    plt.close(fig)
    plt.close(parity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, inputs = historical()
    summary = aggregate(rows)
    pilot_rows, failed = pilot()
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "request_seed_curves.csv", rows)
    write_csv(args.output / "aggregate_curves.csv", summary)
    write_csv(args.output / "parity_pilot.csv", pilot_rows)
    (args.output / "summary.json").write_text(json.dumps({"summary": summary, "parity_pilot": pilot_rows,
        "pilot_failure": failed, "inputs": inputs, "new_model_calls": 0, "new_simulations": 0}, indent=2) + "\n")
    plots(args.output, summary, pilot_rows, failed)
    lines = ["# Self-LLM Improvement by Request", "",
        "Direct supplied-rate LLM candidates only. No RL candidates, fitted descendants or terminal unions enter these curves.", "",
        "## Historical No-Communication Cohorts", "",
        "Twenty seeds per task, five requests of ten proposals. Every saved request context was checked for an empty HOF, withheld SIL/exclusions and no exploration-state file. LLM self-memory remains permitted. Some asynchronous launches preceded earlier feedback; counts below disclose actual availability.", "",
        "Curves are retrospective best-so-far prefixes in launch order, not RL epochs or wall-clock traces. All seeds remain in each summary. Shading is the 25th-75th percentile, not a confidence interval or significance test. The horizontal reference is the existing RL-only median endpoint loss, not a biological-success threshold. Loss values cannot be compared across tasks.", "",
        "| Task | Request | Median best loss | Reaches reference / 20 | Self-feedback present / 20 |",
        "|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['task']} | {r['request']} | {r['median']:.6g} | {r['reaches_rl_reference']} | {r['self_feedback_available']} |")
    lines += ["", "![Historical no-communication curves](SELF_LLM_REQUEST_CURVES.png)", "",
        "## Current Parity Pilot", "",
        "One strictly sequential sequence using the new workspace-loss prompt. This is not pooled with the older prompt cohorts. Four rounds completed; request 4 timed out at 900 seconds. The failed request has no plotted loss, and later planned requests are absent. Correct-proposal counts use complete canonical simulations and all sixteen final-output margins above the evaluator's tolerance; counts are proposals, not distinct topologies.", "",
        "![Parity pilot](PARITY12_SELF_LLM_REQUESTS.png)", "",
        "## Coverage Limitation", "",
        "The audited historical registry contains no matching self-only cohort for dose_hill, dose_ultrasensitive, dose_biphasic, oscillator_mean or oscillator_frequency. Their later full-duplex requests must not be relabeled as self-only. The earlier first-batch difficulty screen remains valid for their empty initial HOF, but cannot provide self-only learning curves.", "",
        "No new experiments, API calls or simulations were performed. Source databases were opened read-only; old results and manuscript files are unchanged."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"historical_runs": len(inputs), "requests": len(rows),
                     "summary": summary, "parity": pilot_rows}, indent=2))


if __name__ == "__main__":
    main()
