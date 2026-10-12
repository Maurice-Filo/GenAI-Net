"""Self-only direct-proposal loss at recorded result arrival times, not launches."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sqlite3

import numpy as np

from comparisons.search_exploration.plot_self_llm_request_curves import (
    PILOT, TASKS, LABELS, LLM_COLOR, RL_COLOR, historical, pilot, read, write_csv,
)


def arrival_curve(events, query):
    events = sorted(events, key=lambda r: (r["arrival_minutes"], r["request"]))
    times = np.array([r["arrival_minutes"] for r in events])
    if np.any(times < 0):
        raise ValueError("Arrival predates run initialization")
    # Recompute prefixes in ARRIVAL order: launch-order bests can contain future arrivals.
    values = np.minimum.accumulate([
        r["batch_best_loss"] if r["batch_best_loss"] is not None else np.inf for r in events
    ])
    indices = np.searchsorted(times, query, side="right") - 1
    return np.where(indices >= 0, values[np.maximum(indices, 0)], np.inf)


def bands(values):
    ordered = np.sort(values, axis=0)
    n = len(ordered)
    # Empirical upper order statistics remain defined when some seeds have no result.
    return tuple(ordered[int(np.ceil(q * (n - 1)))] for q in (.25, .5, .75))


def pilot_arrivals():
    rows, failure = pilot()
    histories = read(PILOT / "history.json")
    starts = [datetime.fromisoformat(read(Path(h["workspace"]) / "run_manifest.json")["created_at"]).timestamp()
              for h in histories]
    with sqlite3.connect((PILOT / "exploration.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        completions = [r[0] for r in db.execute("SELECT created FROM events WHERE kind='evaluation_completed' ORDER BY created")]
    for i, row in enumerate(rows):
        end = starts[i + 1] if i + 1 < len(starts) else float("inf")
        times = [t for t in completions if starts[i] <= t < end]
        if not times:
            raise ValueError("No evaluation completion timestamp for pilot batch")
        row["arrival_minutes"] = (max(times) - starts[0]) / 60
    return rows, failure


def analyze(output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows, inputs = historical()
    parity, failure = pilot_arrivals()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "observed_arrivals.csv", rows)
    write_csv(output / "parity_arrivals.csv", parity)
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), gridspec_kw={"height_ratios": [2.2, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.79, wspace=.32, hspace=.30)
    fig.suptitle("Self-only LLM search: improvement as results arrive", fontsize=17, weight="bold", y=.98)
    fig.text(.5, .925, "Flash | RPA, original Logic and classifier: 20 seeds each | no outward RL/HOF conditioning", ha="center", color="#555555")
    fig.legend(handles=[Line2D([], [], color=LLM_COLOR, lw=2, label="Median best available direct-LLM loss"),
        Patch(color=LLM_COLOR, alpha=.18, label="Across-seed interquartile band"),
        Line2D([], [], color=RL_COLOR, ls="--", label="RL-only terminal median")],
        loc="upper center", bbox_to_anchor=(.5, .90), ncol=3, frameon=False, fontsize=10)
    exported, stats = [], []
    for col, task in enumerate(TASKS):
        task_rows = [r for r in rows if r["task"] == task]
        grid = np.unique([0., *[r["arrival_minutes"] for r in task_rows]])
        values = np.array([arrival_curve([r for r in task_rows if r["seed"] == seed], grid) for seed in range(20)])
        lo, median, hi = bands(values)
        reference = task_rows[0]["rl_median_reference"]
        top, bottom = axes[:, col]
        available = np.isfinite(values).sum(axis=0)
        reached = (values <= reference).sum(axis=0)
        top.fill_between(grid, lo, hi, where=np.isfinite(lo) & np.isfinite(hi),
                         color=LLM_COLOR, alpha=.18, step="post")
        top.step(grid, np.where(np.isfinite(median), median, np.nan), where="post", color=LLM_COLOR, lw=2)
        top.axhline(reference, color=RL_COLOR, ls="--", lw=1.5)
        top.set(title=f"{'ABC'[col]}   {LABELS[task]}", yscale="log", ylabel="Best available loss (log scale)")
        bottom.step(grid, available, where="post", color="#777777", ls=":", lw=1.8, label="With evaluated result")
        bottom.step(grid, reached, where="post", color=LLM_COLOR, lw=2, label="At/below RL reference")
        bottom.set(ylim=(-.7, 21.2), yticks=[0, 10, 20], ylabel="Seeds / 20", xlabel="Elapsed minutes from run initialization")
        if col == 0:
            bottom.legend(loc="lower right", fontsize=8, frameon=False)
        for ax in (top, bottom):
            ax.set_xlim(0, grid[-1] * 1.02)
            ax.grid(axis="y", alpha=.18)
        first = [min(r["arrival_minutes"] for r in task_rows if r["seed"] == seed) for seed in range(20)]
        last = [max(r["arrival_minutes"] for r in task_rows if r["seed"] == seed) for seed in range(20)]
        stats.append({"task": task, "median_first_arrival_minutes": float(np.median(first)),
                      "median_final_arrival_minutes": float(np.median(last)),
                      "final_reaching_reference": int(reached[-1])})
        for i, t in enumerate(grid):
            exported.append({"task": task, "minutes": t, "q25": float(lo[i]) if np.isfinite(lo[i]) else None,
                "upper_median": float(median[i]) if np.isfinite(median[i]) else None,
                "q75": float(hi[i]) if np.isfinite(hi[i]) else None,
                "seeds_with_results": int(available[i]), "seeds_reaching_reference": int(reached[i])})
    fig.text(.075, .035, "Arrival = host collection of an evaluated batch. No backdating to launch; no finite loss before a result arrives.\n"
             "Reference attainment is not functional correctness. Timings include scheduling and host load, not just inference.",
             fontsize=10, color="#555555")
    fig.savefig(output / "SELF_LLM_ARRIVAL_TIME.png", dpi=200)
    fig.savefig(output / "SELF_LLM_ARRIVAL_TIME.pdf")

    pfig, ax = plt.subplots(1, 2, figsize=(11.6, 4.7))
    pfig.subplots_adjust(top=.77, bottom=.24, left=.08, right=.98, wspace=.30)
    pfig.suptitle("Parity-12 pilot: results at canonical evaluation completion", fontsize=15, weight="bold", y=.98)
    pfig.text(.5, .885, "One sequential Flash run | four completed batches | no RL, HOF feedback or rate fitting", ha="center", color="#555555")
    times = [r["arrival_minutes"] for r in parity]
    losses = [r["best_so_far_loss"] for r in parity]
    counts = [r["cumulative_correct_topologies"] for r in parity]
    ax[0].step(times, losses, where="post", color=LLM_COLOR, lw=2)
    ax[0].scatter(times, losses, color=LLM_COLOR, s=28, zorder=3)
    ax[0].set(yscale="log", ylabel="Best available weighted L1 loss", title="A   Evaluated best-so-far")
    ax[1].step([0., *times], [0, *counts], where="post", color=LLM_COLOR, lw=2)
    ax[1].scatter(times, counts, color=LLM_COLOR, s=28, zorder=3)
    ax[1].set(ylabel="Distinct fully classifying topologies", title="B   Correct on all 16 truth-table rows", ylim=(0, max(counts)+2))
    for i, (t, loss) in enumerate(zip(times, losses)):
        ax[0].annotate(str(i), (t, loss), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
    for a in ax:
        a.set(xlabel="Elapsed minutes from first workspace creation", xlim=(0, times[-1]*1.06))
        a.grid(axis="y", alpha=.18)
    pfig.text(.08, .045, "Point labels are request indices. Arrival uses the last logged canonical evaluation in each batch.\n"
              "Request 4 timed out; remaining planned requests have no results. One sequence provides no uncertainty band.",
              fontsize=10, color="#555555")
    pfig.savefig(output / "PARITY12_ARRIVAL_TIME.png", dpi=200)
    pfig.savefig(output / "PARITY12_ARRIVAL_TIME.pdf")
    with PdfPages(output / "SELF_LLM_ARRIVAL_REPORT.pdf") as pdf:
        pdf.savefig(fig)
        pdf.savefig(pfig)
    plt.close(fig)
    plt.close(pfig)
    write_csv(output / "aggregate_arrival_curves.csv", exported)
    (output / "summary.json").write_text(json.dumps({"historical": stats, "parity": parity,
        "pilot_failure": failure, "inputs": inputs, "new_model_calls": 0, "new_simulations": 0}, indent=2) + "\n")
    lines = ["# Self-LLM Results by Arrival Time", "",
        "![Arrival time](SELF_LLM_ARRIVAL_TIME.png)", "",
        "Only direct supplied-rate LLM evaluations from no-communication runs. All 300 historical workspaces have empty HOFs, withheld SIL/exclusions and no outward exploration-state file. Self-feedback is allowed, but asynchronous requests can start before earlier feedback arrives.", "",
        "Historical arrival timestamps are llm_runs.created_at (host collection/recording), relative to training_runs.created_at. These are not provider completion timestamps: generation, queueing, canonical evaluation, scheduling and collection delay all contribute. Recompute best-so-far in arrival order, not launch order. Historical calls were scheduled by independent RL epochs even though the proposal processes did not exchange results. This is descriptive timing, not a controlled standalone speed benchmark.", "",
        "Before any result, a seed has infinite loss. Every seed remains in the aggregate; no conditioning on early responders. Median and quartiles use empirical upper order statistics (indices ceil(q*(n-1))). Missing/infinite bands remain absent. Completed seeds retain their known best while others finish; no unobserved improvement is extrapolated. Bands are across-seed IQR, not confidence intervals or significance tests. The RL endpoint median is a fixed reference, not a functional success threshold.", "",
        "| Task | Median first arrival (min) | Median last arrival (min) | Final seeds reaching reference |",
        "|---|---:|---:|---:|"]
    for r in stats:
        lines.append(f"| {r['task']} | {r['median_first_arrival_minutes']:.2f} | {r['median_final_arrival_minutes']:.2f} | {r['final_reaching_reference']}/20 |")
    lines += ["", "![Parity arrival time](PARITY12_ARRIVAL_TIME.png)", "",
        "Parity is a separate single sequential pilot with the newer prompt. Its arrival proxy is the last SQLite evaluation_completed timestamp in each request window, relative to the first workspace creation. Cache hits have no new completion event; each observed batch includes fresh evaluations. This measures numerical completion, which can slightly precede whole-batch host handoff. Four completed batches only; the later 900-second Harness timeout is retained, not plotted as a new loss observation.", "",
        "No matched self-only dose-response/oscillator cohorts exist in the audited registry; later full-duplex proposals are not substituted. No new experiments, simulations or model calls. Source artifacts and manuscript unchanged."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"historical": stats, "parity": parity}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
