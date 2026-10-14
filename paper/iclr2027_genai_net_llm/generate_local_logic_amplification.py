#!/usr/bin/env python3
"""Reconstruct time-resolved, source-aware portfolios from audited local Logic runs."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.iclr2027_genai_net_llm.generate_local_half_duplex_analysis import (
    ARMS, AUDITS, COLORS, GENERATED, THRESHOLDS, audited_run_roots,
)

PAPER = Path(__file__).resolve().parent
STEM = "local_logic_amplification"
THRESHOLD = THRESHOLDS["logic"]
LABELS = {"none": "Independent union", "rl_to_llm": "RL/HOF to LLM",
          "llm_to_rl": "LLM to RL", "full": "Full duplex"}
CATEGORIES = {
    "llm_sufficient": ("Already successful from LLM", "#D17B24"),
    "rl_rescued": ("RL crosses threshold on LLM-origin topology", "#16845B"),
    "rl_additional": ("RL topology absent from LLM proposals", "#2463A5"),
    "rl_shared": ("Other RL success on a shared topology", "#888888"),
}


def best_by_topology(rows):
    result = {}
    for row in rows:
        if row.get("loss") is None or not np.isfinite(row["loss"]):
            continue
        key = row["topology_hash"]
        if key not in result or row["loss"] < result[key]["loss"]:
            result[key] = row
    return result


def available_proposals(proposals, timestamp):
    # Availability is the recorded completion event, never the request epoch.
    return [row for row in proposals if row["created_at"] <= timestamp]


def pooled_portfolio(hof, proposals, capacity=30):
    best = best_by_topology([*hof, *proposals])
    return sorted(best.values(), key=lambda row: (row["loss"], row["topology_hash"]))[:capacity]


def contributions(portfolio, proposals, threshold=THRESHOLD):
    llm = best_by_topology(proposals)
    counts = dict.fromkeys(CATEGORIES, 0)
    for row in portfolio:
        if row["loss"] >= threshold:
            continue
        prior = llm.get(row["topology_hash"])
        if prior and prior["loss"] < threshold:
            category = "llm_sufficient"
        elif prior and row.get("provenance_class") == "rl_parameter_refinement_of_llm_topology":
            category = "rl_rescued"
        elif prior:
            category = "rl_shared"
        else:
            category = "rl_additional"
        counts[category] += 1
    return counts


def load_run(arm, seed, root, expected_hash, threshold=THRESHOLD):
    database = root / "results.sqlite"
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"Database has uncheckpointed WAL: {database}")
    hasher = hashlib.sha256()
    with database.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"Database differs from passing audit: {database}")
    connection = sqlite3.connect(f"file:{quote(str(database))}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        snapshots = [dict(row) for row in connection.execute(
            "SELECT * FROM hof_snapshots ORDER BY created_at, epoch")]
        entries = defaultdict(list)
        for row in connection.execute("SELECT * FROM hof_snapshot_entries ORDER BY rank"):
            entries[row["snapshot_id"]].append(dict(row))
        proposals = [dict(row) for row in connection.execute(
            """SELECT c.topology_hash, c.loss, r.created_at, r.completed_epoch,
                      r.launched_epoch, 'direct_llm' AS provenance_class
               FROM llm_candidates c JOIN llm_runs r USING(llm_run_id)
               WHERE c.valid = 1 AND c.loss IS NOT NULL ORDER BY r.created_at""")]
        failed = connection.execute("SELECT COUNT(*) FROM llm_failures").fetchone()[0]
    finally:
        connection.close()
    if len([s for s in snapshots if s["boundary"] == "terminal_boundary"]) != 1:
        raise RuntimeError(f"Expected exactly one terminal boundary: {database}")
    records = []
    for snapshot in snapshots:
        hof = entries[snapshot["snapshot_id"]]
        if len(hof) != 30 or len(best_by_topology(hof)) != 30:
            raise RuntimeError(f"HOF is not 30 distinct topologies: {database}")
        visible = available_proposals(proposals, snapshot["created_at"])
        pool = pooled_portfolio(hof, visible)
        terminal = snapshot["boundary"] == "terminal_boundary"
        if terminal:
            actual, reconstructed = best_by_topology(hof), best_by_topology(pool)
            if actual.keys() != reconstructed.keys() or any(
                not np.isclose(actual[key]["loss"], reconstructed[key]["loss"], rtol=1e-9, atol=1e-12)
                for key in actual
            ):
                raise RuntimeError(f"Terminal reconstruction disagrees with stored HOF: {database}")
        counts = contributions(pool, visible, threshold)
        llm_best = best_by_topology(visible)
        improved_shared = sum(
            row.get("provenance_class") == "rl_parameter_refinement_of_llm_topology"
            and row["topology_hash"] in llm_best
            and row["loss"] < llm_best[row["topology_hash"]]["loss"] - 1e-12
            for row in pool
        )
        records.append({
            "arm": arm, "seed": seed, "boundary": snapshot["boundary"],
            "epoch": snapshot["epoch"],
            "completed_epochs": snapshot["epoch"] + (snapshot["boundary"] == "epoch"),
            "evaluations": snapshot["cumulative_total_evaluations"],
            "rl_evaluations": snapshot["cumulative_rl_evaluations"],
            "llm_evaluations": snapshot["cumulative_llm_final_evaluations"],
            "llm_tool_evaluations": snapshot["cumulative_llm_tool_evaluations"],
            "available_llm_candidates": len(visible),
            "llm_only_qualified": min(30, sum(row["loss"] < threshold for row in llm_best.values())),
            "qualified": sum(counts.values()), "best_loss": pool[0]["loss"],
            "improved_llm_origin_topologies": int(improved_shared), **counts,
        })
    return records, {"arm": arm, "seed": seed, "database": str(database),
                     "database_sha256": digest, "model_failures_retained": failed}


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(20260908).integers(0, len(values), size=(2000, len(values)))
    boot = values[indices].mean(axis=1)
    return values.mean(axis=0), *np.quantile(boot, [0.025, 0.975], axis=0)


def step_values(rows, grid):
    rows = sorted(rows, key=lambda row: (row["evaluations"], row["completed_epochs"]))
    x = np.asarray([row["evaluations"] for row in rows])
    y = np.asarray([row["qualified"] for row in rows])
    indices = np.searchsorted(x, grid, side="right") - 1
    if (indices < 0).any() or grid[-1] > x[-1]:
        raise ValueError("Requested evaluation grid would extrapolate")
    return y[indices]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manuscript_figures(training, terminal, summaries, curves, grid, *, arms=ARMS, stem=STEM):
    """Keep labels readable at the manuscript's single-column text width."""
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 6.8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.8))
        fig.subplots_adjust(left=.075, right=.995, bottom=.35, top=.91, wspace=.37)
        for arm in arms:
            paths = []
            for seed in range(20):
                rows = {row["completed_epochs"]: row for row in training[arm, seed]}
                paths.append([rows[e]["qualified"] for e in range(1, 101)])
            mean, lo, hi = bootstrap_mean(paths)
            axes[0].plot(range(1, 101), mean, color=COLORS[arm], lw=1.5, label=LABELS[arm])
            axes[0].fill_between(range(1, 101), lo, hi, color=COLORS[arm], alpha=.09)
            ends = [row["qualified"] for row in terminal if row["arm"] == arm]
            m, l, h = bootstrap_mean(ends)
            offset = (list(arms).index(arm)-(len(arms)-1)/2)*1.6
            axes[0].errorbar(113+offset, m, yerr=[[m-l], [h-m]], fmt="o", color=COLORS[arm], ms=2.5, capsize=1.5)
        axes[0].axvspan(104, 120, color="#eeeeee", zorder=-5)
        axes[0].set(xlabel="Completed epochs", ylabel="Mean successful topologies",
                    xlim=(0,120), xticks=[0,25,50,75,100,114],
                    xticklabels=["0","25","50","75","100","Final"])
        axes[0].tick_params(axis="x", labelsize=7)
        axes[0].set_title("A  Pooled top-30 yield", loc="left", fontweight="bold")
        axes[0].legend(loc="upper left", bbox_to_anchor=(-.07,-.27), frameon=False, ncol=2,
                       columnspacing=.8, handlelength=1.4)
        short_labels = {"llm_sufficient":"Already successful from LLM",
                        "rl_rescued":"RL threshold crossing on LLM topology",
                        "rl_additional":"RL topology absent from LLM proposals",
                        "rl_shared":"Other RL success on shared topology"}
        bottom = np.zeros(len(arms))
        for key, (_, color) in CATEGORIES.items():
            values = [row[key] for row in summaries]
            if not any(values):
                continue
            axes[1].bar(range(len(arms)), values, bottom=bottom, width=.65, color=color, label=short_labels[key])
            bottom += values
        for i, value in enumerate(bottom):
            axes[1].text(i,value+.15,f"{value:.2f}",ha="center",fontsize=8)
        tick_labels = {"none":"Indep.\nunion","rl_to_llm":"RL/HOF\nto LLM",
                       "llm_to_rl":"LLM\nto RL","full":"Full\nduplex"}
        axes[1].set(xticks=range(len(arms)), xticklabels=[tick_labels[arm] for arm in arms],
                    ylabel="Mean successful topologies", ylim=(0,max(bottom)*1.22))
        axes[1].tick_params(axis="x",labelsize=7)
        axes[1].set_title("B  Final source contributions",loc="left",fontweight="bold")
        axes[1].legend(loc="upper left",bbox_to_anchor=(-.12,-.27),frameon=False,handlelength=1.4)
        for ax in axes:
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
        for extension in ("pdf","png"):
            fig.savefig(PAPER / "figures" / f"{stem}_main.{extension}",dpi=300,bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.7,3.2))
        for arm in arms:
            if arm == "none" or arm == "rl_to_llm":
                continue
            mean,lo,hi=bootstrap_mean(curves[arm]-curves["none"])
            ax.plot(grid/1000,mean,color=COLORS[arm],label=LABELS[arm],lw=1.8)
            ax.fill_between(grid/1000,lo,hi,color=COLORS[arm],alpha=.13)
        ax.axhline(0,color="#666666",lw=.8,ls="--")
        ax.set(xlabel="Total candidate evaluations (thousands)",
               ylabel="Mean paired gain in successful topologies",xlim=(0,103))
        ax.legend(loc="upper left",frameon=False,fontsize=8)
        ax.grid(axis="y",alpha=.18)
        fig.tight_layout()
        for extension in ("pdf","png"):
            fig.savefig(PAPER / "figures" / f"{stem}_budget.{extension}",dpi=300,bbox_inches="tight")
        plt.close(fig)


def main():
    trajectories, audit = [], []
    for arm in ARMS:
        roots = audited_run_roots("logic", arm)
        if set(roots) != set(range(20)):
            raise RuntimeError(f"Expected all 20 passing seeds in {arm}")
        hashes = {}
        for name in AUDITS["logic", arm]:
            data = json.loads((GENERATED / name).read_text())
            hashes.update({row["seed"]: row["database_sha256"] for row in data["runs"]
                           if row["task"] == "logic" and row["status"] == "pass"})
        for seed, root in sorted(roots.items()):
            rows, metadata = load_run(arm, seed, root, hashes[seed])
            trajectories.extend(rows)
            audit.append(metadata)
    terminal = [row for row in trajectories if row["boundary"] == "terminal_boundary"]
    training = defaultdict(list)
    for row in trajectories:
        if row["boundary"] != "terminal_boundary":
            training[row["arm"], row["seed"]].append(row)
    start = max(min(row["evaluations"] for row in rows) for rows in training.values())
    stop = min(max(row["evaluations"] for row in rows) for rows in training.values())
    grid = np.linspace(start, stop, 101)
    curves = {arm: np.asarray([step_values(training[arm, seed], grid) for seed in range(20)])
              for arm in ARMS}
    summaries = []
    for arm in ARMS:
        rows = sorted((row for row in terminal if row["arm"] == arm), key=lambda row: row["seed"])
        summaries.append({"arm": arm, "n": len(rows),
                          "mean_qualified": float(np.mean([row["qualified"] for row in rows])),
                          "median_qualified": float(np.median([row["qualified"] for row in rows])),
                          "median_loss": float(np.median([row["best_loss"] for row in rows])),
                          **{key: float(np.mean([row[key] for row in rows])) for key in CATEGORIES},
                          "mean_improved_llm_origin_topologies": float(np.mean([
                              row["improved_llm_origin_topologies"] for row in rows]))})
    gains = []
    baseline = {row["seed"]: row for row in terminal if row["arm"] == "none"}
    for arm in ("llm_to_rl", "full"):
        delta = np.asarray([row["qualified"] - baseline[row["seed"]]["qualified"]
                            for row in terminal if row["arm"] == arm])
        gains.append({"arm": arm, "mean_gain": float(delta.mean()),
                      "median_gain": float(np.median(delta)), "wins": int((delta > 0).sum()),
                      "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), gridspec_kw={"width_ratios": [1.15, 1.1, 1]})
    fig.subplots_adjust(left=.052, right=.985, bottom=.34, top=.80, wspace=.33)
    for arm in ARMS:
        epoch_curves = []
        endpoints = []
        for seed in range(20):
            rows = {row["completed_epochs"]: row for row in training[arm, seed]}
            epoch_curves.append([rows[e]["qualified"] for e in range(1, 101)])
            endpoints.append(next(row["qualified"] for row in terminal if row["arm"] == arm and row["seed"] == seed))
        mean, lo, hi = bootstrap_mean(epoch_curves)
        axes[0].plot(range(1, 101), mean, color=COLORS[arm], label=LABELS[arm], lw=2)
        axes[0].fill_between(range(1, 101), lo, hi, color=COLORS[arm], alpha=.08)
        m, l, h = bootstrap_mean(endpoints)
        offset = (list(ARMS).index(arm)-1.5)*1.2
        axes[0].errorbar(110+offset, m, yerr=[[m-l], [h-m]], fmt="o", color=COLORS[arm], ms=4, capsize=2)
    axes[0].axvspan(103, 115, color="#eeeeee", zorder=-5)
    axes[0].set(xlabel="Completed training epochs", ylabel="Mean successful topologies in pooled top 30",
                xticks=[0, 25, 50, 75, 100, 110], xticklabels=["0", "25", "50", "75", "100", "Final"], xlim=(0,115))
    axes[0].set_title("A   Pooled solution yield", loc="left", fontweight="bold", pad=13)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-.02, -.22), ncol=2, frameon=False, fontsize=9)
    bottom = np.zeros(4)
    for key, (label, color) in CATEGORIES.items():
        values = [row[key] for row in summaries]
        if not any(values):
            continue
        axes[1].bar(range(4), values, bottom=bottom, width=.65, color=color, label=label)
        bottom += values
    for i, value in enumerate(bottom):
        axes[1].text(i, value+.14, f"{value:.2f}", ha="center", fontsize=10)
    axes[1].set(xticks=range(4), xticklabels=["Independent\nunion", "RL/HOF\nto LLM", "LLM\nto RL", "Full\nduplex"],
                ylabel="Mean successful topologies at final pooling", ylim=(0,max(bottom)*1.2))
    axes[1].set_title("B   What RL adds to direct proposals", loc="left", fontweight="bold", pad=13)
    axes[1].legend(loc="upper left", bbox_to_anchor=(-.05, -.26), frameon=False, fontsize=8.5)
    for arm in ("llm_to_rl", "full"):
        mean, lo, hi = bootstrap_mean(curves[arm]-curves["none"])
        axes[2].plot(grid/1000, mean, color=COLORS[arm], lw=2, label=LABELS[arm])
        axes[2].fill_between(grid/1000, lo, hi, color=COLORS[arm], alpha=.13)
    axes[2].axhline(0, color="#666666", lw=.8, ls="--")
    axes[2].set(xlabel="Total candidate evaluations (thousands)", ylabel="Mean paired gain over independent union", xlim=(0,103))
    axes[2].set_title("C   Additional yield at matched budgets", loc="left", fontweight="bold", pad=13)
    axes[2].legend(loc="upper left", bbox_to_anchor=(-.02, -.22), frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Does RL expand the successful LLM proposal portfolio?", fontsize=17, y=.96)
    fig.text(.5,.875, "Local Qwen model | Logic | 20 matched seeds per arm | 100 epochs | Success: loss < 0.04271",
             ha="center", fontsize=11, color="#555555")
    fig.text(.052,.018,"LLM candidates count only after completion. Shading: pointwise 95% seed-bootstrap intervals. Final includes late replies. No structural ancestry is inferred.",fontsize=9,color="#555555")
    for extension in ("png", "pdf"):
        fig.savefig(PAPER / "figures" / f"{STEM}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    manuscript_figures(training, terminal, summaries, curves, grid)
    write_csv(GENERATED / f"{STEM}_trajectories.csv", trajectories)
    write_csv(GENERATED / f"{STEM}_endpoints.csv", terminal)
    output = {"threshold": THRESHOLD, "capacity": 30, "summaries": summaries,
              "paired_gains": gains, "database_audit": audit,
              "evaluation_grid": [float(start),float(stop)],
              "terminal_reconciliation": "All 80 reconstructed terminal portfolios match stored topologies and losses"}
    (GENERATED / f"{STEM}.json").write_text(json.dumps(output,indent=2)+"\n")
    report = ["# Local Logic: proposal amplification diagnostic", "",
              "![Portfolio diagnostic](figures/local_logic_amplification.png)", "",
              "Completed local-model cohort: all four arms, seeds 0-19, 100 epochs. No new simulations.", "",
              "| Arm | Mean successful topologies | Median | LLM already sufficient | RL threshold crossings | Additional RL topologies | Other shared RL |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        report.append(f"| {LABELS[row['arm']]} | {row['mean_qualified']:.2f} | {row['median_qualified']:.1f} | {row['llm_sufficient']:.2f} | {row['rl_rescued']:.2f} | {row['rl_additional']:.2f} | {row['rl_shared']:.2f} |")
    report += ["", "## Interpretation", ""]
    for gain in gains:
        report.append(f"- {LABELS[gain['arm']]} minus independent union: mean gain {gain['mean_gain']:.2f}, median gain {gain['median_gain']:.1f}; paired yield W/T/L {gain['wins']}/{gain['ties']}/{gain['losses']}.")
    report += ["", "Most successful designs in the insertion arms are RL-emitted topologies absent from their own run's completed LLM proposals. The portfolio improvement therefore extends beyond directly retaining successful LLM candidates. Threshold crossings on previously inserted LLM-origin topologies contribute a smaller part of the gain.",
               "", "The yield advantage grows mainly late in the training horizon. This is consistent with a delayed benefit of inserting proposals into RL search; these plots do not isolate elite imitation from every other effect of insertion, nor prove structural descent from an LLM design.",
               "", "These are local-model Logic results. They do not establish an incremental benefit of the HOF return channel, and must not be generalized to Flash or RPA without their own comparisons."]
    report += ["", "## Definitions and checks", "",
        "- Success is the existing frozen Logic threshold, loss < 0.04271267898491677. This counts useful topologies, not a formal structural diversity distance.",
        "- At each snapshot, pool its current HOF with valid completed LLM candidates, retain the lowest loss per complete structural topology hash, and select the best 30. This is a retrospective read-only pool, not an intervention into the uncoupled RL runs.",
        "- LLM availability uses recorded completion timestamps relative to snapshot timestamps, not launch epochs. The training curves stop at the training boundary; the separate Final point includes late replies.",
        "- Panel B gives priority to LLM sufficiency: if any LLM proposal already achieves success on a topology, it belongs to the orange segment even if RL later improves its loss. Exact RL re-emissions do not count as additional designs.",
        "- The green segment requires a successful RL-emitted parameter refinement of a previously inserted LLM-origin topology for which all completed LLM proposals remain above the success threshold. A change of parameters alone is insufficient.",
        "- Blue means a successful RL-emitted topology absent from that run's valid completed LLM proposals. It does not establish ancestry, mutation from an LLM design, or causation for an individual topology. The matched communication intervention supports a cohort-level coupling comparison.",
        "- An optional gray segment covers shared-topology RL successes without the required exposed-LLM provenance. Source categories are operational attribution, not counterfactual mediation estimates.",
        "- Panel C uses completed training snapshots, previous-observation interpolation, and a common evaluation range without extrapolation. Counts include RL, final LLM, and tool evaluations. No terminal waiting response is injected into the training budget curve.",
        "- Confidence bands are pointwise 95% bootstrap intervals of means across 20 seeds (2,000 resamples, RNG seed 20260908); panel C resamples paired seed differences. Bands are descriptive, not a test at every epoch.",
        "- Every database SHA-256 matches its passing audit. All 80 reconstructed terminal pools match the stored terminal topology sets and losses. Model-output failures in passing runs are retained.",
        "- RL records are HOF observations, not every rollout. These panels report a bounded top-30 portfolio, not all successful networks ever evaluated. No dollar-cost or structural-lineage claim is made.",
        "", "## Reproduce", "", "```bash", "MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m paper.iclr2027_genai_net_llm.generate_local_logic_amplification", "```", "",
        "Artifacts: `figures/local_logic_amplification.{png,pdf}`, `generated/local_logic_amplification{,_trajectories,_endpoints}.{json,csv}`."]
    (PAPER / "LOCAL_LOGIC_AMPLIFICATION.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"summaries":summaries,"paired_gains":gains,"verified_databases":len(audit)},indent=2))


if __name__ == "__main__":
    main()
