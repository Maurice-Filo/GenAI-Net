#!/usr/bin/env python3
"""Descriptive threshold and request-accounting checks of 160 frozen runs."""

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np
from matplotlib.patches import Patch

from paper.iclr2027_genai_net_llm.generate_local_logic_amplification import (
    COLORS, GENERATED, LABELS, PAPER, best_by_topology,
    bootstrap_mean, plt, write_csv,
)

THRESHOLDS = json.loads((GENERATED / "analysis_plan_v2.json").read_text())["quality_thresholds"]

COHORTS = {
    "local_logic": ("Logic / local model", "logic", ("none", "rl_to_llm", "llm_to_rl", "full")),
    "flash_rpa": ("RPA / Flash", "rpa", ("none", "full")),
    "flash_classifier": ("Classifier / Flash", "classifier", ("none", "full")),
}
SENSITIVITY_FIGURE_COHORTS = ("local_logic", "flash_rpa")
SHORT = {"none": "Independent", "rl_to_llm": "RL/HOF\nto LLM",
         "llm_to_rl": "LLM to RL", "full": "Full duplex"}
STATES = {
    "available": ("Valid batch, during training", "#2463A5"),
    "late_valid": ("Valid batch, after training", "#D17B24"),
    "empty": ("No usable candidates, during training", "#A5A5A5"),
    "late_empty": ("No usable candidates, after training", "#E0E0E0"),
}
MULTIPLIERS = np.unique(np.r_[np.geomspace(.25, 2, 81), .5, 1.0])


def qualified_counts(entries, threshold, multipliers):
    losses = np.asarray([row["loss"] for row in best_by_topology(entries).values()])
    losses = np.sort(losses)[:30]
    return (losses[:, None] < threshold * np.asarray(multipliers)[None, :]).sum(axis=0)


def count_support(values):
    values = np.asarray(values)
    return {"n_seeds": len(values), "median_qualified": float(np.median(values)),
            "std_qualified": float(np.std(values, ddof=1)),
            "min_qualified": int(values.min()), "max_qualified": int(values.max()),
            "zero_fraction": float((values == 0).mean())}


def classify_request(row, training_timestamp, epochs=100):
    # Collection occurs before the same epoch's agent.update; terminal waits do not.
    in_training = row["completed_epoch"] < epochs and row["created_at"] <= training_timestamp
    usable = row["finite_valid"] > 0
    state = ("available" if usable else "empty") if in_training else (
        "late_valid" if usable else "late_empty")
    return state, epochs - row["completed_epoch"] if in_training else 0


def read_database(meta):
    database = Path(meta["database"])
    wal = database.with_name(database.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError(f"Uncheckpointed WAL: {database}")
    digest = hashlib.sha256()
    with database.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != meta["database_sha256"]:
        raise RuntimeError(f"Audited database changed: {database}")
    connection = sqlite3.connect(f"file:{quote(str(database))}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        boundaries = [dict(row) for row in connection.execute(
            "SELECT * FROM hof_snapshots WHERE boundary IN ('training_boundary','terminal_boundary')")]
        if sorted(row["boundary"] for row in boundaries) != ["terminal_boundary", "training_boundary"]:
            raise RuntimeError(f"Missing or duplicate boundaries: {database}")
        training = next(row for row in boundaries if row["boundary"] == "training_boundary")
        terminal = next(row for row in boundaries if row["boundary"] == "terminal_boundary")
        entries = [dict(row) for row in connection.execute(
            "SELECT * FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank",
            (terminal["snapshot_id"],))]
        if len(entries) != 30 or len(best_by_topology(entries)) != 30:
            raise RuntimeError(f"Unexpected terminal portfolio: {database}")
        requests = []
        for row in connection.execute("SELECT * FROM llm_runs ORDER BY launched_epoch"):
            row = dict(row)
            candidates = [dict(c) for c in connection.execute(
                "SELECT valid,loss FROM llm_candidates WHERE llm_run_id=?", (row["llm_run_id"],))]
            if sum(c["valid"] for c in candidates) != row["valid_count"]:
                raise RuntimeError(f"Candidate validity accounting mismatch: {database}")
            row["finite_valid"] = sum(c["valid"] and c["loss"] is not None and np.isfinite(c["loss"])
                                      for c in candidates)
            row["failed"] = False
            requests.append(row)
        for row in connection.execute("SELECT * FROM llm_failures ORDER BY launched_epoch"):
            requests.append({**dict(row), "finite_valid": 0, "failed": True})
        # Count observed launch/outcome records, not the schedule alone.
        if sorted(row["launched_epoch"] for row in requests) != [0, 20, 40, 60, 80]:
            raise RuntimeError(f"Expected five distinct observed launches: {database}")
        for row in requests:
            if not row["launched_epoch"] <= row["completed_epoch"] <= 100:
                raise RuntimeError(f"Invalid completion epoch: {database}")
            state, updates = classify_request(row, training["created_at"])
            row["state"], row["remaining_updates"] = state, updates
        return entries, requests
    finally:
        connection.close()


def figures(sensitivity, endpoints, summaries):
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
        fig.subplots_adjust(left=.095, right=.985, top=.93, bottom=.18, hspace=.42, wspace=.26)
        for column, cohort in enumerate(SENSITIVITY_FIGURE_COHORTS):
            title, _, arms = COHORTS[cohort]
            upper, lower = axes[:, column]
            for arm in arms:
                rows = [r for r in sensitivity if r["cohort"] == cohort and r["arm"] == arm]
                x = [r["multiplier"] for r in rows]
                if arm != "none":
                    upper.plot(x, [r["mean_paired_gain"] for r in rows], color=COLORS[arm])
                    upper.fill_between(x, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows],
                                       color=COLORS[arm], alpha=.12)
                lower.plot(x, [r["mean_qualified"] for r in rows], color=COLORS[arm])
                lower.fill_between(
                    x, [max(0, r["mean_qualified"]-r["std_qualified"]) for r in rows],
                    [min(30, r["mean_qualified"]+r["std_qualified"]) for r in rows],
                    color=COLORS[arm], alpha=.12)
            upper.set_title(title)
            upper.axhline(0, color="#999999", lw=.7)
            lower.set_ylim(-.5, 30.5)
            lower.set_yticks([0, 10, 20, 30])
            lower.set_xlabel("Threshold / frozen threshold")
            for ax in (upper, lower):
                ax.set_xscale("log")
                ax.set_xticks([.25, .5, 1, 2], ["0.25", "0.5", "1", "2"])
                ax.minorticks_off()
                ax.axvline(1, color="#444444", ls=":", lw=1)
                ax.grid(axis="y", alpha=.15)
            if column == 0:
                upper.set_ylabel("Mean paired yield gain\nover independent union")
                lower.set_ylabel("Qualified topologies\n(mean $\\pm$ 1 SD)")
        fig.legend(handles=[Patch(color=COLORS[a], label=LABELS[a]) for a in COHORTS["local_logic"][2]],
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .015))
        save(fig, "portfolio_threshold_sensitivity")

        fig, axes = plt.subplots(2, 3, figsize=(7.0, 6.0))
        fig.subplots_adjust(left=.075, right=.99, top=.93, bottom=.20, hspace=.55, wspace=.32)
        for column, (cohort, (title, _, arms)) in enumerate(COHORTS.items()):
            upper, lower = axes[:, column]
            for index, arm in enumerate(arms):
                row = next(r for r in summaries if r["cohort"] == cohort and r["arm"] == arm)
                bottom = 0
                for state, (_, color) in STATES.items():
                    height = row[f"mean_{state}"]
                    upper.bar(index, height, bottom=bottom, color=color, width=.65,
                              edgecolor="white", linewidth=.4)
                    bottom += height
                values = np.asarray([r["qualified"] for r in endpoints
                                     if r["cohort"] == cohort and r["arm"] == arm])
                lower.bar(index, values.mean(), color=COLORS[arm], alpha=.25, width=.65)
                offsets = np.random.default_rng(20260908).uniform(-.20, .20, len(values))
                lower.scatter(index+offsets, values, s=12, color=COLORS[arm], alpha=.8, linewidths=0)
                lower.plot([index-.27,index+.27], [values.mean()]*2, color=COLORS[arm], lw=2)
            upper.set_title(title)
            upper.set_ylim(0, 5.3)
            upper.set_yticks([0,1,2,3,4,5])
            lower.set_ylim(-.6,30.8)
            for ax in (upper, lower):
                ax.set_xticks(range(len(arms)), [SHORT[a] for a in arms], rotation=30, ha="right", fontsize=7)
                ax.grid(axis="y", alpha=.15)
                ax.set_axisbelow(True)
            if column == 0:
                upper.set_ylabel("Proposal batches per run")
                lower.set_ylabel("Terminal successful topologies")
        fig.legend(handles=[Patch(color=c, label=label) for label,c in STATES.values()],
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .015))
        save(fig, "portfolio_request_opportunities")


def save(fig, stem):
    for extension in ("pdf", "png"):
        fig.savefig(PAPER / "figures" / f"{stem}.{extension}", dpi=240)
    plt.close(fig)


def main():
    local = json.loads((GENERATED / "local_logic_amplification.json").read_text())
    hosted = json.loads((GENERATED / "flash_portfolio_amplification.json").read_text())
    metadata = [{"cohort":"local_logic", **r} for r in local["database_audit"]]
    metadata += [{"cohort":f"flash_{r['task']}", **r} for r in hosted["database_audit"]]
    reference = {}
    for name in ("local_logic_amplification_endpoints.csv", "flash_portfolio_amplification_endpoints.csv"):
        with (GENERATED / name).open() as handle:
            for row in csv.DictReader(handle):
                cohort = f"flash_{row['task']}" if "task" in row else "local_logic"
                reference[cohort, row["arm"], int(row["seed"])] = int(row["qualified"])
    counts, endpoints, requests = {}, [], []
    for meta in metadata:
        cohort, arm, seed = meta["cohort"], meta["arm"], meta["seed"]
        threshold = THRESHOLDS[COHORTS[cohort][1]]
        entries, batches = read_database(meta)
        curve = qualified_counts(entries, threshold, MULTIPLIERS)
        count = int(curve[np.flatnonzero(MULTIPLIERS == 1)[0]])
        if count != reference[cohort, arm, seed]:
            raise RuntimeError(f"Frozen endpoint mismatch: {cohort}, {arm}, {seed}")
        counts[cohort, arm, seed] = curve
        base = {"cohort":cohort, "arm":arm, "seed":seed}
        records = []
        for row in batches:
            record = {**base, **{key:row[key] for key in (
                "launched_epoch", "completed_epoch", "elapsed_seconds", "finite_valid", "failed",
                "state", "remaining_updates")}}
            record["offered_to_rl_during_training"] = (
                row["finite_valid"] if arm in ("full", "llm_to_rl") and row["state"] == "available" else 0)
            records.append(record)
        requests.extend(records)
        endpoints.append({**base, "qualified":count, "launched":len(batches),
            **{state:sum(r["state"] == state for r in records) for state in STATES},
            "failed":sum(r["failed"] for r in records),
            "training_candidates":sum(r["finite_valid"] for r in records if r["state"] == "available"),
            "offered_to_rl":sum(r["offered_to_rl_during_training"] for r in records)})

    sensitivity, summaries = [], []
    for cohort, (_, task, arms) in COHORTS.items():
        baseline = np.asarray([counts[cohort,"none",s] for s in range(20)])
        for arm in arms:
            values = np.asarray([counts[cohort,arm,s] for s in range(20)])
            mean, low, high = bootstrap_mean(values-baseline)
            for i, multiplier in enumerate(MULTIPLIERS):
                sensitivity.append({"cohort":cohort,"arm":arm,"multiplier":float(multiplier),
                    "threshold":float(multiplier*THRESHOLDS[task]),"mean_qualified":float(values[:,i].mean()),
                    "mean_paired_gain":float(mean[i]),"ci_low":float(low[i]),"ci_high":float(high[i]),
                    "cap_fraction":float((values[:,i] == 30).mean()),
                    **count_support(values[:,i])})
            rows = [r for r in endpoints if r["cohort"] == cohort and r["arm"] == arm]
            batches = [r for r in requests if r["cohort"] == cohort and r["arm"] == arm]
            summaries.append({"cohort":cohort,"arm":arm,"seeds":len(rows),"launched":len(batches),
                "failed":sum(r["failed"] for r in rows),
                **{f"mean_{state}":float(np.mean([r[state] for r in rows])) for state in STATES},
                "mean_training_candidates":float(np.mean([r["training_candidates"] for r in rows])),
                "mean_offered_to_rl":float(np.mean([r["offered_to_rl"] for r in rows])),
                "median_collection_minutes":float(np.median([r["elapsed_seconds"] for r in batches])/60),
                "mean_qualified":float(np.mean([r["qualified"] for r in rows]))})
    figures(sensitivity, endpoints, summaries)
    write_csv(GENERATED / "portfolio_threshold_sensitivity.csv", sensitivity)
    write_csv(GENERATED / "portfolio_request_opportunities.csv", requests)
    write_csv(GENERATED / "portfolio_opportunity_endpoints.csv", endpoints)
    write_csv(GENERATED / "portfolio_opportunity_summary.csv", summaries)
    (GENERATED / "portfolio_sensitivity_opportunities.json").write_text(json.dumps({
        "database_audit":metadata,"summaries":summaries,"multipliers":MULTIPLIERS.tolist(),
        "bootstrap":{"resamples":2000,"seed":20260908,"interval":"pointwise percentile 95%"},
        "absolute_count_shading":{"statistic":"sample standard deviation across seeds", "ddof":1,
                                  "width_sd":1, "display_clip":[0,30], "confidence_interval":False},
        "endpoint_reconciliation":"All 160 frozen-threshold counts match source-aware endpoint CSVs",
        "sensitivity_figure_cohorts":list(SENSITIVITY_FIGURE_COHORTS),
        "analysis_cohorts":list(COHORTS),
    }, indent=2)+"\n")
    report(sensitivity, summaries)
    print(json.dumps(summaries, indent=2))


def report(sensitivity, summaries):
    lines = ["# Portfolio robustness and proposal opportunities", "",
        "Read-only analysis of 160 completed, hash-verified databases. All 160 frozen-threshold counts reproduce the existing endpoint analysis; all 800 launched batches have recorded outcomes. No model calls, simulations, or running-campaign changes.", "",
        "## Threshold sensitivity", "",
        "![Threshold sensitivity](figures/portfolio_threshold_sensitivity.png)", "",
        "The threshold sweep is 0.25 to 2 times each frozen threshold (strict loss < threshold). It retains the same terminal best-30 structural topologies and 20 paired seeds. Rates do not define additional topologies. Upper bands are pointwise 95% paired-seed bootstrap intervals, not simultaneous bands or new significance tests. The sweep is descriptive, not threshold selection.", "",
        "Upper panels show mean paired count gains over independent union. Lower panels show mean absolute qualified-topology counts with +/-1 sample standard deviation across seeds (ddof=1), clipped for display to the feasible range 0-30. These lower bands show between-seed variability, not confidence intervals or a significance test based on overlap; raw standard deviations are retained in the CSV.", "",
        "The sensitivity figure displays local Logic and Flash RPA only. Classifier remains in the complete numerical analysis below and in the CSV; this presentation choice does not invalidate its estimates or remove its threshold-dependent reversal. Request-opportunity accounting still includes all three cohorts.", "",
        "| Cohort | Arm | Gain at 0.25x | Gain at 0.5x | Gain at 1x | Gain at 2x | Cap fraction at 1x |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for cohort, (_, _, arms) in COHORTS.items():
        for arm in arms:
            if arm == "none":
                continue
            selected = [min((r for r in sensitivity if r["cohort"] == cohort and r["arm"] == arm),
                            key=lambda r: abs(r["multiplier"]-m)) for m in (.25,.5,1,2)]
            lines.append(f"| {cohort} | {LABELS[arm]} | " + " | ".join(f"{r['mean_paired_gain']:.2f}" for r in selected)
                         + f" | {selected[2]['cap_fraction']:.0%} |")
    lines += ["", "At half the frozen Logic threshold, all four arms have zero successful topologies; at the frozen threshold the insertion/full-duplex gains are 6.15/5.85. At twice the threshold, 90-95% of seeds reach the cap and the gains contract. Classifier full duplex gains 2.25 at half threshold and 5.40 at the frozen threshold, but loses 2.35 at twice the threshold (cap fractions: independent 70%, full duplex 40%). RPA's mean gain changes sign across the sweep and its pointwise interval includes zero throughout. These results do not support threshold-independent dominance.",
        "", "## Count-support audit", "",
        "Exact within-archive counts at the frozen threshold, not fitted estimates from individual topologies. The independent sampling unit for bootstrap intervals is the matched seed (20 pairs), not each topology. Low counts make this outcome discrete and threshold-sensitive, but do not alone invalidate seed-level estimates. Local Logic is sparser than classifier in these cohorts. All-zero counts can yield a zero-width empirical bootstrap interval without establishing population equivalence.", "",
        "| Cohort | Arm | Seeds | Mean | Median | Min-max | Zero-yield seeds |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in sensitivity:
        if r["multiplier"] == 1.0:
            lines.append(f"| {r['cohort']} | {LABELS[r['arm']]} | {r['n_seeds']} | "
                         f"{r['mean_qualified']:.2f} | {r['median_qualified']:.2f} | "
                         f"{r['min_qualified']}-{r['max_qualified']} | "
                         f"{round(r['zero_fraction']*r['n_seeds'])} |")
    lines += ["", "## Opportunity accounting", "",
        "![Opportunities and yield](figures/portfolio_request_opportunities.png)", "",
        "A batch is one scheduled multi-proposal Decider/Writer round, not one provider call. All five actual launch/outcome epochs are verified (0,20,40,60,80), including failures. A usable batch contains at least one valid candidate with finite evaluated loss. Training availability requires collection before the training-boundary timestamp and at completed_epoch < 100. The same epoch's policy update follows collection; epoch 99 still offers one update, epoch 100 none.", "",
        "| Cohort | Arm | Launched | Failed | Usable batches/run in training | Late usable/run | Valid training candidates/run | Offered to RL/run | Median collection min | Mean yield |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['cohort']} | {LABELS[r['arm']]} | {r['launched']} | {r['failed']} | "
                     f"{r['mean_available']:.2f} | {r['mean_late_valid']:.2f} | {r['mean_training_candidates']:.2f} | "
                     f"{r['mean_offered_to_rl']:.2f} | {r['median_collection_minutes']:.1f} | {r['mean_qualified']:.2f} |")
    lines += ["", "All 800 batch outcomes were collected during training; five local-model batches failed, with no late completions in these cohorts. Hosted arms have identical usable-batch and candidate counts despite different terminal yields. Local Logic insertion has slightly fewer usable batches than the independent arm, yet more successful topologies. Thus larger batch counts or missing terminal-only replies do not explain the observed gains; timing within training and proposal content still differ, so this is not a causal latency ablation.",
        "", "## Interpretation limits", "",
        "Training-available does not mean inserted, retained in the HOF, sampled for elite imitation, or causally responsible for an RL descendant. Offered-to-RL counts are valid finite candidates passed to the HOF path in inward-enabled arms, not retained insertions. Independent/outward-only candidates are isolated from RL until terminal pooling. Repeated topologies count as separate candidate opportunities but only once in terminal yield. Collection latency includes background work and waiting for the trainer's collection hook; it is not pure model inference time or monetary cost.", "",
        "The yield panel uses each task's frozen threshold and is paired by seed in the sensitivity analysis. Raw counts across tasks do not imply equal difficulty. Empty/failed outcomes are retained; no resampling of failed runs. These plots do not identify latency as a causal mechanism, and the top-30 ceiling does not establish what an unbounded archive would contain.", "",
        "Regenerate from the repository root: `MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m paper.iclr2027_genai_net_llm.generate_portfolio_sensitivity_opportunities`."]
    (PAPER / "PORTFOLIO_SENSITIVITY_OPPORTUNITIES.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
