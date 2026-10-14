"""LLM-only proposal diversity under self-feedback versus RL/HOF conditioning."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper.iclr2027_genai_net_llm import analyze_contract_v2_primary as primary
from paper.iclr2027_genai_net_llm.generate_flash_logic_directional import sources as flash_logic_sources
from paper.iclr2027_genai_net_llm.generate_local_half_duplex_analysis import AUDITS, ARMS, COLORS
from paper.iclr2027_genai_net_llm.generate_local_logic_amplification import LABELS, bootstrap_mean, write_csv

OUT = primary.GENERATED
STEM = "llm_proposal_diversity"
COHORTS = {"flash_logic": "logic", "local_logic": "logic", "local_rpa": "rpa",
           "flash_rpa": "rpa", "flash_classifier": "classifier"}


def sources():
    result = {"flash_logic": flash_logic_sources()[0]}
    for task in ("logic", "rpa"):
        by_arm = {}
        for arm in ARMS:
            rows = {}
            for name in AUDITS[task, arm]:
                audit = primary.read_json(OUT / name)
                for row in audit["runs"]:
                    if row["task"] == task and row["status"] == "pass":
                        if row["seed"] in rows:
                            raise RuntimeError("Duplicate seed in local audit")
                        rows[row["seed"]] = row
            by_arm[arm] = rows
        common = set.intersection(*(set(rows) for rows in by_arm.values()))
        if len(common) != (17 if task == "rpa" else 20):
            raise RuntimeError("Unexpected local complete-case sample")
        result[f"local_{task}"] = {(a, s): by_arm[a][s] for a in ARMS for s in sorted(common)}
    registry = primary.read_json(primary.REGISTRY_PATH)
    index = {}
    for name in registry["audit_files"]:
        audit = primary.read_json(primary.ROOT / name)
        if audit["status"] != "pass":
            raise RuntimeError("Primary audit does not pass")
        index.update({str(Path(r["run_root"])/"results.sqlite"): r for r in audit["runs"] if r["status"] == "pass"})
    for task in ("rpa", "classifier"):
        result[f"flash_{task}"] = {
            (a, s): index[str(primary.database_path(registry, reg, task, s))]
            for a, reg in (("none", "no_communication"), ("full", "full_duplex")) for s in range(20)}
    return result


def proposal_metrics(rows, threshold):
    usable = [r for r in rows if r["valid"] and r["loss"] is not None and np.isfinite(r["loss"])]
    best = {}
    for row in usable:
        if not row["topology_hash"] or row.get("structure_json") is None:
            raise RuntimeError("Usable proposal lacks complete structure")
        key = row["topology_hash"]
        if key not in best or row["loss"] < best[key]["loss"]:
            best[key] = row
    return {"valid_proposals": len(usable), "unique_topologies": len(best),
            "qualified_topologies": sum(r["loss"] < threshold for r in best.values()),
            "repeat_fraction": 1-len(best)/len(usable) if usable else None}, list(best.values())


def read_run(cohort, arm, seed, audit):
    task = COHORTS[cohort]
    path = Path(audit["run_root"])/"results.sqlite"
    wal = path.with_name(path.name+"-wal")
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError("Uncheckpointed database is not a frozen input")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != audit["database_sha256"]:
        raise RuntimeError(f"Database changed since its passing audit: {path}")
    with primary.connect(path) as c:
        rows = [dict(r) for r in c.execute("""SELECT p.topology_hash, p.loss, p.valid,
            p.candidate_index, r.launched_epoch, r.created_at, t.structure_json
            FROM llm_candidates p JOIN llm_runs r USING(llm_run_id)
            LEFT JOIN crns t USING(topology_hash)
            ORDER BY r.launched_epoch, p.candidate_index""")]
        served = [r[0] for r in c.execute("SELECT launched_epoch FROM llm_runs")]
        failed = [r[0] for r in c.execute("SELECT launched_epoch FROM llm_failures")]
    schedule = [0, 20, 40, 60, 80]
    if sorted(served+failed) != schedule:
        raise RuntimeError(f"Request accounting differs from five rounds: {path}")
    threshold = primary.read_json(primary.PLAN_PATH)["quality_thresholds"][task]
    metrics, unique = proposal_metrics(rows, threshold)
    metrics["mean_jaccard"] = primary.mean_jaccard(task, unique)
    metrics.update(cohort=cohort, arm=arm, seed=seed, threshold=threshold,
                   returned_batches=len(served), failed_batches=len(failed), requested_slots=50,
                   database=str(path), database_sha256=digest.hexdigest())
    curves = []
    for i, epoch in enumerate(schedule):
        cumulative, _ = proposal_metrics([r for r in rows if r["launched_epoch"] <= epoch], threshold)
        batch, _ = proposal_metrics([r for r in rows if r["launched_epoch"] == epoch], threshold)
        curves.append({"cohort": cohort, "arm": arm, "seed": seed, "round": i,
                       "request_epoch": epoch, **cumulative,
                       "batch_unique_topologies": batch["unique_topologies"]})
    return metrics, curves


def summarize(endpoints):
    summaries = []
    metrics = ("unique_topologies", "qualified_topologies", "mean_jaccard", "repeat_fraction", "valid_proposals")
    for cohort in COHORTS:
        for arm in ARMS:
            rows = [r for r in endpoints if r["cohort"] == cohort and r["arm"] == arm]
            if not rows:
                continue
            summary = {"cohort": cohort, "arm": arm, "n": len(rows),
                       "failed_batches": sum(r["failed_batches"] for r in rows)}
            for m in metrics:
                values = [r[m] for r in rows if r[m] is not None]
                summary[f"mean_{m}"] = float(np.mean(values)) if values else None
                summary[f"median_{m}"] = float(np.median(values)) if values else None
            summaries.append(summary)
    contrasts = []
    for cohort in COHORTS:
        index = {(r["arm"], r["seed"]): r for r in endpoints if r["cohort"] == cohort}
        directional = cohort in {"flash_logic", "local_logic", "local_rpa"}
        pairs = [("rl_to_llm", "none"), ("full", "llm_to_rl")] if directional else [("full", "none")]
        for left, right in pairs:
            seeds = sorted(s for a, s in index if a == left)
            for m in ("unique_topologies", "qualified_topologies", "mean_jaccard"):
                valid = [s for s in seeds if index[left, s][m] is not None and index[right, s][m] is not None]
                x, y = [[index[a, s][m] for s in valid] for a in (left, right)]
                relations = Counter(primary.paired_relation(-a, -b) for a, b in zip(x, y))
                contrasts.append({"cohort": cohort, "left": left, "right": right,
                    "isolates_return_channel": directional, "metric": m, "n": len(valid),
                    "mean_delta": float(np.mean(np.array(x)-y)) if valid else None,
                    "wins": relations["win"], "ties": relations["tie"], "losses": relations["loss"],
                    "p_unadjusted": primary.wilcoxon_p(x, y) if valid else 1.0})
    for r, p in zip(contrasts, primary.holm([r["p_unadjusted"] for r in contrasts])):
        r["p_holm_24_exploratory"] = p
    return summaries, contrasts


def figure(curves):
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, layout="constrained")
        for col, cohort in enumerate(["flash_logic", "local_logic"]):
            for row, metric in enumerate(["unique_topologies", "qualified_topologies"]):
                ax = axes[row, col]
                for arm in ARMS:
                    index = {(r["seed"], r["round"]): r for r in curves if r["cohort"] == cohort and r["arm"] == arm}
                    values = [[index[s, i][metric] for i in range(5)] for s in range(20)]
                    mean, lo, hi = bootstrap_mean(values)
                    ax.plot(range(5), mean, color=COLORS[arm], label=LABELS[arm])
                    ax.fill_between(range(5), lo, hi, color=COLORS[arm], alpha=.12)
                ax.set_title(f"{'ABCD'[row*2+col]}  Logic / {'Flash' if col == 0 else 'local model'}")
                ax.set_ylabel("Distinct LLM topologies" if row == 0 else "Distinct successful LLM topologies")
                ax.set_xticks(range(5))
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
                if row == 1:
                    ax.set_xlabel("Scheduled request index (cumulative)")
        axes[0, 0].legend(fontsize=7, frameon=False)
        for ext in ("pdf", "png"):
            fig.savefig(primary.FIGURES / f"{STEM}.{ext}", dpi=220)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.65), sharey=True, layout="constrained")
        for ax, cohort, title in zip(axes, ("flash_logic", "local_logic"),
                                     ("A  Logic / Flash", "B  Logic / local model")):
            for arm in ARMS:
                index = {(r["seed"], r["round"]): r for r in curves
                         if r["cohort"] == cohort and r["arm"] == arm}
                mean, lo, hi = bootstrap_mean([
                    [index[s, i]["unique_topologies"] for i in range(5)] for s in range(20)])
                ax.plot(range(5), mean, color=COLORS[arm], label=LABELS[arm])
                ax.fill_between(range(5), lo, hi, color=COLORS[arm], alpha=.12)
            ax.set(title=title, xlabel="Scheduled request index", xticks=range(5), ylim=(0, 40))
            ax.grid(axis="y", alpha=.2)
        axes[0].set_ylabel("Cumulative distinct LLM topologies")
        axes[0].legend(fontsize=7, frameon=False, loc="upper left")
        for ext in ("pdf", "png"):
            fig.savefig(primary.FIGURES / f"{STEM}_main.{ext}", dpi=220)
        plt.close(fig)


def latex_tables(summaries, contrasts):
    cohort_labels = {"flash_logic": "Flash Logic", "local_logic": "Local Logic",
                     "local_rpa": "Local RPA", "flash_rpa": "Flash RPA",
                     "flash_classifier": "Flash classifier"}
    arm_labels = {"none": "None", "full": "Full", "llm_to_rl": r"LLM$\to$RL",
                  "rl_to_llm": r"RL/HOF$\to$LLM"}
    lines = [r"% Generated by analyze_llm_proposal_diversity.py; do not edit.",
             r"\begin{table}[p]\centering\small",
             r"\begin{tabular}{llrrrrr}\toprule",
             r"Cohort & Arm & $n$ & Distinct & Successful & Distance & Repeat (\%)\\\midrule"]
    for r in summaries:
        lines.append(f"{cohort_labels[r['cohort']]} & {arm_labels[r['arm']]} & {r['n']} & "
                     f"{r['mean_unique_topologies']:.2f} & {r['mean_qualified_topologies']:.2f} & "
                     f"{r['mean_mean_jaccard']:.3f} & {100*r['mean_repeat_fraction']:.1f}" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Direct LLM proposals across five scheduled rounds. Entries are means per seed; distance is mean within-run pairwise added-reaction Jaccard distance. Successful means strict loss below the frozen task threshold. No terminal archive cap is applied.}",
              r"\label{tab:llm-diversity-endpoints}\end{table}",
              r"\begin{table}[p]\centering\scriptsize",
              r"\begin{tabular}{lllrrrr}\toprule",
              r"Cohort & Contrast & Metric & Mean $\Delta$ & W/T/L & Raw $p$ & Holm $p$\\\midrule"]
    names = {"unique_topologies": "Distinct", "qualified_topologies": "Successful",
             "mean_jaccard": "Distance"}
    for r in contrasts:
        lines.append(f"{cohort_labels[r['cohort']]} & {arm_labels[r['left']]} vs {arm_labels[r['right']]} & "
                     f"{names[r['metric']]} & {r['mean_delta']:+.3f} & "
                     f"{r['wins']}/{r['ties']}/{r['losses']} & "
                     f"{r['p_unadjusted']:.5f} & {r['p_holm_24_exploratory']:.5f}" + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Exploratory paired proposal-distribution contrasts. Positive differences favor the left arm. Two-sided Wilcoxon tests use Pratt zeros; Holm correction covers all 24 tests shown. Flash RPA and classifier vary both channels and do not isolate return conditioning. Pair counts are 17 for local RPA and 20 otherwise.}",
              r"\label{tab:llm-diversity-contrasts}\end{table}"]
    (OUT / f"{STEM}_tables.tex").write_text("\n".join(lines)+"\n")


def main():
    endpoints, curves = [], []
    for cohort, runs in sources().items():
        for (arm, seed), audit in runs.items():
            row, curve = read_run(cohort, arm, seed, audit)
            endpoints.append(row); curves.extend(curve)
        print(cohort, len(runs), "hash-verified databases", flush=True)
    summaries, contrasts = summarize(endpoints)
    figure(curves)
    latex_tables(summaries, contrasts)
    write_csv(OUT / f"{STEM}_endpoints.csv", endpoints)
    write_csv(OUT / f"{STEM}_rounds.csv", curves)
    write_csv(OUT / f"{STEM}_contrasts.csv", contrasts)
    payload = {"summaries": summaries, "contrasts": contrasts, "runs": endpoints,
               "scope": "Valid evaluated LLM proposals only; uncapped by the terminal top-30 archive; rates do not define distinct topologies"}
    (OUT / f"{STEM}.json").write_text(json.dumps(payload, indent=2)+"\n")
    lines = ["# Does RL/HOF feedback diversify the LLM's own proposals?", "",
             "Read-only analysis of 308 completed, hash-verified databases. No model calls, simulations, active parity data or manuscript edits.", "",
             "Every arm retains evaluator feedback on its own LLM proposals. Return-only versus none, and full versus insertion-only, isolate outward RL/HOF conditioning. Flash RPA and classifier have only full-versus-none comparisons here: both channels vary, so those contrasts do not isolate the return channel.", "",
             "## Takeaway", "",
             "The return channel broadens the LLM proposal distribution in several completed cohorts. On local Logic, mean distinct topology counts increase from 26.45 to 33.10 for return-only versus none, and from 27.10 to 34.60 for full versus insertion-only. Both count contrasts survive the 24-test exploratory Holm correction. Flash Logic shows the same descriptive count pattern; its return-only structural-distance contrast survives correction, but its topology-count contrasts do not. Both local RPA structural-distance contrasts also survive correction.", "",
             "No successful-topology count contrast survives that correction. Thus broader structural exploration is supported more clearly than improved direct-LLM solution yield. Flash RPA does not show a universal diversification benefit in its joint-channel comparison. This analysis does not establish full-duplex performance synergy, nor distinguish copying RL structures from generating novel structures in response to them.", "",
             "Request-0 cohorts already differ despite withholding the initial HOF. Descriptively, local Logic adds 19.90 versus 24.60 new topologies after request 0 (none versus return-only), and 20.00 versus 27.00 (insertion-only versus full). The endpoint pattern is therefore not solely an initial-batch count difference; these increments are not separately significance-tested.", "",
             "## LLM-only endpoints", "",
             "Means per seed, after five scheduled batches of ten proposals. Validity failures remain in the allowance; invalid candidates are not assigned fictional structures. Local RPA uses 17 complete matched seeds (four storage-invalid arm-seed records leave three seeds outside the complete-case intersection); other cohorts have 20. No terminal top-30 cap is applied.", "",
             "| Cohort | Arm | Distinct topologies | Successful distinct | Mean Jaccard distance | Repeated fraction |", "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['cohort']} | {LABELS[r['arm']]} | {r['mean_unique_topologies']:.2f} | {r['mean_qualified_topologies']:.2f} | {r['mean_mean_jaccard']:.3f} | {100*r['mean_repeat_fraction']:.1f}% |")
    lines += ["", "## Paired return-channel contrasts", "",
              "W/T/L favors the left arm (more distinct proposals, more successful proposals, or greater distance). Tests are two-sided paired Wilcoxon with Pratt zeros; raw and Holm-adjusted values across all 24 exploratory tests are retained. No new confirmatory family is claimed.", "",
              "| Cohort | Contrast | Metric | Mean delta | W/T/L | Raw p | Holm p |", "|---|---|---|---:|---:|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {r['cohort']} | {LABELS[r['left']]} vs {LABELS[r['right']]} | {r['metric']} | {r['mean_delta']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} | {r['p_unadjusted']:.4g} | {r['p_holm_24_exploratory']:.4g} |")
    lines += ["", "![LLM-only cumulative proposal diversity](figures/llm_proposal_diversity.png)", "",
              "## Interpretation limits", "",
              "Distinctness uses complete labelled reaction topology hashes, ignoring rate-only changes and reaction ordering. Jaccard distance compares complete added-reaction records after subtracting exact fixed-template records, preserving input-channel identity. It measures structural dispersion, not usefulness. Successful proposals use each task's already-frozen RL-only threshold, with strict loss < threshold. These counts concern direct LLM proposals before any RL refinement, not final mixed-archive contributions.", "",
              "Curves group eventual outcomes by scheduled request index, not wall time or availability at a training epoch. Request 0 has no initial HOF in any arm. Bands are pointwise 95% seed-bootstrap intervals; no independent test of every plotted point. Run-level failed batches are retained as zero additions. Conditional distance and repeat fractions are undefined when there are too few valid structures, and such pairs are excluded only for that metric. Counts retain zero-yield runs. Hosted weights and asynchronous timing are not controlled by the seed alone. Lack of a detected difference is not equivalence.", "",
              "Reproduce: `MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m paper.iclr2027_genai_net_llm.analyze_llm_proposal_diversity`."]
    (primary.PAPER / "LLM_PROPOSAL_DIVERSITY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"summaries": summaries, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
