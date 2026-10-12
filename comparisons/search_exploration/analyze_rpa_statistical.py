"""Read-only paired RPA analysis. Never launches an evaluator or fills missing runs."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.stats import PermutationMethod, wilcoxon

from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json

LABELS = {"julia": "Julia-fit hybrid", "evolution": "Evolution-fit hybrid", "ga": "Standalone GA"}
COLORS = {"julia": "#168577", "evolution": "#376ab3", "ga": "#bd5b71"}


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def paired(a, b, *, smaller, resamples=10000, seed=20260915, inference=True):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError("Need finite nonempty matched observations")
    d = a - b
    ties = np.isclose(a, b, rtol=1e-9, atol=1e-12) if smaller else a == b
    signed = np.where(ties, 0., d)
    wins = signed < 0 if smaller else signed > 0
    result = {"n": len(d), "wins": int(wins.sum()), "ties": int(ties.sum()),
        "losses": int(len(d) - wins.sum() - ties.sum()), "mean_difference": float(np.mean(d))}
    ratios = a / b if (b > 0).all() else None
    result["median_paired_ratio"] = float(np.median(ratios)) if ratios is not None else None
    if inference:
        result["p"] = float(wilcoxon(signed, zero_method="pratt", alternative="two-sided",
            method=PermutationMethod(n_resamples=np.inf, batch=4096)).pvalue) if np.any(signed) else 1.
        indices = np.random.default_rng(seed).integers(0, len(d), size=(resamples, len(d)))
        result["mean_difference_ci95"] = np.quantile(d[indices].mean(axis=1), [.025, .975]).tolist()
        result["median_ratio_ci95"] = np.quantile(np.median(ratios[indices], axis=1), [.025, .975]).tolist() if ratios is not None else None
    return result


def endpoint(root, job, plan, runtime):
    folder = Path(job.get("output", root / "runs" / job["name"]))
    status = read_status(folder / "status.json") or {}
    completed = read_status(folder / "completed.json")
    row = {"arm": job["arm"], "seed": job["seed"], "status": runtime.get("status", status.get("status", "queued")),
        "epochs": status.get("epochs", status.get("epoch", 0)), "loss": None,
        "qualified_discovered": None, "qualified_retained": None, "evaluations": None,
        "canonical_evaluations": None, "joint_evaluations": None, "minutes": None, "audit_error": None,
        "deadline_policy": job.get("deadline_policy", "original"),
        "loss_disagreement_policy": job.get("loss_disagreement_policy", "original"),
        "previous_attempt_evaluations": sum(a["charged_evaluations"] for a in job.get("previous_attempts", []))}
    if not completed or completed.get("status") != "complete" or runtime.get("status") != "complete":
        return row
    manifest = read_status(folder / "manifest.json")
    try:
        cfg = manifest["config"]
        if job.get("deadline_policy") == "disabled" and (cfg["evaluation_seconds"] is not None or cfg["fit_seconds"] is not None):
            raise ValueError("Amended run still has scientific wall-clock cutoffs")
        expected_schema = {"julia": "search-streams-julia-v1", "evolution": "search-streams-v2", "ga": "standalone-ga-bounded-v1"}
        if (manifest["schema"] != expected_schema[job["arm"]] or manifest["task"] != "rpa" or
                cfg["seed"] != job["seed"] or cfg["evaluation_budget"] != plan["attempt_ceiling"] or
                cfg["quality_threshold"] != plan["threshold"] or cfg["rate_min"] != .001 or cfg["rate_max"] != 100. or
                cfg["batch_size"] != 1023 or manifest.get("harness") is not False):
            raise ValueError("Run identity/settings differ from frozen campaign")
        counters, metrics = completed["counters"], completed["metrics"]
        total = counters["expensive_evaluations"]
        joint = counters.get("julia/joint_evaluations", 0)
        if (total > plan["attempt_ceiling"] or completed["pending_optimization_jobs"] != 0 or
                counters.get("julia/reserved", 0) or counters.get("julia/unknown_evaluations_upper_bound", 0)):
            raise ValueError("Budget exceeded or fitting not drained")
        if job["arm"] == "ga":
            if total != plan["attempt_ceiling"]:
                raise ValueError("GA did not consume its declared evaluation budget")
        elif completed["epochs"] != 100:
            raise ValueError("Hybrid did not finish 100 epochs")
        if job["arm"] != "ga":
            expected_config = read_status(Path(job.get("config_path", root / "configs" / (job["name"] + ".json"))))
            if cfg != expected_config or manifest["policy"]["deep_layer_size"] != 1024:
                raise ValueError("Hybrid configuration differs from frozen settings")
            if manifest["task_config"] != read_status(root / "configs/task.json"):
                raise ValueError("Task configuration changed")
            if job["arm"] == "julia" and (manifest.get("post_prune") is not False or
                    manifest["julia_recipe"] != read_status(Path(job.get("recipe_path", root / "configs/julia_recipe.json")))):
                raise ValueError("Julia fitting or pruning recipe differs")
        loss = metrics["Archive/best_loss"]
        if loss is None or not np.isfinite(loss) or loss < 0:
            raise ValueError("Unscorable endpoint loss")
        with sqlite3.connect(f"file:{folder / 'exploration.sqlite'}?mode=ro", uri=True) as db:
            discovered = db.execute("SELECT count(DISTINCT json_extract(payload,'$.topology_hash')) "
                "FROM designs WHERE json_extract(payload,'$.qualified')=1").fetchone()[0]
        row.update(status="complete", epochs=completed["epochs"], loss=loss, qualified_discovered=discovered,
            qualified_retained=metrics["Archive/qualified_topologies"], evaluations=total,
            joint_evaluations=joint, canonical_evaluations=total-joint, minutes=completed["elapsed_seconds"]/60)
    except (KeyError, TypeError, ValueError, sqlite3.Error) as error:
        row.update(status="invalid", audit_error=str(error))
    return row


def summarize(rows, plan):
    complete = {(r["arm"], r["seed"]): r for r in rows if r["status"] == "complete"}
    expected = {(arm, seed) for arm in plan["arms"] for seed in plan["seeds"]}
    inference = set(complete) == expected and len(rows) == len(expected)
    contrasts = []
    for arm, control in plan["contrasts"]:
        seeds = [s for s in plan["seeds"] if (arm, s) in complete and (control, s) in complete]
        for metric in ("qualified_discovered", "loss"):
            values = paired([complete[arm, s][metric] for s in seeds], [complete[control, s][metric] for s in seeds],
                smaller=metric == "loss", inference=inference, resamples=plan["bootstrap"]["resamples"],
                seed=plan["bootstrap"]["seed"]) if seeds else {"n": 0}
            contrasts.append({"arm": arm, "control": control, "metric": metric, "seeds": seeds, **values})
    if inference:
        for row, adjusted in zip(contrasts, holm([r["p"] for r in contrasts])):
            row["p_holm"] = adjusted
    return {"inference_ready": inference, "endpoints": rows, "contrasts": contrasts}


def plot(output, rows, plan):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    rng = np.random.default_rng(20260915)
    jitter = dict(zip(plan["seeds"], rng.uniform(-.09, .09, len(plan["seeds"]))))
    for ax, key, label in zip(axes.flat,
            ("loss", "qualified_discovered", "evaluations", "minutes"),
            ("Best canonical loss (lower is better)", "Qualifying topologies discovered", "Consumed canonical + joint attempts", "Elapsed minutes (descriptive)")):
        complete = [r for r in rows if r["status"] == "complete"]
        for seed in plan["seeds"]:
            pairs = [(i, next((r[key] for r in complete if r["seed"] == seed and r["arm"] == arm), None))
                for i, arm in enumerate(plan["arms"])]
            if all(value is not None for _, value in pairs):
                ax.plot([i+jitter[seed] for i, _ in pairs], [v for _, v in pairs], color=".83", lw=.7, zorder=1)
        for i, arm in enumerate(plan["arms"]):
            valid = [r for r in complete if r["arm"] == arm]
            ax.scatter([i+jitter[r["seed"]] for r in valid], [r[key] for r in valid],
                color=COLORS[arm], s=20, alpha=.8, zorder=2)
            if valid:
                ax.hlines(np.median([r[key] for r in valid]), i-.18, i+.18, color=COLORS[arm], lw=3)
        if key == "loss":
            if complete and all(r[key] > 0 for r in complete):
                ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(range(3), ["Julia-fit\nhybrid", "Evolution-fit\nhybrid", "Standalone\nGA"])
        ax.set_ylabel(label)
        ax.set_xlim(-.45, 2.45)
        ax.spines[["top", "right"]].set_visible(False)
        if not complete:
            ax.text(.5, .5, "Awaiting completed runs", transform=ax.transAxes, ha="center")
    fig.savefig(output / "endpoints.png", dpi=180)
    fig.savefig(output / "endpoints.pdf")
    plt.close(fig)


def analyze(root):
    root = Path(root)
    plan = json.loads((root / ("active_plan.json" if (root / "active_plan.json").exists() else "plan.json")).read_text())
    progress = read_status(root / plan.get("progress_file", "progress.json")) or {}
    runtime = {j["name"]: j for j in progress.get("jobs", [])}
    rows = [endpoint(root, j, plan, runtime.get(j["name"], {})) for j in plan["jobs"]]
    result = summarize(rows, plan)
    output = root / plan.get("analysis_dir", "analysis")
    output.mkdir(exist_ok=True)
    atomic_json(output / "summary.json", result)
    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# RPA: Matched-Seed Workflow Comparison", "", f"Campaign: `{plan['id']}`", "",
        f"Completed and validated: {sum(r['status']=='complete' for r in rows)}/{len(rows)}.",
        f"Inference ready: {result['inference_ready']}. Partial results are descriptive, with no interim p-values.", "",
        "| Workflow | Complete | Median best loss | Mean qualifying discoveries | Median minutes |",
        "|---|---:|---:|---:|---:|"]
    for arm in plan["arms"]:
        valid = [r for r in rows if r["arm"] == arm and r["status"] == "complete"]
        values = [f"{f([r[k] for r in valid]):.6g}" if valid else "pending"
            for k, f in (("loss", np.median), ("qualified_discovered", np.mean), ("minutes", np.median))]
        lines.append(f"| {LABELS[arm]} | {len(valid)} | " + " | ".join(values) + " |")
    lines += ["", "## Paired Comparisons", "", "Effects are Julia minus control; loss ratios are Julia/control.", ""]
    if plan.get("concurrency_amendment"):
        lines += ["**Concurrency amendment:** two hybrid runs and two GA runs may overlap (128 worker slots). "
            "Per-run scientific settings and budgets are unchanged. Runtime and asynchronous fitting admission "
            "can depend on concurrent load; this is not an identical wall-clock schedule. "
            "Existing active runs were adopted without interruption. See amendments/concurrency/AMENDMENT.md.", ""]
    if plan.get("amendment"):
        lines += ["**Protocol amendment:** evaluation/fitting wall-clock cutoffs are disabled for remaining runs. "
            "Four completed runs are retained after checking that no cutoff bound their work. "
            "GA seed 2 restarts in a separate attempt folder; its original timeout remains preserved. "
            f"Previously charged failed-attempt work: {sum(r['previous_attempt_evaluations'] for r in rows)} evaluations, "
            "reported separately from each completed attempt's budget. No replacement seeds or best-attempt selection. "
            "This is an amended, mixed deadline-policy cohort, not an untouched preregistered campaign. "
            "See amendments/no-timeouts/AMENDMENT.md. Exact global compute equality is not claimed.", ""]
    if plan.get("loss_disagreement_amendment"):
        lines += ["**Agreement-policy amendment:** finite Julia/canonical score disagreements are logged, "
            "not fatal, for remaining Julia runs. The canonical score alone determines archive admission, "
            "qualification and learning. Both scores, relative differences and large-disagreement counts "
            "remain saved. Julia seed 6 restarts in a separate attempt; its original failure and consumed "
            "work remain included in previous-attempt overhead. Completed results are preserved. "
            "See amendments/canonical-warnings/AMENDMENT.md.", ""]
    if plan.get("disk_recovery_amendment"):
        lines += ["**Disk-full recovery:** the campaign was relocated to /local1/rossin. "
            "All 26 previously completed results passed SQLite integrity and endpoint checks and were retained. "
            "GA seeds 9 and 10, Julia seed 10, and evolution seed 10 restart with the same seeds and settings "
            "in separate folders; the disk-failed attempts remain preserved and are not pooled with the new attempts. "
            "No exact asynchronous checkpoint resume or best-attempt selection is claimed. Recorded failed-attempt "
            "compute is a lower bound because disk exhaustion can prevent final counter persistence. "
            "Budgets, concurrency and no-timeout policies are unchanged. See amendments/disk-recovery/AMENDMENT.md.", ""]
    for contrast in result["contrasts"]:
        lines.append(f"- Julia vs {LABELS[contrast['control']]}, {contrast['metric']}: n={contrast['n']}" +
            (f", W/T/L={contrast['wins']}/{contrast['ties']}/{contrast['losses']}, mean difference={contrast['mean_difference']:.6g}" if contrast["n"] else "") +
            (f", 95% paired bootstrap interval={contrast['mean_difference_ci95']}, p(Holm)={contrast['p_holm']:.6g}" if result["inference_ready"] else ""))
    lines += ["", "![Matched endpoints and resource use](endpoints.png)", "",
        "Points are completed seeds; lines connect complete three-arm seed blocks; thick marks are medians. "
        "Unequal interim completion can be biased. Matching RNG seeds does not imply identical proposals or asynchronous completion times.", "",
        "Both hybrids stop after 100 epochs and drain fitting; GA spends its 214793-attempt budget. "
        "A common ceiling does not mean equal expenditure or compute: Julia joint gradients cost more than state-only evaluations. "
        "See endpoints.csv for separate canonical and joint counts. GA generations are not RL epochs. "
        "GA retains 30 parents versus 2000 discovery entries in hybrids; all historically recorded qualifying topology hashes are counted. "
        "These counts do not prove functional/mechanistic diversity. Pruning is disabled and no LLM is used.", "",
        "All 60 jobs must complete and pass validation before the four planned tests and their Holm correction are reported. "
        "Failures remain listed; no automatic replacement seeds or silent complete-case inference. "
        "Confidence intervals are marginal, not simultaneous; failure to reject is not equivalence. "
        "This tests complete workflows, not the causal effect of changing only the rate optimizer.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    plot(output, rows, plan)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    analyze(args.root)
