"""Frozen completed-archive audit: structural nulls and single-reaction deletion."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from comparisons.llm_seeded_evolution.fixed_stream_report import collect
from comparisons.llm_seeded_evolution.replay import sha256
from comparisons.llm_seeded_evolution.run import build_evaluator
from comparisons.llm_seeded_evolution.search import Candidate, ScoredCandidate
from comparisons.llm_seeded_evolution.analysis.live_hof_diversity import structural_summary
from paper.iclr2027_genai_net_llm import analyze_logic_correctness_pruning as pruning

EPSILONS = (1e-8, .001, .01, .05)


def is_null(reaction):
    return not any(reaction.get_stoichiometry_dict().values())


def eligible(row, epsilon, threshold):
    return (row["status"] == "ok" and row["all16_correct"]
            and row["loss"] < threshold and row["max_output_deviation"] <= epsilon)


def init(config):
    pruning.EVALUATOR = build_evaluator(config)
    pruning.replay.cache_clear()


def evaluate(job):
    entry = job["entry"]
    candidate = entry["candidate"]
    actions = tuple((int(i), (float(rate),)) for i, rate in
                    zip(candidate["reaction_ids"], candidate["rates"]))
    original = pruning.replay(actions)
    if (original["status"] != "ok" or original["topology_hash"] != entry["topology_hash"]
            or not math.isclose(original["loss"], entry["loss"], rel_tol=1e-7, abs_tol=1e-10)):
        raise ValueError(f"Original does not reproduce: {job['cohort']}/{job['seed']}/{job['rank']}")
    base = pruning.subset_record(original, tuple(range(5)), original["outputs"])
    deletions = []
    for removed in range(5):
        retained = tuple(i for i in range(5) if i != removed)
        reduced = pruning.replay(tuple(actions[i] for i in retained))
        row = pruning.subset_record(reduced, retained, original["outputs"])
        row["removed_reaction_id"] = actions[removed][0]
        deletions.append(row)
    return {k: v for k, v in job.items() if k != "entry"} | {
        "candidate": candidate, "original": base, "deletions": deletions}


def summarize(records, threshold):
    correct = [r for r in records if r["original"]["all16_correct"]
               and r["original"]["loss"] < threshold]
    result = {"archive_size": len(records), "qualified_and_all16_correct": len(correct),
              "deletion_solver_failures": sum(d["status"] != "ok" for r in records for d in r["deletions"]),
              "tolerances": {}}
    for epsilon in EPSILONS:
        choices = []
        for r in correct:
            passing = [d for d in r["deletions"] if eligible(d, epsilon, threshold)]
            chosen = min(passing, key=lambda d: d["structure_json"]) if passing else r["original"]
            choices.append(chosen)
        result["tolerances"][str(epsilon)] = {
            "designs_with_removable_reaction": sum(r["retained_count"] == 4 for r in choices),
            "distinct_selected_4_or_5_reaction_structures": len({r["topology_hash"] for r in choices}),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    plan = json.loads((args.campaign / "plan.json").read_text())
    checked = {(r["cohort"], r["seed"]) for r in collect(plan)}
    sources, tasks, configs, hashes = [], [], [], {}
    evaluator = None
    for job in plan["jobs"]:
        if (job["cohort"], job["seed"]) not in checked:
            continue
        root = Path(job["output"])
        config = json.loads((root / "config.json").read_text())
        configs.append(config)
        if evaluator is None:
            evaluator = build_evaluator(config)
            library = {r.ID: r for r in evaluator.library.reactions}
        if config["logic"] != configs[0]["logic"] or config["search"]["max_added_reactions"] != 5:
            raise ValueError("Evaluator configurations differ")
        archive = json.loads((root / "terminal_archive.json").read_text())
        completion = json.loads((root / "completed.json").read_text())
        threshold = completion["quality_threshold"]
        if sources and threshold != sources[0]["threshold"]:
            raise ValueError("Qualification thresholds differ")
        records = [ScoredCandidate(Candidate(**r["candidate"]), r["loss"], r["topology_hash"]) for r in archive]
        ids = [r.candidate.reaction_ids for r in records]
        if any(len(x) != 5 or len(set(x)) != 5 or set(x) & evaluator.forbidden_reaction_ids for x in ids):
            raise ValueError("Unexpected reaction count, duplicate, or forbidden reaction")
        if len({tuple(sorted(x)) for x in ids}) != len(records):
            raise ValueError("Reaction ordering inflated topology count")
        structural = structural_summary(records)
        structural["zero_net_reaction_occurrences"] = sum(is_null(library[i]) for x in ids for i in x)
        rates = [v for r in records for v in r.candidate.rates]
        structural.update(min_rate=min(rates), max_rate=max(rates))
        sources.append({"cohort": job["cohort"], "seed": job["seed"], "output": str(root),
                        "threshold": threshold, "structural": structural})
        for rank, entry in enumerate(archive, 1):
            tasks.append({"cohort": job["cohort"], "seed": job["seed"], "rank": rank, "entry": entry})
        for name in ("terminal_archive.json", "replay.json", "config.json", "completed.json"):
            hashes[str(root / name)] = sha256(root / name)
    if not sources:
        raise ValueError("No completed runs")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"captured_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "source_hashes": hashes, "analysis_source_hashes": {
                    str(Path(p).resolve()): sha256(Path(p)) for p in (__file__, pruning.__file__)},
                "sources": sources, "selection": "All completed runs at capture; all 30 archive designs",
                "threshold": threshold, "epsilons": EPSILONS, "new_model_calls": 0,
                "live_search_modified": False, "workers": args.workers,
                "planned_diagnostic_evaluations": len(tasks) * 6,
                "definition": "One deletion, unchanged surviving rates and fixed input reactions; maximum absolute OUT difference across 16x1000 points; frozen loss cutoff and all-row correctness with margin >1e-6",
                "limitations": ["Post-hoc incomplete cohort, descriptive only",
                    "Single-deletion screen, not exhaustive minimal-core enumeration",
                    "Selected reduced structures can still be dynamically similar",
                    "Preserves original unbounded rates; no rate clipping or refitting",
                    "Reduced circuits are diagnostics, not five-reaction search submissions"]}
    pruning.atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"runs": len(sources), "designs": len(tasks), "evaluations": len(tasks)*6}), flush=True)
    records = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(configs[0],),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = [pool.submit(evaluate, task) for task in tasks]
            for future in as_completed(pending):
                row = future.result()
                pruning.atomic_json(args.output / "records" / f"{row['cohort']}_{row['seed']}_{row['rank']}.json", row)
                records.append(row)
                if len(records) % 30 == 0:
                    print(f"Checked {len(records)}/{len(tasks)} designs", flush=True)
        for source in sources:
            group = sorted([r for r in records if (r["cohort"], r["seed"]) == (source["cohort"], source["seed"])], key=lambda r: r["rank"])
            source["deletion_audit"] = summarize(group, threshold)
        for path, expected in hashes.items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Source changed: {path}")
        manifest.update(status="complete", sources=sources)
        pruning.atomic_json(args.output / "summary.json", manifest)
        lines = ["# Completed Logic GA: null and redundancy audit", "",
                 f"Captured {manifest['captured_at']}; {len(sources)} runs / {len(tasks)} designs.", "",
                 "No literal null reactions, zero-net-change reactions, duplicate reaction IDs or reaction-order duplicates were found.", "",
                 "| Source | Seed | Mean Jaccard distance | One-replacement pairs | All-16 correct + qualified | Removable at .01 | Distinct selected structures |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for s in sources:
            d, a = s["structural"], s["deletion_audit"]
            t = a["tolerances"]["0.01"]
            lines.append(f"| {s['cohort']} | {s['seed']} | {d['mean_pairwise_jaccard_distance']:.3f} | {d['one_replacement_pair_fraction']:.1%} | {a['qualified_and_all16_correct']} | {t['designs_with_removable_reaction']} | {t['distinct_selected_4_or_5_reaction_structures']} |")
        lines.extend(["", manifest["definition"], "", "Each design retains its original rates. A removed reaction is called negligible only at the stated trajectory tolerance, not mathematically null. All simulation failures remain recorded; original replay failures stop the audit.", "", *manifest["limitations"]])
        (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
        print("Complete: " + str(args.output / "REPORT.md"), flush=True)
    except Exception as error:
        pruning.atomic_json(args.output / "failed.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
