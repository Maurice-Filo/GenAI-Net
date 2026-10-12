"""Export paper comparison tables for matched full-budget replay runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from comparisons.llm_seeded_evolution import METHOD
from comparisons.llm_seeded_evolution.replay import GENERATED, ROOT, sha256

METHOD_CHOICES = ("no_communication", "llm_to_rl", "rl_to_llm", "full", METHOD)
METHOD_LABELS = {"no_communication": "No communication (terminal union)",
                 "llm_to_rl": "LLM-to-RL insertion", "rl_to_llm": "RL/HOF-to-LLM",
                 "full": "Full duplex", METHOD: "LLM-seeded evolution"}


def collect_runs(raw_root: Path, task: str, seeds: list[int]) -> list[tuple[dict, dict, dict]]:
    records = {}
    plan = json.loads((GENERATED / "analysis_plan_v2.json").read_text())
    for path in sorted((raw_root / METHOD).glob("*/completed.json")):
        result = json.loads(path.read_text())
        if result.get("task") != task or result.get("seed") not in seeds:
            continue
        if result.get("status") != "complete" or result.get("pilot") is not False:
            continue
        replay_path, config_path = path.parent / "replay.json", path.parent / "config.json"
        if result["replay_sha256"] != sha256(replay_path) or result["config_sha256"] != sha256(config_path):
            raise ValueError(f"Changed run inputs: {path.parent}")
        replay = json.loads(replay_path.read_text())
        config = json.loads(config_path.read_text())
        archive_path = path.parent / "terminal_archive.json"
        if result.get("terminal_archive_sha256") != sha256(archive_path):
            raise ValueError(f"Changed or unverified terminal archive: {path.parent}")
        archive = json.loads(archive_path.read_text())
        if (not archive or min(x["loss"] for x in archive) != result["best_loss"]
                or len({x["topology_hash"] for x in archive if x["loss"] < result["quality_threshold"]})
                != result["qualified_topologies"]):
            raise ValueError(f"Terminal archive disagrees with summary: {path.parent}")
        if (replay["communication_mode"] != "none" or replay.get("expose_state_to_llm") is not False
                or replay.get("insert_llm_into_rl") is not False):
            raise ValueError("Comparison only admits No communication proposal streams.")
        if (result["search_evaluations"] != replay["search_budget"]
                or result["llm_evaluations"] != replay["llm_evaluations"]
                or result["candidate_evaluations"] != result["search_evaluations"] + result["llm_evaluations"]
                or result["quality_threshold"] != plan["quality_thresholds"][task]
                or result["portfolio_top_k"] != plan["portfolio_top_k"]):
            raise ValueError(f"Budget or estimand mismatch: {path.parent}")
        if result["seed"] in records:
            raise ValueError(f"Multiple complete baseline runs for seed {result['seed']}; choose an isolated raw root.")
        records[result["seed"]] = (result, replay, config)
    missing = set(seeds) - records.keys()
    if missing:
        raise ValueError(f"Missing full-budget baseline runs for seeds {sorted(missing)}; pilots are excluded.")
    # Worker count is operational; all search-defining settings must match.
    signatures = {json.dumps({k: v for k, v in config["evolution"].items() if k != "workers"}, sort_keys=True)
                  for _, _, config in records.values()}
    if len(signatures) != 1:
        raise ValueError("Cannot pool seeds with different evolutionary search protocols.")
    return [records[seed] for seed in seeds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("logic", "rpa", "classifier"), default="logic")
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES,
                        default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--raw-root", type=Path, default=ROOT / "comparisons/rpa_search/data/raw")
    parser.add_argument("--output-dir", type=Path, default=GENERATED / "llm_seeded_evolution")
    args = parser.parse_args(argv)
    if args.methods is None:
        args.methods = (["no_communication", METHOD] if args.task == "classifier" else
                        ["no_communication", "llm_to_rl", METHOD])
    if args.task == "classifier" and set(args.methods) & {"llm_to_rl", "rl_to_llm", "full"}:
        raise ValueError("Classifier exports support no_communication and llm_seeded_evolution; local communication-arm endpoints do not cover Classifier.")
    if len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        raise ValueError("Seed or method list contains duplicates.")
    records = collect_runs(args.raw_root, args.task, args.seeds)
    contextual = {}
    if set(args.methods) & {"llm_to_rl", "rl_to_llm", "full"}:
        with (GENERATED / "local_half_duplex_endpoints.csv").open() as stream:
            for row in csv.DictReader(stream):
                if row["task"] == args.task:
                    key = (row["arm"], int(row["seed"]))
                    if key in contextual:
                        raise ValueError(f"Duplicate published endpoint: {key}")
                    contextual[key] = row
    rows = []
    for result, replay, _ in records:
        threshold, top_k = result["quality_threshold"], result["portfolio_top_k"]
        source = replay["source_terminal_archive"][:top_k]
        for method, loss, qualified in (
            ("no_communication", min(x["loss"] for x in source),
             len({x["topology_hash"] for x in source if x["loss"] < threshold})),
            (METHOD, result["best_loss"], result["qualified_topologies"]),
        ):
            if method in args.methods:
                rows.append({"task": args.task, "seed": result["seed"], "method": method,
                             "best_loss": loss, "qualified_topologies": qualified,
                             "candidate_evaluations": result["candidate_evaluations"],
                             "quality_threshold": threshold, "proposal_stream": "same_no_communication_stream"})
        for method in args.methods:
            if method not in {"llm_to_rl", "rl_to_llm", "full"}:
                continue
            row = contextual.get((method, result["seed"]))
            if row is None:
                raise ValueError(f"Missing published {method} endpoint for seed {result['seed']}.")
            rows.append({"task": args.task, "seed": result["seed"], "method": method,
                         "best_loss": float(row["rank_one_loss"]),
                         "qualified_topologies": int(row["qualified_topologies"]),
                         "candidate_evaluations": replay["search_budget"] + int(row["accepted_llm_candidates"]),
                         "quality_threshold": threshold, "proposal_stream": "different_recorded_stream"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.task}_endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"task": args.task, "seeds": args.seeds, "exploratory": True,
               "comparison": "evolution_and_no_communication_share_proposals;_other_arms_are_contextual",
               "note": "Published communication arms used different proposal streams; they are not identical-proposal optimizer controls.",
               "methods": {}}
    for method in args.methods:
        members = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            "median_best_loss": float(np.median([r["best_loss"] for r in members])),
            "median_qualified_topologies": float(np.median([r["qualified_topologies"] for r in members])),
        }
    if {"no_communication", METHOD} <= set(args.methods):
        summary["paired_evolution_minus_independent_union"] = {}
        for metric, sign in (("best_loss", -1), ("qualified_topologies", 1)):
            left = np.array([next(r[metric] for r in rows if r["seed"] == s and r["method"] == METHOD)
                             for s in args.seeds])
            right = np.array([next(r[metric] for r in rows if r["seed"] == s and r["method"] == "no_communication")
                              for s in args.seeds])
            delta = left - right
            tied = (np.isclose(left, right, rtol=1e-9, atol=1e-12) if metric == "best_loss"
                    else left == right)
            summary["paired_evolution_minus_independent_union"][metric] = {"median_delta": float(np.median(delta)),
                "wins": int(np.sum((sign * delta > 0) & ~tied)), "ties": int(np.sum(tied)),
                "losses": int(np.sum((sign * delta < 0) & ~tied))}
    (args.output_dir / f"{args.task}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["% Exploratory comparison; only No communication shares evolution's proposal stream.",
             "\\begin{tabular}{lrr}", "\\toprule", "Method & Best loss (median) & Qualified yield (median) \\\\", "\\midrule"]
    for method, values in summary["methods"].items():
        name = METHOD_LABELS[method]
        if method in {"llm_to_rl", "rl_to_llm", "full"}:
            name += "$^{\\dagger}$"
        lines.append(f"{name} & {values['median_best_loss']:.5g} & {values['median_qualified_topologies']:.1f} \\\\")
    if set(args.methods) & {"llm_to_rl", "rl_to_llm", "full"}:
        lines.append("\\multicolumn{3}{l}{\\footnotesize $\\dagger$ Separate recorded proposal streams.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (args.output_dir / f"{args.task}_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
