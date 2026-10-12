"""Read-only reconstruction of live GA HOFs at committed progress checkpoints."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

from comparisons.llm_seeded_evolution.replay import GENERATED, sha256
from comparisons.llm_seeded_evolution.search import Candidate, EliteArchive, ScoredCandidate


def complete_csv(path):
    # Only read the captured file length, and exclude an in-progress final line.
    with path.open("rb") as stream:
        length = path.stat().st_size
        raw = stream.read(length)
    raw = raw[:raw.rfind(b"\n") + 1]
    return raw, list(csv.DictReader(io.StringIO(raw.decode())))


def structural_summary(records):
    sets = [set(r.candidate.reaction_ids) for r in records]
    pairs = list(itertools.combinations(sets, 2))
    distances = [1 - len(a & b) / len(a | b) for a, b in pairs]
    frequency = Counter(i for ids in sets for i in ids)
    return {
        "topologies": len(records),
        "mean_pairwise_jaccard_distance": sum(distances) / len(distances) if distances else None,
        "one_replacement_pair_fraction": sum(len(a - b) == len(b - a) == 1 for a, b in pairs) / len(pairs) if pairs else None,
        "distinct_added_reactions": len(frequency),
        "universal_added_reactions": sorted(set.intersection(*sets)) if sets else [],
        "reaction_frequency": dict(sorted(frequency.items(), key=lambda item: (-item[1], item[0]))),
        "direct_llm": sum(r.candidate.source == "llm" for r in records),
        "ga_with_llm_ancestry": sum(bool(r.candidate.proposal_id) and r.candidate.source != "llm" for r in records),
        "no_recorded_llm_ancestry": sum(not r.candidate.proposal_id for r in records),
        "retained_llm_lineages": dict(Counter(r.candidate.proposal_id for r in records if r.candidate.proposal_id)),
    }


def reconstruct(rows, checkpoints, *, threshold, forbidden, capacity=30):
    targets = {int(p["candidate_evaluations"]): p for p in checkpoints}
    archive = EliteArchive(capacity)
    qualified_seen, llm_hashes = set(), set()
    results = {}
    for expected_id, row in enumerate(rows, 1):
        total = int(row["candidate_evaluations"])
        if total > max(targets):
            break
        if total != expected_id or total != int(row["search_evaluations"]) + int(row["llm_evaluations"]):
            raise ValueError("Candidate sequence is incomplete or accounting changed")
        ids = tuple(json.loads(row["reaction_ids"]))
        if len(set(ids)) != 5 or set(ids) & forbidden:
            raise ValueError("Expected five distinct, unmodulated, non-template library reactions")
        candidate = Candidate(ids, tuple(json.loads(row["rate_constants"])), row["source"],
                              row["parent_hash"] or None, row["proposal_id"] or None)
        item = ScoredCandidate(candidate, float(row["loss"]), row["topology_hash"])
        archive.add(item)
        if item.loss < threshold:
            qualified_seen.add(item.topology_hash)
        if candidate.source == "llm":
            llm_hashes.add(item.topology_hash)
        if total not in targets:
            continue
        checkpoint = targets[total]
        ranked = archive.ranked()
        qualified = [r for r in ranked if r.loss < threshold]
        if (not math.isclose(ranked[0].loss, float(checkpoint["best_so_far_loss"]), rel_tol=1e-12)
                or len(qualified) != int(checkpoint["qualified_topologies"])):
            raise ValueError("Reconstructed archive disagrees with saved checkpoint")
        signatures = {tuple(sorted(r.candidate.reaction_ids)) for r in ranked}
        if len(signatures) != len(ranked):
            raise ValueError("Topology hashes disagree with added-reaction set equivalence")
        results[total] = {
            "epoch": int(checkpoint["search_evaluations"]) // 1023,
            "search_evaluations": int(checkpoint["search_evaluations"]),
            "llm_evaluations": int(checkpoint["llm_evaluations"]),
            "best_loss": ranked[0].loss, "worst_hof_loss": ranked[-1].loss,
            "all_hof": structural_summary(ranked), "qualified_hof": structural_summary(qualified),
            "qualified_topologies_seen_so_far": len(qualified_seen),
            "hof_topologies_absent_from_received_llm_proposals": sum(r.topology_hash not in llm_hashes for r in ranked),
            "archive": [asdict(r) for r in ranked],
        }
    if set(results) != set(targets):
        raise ValueError("Captured candidate log does not reach selected checkpoint")
    return results


def analyze(root, output):
    state = json.loads((root / "status.json").read_text())
    jobs = state["active"]
    if not jobs:
        raise ValueError("No active GA runs")
    progress = {}
    for job in jobs:
        raw, rows = complete_csv(Path(job["output"]) / "progress.csv")
        if not rows:
            raise ValueError("Run has no committed checkpoint")
        progress[job["run_id"]] = (rows, hashlib.sha256(raw).hexdigest())
    common = min(int(rows[-1]["search_evaluations"]) for rows, _ in progress.values())
    threshold = json.loads((GENERATED / "analysis_plan_v2.json").read_text())["quality_thresholds"]["logic"]
    summary = {"captured_at": datetime.now(timezone.utc).isoformat(), "interim": True,
        "campaign": str(root), "common_epoch": common // 1023, "threshold": threshold,
        "definition": "Mean 1-|A intersection B|/|A union B| over distinct labelled added-reaction sets; rates and order ignored; fixed input reactions excluded by validated mask",
        "caveats": ["Four active runs, two seeds per proposal source; no inferential tests.",
            "Top-30 archive cap; qualified counts do not measure mechanistic or functional independence.",
            "Inherited proposal IDs indicate recorded ancestry, not a counterfactual causal effect.",
            "No simulations, model calls, or changes to live search files."], "runs": []}
    for job in jobs:
        run = Path(job["output"])
        rows, progress_hash = progress[job["run_id"]]
        common_row = next(p for p in rows if int(p["search_evaluations"]) == common)
        latest = rows[-1]
        raw, candidates = complete_csv(run / "candidates.csv")
        replay = json.loads((run / "replay.json").read_text())
        if sha256(run / "replay.json") != job["replay_sha256"]:
            raise ValueError("Replay changed")
        found = reconstruct(candidates, [common_row, latest], threshold=threshold,
                            forbidden=set(replay["proposal_space"]["forbidden_reaction_ids"]))
        summary["runs"].append({"cohort": job["cohort"], "seed": job["seed"], "output": str(run),
            "candidate_prefix_bytes": len(raw), "candidate_prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "progress_prefix_sha256": progress_hash,
            "common": found[int(common_row["candidate_evaluations"])],
            "latest": found[int(latest["candidate_evaluations"])]})
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Interim GA HOF diversity", "", f"Captured: {summary['captured_at']}", "",
        f"Frozen Logic quality cutoff: loss < {threshold:.12g}.", ""]
    for key, label in (("common", "Matched search budget"), ("latest", "Latest completed batch")):
        lines.extend([f"## {label}", "", "| Source | Seed | Epoch | Best loss | Qualified HOF | Jaccard distance | Added-reaction union | Universal core | Direct LLM / descendants / other |", "|---|---:|---:|---:|---:|---:|---:|---|---|"])
        for row in summary["runs"]:
            snapshot = row[key]
            d = snapshot["all_hof"]
            lines.append(f"| {row['cohort']} | {row['seed']} | {snapshot['epoch']} | {snapshot['best_loss']:.6g} | {snapshot['qualified_hof']['topologies']}/30 | {d['mean_pairwise_jaccard_distance']:.3f} | {d['distinct_added_reactions']} | {d['universal_added_reactions']} | {d['direct_llm']} / {d['ga_with_llm_ancestry']} / {d['no_recorded_llm_ancestry']} |")
        lines.append("")
    lines.extend(["Jaccard **distance**: 0 = identical reaction sets; 1 = disjoint. All current HOF designs have five added reactions. Rates, reaction ordering, and shared fixed input reactions are excluded.", "", *summary["caveats"]])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign.resolve(), args.output.resolve())
    print((args.output / "REPORT.md").read_text())
