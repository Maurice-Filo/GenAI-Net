"""Descriptive emitter/parent attribution from recorded events, without evaluations."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def summarize(designs, events, threshold, require_all_rows=False):
    first_qualified = {}
    gains = defaultdict(float)
    counts = Counter()
    mutation = Counter()
    qualified_by_source = defaultdict(set)
    for event in events:
        source, stream = event["source"], event.get("stream", "unknown")
        group = source + "/" + stream
        counts[group] += 1
        gains[group] += event.get("gain", 0.)
        record = designs.get(event["candidate_hash"])
        if record is None:
            raise ValueError("Event candidate is missing from design ledger")
        qualified = (record["simulation_complete"] and record["loss"] < threshold
            and (not require_all_rows or record.get("all_rows_correct") is True))
        topology = record["topology_hash"]
        if qualified:
            qualified_by_source[group].add(topology)
            first_qualified.setdefault(topology, group)
        if source == "ga" and stream == "search":
            mutation["proposals"] += 1
            mutation["origin/" + str(event.get("origin"))] += 1
            parent = designs.get(event.get("parent_hash"))
            if parent is None:
                mutation["missing_parent"] += 1
                continue
            mutation["qualified_outputs"] += int(qualified)
            if qualified:
                mutation["qualified_origin/" + str(event.get("origin"))] += 1
            mutation["improved_parent_loss"] += int(record["simulation_complete"]
                and record["loss"] < parent["loss"])
            mutation["changed_topology"] += int(topology != parent["topology_hash"])
    first_counts = Counter(first_qualified.values())
    return {"distinct_qualifying_topologies": len(first_qualified),
        "groups": [{"group": g, "events": counts[g], "archive_region_utility_gain": gains[g],
            "qualifying_topologies_seen": len(qualified_by_source[g]),
            "first_qualifications": first_counts[g]} for g in sorted(counts)],
        "direct_archive_mutation": dict(mutation)}


def analyze(run, output):
    run, output = Path(run), Path(output)
    manifest = json.loads((run / "manifest.json").read_text())
    with sqlite3.connect("file:" + str(run / "exploration.sqlite") + "?mode=ro", uri=True) as db:
        designs = {key: json.loads(payload) for key, payload in db.execute("SELECT key,payload FROM designs")}
        events = (json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM events WHERE kind='archive_insertion' ORDER BY id"))
        result = summarize(designs, events, manifest["config"]["quality_threshold"],
            manifest["config"]["require_all_rows"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_interaction.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Recorded Source Interaction", "", f"Run: `{run}`", "",
        "Descriptive saved-event analysis; no new simulations or model calls.", "",
        "| Source/stream | Recorded events | First qualifying topologies | Qualifying topologies seen |",
        "| --- | ---: | ---: | ---: |"]
    for row in result["groups"]:
        lines.append(f"| {row['group']} | {row['events']} | {row['first_qualifications']} | {row['qualifying_topologies_seen']} |")
    lines += ["", "First qualifications are assigned in recorded merge order and sum to the",
        "distinct total. 'Seen' counts can overlap between groups. Optimization keeps",
        "the initiating structural source label; it does not mean the neural policy fits rates.",
        "", "## Archive Mutation", "", "```json",
        json.dumps(result["direct_archive_mutation"], indent=2), "```", "",
        "Origin tracks explicit mutation ancestry, not the ancestry of learned policy weights.",
        "Parent loss comparisons here use the mutation's initial inherited rates, before",
        "its own rate optimization. Repeated proposals are counted as events, not unique designs.",
        "Archive utility gains refer to descriptor-region improvements, not raw loss or",
        "a formal diversity metric. Source opportunity and optimization selection differ.",
        "GA here means a single-reaction archive mutation plus a separate evolutionary rate",
        "optimizer; it is not the earlier standalone population-GA experiment.",
        "The architecture allows GA descendants into imitation and RL designs into mutation.",
        "These logs do not identify which imitation examples caused later policy proposals.",
        "Neither attribution nor parentage establishes a causal synergy. A no-RL control",
        "tests the learned emitter contribution; reciprocal removal of archive mutation",
        "and replicated seeds would be needed for stronger interaction claims.", ""]
    (output / "SOURCE_INTERACTION.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
