"""Audit paired RL-LLM archive rates; reevaluate only out-of-bound designs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from comparisons.llm_seeded_evolution.analysis import clip_hof_rates
from comparisons.llm_seeded_evolution.replay import GENERATED, sha256


def summarize(records):
    rates = np.array([v for r in records for v in r["rates"]])
    if not records:
        return {"designs": 0}
    return {"designs": len(records), "rates": len(rates), "min": float(rates.min()),
        "median": float(np.median(rates)), "p95": float(np.quantile(rates, .95)), "max": float(rates.max()),
        "rates_below_001": int((rates < .001).sum()), "rates_above_100": int((rates > 100).sum()),
        "designs_outside_bounds": sum(any(v < .001 or v > 100 for v in r["rates"]) for r in records),
        "emitters": dict(Counter(r["emitter"] for r in records)),
        "provenance_classes": dict(Counter(r["provenance_class"] for r in records))}


def entries(connection, snapshot, job, replay):
    records = []
    fixed = set(replay["proposal_space"]["template_reaction_ids"])
    for row in connection.execute("SELECT * FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank", (snapshot["snapshot_id"],)):
        params = json.loads(row["parameters_json"])
        added = [p for p in params if p["reaction_id"] not in fixed]
        template = [p for p in params if p["reaction_id"] in fixed]
        if (len(added) != 5 or len(template) != 4 or len({p["reaction_id"] for p in added}) != 5
                or any(any(v is not None for v in p["structure"]["inputs"]) for p in added)
                or any(p["parameters"] != [1.] for p in template)):
            raise ValueError("Unexpected reaction contract; fixed versus added rates require review")
        rates = [p["parameters"][0] for p in added]
        if any(len(p["parameters"]) != 1 for p in added) or any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError("Invalid scalar rates")
        records.append({"cohort": job["cohort"], "seed": job["seed"], "rank": row["rank"],
            "loss": row["loss"], "topology_hash": row["topology_hash"], "rates": rates,
            "reaction_ids": [p["reaction_id"] for p in added], "emitter": row["emitter"],
            "provenance_class": row["provenance_class"], "related_llm_proposal_id": row["related_llm_proposal_id"],
            "search_evaluations": snapshot["cumulative_rl_evaluations"]})
    if len(records) != 30:
        raise ValueError("Expected top-30 HOF")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--ga-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.campaign / "plan.json").read_text())
    ga = json.loads(args.ga_snapshot.read_text())
    threshold = json.loads((GENERATED / "analysis_plan_v2.json").read_text())["quality_thresholds"]["logic"]
    selected = {(r["cohort"], r["seed"]) for r in ga["runs"]}
    all_records, matched, checks, sources = [], [], [], []
    for job in plan["jobs"]:
        path = Path(job["replay"])
        if sha256(path) != job["replay_sha256"]:
            raise ValueError("Frozen replay changed")
        replay = json.loads(path.read_text())
        source = Path(replay["source_run"])
        database = source / "results.sqlite"
        wal = source / "results.sqlite-wal"
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Source has nonempty WAL")
        if sha256(database) != replay["source_database_sha256"]:
            raise ValueError("Source no longer matches audited replay")
        sources.append({"cohort": job["cohort"], "seed": job["seed"], "path": str(source),
                        "database_sha256": replay["source_database_sha256"]})
        with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            terminal = connection.execute("SELECT * FROM hof_snapshots WHERE boundary='terminal_boundary' ORDER BY created_at DESC LIMIT 1").fetchone()
            if terminal["cumulative_rl_evaluations"] != 102300:
                raise ValueError("Not a complete 100-epoch source")
            rows = entries(connection, terminal, job, replay)
            all_records.extend(rows)
            if (job["cohort"], job["seed"]) in selected:
                checkpoint = connection.execute("SELECT * FROM hof_snapshots WHERE boundary='epoch' AND cumulative_rl_evaluations=? ORDER BY created_at DESC LIMIT 1", (ga["common_epoch"] * 1023,)).fetchone()
                matched.extend(entries(connection, checkpoint, job, replay))
        outliers = [r for r in rows if any(v < .001 or v > 100 for v in r["rates"])]
        if outliers:
            clip_hof_rates.init_worker(json.loads((source / "config.json").read_text()))
        for row in outliers:
            result = clip_hof_rates.evaluate({"candidate": {"rates": row["rates"], "reaction_ids": row["reaction_ids"]},
                                              "topology_hash": row["topology_hash"], "loss": row["loss"]})
            checks.append({"cohort": row["cohort"], "seed": row["seed"], "rank": row["rank"], **result})
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "source_mode": "llm-to-rl",
        "scope": "40 completed Logic RL-LLM runs paired with the fixed-proposal GA campaign; not full-duplex or parity",
        "threshold": threshold, "common_epoch": ga["common_epoch"], "sources": sources,
        "cohorts": {}, "matched_runs": [], "clipping_checks": checks,
        "diagnostic_evaluations": 2 * len(checks), "terminal_records": all_records,
        "matched_records": matched}
    for cohort in ("local_logic", "flash_logic"):
        records = [r for r in all_records if r["cohort"] == cohort]
        summary["cohorts"][cohort] = {"all": summarize(records),
            "qualified": summarize([r for r in records if r["loss"] < threshold]),
            "rank_one": summarize([r for r in records if r["rank"] == 0]),
            "by_emitter": {e: summarize([r for r in records if r["emitter"] == e]) for e in ("RL", "LLM")}}
    for cohort, seed in sorted(selected):
        records = [r for r in matched if r["cohort"] == cohort and r["seed"] == seed]
        summary["matched_runs"].append({"cohort": cohort, "seed": seed, **summarize(records)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sources", "terminal_records", "matched_records")}, indent=2))


if __name__ == "__main__":
    main()
