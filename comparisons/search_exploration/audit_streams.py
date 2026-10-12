"""Read-only artifact audit; completed runs must have no undrained work or lost charges."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile


def audit(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    status = json.loads((path / "status.json").read_text())
    source_errors = []
    with tarfile.open(path / "source.tar.gz") as archive:
        for filename, expected in manifest["new_source"].items():
            if hashlib.sha256(archive.extractfile(filename).read()).hexdigest() != expected:
                source_errors.append(filename)
    with sqlite3.connect("file:" + str(path / "exploration.sqlite") + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        counts = dict(db.execute("SELECT key,n FROM visits WHERE key NOT LIKE 'region:%' AND key NOT LIKE 'topology:%'"))
        completed = db.execute("SELECT count(*) FROM evaluation_summaries").fetchone()[0]
        failures = db.execute("SELECT count(*) FROM events WHERE kind='evaluation_failed'").fetchone()[0]
        incomplete = db.execute("SELECT count(*) FROM evaluation_summaries WHERE json_extract(payload,'$.simulation_complete')=0").fetchone()[0]
        diagnostics = db.execute("SELECT count(*) FROM diagnostics").fetchone()[0]
        dispatches = db.execute("SELECT count(*) FROM events WHERE kind='optimization_dispatch'").fetchone()[0]
        merges = db.execute("SELECT count(*) FROM events WHERE kind='optimization_result'").fetchone()[0]
        records = [json.loads(r[0]) for r in db.execute("SELECT payload FROM designs")]
        db.rollback()
    cfg = manifest["config"]
    bad_bounds = [r["candidate_hash"] for r in records if not all(cfg["rate_min"] <= v <= cfg["rate_max"] for v in r["rates"])]
    reaction_count = manifest["task_config"]["search"]["max_added_reactions"]
    bad_structures = [r["candidate_hash"] for r in records if len(r["reaction_ids"]) != reaction_count
        or len(set(r["reaction_ids"])) != reaction_count or len(r["rates"]) != reaction_count]
    result = {"status": status["status"], "source_snapshot_errors": source_errors,
        "identity_matches": identity == manifest["identity"], "counters": counts,
        "completed_evaluations": completed, "failed_evaluation_events": failures,
        "incomplete_simulations": incomplete, "retained_diagnostics": diagnostics,
        "optimization_dispatches": dispatches, "optimization_merges": merges,
        "bad_archived_rate_bounds": bad_bounds, "bad_recorded_structures": bad_structures,
        "harness": manifest["harness"],
        "notes": "Incomplete simulations are retained scalar outcomes, not successful trajectories; rate audit covers recorded designs, not every intermediate vector."}
    assert not source_errors and result["identity_matches"] and not bad_bounds and not bad_structures
    assert counts["expensive_evaluations"] <= cfg["evaluation_budget"]
    assert counts.get("evaluations/search", 0) <= manifest["search_evaluation_reserve"]
    assert counts.get("evaluations/optimization", 0) + counts.get("evaluations/llm", 0) <= cfg["evaluation_budget"] - manifest["search_evaluation_reserve"]
    assert counts["expensive_evaluations"] == sum(counts.get("evaluations/" + s, 0) for s in ("search", "optimization", "llm"))
    if status["status"] == "complete":
        if manifest["schema"] == "standalone-ga-bounded-v1":
            assert counts["expensive_evaluations"] == cfg["evaluation_budget"]
            assert manifest["replay_events"] == [] and not manifest["policy_training_enabled"]
            assert all(set(r["reaction_ids"]) <= set(manifest["allowed_ids"]) for r in records)
        else:
            assert status["epochs"] == cfg["epochs"]
        assert status["pending_optimization_jobs"] == 0 and dispatches == merges
        assert completed + failures == counts["expensive_evaluations"]
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audit(args.run)
