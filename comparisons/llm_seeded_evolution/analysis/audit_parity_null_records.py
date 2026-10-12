"""Count saved RL null-reaction records without touching search outputs."""

import argparse
import json
from pathlib import Path
import sqlite3

from comparisons.llm_seeded_evolution.replay import GENERATED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads((GENERATED / "parity12_communication_300epoch_20260909_schedule.json").read_text())
    report = {"scope": "Seeds 0..4, RL-emitted epoch HOFs, distinct topology/parameter/loss records within each arm and seed; not all generated proposals", "arms": {}}
    for arm, settings in schedule["campaigns"].items():
        result = {"rl_parameter_records": 0, "null_records": []}
        for done_path in (Path(settings["audit_command"][2]) / "runs").glob("*/*/completed.json"):
            done = json.loads(done_path.read_text())
            if done["seed"] not in range(5):
                continue
            database = done_path.parent / "results.sqlite"
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as c:
                rows = c.execute("SELECT DISTINCT e.topology_hash,e.parameters_json,e.loss FROM hof_snapshot_entries e JOIN hof_snapshots h ON e.snapshot_id=h.snapshot_id WHERE e.emitter='RL' AND h.boundary='epoch'").fetchall()
            result["rl_parameter_records"] += len(rows)
            for topology, params, loss in rows:
                if any(r["reaction_id"] == 0 for r in json.loads(params)):
                    result["null_records"].append({"seed": done["seed"], "topology_hash": topology,
                        "loss": loss, "parameters": json.loads(params), "database": str(database)})
        result["null_count"] = len(result["null_records"])
        report["arms"][arm] = result
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({arm: {k: v for k, v in values.items() if k != "null_records"} for arm, values in report["arms"].items()})


if __name__ == "__main__":
    main()
