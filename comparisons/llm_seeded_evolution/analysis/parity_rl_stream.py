"""Separate RL batch minima from LLM/pooled endpoints for completed parity seeds."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import statistics

from comparisons.llm_seeded_evolution.replay import GENERATED, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule_path = GENERATED / "parity12_communication_300epoch_20260909_schedule.json"
    schedule = json.loads(schedule_path.read_text())
    records, histories = [], {}
    for arm, settings in schedule["campaigns"].items():
        campaign = Path(settings["audit_command"][2])
        for completed_path in (campaign / "runs").glob("*/*/completed.json"):
            done = json.loads(completed_path.read_text())
            seed = done["seed"]
            if seed not in range(5):
                continue
            root = completed_path.parent
            if (done["rl_candidate_evaluations"] != 306900 or done["task"] != "logic_parity"
                    or done["communication_mode"] != arm):
                raise ValueError("Wrong completed run or budget")
            with (root / "progress.csv").open() as stream:
                rows = [r for r in csv.DictReader(stream) if 1 <= int(r["step"]) <= 300]
            if [int(r["step"]) for r in rows] != list(range(1, 301)):
                raise ValueError("Incomplete RL batch history")
            batch = [float(r["loss"]) for r in rows]
            if not all(math.isfinite(x) for x in batch):
                raise ValueError("Nonfinite RL batch loss")
            histories[arm, seed] = batch
            database = root / "results.sqlite"
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                connection.execute("BEGIN")
                llm = connection.execute("SELECT MIN(loss) FROM llm_candidates WHERE valid=1").fetchone()[0]
                retained_rl = connection.execute("SELECT MIN(e.loss) FROM hof_snapshot_entries e JOIN hof_snapshots h ON e.snapshot_id=h.snapshot_id WHERE e.emitter='RL' AND h.boundary='epoch'").fetchone()[0]
            if not math.isclose(min(min(batch), llm), done["best_loss"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Source minima do not reconcile with terminal endpoint")
            records.append({"arm": arm, "seed": seed, "run": str(root),
                "completed_sha256": sha256(completed_path), "progress_sha256": sha256(root / "progress.csv"),
                "rl_at100": min(batch[:100]), "rl_at200": min(batch[:200]), "rl_at300": min(batch),
                "llm_at300": llm, "combined_at300": done["best_loss"],
                "retained_rl_min_over_snapshots": retained_rl,
                "endpoint_winner": "LLM" if llm < min(batch) else "RL" if min(batch) < llm else "tie"})
    expected = {(arm, seed) for arm in schedule["campaigns"] for seed in range(5)}
    if {(r["arm"], r["seed"]) for r in records} != expected or len(records) != 20:
        raise ValueError("Expected exactly five completed matched seeds per arm")
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "seeds": list(range(5)),
        "task": "logic_parity", "added_reactions": 12, "epochs": 300, "schedule_sha256": sha256(schedule_path),
        "definition": "RL best-so-far = running minimum of progress.loss over steps 1..300; step 301 is pooled and excluded. progress.best_so_far_loss is mixed and not used for RL isolation.",
        "none_return_identical_rl_histories": all(histories["none", s] == histories["rl-to-llm", s] for s in range(5)),
        "no_new_evaluations": True, "interim": True, "rows": records, "means": {}}
    for arm in schedule["campaigns"]:
        members = [r for r in records if r["arm"] == arm]
        summary["means"][arm] = {key: statistics.mean(r[key] for r in members)
            for key in ("rl_at100", "rl_at200", "rl_at300", "llm_at300", "combined_at300")}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
