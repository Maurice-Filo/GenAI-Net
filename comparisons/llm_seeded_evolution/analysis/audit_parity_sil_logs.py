"""Read-only Comet and archive audit for completed parity SIL runs."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics

from comparisons.llm_seeded_evolution.replay import GENERATED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    schedule = json.loads((GENERATED / "parity12_communication_300epoch_20260909_schedule.json").read_text())
    api = None
    if args.comet:
        from comet_ml import API
        api = API(cache=False)
    report = {"runs": []}
    wanted = {"Loss: SIL", "Temperature", "Entropy: Batch", "Entropy: Global Weight",
              "LogP: Batch", "LogP: Top-51", "Loss: Batch Best"}
    for arm, settings in schedule["campaigns"].items():
        campaign = Path(settings["audit_command"][2])
        for done_path in sorted((campaign / "runs").glob("*/*/completed.json")):
            done = json.loads(done_path.read_text())
            if done["seed"] not in (0, 4):
                continue
            root = done_path.parent
            manifest = json.loads((root / "run_manifest.json").read_text())
            with (root / "progress.csv").open() as f:
                batch = {int(r["step"]) - 1: float(r["loss"]) for r in csv.DictReader(f) if int(r["step"]) <= 300}
            counts = []
            with sqlite3.connect((root / "results.sqlite").as_uri() + "?mode=ro", uri=True) as c:
                c.row_factory = sqlite3.Row
                snapshots = c.execute("SELECT * FROM hof_snapshots WHERE boundary='epoch' ORDER BY epoch").fetchall()
                for snap in snapshots:
                    rows = c.execute("SELECT loss,emitter FROM hof_snapshot_entries WHERE snapshot_id=?", (snap["snapshot_id"],)).fetchall()
                    good = [r for r in rows if r["loss"] < batch[snap["epoch"]]]
                    counts.append({"epoch": snap["epoch"], "positive_advantage": len(good),
                        "llm_positive_advantage": sum(r["emitter"] == "LLM" for r in good),
                        "llm_in_hof": sum(r["emitter"] == "LLM" for r in rows)})
            row = {"arm": arm, "seed": done["seed"], "run": str(root), "snapshot_counts": counts,
                "mean_positive_advantage": statistics.mean(r["positive_advantage"] for r in counts),
                "mean_llm_positive_advantage": statistics.mean(r["llm_positive_advantage"] for r in counts),
                "final_llm_in_hof": counts[-1]["llm_in_hof"], "metrics": {}}
            if api:
                experiment = api.get_experiment_by_key(manifest["comet_experiment_key"])
                for name in sorted(wanted):
                    values = experiment.get_metrics(name)
                    row["metrics"][name] = [{"step": x.get("step"), "value": x.get("metricValue")} for x in values]
                row["assets"] = [{"fileName": x.get("fileName"), "type": x.get("type")} for x in experiment.get_asset_list()]
            (args.output / f"{arm}_seed{done['seed']}.json").write_text(json.dumps(row, indent=2) + "\n")
            report["runs"].append(row)
            brief = {k: v for k, v in row.items() if k not in ("metrics", "snapshot_counts", "assets")}
            brief["metrics"] = {}
            for name, values in row["metrics"].items():
                vals = [float(x["value"]) for x in values if x["value"] is not None]
                if vals:
                    brief["metrics"][name] = {"n": len(vals), "first": vals[0], "last": vals[-1],
                        "min": min(vals), "max": max(vals), "zero": sum(v == 0 for v in vals)}
            print(json.dumps(brief), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
