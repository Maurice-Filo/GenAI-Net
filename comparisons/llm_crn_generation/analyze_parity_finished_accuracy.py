"""Post-hoc endpoint accuracy for the first completed parity seed block."""

import csv
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np

from comparisons.llm_crn_generation.run_mmc2_harness_smoke import build_evaluator
from RL4CRN.llm.schemas import LLMCandidate
from RL4CRN.utils.results_database import serialize_crn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "comparisons/llm_crn_generation/generated/parity_accuracy_seeds0to4_20260909"
BASE = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
ARMS = ("none", "rl-to-llm", "llm-to-rl", "full")


def classify(inputs, outputs):
    inputs, outputs = np.asarray(inputs), np.asarray(outputs, dtype=float)
    if inputs.shape != (16, 4) or outputs.shape != (16,) or not np.isfinite(outputs).all():
        raise ValueError("Expected 16 finite scalar outputs and four binary inputs")
    if not np.isin(inputs, [0, 1]).all() or len(set(map(tuple, inputs))) != 16:
        raise ValueError("Truth table is incomplete or non-binary")
    targets = inputs.sum(axis=1).astype(int) % 2
    predictions = (outputs >= 0.5).astype(int)
    return targets, predictions, predictions == targets


def cached_outputs(root, entry, reactions):
    for path in sorted((root / "harness-workspaces").glob("*/CONTEXT/HALL_OF_FAME.json"), reverse=True):
        for item in json.loads(path.read_text()).get("entries", []):
            if item.get("topology_hash") != entry["topology_hash"] or item.get("loss") != entry["loss"]:
                continue
            if not item.get("crn", "").rstrip().endswith("\n".join(reactions)):
                continue
            relative = item.get("cached_diagnostics")
            if not relative:
                continue
            diagnostic = path.parent.parent / relative
            d = json.loads(diagnostic.read_text())
            rows = sorted(d.get("trajectories", []), key=lambda r: r["scenario"])
            if [r["scenario"] for r in rows] == list(range(16)) and all(
                r.get("finite") and len(r.get("terminal_values", [])) == 1 for r in rows
            ):
                return [r["terminal_values"][0] for r in rows], str(diagnostic)
    return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluator = build_evaluator("logic_parity", solver="CVODE")
    # Replay stored parameters verbatim, including the RL stream's unbounded rates.
    evaluator.enforce_parameter_bounds = False
    evaluator.min_parameter_value = 0.0
    evaluator.max_parameter_value = None
    records, table = [], []
    for arm in ARMS:
        campaign = BASE / f"flash-parity4-v2-workspace-loss-{arm}-300epoch-seeds0to19-20260908"
        status = json.loads((campaign / "status.json").read_text())
        completed = {r["seed"] for r in status["completed"]}
        if not set(range(5)) <= completed:
            raise RuntimeError(f"First five seeds are not complete for {arm}")
        for seed in range(5):
            root, = campaign.glob(f"runs/*seed{seed}/logic_parity_*")
            db = root / "results.sqlite"
            with sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True) as c:
                c.row_factory = sqlite3.Row
                entry = dict(c.execute("""SELECT e.* FROM hof_snapshot_entries e
                    JOIN hof_snapshots s USING(snapshot_id)
                    WHERE s.boundary='terminal_boundary' ORDER BY e.rank LIMIT 1""").fetchone())
            best = json.loads((root / "best_network.json").read_text())
            parameters = json.loads(entry["parameters_json"])
            if best["reaction_ids"] != [r["reaction_id"] for r in parameters] or not np.array_equal(
                best["rate_constants"], [r["parameters"][0] for r in parameters]
            ):
                raise RuntimeError("Best-network export and terminal database disagree")
            info = json.loads(entry["task_info_json"])
            inputs = info["inputs"]
            outputs, cache = cached_outputs(root, entry, best["reactions"])
            replay_loss = None
            if outputs is None:
                added = [r for r in parameters if r["reaction_id"] not in evaluator.initial_reaction_ids]
                if not all(np.isfinite(r["parameters"]).all() and min(r["parameters"]) > 0 for r in added):
                    raise RuntimeError("Nonpositive/nonfinite stored parameter")
                result = evaluator.evaluate(LLMCandidate(
                    reaction_ids=[r["reaction_id"] for r in added],
                    parameter_values=[r["parameters"] for r in added]))
                if not result.valid or not np.isclose(result.loss, entry["loss"], rtol=1e-7, atol=1e-9):
                    raise RuntimeError(f"Replay disagrees: {arm}/{seed}: {result.message}, {result.loss} vs {entry['loss']}")
                if serialize_crn(result.env.state)["topology_hash"] != entry["topology_hash"]:
                    raise RuntimeError("Replayed topology differs")
                if not np.array_equal(result.task_info["inputs"], inputs):
                    raise RuntimeError("Scenario order differs")
                outputs = [np.asarray(o).reshape(-1, 1000)[0, -1] for o in result.task_info["outputs"]]
                replay_loss = result.loss
            target, prediction, correct = classify(inputs, outputs)
            if not np.array_equal(target, np.asarray(info["setpoint"]).reshape(-1)):
                raise RuntimeError("Parity labels differ from the executed task")
            row = {"arm": arm, "seed": seed, "loss": entry["loss"],
                   "correct": int(correct.sum()), "accuracy": float(correct.mean()),
                   "false_positive": int(((prediction == 1) & (target == 0)).sum()),
                   "false_negative": int(((prediction == 0) & (target == 1)).sum()),
                   "topology_hash": entry["topology_hash"], "database": str(db),
                   "source": "cached_live_run" if cache else "posthoc_cvode_replay",
                   "cache": cache, "replay_loss": replay_loss,
                   "emitter": entry["emitter"], "provenance": entry["provenance_class"]}
            records.append(row)
            for u, out, y, pred, ok in zip(inputs, outputs, target, prediction, correct):
                table.append({"arm": arm, "seed": seed, "inputs": "".join(str(int(v)) for v in u),
                              "output_t100": float(out), "target": int(y),
                              "prediction": int(pred), "correct": bool(ok)})
            print(arm, seed, f"{row['correct']}/16", row["source"], flush=True)
    payload = {"selection": "Terminal rank-one network by canonical loss, seeds 0..4 per arm; not selected by accuracy",
               "rule": "Predict 1 iff OUT(t=100) >= 0.5; fixed threshold, all 16 rows equally weighted",
               "diagnostic_only": True, "records": records, "truth_table_rows": table}
    (OUT / "accuracy.json").write_text(json.dumps(payload, indent=2)+"\n")
    for name, rows in [("endpoints.csv", records), ("truth_table.csv", table)]:
        with (OUT / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["# Parity endpoint classification: completed seeds 0-4", "", payload["selection"], "", payload["rule"], "",
             "| Arm | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        lines.append(f"| {arm} | "+" | ".join(f"{r['correct']}/16" for r in rows)+f" | {100*np.mean([r['accuracy'] for r in rows]):.2f}% |")
    lines += ["", "Post-hoc diagnostic only, not a replacement objective. Terminal classification is not a test of transient tracking or robustness. Cached values require identical topology, loss and reaction text/parameters; other winners are replayed with unchanged parameters through CVODE and their loss and topology must agree. No model calls, candidate changes, or feedback into the running search. This checks 20 loss-selected winners, not every candidate or every terminal HOF member."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("Report:", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
