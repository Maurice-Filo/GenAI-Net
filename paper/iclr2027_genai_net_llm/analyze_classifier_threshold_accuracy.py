"""Post-hoc label accuracy of frozen classifier archives, without search updates."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from paper.iclr2027_genai_net_llm.generate_portfolio_sensitivity_opportunities import (
    GENERATED, PAPER, THRESHOLDS, read_database, write_csv,
)

STEM = "classifier_threshold_accuracy"


def endpoint_accuracy(outputs, targets):
    y, target = np.asarray(outputs, dtype=float), np.asarray(targets, dtype=float)
    if y.shape != (32, 2) or target.shape != (32, 2) or not np.isfinite(y).all():
        raise ValueError("Expected 32 finite two-output scenarios")
    if not np.isin(target, [0, 1]).all() or not (target.sum(axis=1) == 1).all():
        raise ValueError("Expected one-hot target vectors")
    expected = target.argmax(axis=1)
    margin = y[np.arange(32), expected] - y[np.arange(32), 1-expected]
    return {"winner_correct": int((margin > 1e-6).sum()),
            "binary_correct": int(((y >= .5) == target).all(axis=1).sum()),
            "ambiguous": int((np.abs(margin) <= 1e-6).sum()),
            "max_terminal_error": float(np.abs(y-target).max())}


def cached_index(root):
    found = {}
    for path in sorted(root.glob("harness-workspaces/*/CONTEXT/HALL_OF_FAME.json")):
        for entry in json.loads(path.read_text()).get("entries", []):
            actions = entry.get("actions", [])
            key = (entry.get("topology_hash"), entry.get("loss"),
                   tuple(sorted((a["reaction_id"], tuple(a["parameter_values"])) for a in actions)))
            relative = entry.get("cached_diagnostics")
            if not relative:
                continue
            diagnostic = path.parent.parent / relative
            if not diagnostic.is_file():
                continue
            info = json.loads(diagnostic.read_text())
            rows = sorted(info.get("trajectories", []), key=lambda r: r["scenario"])
            if info.get("loss") == entry.get("loss") and [r["scenario"] for r in rows] == list(range(32)) and all(
                    r.get("finite") and len(r.get("terminal_values", [])) == 2 for r in rows):
                found[key] = ([r["terminal_values"] for r in rows], str(diagnostic))
    return found


def analyze_run(meta):
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import build_evaluator
    from RL4CRN.environments.environment import Environment
    from RL4CRN.utils.results_database import serialize_crn

    entries, _ = read_database(meta)
    evaluator = build_evaluator("classifier", solver="CVODE")
    cache = cached_index(Path(meta["database"]).parent)
    records = []
    for row in entries:
        params = json.loads(row["parameters_json"])
        archived = json.loads(row["task_info_json"])
        key = (row["topology_hash"], row["loss"],
               tuple(sorted((p["reaction_id"], tuple(p["parameters"])) for p in params)))
        outputs, cached_path = cache.get(key, (None, None))
        replay_loss = None
        if outputs is None:
            # Replay stored RL actions verbatim, not through new-LLM admission rules.
            env = Environment(evaluator.crn_template, evaluator.max_added_reactions)
            env.reset()
            for p in params:
                raw = {"reaction index": p["reaction_id"], "parameters": p["parameters"],
                       "continuous parameters": p["parameters"], "discrete parameters": None}
                env.step(action=evaluator.actuator.actuate(raw), stepper=evaluator.stepper, raw_action=raw)
            replay_loss, info = evaluator._compute_loss(env)
            if not np.isclose(replay_loss, row["loss"], rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Loss replay mismatch: {meta['arm']}/{meta['seed']}/{row['rank']}")
            if serialize_crn(env.state)["topology_hash"] != row["topology_hash"]:
                raise RuntimeError("Topology replay mismatch")
            for field in ("setpoint", "initial_conditions"):
                if not np.array_equal(info[field], archived[field]):
                    raise RuntimeError(f"Scenario mismatch: {field}")
            outputs = np.asarray(info["outputs"])[:, :, -1].tolist()
        metrics = endpoint_accuracy(outputs, archived["setpoint"])
        records.append({"arm": meta["arm"], "seed": meta["seed"], "rank": row["rank"],
                        "topology_hash": row["topology_hash"], "loss": row["loss"],
                        "emitter": row["emitter"], "provenance": row["provenance_class"],
                        **metrics, "replay_loss": replay_loss, "cached_path": cached_path,
                        "outputs": outputs, "targets": archived["setpoint"],
                        "database": meta["database"], "database_sha256": meta["database_sha256"]})
    return records


def summarize(records, multipliers):
    rows = []
    for multiplier in multipliers:
        for arm in ("none", "full"):
            chosen = [r for r in records if r["arm"] == arm and r["loss"] < multiplier*THRESHOLDS["classifier"]]
            rows.append({"multiplier": multiplier, "loss_cutoff": multiplier*THRESHOLDS["classifier"],
                         "arm": arm, "qualified": len(chosen), "mean_qualified": len(chosen)/20,
                         "all_labels_correct": sum(r["winner_correct"] == 32 for r in chosen),
                         "mean_all_labels_correct": sum(r["winner_correct"] == 32 for r in chosen)/20,
                         "all_binary_outputs_correct": sum(r["binary_correct"] == 32 for r in chosen),
                         "mean_all_binary_outputs_correct": sum(r["binary_correct"] == 32 for r in chosen)/20,
                         "min_correct_labels": min(r["winner_correct"] for r in chosen)})
    return rows


def main():
    audited = json.loads((GENERATED / "portfolio_sensitivity_opportunities.json").read_text())
    metadata = [m for m in audited["database_audit"] if m["cohort"] == "flash_classifier"]
    assert {(m["arm"], m["seed"]) for m in metadata} == {(a, s) for a in ("none", "full") for s in range(20)}
    with (GENERATED / "portfolio_threshold_sensitivity.csv").open() as f:
        sensitivity = list(csv.DictReader(f))
    inversion = min(float(r["multiplier"]) for r in sensitivity if r["cohort"] == "flash_classifier"
                    and r["arm"] == "full" and float(r["multiplier"]) > 1 and float(r["mean_paired_gain"]) < 0)
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(analyze_run, metadata):
            records.extend(rows)
            print(rows[0]["arm"], rows[0]["seed"], "checked", len(rows), flush=True)
    summary = summarize(records, [.5, 1., inversion, 2.])
    assert len(records) == 1200
    payload = {"records": records, "summary": summary, "first_negative_sweep_multiplier": inversion,
               "rule": "At t=100, target species exceeds the other by >1e-6; ties are incorrect. Also report both outputs thresholded at 0.5.",
               "scope": "Post-hoc diagnostic of the 32 supplied cases, not held-out generalization, steady-state proof, or a replacement loss."}
    (GENERATED / f"{STEM}.json").write_text(json.dumps(payload, indent=2)+"\n")
    write_csv(GENERATED / f"{STEM}.csv", summary)
    lines = ["# Classifier correctness at the threshold inversion", "", payload["rule"], "", payload["scope"], "",
             f"All 1200 terminal records in 40 audited databases checked; {sum(r['cached_path'] is not None for r in records)} cached, the remainder replayed through CVODE with unchanged actions/rates, matching topology, loss, and scenarios. No optimization, model calls, archive updates, or manuscript edits.", "",
             "| Cutoff / frozen | Arm | Mean qualifying | Mean with 32/32 winning labels | Mean with 32/32 binary targets | Worst labels correct |",
             "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['multiplier']:.4f} | {r['arm']} | {r['mean_qualified']:.2f} | "
                     f"{r['mean_all_labels_correct']:.2f} | {r['mean_all_binary_outputs_correct']:.2f} | {r['min_correct_labels']}/32 |")
    lines += ["", "These are topology-by-seed records, not globally unique networks or independent observations within a seed. The first negative multiplier is the first sampled sweep point, not an exact continuous crossing. Endpoint labels do not measure transient tracking, precise one-hot concentrations, robustness, or unseen initial conditions. Correct-label counts are post-hoc diagnostics and do not redefine the frozen portfolio threshold."]
    (PAPER / "CLASSIFIER_THRESHOLD_ACCURACY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
