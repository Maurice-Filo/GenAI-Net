"""Reevaluate frozen GA HOFs before/after clipping; never modify live archives."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from comparisons.llm_seeded_evolution.replay import sha256
from comparisons.llm_seeded_evolution.run import build_evaluator

EVALUATOR = None


def init_worker(config):
    global EVALUATOR
    EVALUATOR = build_evaluator(config)


def clipped_rates(rates):
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Source contains invalid rates")
    return [min(100., max(.001, rate)) for rate in rates]


def functional_accuracy(info):
    targets = np.asarray(info["setpoint"]).reshape(-1)
    inputs = np.asarray(info["inputs"])
    if inputs.shape != (16, 4) or len(set(map(tuple, inputs))) != 16 or targets.shape != (16,):
        raise ValueError("Incomplete Logic truth table")
    expected = ((inputs[:, 0] * inputs[:, 1]) + (inputs[:, 1] * inputs[:, 2])
                + (inputs[:, 2] * inputs[:, 3])) > 0
    if not np.array_equal(targets, expected):
        raise ValueError("Unexpected Logic target")
    outputs = [np.asarray(y) for y in info["outputs"]]
    if len(outputs) != 16 or any(y.shape != (1, 1000) or not np.isfinite(y).all()
                                 or np.all(y == 1e4) for y in outputs):
        return {"simulation_complete": False, "correct_rows": None}
    terminal = np.array([y[0, -1] for y in outputs])
    return {"simulation_complete": True, "correct_rows": int(((terminal >= .5) == targets).sum()),
            "terminal_outputs": terminal.tolist()}


def evaluate(entry):
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.utils.results_database import serialize_crn

    rates = entry["candidate"]["rates"]
    clipped = clipped_rates(rates)
    result = {"original_rates": rates, "clipped_rates": clipped,
              "clipped_parameters": sum(a != b for a, b in zip(rates, clipped)),
              "reaction_ids": entry["candidate"]["reaction_ids"], "topology_hash": entry["topology_hash"]}
    for label, values in (("original", rates), ("clipped", clipped)):
        evaluated = EVALUATOR.evaluate(LLMCandidate(result["reaction_ids"], [[v] for v in values]))
        if not evaluated.valid or evaluated.env is None or not math.isfinite(float(evaluated.loss)):
            raise RuntimeError(f"Canonical evaluator failure: {evaluated.message}")
        if serialize_crn(evaluated.env.state)["topology_hash"] != entry["topology_hash"]:
            raise ValueError("Reevaluation changed topology")
        if label == "original" and not math.isclose(evaluated.loss, entry["loss"], rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"Original loss does not reproduce: {entry['loss']} versus {evaluated.loss}")
        result[label] = {"loss": float(evaluated.loss), **functional_accuracy(evaluated.task_info)}
    result["original_reproduction_error"] = abs(result["original"]["loss"] - entry["loss"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    config_paths = [Path(r["output"]) / "config.json" for r in snapshot["runs"]]
    configs = [json.loads(p.read_text()) for p in config_paths]
    if any(c["benchmark"]["task"] != "logic" or c["logic"] != configs[0]["logic"]
           or c["search"]["max_added_reactions"] != 5 for c in configs):
        raise ValueError("Source evaluator configurations differ")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": str(args.snapshot.resolve()), "snapshot_sha256": sha256(args.snapshot),
        "source_config_hashes": {str(p): sha256(p) for p in config_paths},
        "bounds": [.001, 100.], "snapshot_selection": "latest", "optimization": False,
        "live_search_modified": False, "threshold": snapshot["threshold"], "runs": [],
        "scientific_config": {"logic": configs[0]["logic"], "search": {"max_added_reactions": 5}}}
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(configs[0],),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for source in snapshot["runs"]:
                evaluated = list(pool.map(evaluate, source["latest"]["archive"]))
                row = {"cohort": source["cohort"], "seed": source["seed"], "epoch": source["latest"]["epoch"],
                       "evaluations": 2 * len(evaluated), "candidates": evaluated}
                for mode in ("original", "clipped"):
                    row[mode] = {"best_loss": min(r[mode]["loss"] for r in evaluated),
                        "qualified": sum(r[mode]["loss"] < report["threshold"] for r in evaluated),
                        "all16_correct": sum(r[mode]["correct_rows"] == 16 for r in evaluated),
                        "incomplete_simulations": sum(not r[mode]["simulation_complete"] for r in evaluated)}
                row["original_winner_after_clipping"] = evaluated[0]["clipped"]
                report["runs"].append(row)
                (args.output / f"{source['cohort']}_seed{source['seed']}.json").write_text(json.dumps(row, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "candidates"}), flush=True)
        report["diagnostic_evaluations"] = sum(r["evaluations"] for r in report["runs"])
        report["status"] = "complete"
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        (args.output / "failed.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
