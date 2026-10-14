"""Logic replay and reduction helpers used by the current portfolio analysis.

Function bodies are preserved from the original scientific implementation.
"""
from collections import Counter
from functools import lru_cache
import itertools
import json
import numpy as np

EVALUATOR = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def structural_record(record):
    return {"reaction_id": record["reaction_id"], "structure": record["structure"]}


def added_records(records, template):
    """Subtract exact template records, including rates and input channels.

    Numeric reaction IDs alone are insufficient: a library reaction can share
    an ID with an input-modulated template reaction.
    """
    key = lambda r: canonical({**structural_record(r), "parameters": r["parameters"]})
    remaining = Counter(key(r) for r in template)
    added = []
    for record in records:
        identity = key(record)
        if remaining[identity]:
            remaining[identity] -= 1
        else:
            added.append(record)
    if any(remaining.values()):
        raise ValueError("Stored candidate does not contain the exact fixed template")
    if len(added) != 5:
        raise ValueError(f"Expected five added reactions, received {len(added)}")
    return added


def logic_accuracy(outputs, targets):
    y, target = np.asarray(outputs, dtype=float), np.asarray(targets, dtype=float)
    if y.shape != (16, 1, 1000) or target.shape != (16, 1):
        raise ValueError("Expected 16 single-output scenarios at 1000 times")
    if not np.isfinite(y).all() or not np.isin(target, [0, 1]).all():
        raise ValueError("Nonfinite outputs or nonbinary targets")
    margins = (2 * target - 1)[:, :, None] * (y - 0.5)
    terminal = margins[:, 0, -1]
    late = margins[:, 0, 800:]
    return {
        "correct_rows": int((terminal > 1e-6).sum()),
        "all16_correct": bool(np.all(terminal > 1e-6)),
        "minimum_margin": float(terminal.min()),
        "row_margins": terminal.tolist(),
        "terminal_outputs": y[:, 0, -1].tolist(),
        "late_all16_correct": bool(np.all(late > 1e-6)),
        "minimum_late_margin": float(late.min()),
        "max_terminal_target_error": float(np.abs(y[:, :, -1] - target).max()),
    }


def select_core(subsets, epsilon, threshold):
    eligible = [r for r in subsets if r["status"] == "ok"
                and r["all16_correct"] and np.isfinite(r["loss"])
                and np.isfinite(r["max_output_deviation"])
                and r["loss"] < threshold and r["max_output_deviation"] <= epsilon]
    if not eligible:
        raise ValueError("An eligible original candidate must remain in its pruning set")
    return min(eligible, key=lambda r: (r["retained_count"], r["structure_json"]))


@lru_cache(maxsize=256)
def replay(action_key):
    from RL4CRN.environments.environment import Environment
    from RL4CRN.utils.results_database import serialize_crn
    env = Environment(EVALUATOR.crn_template, 5)
    for reaction_id, parameters in action_key:
        raw = {"reaction index": reaction_id, "parameters": list(parameters),
               "continuous parameters": list(parameters), "discrete parameters": None}
        env.step(action=EVALUATOR.actuator.actuate(raw), stepper=EVALUATOR.stepper, raw_action=raw)
    # Replay raw RL actions without applying the direct-LLM clamp or five-reaction validator.
    loss, info = EVALUATOR._compute_loss(env)
    inputs = np.asarray(info["inputs"])
    expected_inputs = np.asarray(list(itertools.product([0., 1.], repeat=4)))
    expected_targets = np.asarray([(u[0] and u[1]) or (u[1] and u[2]) or (u[2] and u[3])
                                  for u in expected_inputs], dtype=float)[:, None]
    if not np.array_equal(inputs, expected_inputs) or not np.array_equal(info["setpoint"], expected_targets):
        raise RuntimeError("Canonical scenarios or truth-table targets changed")
    if not np.allclose(info["time_horizon"], np.linspace(0, 100, 1000), rtol=1e-7, atol=1e-7):
        raise RuntimeError("Canonical time grid changed")
    y = np.asarray(info["outputs"], dtype=float)
    identity = serialize_crn(env.state)
    # The canonical solver fills failed/truncated outputs with its 1e4 penalty.
    status = "ok" if np.isfinite(loss) and np.isfinite(y).all() and not np.any(y == 1e4) else "solver_failure"
    return {"loss": float(loss), "outputs": y, "targets": expected_targets,
            "inputs": expected_inputs, "initial_conditions": np.asarray(info["initial_conditions"]),
            "status": status, "topology_hash": identity["topology_hash"],
            "structure_json": identity["structure_json"], "candidate_hash": identity["candidate_hash"]}


def subset_record(result, retained, original_outputs):
    row = {"retained_indices": list(retained), "retained_count": len(retained),
           "status": result["status"], "loss": result["loss"] if np.isfinite(result["loss"]) else None,
           "topology_hash": result["topology_hash"], "structure_json": result["structure_json"]}
    if result["status"] == "ok":
        row.update(logic_accuracy(result["outputs"], result["targets"]))
        row["max_output_deviation"] = float(np.abs(result["outputs"] - original_outputs).max())
    return row


def seed_summary(records, epsilon):
    qualified = [r for r in records if r["qualified"]]
    correct = [r for r in qualified if r["all16_correct"]]
    cores = [r["selected_cores"][str(epsilon)] for r in correct]
    return {"qualified": len(qualified), "qualified_all16_correct": len(correct),
            "qualified_late_all16_correct": sum(r["late_all16_correct"] for r in qualified),
            "distinct_pruned_cores": len({r["topology_hash"] for r in cores}),
            "prunable_candidates": sum(r["retained_count"] < 5 for r in cores),
            "removed_reactions": sum(5 - r["retained_count"] for r in cores),
            "merged_candidates": len(correct) - len({r["topology_hash"] for r in cores})}
