"""Replay audited Logic portfolios and exhaustively prune output-negligible reactions.

The immutable post-hoc plan is written before outcomes are computed. Original
databases are read-only. All surviving rates and fixed template reactions are
preserved; reduced circuits are diagnostics, not new admitted search candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from paper.iclr2027_genai_net_llm.analyze_contract_v2_primary import holm, wilcoxon_p, write_csv
from paper.iclr2027_genai_net_llm.generate_portfolio_sensitivity_opportunities import read_database

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
GENERATED = PAPER / "generated"
STEM = "logic_correctness_pruning"
PLAN_PATH = GENERATED / f"{STEM}_plan.json"
SOURCES = {"flash_logic": "flash_logic_directional.json", "local_logic": "local_logic_amplification.json"}
ARMS = ("none", "rl_to_llm", "llm_to_rl", "full")
EPSILONS = (0.001, 0.01, 0.05)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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


def worker_init(plan):
    global EVALUATOR, TEMPLATE, THRESHOLD
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import build_evaluator
    from RL4CRN.utils.results_database import serialize_crn
    EVALUATOR = build_evaluator("logic", solver="CVODE")
    TEMPLATE = json.loads(serialize_crn(EVALUATOR.crn_template)["parameters_json"])
    THRESHOLD = plan["quality_threshold"]


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


def analyze_run(meta, fingerprint, plan, cache_dir):
    started = time.monotonic()
    entries, _ = read_database(meta)  # Recheck immutable input hash even when resuming.
    cache_path = cache_dir / f"{meta['cohort']}_{meta['arm']}_{meta['seed']:02d}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") == fingerprint and cached.get("database_sha256") == meta["database_sha256"]:
            if len(cached["records"]) != 30 or cached.get("status") != "complete":
                raise RuntimeError("Invalid partial cache")
            return cached
    records = []
    for entry in entries:
        stored = json.loads(entry["parameters_json"])
        archived = json.loads(entry["task_info_json"])
        added = added_records(stored, TEMPLATE)
        action_key = tuple((int(a["reaction_id"]), tuple(float(x) for x in a["parameters"])) for a in added)
        original = replay(action_key)
        if original["status"] != "ok":
            raise RuntimeError(f"Original candidate replay failed: {meta['cohort']}/{meta['arm']}/{meta['seed']}/{entry['rank']}")
        if original["topology_hash"] != entry["topology_hash"] or not np.isclose(
                original["loss"], entry["loss"], **plan["replay_tolerance"]):
            raise RuntimeError("Original candidate topology/loss replay mismatch")
        for field in ("setpoint", "initial_conditions", "inputs"):
            actual = original["targets"] if field == "setpoint" else original[field]
            if not np.array_equal(actual, archived[field]):
                raise RuntimeError(f"Archived scenario mismatch: {field}")
        record = {"cohort": meta["cohort"], "arm": meta["arm"], "seed": meta["seed"],
                  "rank": entry["rank"], "topology_hash": entry["topology_hash"],
                  "candidate_hash": original["candidate_hash"], "loss": entry["loss"],
                  "replay_loss": original["loss"], "emitter": entry["emitter"],
                  "qualified": bool(entry["loss"] < THRESHOLD), "added_reactions": added,
                  "targets": original["targets"][:, 0].tolist(),
                  **logic_accuracy(original["outputs"], original["targets"]),
                  "subsets": [], "selected_cores": {}}
        if record["qualified"] and record["all16_correct"]:
            for count in range(6):
                for retained in itertools.combinations(range(5), count):
                    result = replay(tuple(action_key[i] for i in retained))
                    record["subsets"].append(subset_record(result, retained, original["outputs"]))
            if len(record["subsets"]) != 32:
                raise RuntimeError("Incomplete exhaustive subset enumeration")
            for epsilon in EPSILONS:
                core = select_core(record["subsets"], epsilon, THRESHOLD)
                record["selected_cores"][str(epsilon)] = {k: core[k] for k in (
                    "topology_hash", "retained_count", "retained_indices", "loss", "minimum_margin", "max_output_deviation")}
        records.append(record)
    payload = {**meta, "status": "complete", "fingerprint": fingerprint,
               "elapsed_seconds": time.monotonic() - started, "records": records}
    atomic_json(cache_path, payload)
    return payload


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


def summarize(runs, plan):
    endpoints = []
    for run in runs:
        for epsilon in EPSILONS:
            endpoints.append({"cohort": run["cohort"], "arm": run["arm"], "seed": run["seed"],
                              "epsilon": epsilon, **seed_summary(run["records"], epsilon)})
    indexed = {(r["cohort"], r["arm"], r["seed"], r["epsilon"]): r for r in endpoints}
    contrasts, summaries = [], []
    for cohort in SOURCES:
        for arm in ARMS:
            for epsilon in EPSILONS:
                rows = [indexed[cohort, arm, seed, epsilon] for seed in range(20)]
                summary = {"cohort": cohort, "arm": arm, "epsilon": epsilon, "n": 20}
                for field in ("qualified", "qualified_all16_correct", "qualified_late_all16_correct",
                              "distinct_pruned_cores", "prunable_candidates", "removed_reactions", "merged_candidates"):
                    summary[f"mean_{field}"] = float(np.mean([r[field] for r in rows]))
                    summary[f"median_{field}"] = float(np.median([r[field] for r in rows]))
                summaries.append(summary)
        for metric in ("qualified_all16_correct", "distinct_pruned_cores"):
            left, right = [np.array([indexed[cohort, arm, seed, .01][metric] for seed in range(20)])
                           for arm in ("llm_to_rl", "none")]
            delta = left - right
            resamples = np.random.default_rng(20260909).integers(0, 20, (20000, 20))
            low, high = np.quantile(delta[resamples].mean(axis=1), [.025, .975])
            contrasts.append({"cohort": cohort, "left": "llm_to_rl", "right": "none", "metric": metric,
                              "mean_left": float(left.mean()), "mean_right": float(right.mean()),
                              "mean_delta": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
                              "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()),
                              "losses": int((delta < 0).sum()), "p_raw": wilcoxon_p(left.tolist(), right.tolist())})
    for row, p in zip(contrasts, holm([r["p_raw"] for r in contrasts])):
        row["p_holm_4"] = p
    return endpoints, summaries, contrasts


def report(payload):
    lines = ["# Logic correctness and output-preserving reaction pruning", "",
             "Post-hoc diagnostic; rules fixed before computation in `generated/logic_correctness_pruning_plan.json`.", "",
             "All 4800 terminal topology-by-seed records are replayed with the canonical CVODE evaluator. "
             "Original topology, loss, and scenarios must match their frozen records. Correctness requires "
             "all 16 signed terminal margins relative to 0.5 to exceed 1e-6.", "",
             "For each originally qualifying, correct candidate, all 32 subsets of its five added reactions "
             "are checked with fixed template and surviving rates unchanged. A retained subset must remain "
             "qualified and correct, with maximum OUT deviation over all 16 x 1000 observations at most "
             "0.01 (primary), 0.001 or 0.05 (descriptive sensitivity). The fewest-reaction subset is chosen; "
             "ties use canonical structural serialization. Selected core hashes are deduplicated within each seed.", "",
             f"Replay audit: {payload['audit']['original_records']} original records matched; "
             f"{payload['audit']['eligible_candidates']} candidates underwent "
             f"{payload['audit']['subset_evaluations']} subset checks, including full and empty subsets. "
             f"{payload['audit']['failed_subsets']} failed subset evaluations were ineligible.", "",
             "## Primary pruning rule (0.01)", "",
             "| Model | Arm | Mean qualifying | Mean correct | Mean distinct cores | Mean prunable | Mean merged |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in payload["summaries"]:
        if r["epsilon"] == .01:
            lines.append(f"| {r['cohort']} | {r['arm']} | {r['mean_qualified']:.2f} | "
                         f"{r['mean_qualified_all16_correct']:.2f} | {r['mean_distinct_pruned_cores']:.2f} | "
                         f"{r['mean_prunable_candidates']:.2f} | {r['mean_merged_candidates']:.2f} |")
    lines += ["", "## Insertion versus independent union", "",
              "| Model | Metric | Mean gain | 95% paired bootstrap interval | W/T/L | Holm p (4 tests) |",
              "|---|---|---:|---|---|---:|"]
    for r in payload["contrasts"]:
        lines.append(f"| {r['cohort']} | {r['metric']} | {r['mean_delta']:.2f} | "
                     f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['wins']}/{r['ties']}/{r['losses']} | {r['p_holm_4']:.6g} |")
    lines += ["", "## Descriptive pruning sensitivity", "",
              "| Model | Tolerance | Independent mean cores | Insertion mean cores | Mean gain |",
              "|---|---:|---:|---:|---:|"]
    for cohort in SOURCES:
        for eps in EPSILONS:
            index = {r["arm"]: r for r in payload["summaries"] if r["cohort"] == cohort and r["epsilon"] == eps}
            a, b = [index[arm]["mean_distinct_pruned_cores"] for arm in ("none", "llm_to_rl")]
            lines.append(f"| {cohort} | {eps} | {a:.2f} | {b:.2f} | {b-a:.2f} |")
    lines += ["", "Pruned cores are derived diagnostics outside the original exact-five-reaction proposal "
              "space. Counts do not add new candidates to any search archive. Fixed surviving rates and "
              "pointwise output preservation make this a test of removable reactions, not a general "
              "mechanistic equivalence test. Endpoint correctness and finite-horizon trajectory similarity "
              "do not establish asymptotic stability, parameter robustness, or biological implementability. "
              "Confidence intervals are marginal; only the four stated insertion contrasts form the new Holm family."]
    (PAPER / "LOGIC_CORRECTNESS_PRUNING.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-runs", type=int, help="Implementation check only; never emits study summaries")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    for name, expected in plan["input_hashes"].items():
        if digest(GENERATED / name) != expected:
            raise RuntimeError(f"Frozen plan input changed: {name}")
    source_paths = [*sorted((ROOT / "RL4CRN").rglob("*.py")),
                    *sorted((ROOT / "comparisons/rpa_search/src/common").glob("*.py")),
                    ROOT / "comparisons/rpa_search/configs/logic_100k.json",
                    ROOT / "comparisons/llm_crn_generation/run_mmc2_harness_smoke.py",
                    Path(__file__), PAPER / "analyze_contract_v2_primary.py",
                    PAPER / "generate_portfolio_sensitivity_opportunities.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    fingerprint = hashlib.sha256(canonical({"plan": digest(PLAN_PATH), "code": hashes}).encode()).hexdigest()
    metadata = [{"cohort": cohort, **r} for cohort, name in SOURCES.items()
                for r in json.loads((GENERATED / name).read_text())["database_audit"]]
    if len(metadata) != 160 or {(m['cohort'], m['arm'], m['seed']) for m in metadata} != {
            (c, a, s) for c in SOURCES for a in ARMS for s in range(20)}:
        raise RuntimeError("Expected complete two-model four-arm 20-seed grid")
    if args.limit_runs is not None:
        metadata = metadata[:args.limit_runs]
    cache_dir = GENERATED / f"{STEM}_runs"
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(plan,)) as pool:
        futures = {pool.submit(analyze_run, m, fingerprint, plan, cache_dir): m for m in metadata}
        for future in as_completed(futures):
            run = future.result()
            runs.append(run)
            print(f"{len(runs)}/{len(metadata)} {run['cohort']} {run['arm']} seed={run['seed']} "
                  f"checked={len(run['records'])} seconds={run['elapsed_seconds']:.1f}", flush=True)
    if args.limit_runs is not None:
        print("Implementation check complete; full-study summaries not emitted.", flush=True)
        return
    runs.sort(key=lambda r: (r["cohort"], r["arm"], r["seed"]))
    endpoints, summaries, contrasts = summarize(runs, plan)
    payload = {"status": "complete", "plan_sha256": digest(PLAN_PATH), "source_hashes": hashes,
               "fingerprint": fingerprint, "plan": plan, "database_audit": [
                   {k: r[k] for k in ("cohort", "arm", "seed", "database", "database_sha256")} for r in runs],
               "records": [r for run in runs for r in run["records"]],
               "endpoints": endpoints, "summaries": summaries, "contrasts": contrasts}
    if len(payload["records"]) != 4800:
        raise RuntimeError("Missing terminal records")
    payload["audit"] = {
        "original_records": len(payload["records"]),
        "eligible_candidates": sum(r["qualified"] and r["all16_correct"] for r in payload["records"]),
        "subset_evaluations": sum(len(r["subsets"]) for r in payload["records"]),
        "failed_subsets": sum(s["status"] != "ok" for r in payload["records"] for s in r["subsets"]),
    }
    atomic_json(GENERATED / f"{STEM}.json", payload)
    for suffix, rows in (("endpoints", endpoints), ("summary", summaries), ("contrasts", contrasts)):
        write_csv(GENERATED / f"{STEM}_{suffix}.csv", rows)
    report(payload)
    print(json.dumps(contrasts, indent=2), flush=True)


if __name__ == "__main__":
    main()
