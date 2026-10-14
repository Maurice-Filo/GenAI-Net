"""Frozen post-hoc Logic portfolio pruning and kinetic robustness, without refitting."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from comparisons.search_exploration.analyze_logic5_four_arm import extract, read, sha
from comparisons.search_exploration.analyze_rpa_communication_families import search_records, union_records
from comparisons.search_exploration.analyze_rpa_directional_streams import direct_records
from comparisons.search_exploration.run_rpa_perturbations import perturb, RADII
from RL4CRN.search_exploration.engine import atomic_json

ROOT = Path(__file__).resolve().parents[2]
EPSILONS = (.001, .01, .05)
PLAN = DIRECTIONS = None


def canonical_actions(ids, rates):
    if len(ids) != len(rates):
        raise ValueError("Reaction IDs and rates must have equal lengths")
    return sorted((int(i), float(k)) for i, k in zip(ids, rates))


def candidate_key(ids, rates):
    actions = canonical_actions(ids, rates)
    return hashlib.sha256(json.dumps(actions, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare(campaign, output, workers, previous_study=None):
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.utils.results_database import serialize_crn
    amendment = None
    if previous_study is not None:
        amendment = {"previous_study": str(previous_study),
                     "previous_plan_sha256": sha(previous_study / "plan.json"),
                     "previous_failure": read(previous_study / "controller_failed.json"),
                     "reuse_previous_diagnostics": False}
    endpoints = extract(campaign)
    source_plan = read(campaign / "plan.json")
    manifest = read(Path(source_plan["jobs"][0]["output"]) / "manifest.json")
    evaluator, _, _ = build("logic", manifest["task_config"], .001, 100.)
    catalog = {}
    for i in range(len(evaluator.library)):
        reaction = evaluator.library.get_reaction(i)
        catalog[str(i)] = {"reaction_id": i, "structure": {"type": type(reaction).__name__,
            "reactants": list(reaction.reactant_labels), "products": list(reaction.product_labels),
            "inputs": list(reaction.input_channels)}}
    fixed = json.loads(serialize_crn(evaluator.crn_template)["parameters_json"])
    runs, candidates, inputs = [], {}, {str(campaign / "plan.json"): sha(campaign / "plan.json")}
    for endpoint in endpoints:
        folder = Path(endpoint["output"])
        records = union_records(search_records(folder), direct_records(folder))
        qualified = [r for r in records if r["loss"] < endpoint["threshold"]]
        if len(qualified) != endpoint["qualified"] or not np.isclose(records[0]["loss"], endpoint["loss"], rtol=1e-12):
            raise ValueError(f"Database union does not match endpoint: {folder}")
        members = []
        for rank, record in enumerate(qualified, 1):
            ids, rates = record["reaction_ids"], record["rates"]
            if (len(ids) != 5 or len(set(ids)) != 5 or set(ids) & evaluator.forbidden_reaction_ids or
                    len(rates) != 5 or not all(.001 <= k <= 100 for k in rates)):
                raise ValueError("Invalid original five-reaction design")
            saved_ids = [int(i) for i in ids]
            actions = canonical_actions(ids, rates)
            ids, rates = [a[0] for a in actions], [a[1] for a in actions]
            key = candidate_key(ids, rates)
            candidate = {"id": key, "reaction_ids": [int(i) for i in ids],
                         "rates": [float(k) for k in rates], "loss": record["loss"],
                         "recorded_losses": [record["loss"]],
                         "saved_orders": [saved_ids],
                         "recorded_correctness": [record["all_rows_correct"]] if record.get("all_rows_correct") is not None else [],
                         "topology_hash": record["topology_hash"], "winner": rank == 1}
            if key in candidates:
                if candidates[key]["topology_hash"] != candidate["topology_hash"]:
                    raise ValueError("Same exact candidate has inconsistent topology identity")
                candidates[key]["recorded_losses"] = sorted(set(candidates[key]["recorded_losses"] + candidate["recorded_losses"]))
                candidates[key]["recorded_correctness"] = sorted(set(candidates[key]["recorded_correctness"] + candidate["recorded_correctness"]))
                candidates[key]["saved_orders"] = sorted(map(list, set(map(tuple,
                    candidates[key]["saved_orders"] + candidate["saved_orders"]))))
                candidates[key]["winner"] |= rank == 1
            else:
                candidates[key] = candidate
            members.append({"candidate": key, "topology_hash": record["topology_hash"], "rank": rank,
                            "loss": record["loss"], "source": record.get("source"), "origin": record.get("origin"),
                            "saved_reaction_ids": saved_ids})
        runs.append({**endpoint, "members": members})
        for name in ("manifest.json", "completed.json", "terminal_union.json", "direct_llm_proposals.jsonl"):
            p = folder / name
            if p.exists():
                inputs[str(p)] = sha(p)
        print(f"Extracted {len(runs)}/80 portfolios: {len(members)} qualifying", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records").mkdir()
    with gzip.open(output / "selection.json.gz", "wt") as stream:
        json.dump({"runs": runs, "candidates": candidates, "catalog": catalog, "fixed": fixed}, stream, allow_nan=False)
    z = np.random.default_rng(20260917).uniform(-1, 1, (16, len(catalog)))
    paths = [Path(__file__), ROOT / "reproduce/diagnostics/logic_pruning.py",
             ROOT / "comparisons/search_exploration/run_rpa_perturbations.py",
             ROOT / "comparisons/search_exploration/analyze_logic5_four_arm.py",
             ROOT / "comparisons/search_exploration/analyze_logic5_portfolio_diagnostics.py",
             ROOT / "comparisons/search_exploration/analyze_rpa_mechanisms.py"]
    paths += list((ROOT / "RL4CRN").rglob("*.py"))
    paths += list((ROOT / "comparisons/rpa_search/src/common").glob("*.py"))
    plan = {"schema": "logic5-portfolio-pruning-robustness-v3-order-fallback", "created": time.time(), "campaign": str(campaign),
        "ordering_amendment": amendment,
        "workers": workers, "task_config": manifest["task_config"], "threshold": endpoints[0]["threshold"],
        "epsilons": list(EPSILONS), "primary_epsilon": .01, "replay_tolerance": {"rtol": 1e-7, "atol": 1e-7},
        "strict_prior_replay_tolerance": {"rtol": 1e-7, "atol": 1e-9},
        "replay_audit_note": "Post-hoc numerical-order recovery: try ascending added-reaction IDs first, then distinct recorded orders in lexicographic order, keeping rates aligned and the fixed template unchanged. Baselines require unchanged topology, all saved losses within rtol1e-7/atol1e-7, qualification and saved correctness. If none passes, retain an unresolved record and continue; do not substitute another candidate. For subsets and perturbations retry only numerical solver failures, never unfavorable loss or classification. Use the first numerically successful solve, not the minimum loss. Log every attempted order, status and loss. Unresolved baseline counts are explicit; inferred count contrasts are suppressed when any baseline is unresolved. Original endpoint scores are never overwritten; previous diagnostic records are not reused. This protocol does not establish order robustness.",
        "directions": np.concatenate((z, -z)).tolist(), "winner_radii": list(RADII), "portfolio_radius": 1.1,
        "qualification": "loss below frozen threshold; functional validity also requires all16 margins >1e-6 at t=100",
        "pruning": "All 32 subsets for every qualifying all16-correct candidate. Fixed template and surviving rates unchanged. Minimum retained count, then structural serialization. Output tolerance across all 16x1000 points.",
        "perturbations": "32 shared antithetic log-uniform directions keyed by reaction ID, no clipping/refitting. Whole correct portfolio and selected .01 cores at radius1.1; rank-one winners also at all five prior radii plus 10 OAT changes. No RPA-specific balance-preserving experiment for Logic.",
        "robust_diagnostic_cutoff": .9, "robust_diagnostic_note": "Descriptive: at least 29/32 draws stay loss-qualified and all16-correct at 10%; report continuous pass fractions and failure counts too. Not a confidence guarantee.",
        "original_records": sum(len(r["members"]) for r in runs), "unique_candidates": len(candidates),
        "refit": False, "clip": False, "scientific_timeout": None, "new_model_calls": 0,
        "analysis_unit": "matched seed; perturbation draws and repeated topologies are not independent replicates",
        "replay_order": "Canonical first, then lexicographically sorted distinct recorded orders; restrict recorded orders to retained IDs for subsets. Preserve aligned rates. Baseline recovery requires reference agreement; draw/subset recovery requires numerical success only. Unresolved numerical cases do not stop other jobs.",
        "source_hashes": {str(p.resolve()): sha(p) for p in paths},
        "input_hashes": {**inputs, str(output / "selection.json.gz"): sha(output / "selection.json.gz")}}
    atomic_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "\n")
    atomic_json(output / "status.json", {"phase": "prepared", "expected": len(candidates), "completed": 0})
    print(f"Prepared {plan['original_records']} records / {len(candidates)} unique rate-specific candidates", flush=True)
    return plan


def initialize(plan):
    global PLAN, DIRECTIONS
    import torch
    torch.set_num_threads(1)
    from RL4CRN.search_exploration.evaluation import build
    from reproduce.diagnostics import logic_pruning as pruning
    pruning.EVALUATOR, _, _ = build("logic", plan["task_config"], .001, 100.)
    pruning.replay.cache_clear()
    PLAN, DIRECTIONS = plan, np.asarray(plan["directions"])


def action_key(ids, rates):
    return tuple((i, (k,)) for i, k in canonical_actions(ids, rates))


def replay_with_orders(actions, saved_orders, accept=None):
    from reproduce.diagnostics import logic_pruning as pruning
    rates = dict(actions)
    canonical = tuple(sorted(rates))
    orders = [canonical] + sorted({tuple(i for i in order if i in rates)
                                  for order in saved_orders} - {canonical})
    attempts = []
    for order in orders:
        if len(order) != len(rates) or set(order) != set(rates):
            raise ValueError("Saved order does not cover the evaluated reactions")
        value = pruning.replay(tuple((i, rates[i]) for i in order))
        accepted = value["status"] == "ok" and (accept is None or accept(value))
        attempts.append({"reaction_ids": list(order), "status": value["status"],
                         "loss": float(value["loss"]) if np.isfinite(value["loss"]) else None,
                         "accepted": bool(accepted)})
        if accepted:
            return value, attempts, True
    return value, attempts, False


def evaluate_candidate(candidate):
    from reproduce.diagnostics import logic_pruning as pruning
    started = time.monotonic()
    key = action_key(candidate["reaction_ids"], candidate["rates"])
    before = pruning.replay.cache_info()
    references = candidate.get("recorded_losses", [candidate["loss"]])
    saved_orders = candidate.get("saved_orders", [candidate["reaction_ids"]])
    def matches_reference(value):
        correct = pruning.logic_accuracy(value["outputs"], value["targets"])["all16_correct"]
        return (value["topology_hash"] == candidate["topology_hash"] and
                all(np.isclose(value["loss"], loss, **PLAN["replay_tolerance"]) for loss in references) and
                value["loss"] < PLAN["threshold"] and
                all(v == correct for v in candidate.get("recorded_correctness", [])))
    original, baseline_attempts, resolved = replay_with_orders(key, saved_orders, matches_reference)
    if not resolved:
        after = pruning.replay.cache_info()
        return {"id": candidate["id"], "status": "complete", "baseline_unresolved": True,
                "baseline": {"status": "unresolved", "qualified": None, "all16_correct": None,
                             "stored_losses": references, "order_attempts": baseline_attempts},
                "subsets": [], "cores": {}, "perturbations": [],
                "elapsed_seconds": time.monotonic()-started,
                "evaluator_calls": after.misses-before.misses, "cache_hits": after.hits-before.hits}
    baseline = pruning.subset_record(original, tuple(range(5)), original["outputs"])
    baseline["order_attempts"] = baseline_attempts
    baseline["qualified"] = bool(original["loss"] < PLAN["threshold"])
    baseline["stored_losses"] = references
    baseline["max_replay_loss_deviation"] = max(abs(original["loss"]-loss) for loss in references)
    baseline["strict_prior_tolerance_pass"] = all(np.isclose(original["loss"], loss,
        **PLAN.get("strict_prior_replay_tolerance", PLAN["replay_tolerance"])) for loss in references)
    result = {"id": candidate["id"], "baseline_unresolved": False, "baseline": baseline,
              "subsets": [], "cores": {}, "perturbations": []}
    valid = baseline["qualified"] and baseline["all16_correct"]
    if valid:
        for n in range(6):
            for retained in itertools.combinations(range(5), n):
                if len(retained) == 5:
                    replayed, attempts = original, baseline_attempts
                else:
                    replayed, attempts, _ = replay_with_orders(tuple(key[i] for i in retained), saved_orders)
                subset = pruning.subset_record(replayed, retained, original["outputs"])
                subset["order_attempts"] = attempts
                result["subsets"].append(subset)
        for epsilon in PLAN["epsilons"]:
            result["cores"][str(epsilon)] = pruning.select_core(result["subsets"], epsilon, PLAN["threshold"])

    def sensitivity(actions, radius, mode, directions):
        ids, rates = [x[0] for x in actions], [x[1][0] for x in actions]
        for draw, direction in enumerate(directions):
            perturbed = perturb(ids, rates, radius, direction)
            value, attempts, _ = replay_with_orders(action_key(ids, perturbed), saved_orders)
            row = {"mode": mode, "radius": radius, "draw": draw, "status": value["status"],
                   "order_attempts": attempts,
                   "loss": float(value["loss"]) if np.isfinite(value["loss"]) else None,
                   "qualified": False, "all16_correct": False, "valid": False}
            if value["status"] == "ok":
                accuracy = pruning.logic_accuracy(value["outputs"], value["targets"])
                row.update({k: accuracy[k] for k in ("all16_correct", "correct_rows", "minimum_margin", "late_all16_correct")})
                row["qualified"] = bool(value["loss"] < PLAN["threshold"])
                row["valid"] = row["qualified"] and row["all16_correct"]
            result["perturbations"].append(row)

    if valid or candidate["winner"]:
        radii = PLAN["winner_radii"] if candidate["winner"] else [PLAN["portfolio_radius"]]
        for radius in radii:
            sensitivity(key, radius, "original", DIRECTIONS)
    if candidate["winner"]:
        oat = np.zeros((10, DIRECTIONS.shape[1]))
        for i, (rid, _) in enumerate(key):
            oat[2*i, rid], oat[2*i+1, rid] = -1, 1
        sensitivity(key, 1.1, "oat", oat)
    if valid:
        core = result["cores"]["0.01"]
        retained = core["retained_indices"]
        core_key = tuple(key[i] for i in retained)
        if len(retained) == 5:
            result["perturbations"] += [{**r, "mode": "core"} for r in result["perturbations"]
                                        if r["mode"] == "original" and r["radius"] == 1.1]
        else:
            sensitivity(core_key, 1.1, "core", DIRECTIONS)
    after = pruning.replay.cache_info()
    result.update(status="complete", elapsed_seconds=time.monotonic()-started,
                  evaluator_calls=after.misses-before.misses, cache_hits=after.hits-before.hits)
    return result


def load_plan(output):
    plan = read(output / "plan.json")
    if sha(output / "plan.json") != (output / "plan.sha256").read_text().strip():
        raise ValueError("Plan hash changed")
    for name, expected in {**plan["source_hashes"], **plan["input_hashes"]}.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen source/input changed: {name}")
    with gzip.open(output / "selection.json.gz", "rt") as stream:
        return plan, json.load(stream)


def execute(output, smoke=False, comet_project="genai-net-exploration"):
    plan, selection = load_plan(output)
    fingerprint = sha(output / "plan.json")
    candidates = selection["candidates"]
    done, calls, unresolved, recovered = set(), 0, 0, 0
    for file in (output / "records").glob("*.json"):
        record = read(file)
        if record.get("plan_sha256") != fingerprint or record.get("status") != "complete" or record["id"] not in candidates:
            raise ValueError(f"Invalid cached diagnostic: {file}")
        done.add(record["id"])
        calls += record["evaluator_calls"]
        unresolved += record.get("baseline_unresolved", False)
        recovered += not record.get("baseline_unresolved", False) and len(record["baseline"].get("order_attempts", [])) > 1
    todo = sorted((r for k, r in candidates.items() if k not in done), key=lambda r: (not r["winner"], r["id"]))
    if smoke:
        todo = todo[:2]
    workers = min(plan["workers"], 2) if smoke else plan["workers"]
    logger = None
    if not smoke and comet_project:
        os.environ.pop("COMET_API_KEY", None)
        from comet_ml import Experiment
        logger = Experiment(project_name=comet_project, auto_param_logging=False, auto_metric_logging=False,
            log_code=False, log_git_patch=False, log_graph=False, log_git_metadata=False,
            log_env_details=False, log_env_host=False, log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
        if not logger.get_key() or not getattr(logger, "alive", False):
            raise RuntimeError("Comet connection failed")
        logger.set_name(output.name)
        logger.add_tags(["logic5", "post-hoc", "perturbation", "exhaustive-pruning", "no-model-calls"])
        logger.log_parameters({"workers": workers, "unique_candidates": len(candidates), "plan_sha256": fingerprint,
                               "refit": False, "clip": False, "portfolio_radius": 1.1, "draws": 32})
        atomic_json(output / "comet.json", {"experiment_key": logger.get_key(), "project": comet_project})
    pending, iterator = {}, iter(todo)
    started = time.time()
    def progress(phase):
        payload = {"phase": phase, "completed": len(done), "expected": len(candidates), "evaluator_calls": calls,
                   "updated": time.time(), "started": started, "pid": os.getpid(), "workers": workers,
                   "in_flight": len(pending), "plan_sha256": fingerprint}
        payload.update(baselines_unresolved=unresolved, baselines_recovered=recovered)
        atomic_json(output / "status.json", payload)
        if logger:
            logger.log_metrics({"Diagnostics/completed": len(done), "Diagnostics/remaining": len(candidates)-len(done),
                                "Diagnostics/evaluator_calls": calls, "Diagnostics/baselines_unresolved": unresolved,
                                "Diagnostics/baselines_recovered": recovered}, step=len(done))
    progress("smoke" if smoke else "running")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(plan,)) as pool:
            exhausted, last = False, 0.
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    candidate = next(iterator, None)
                    if candidate is None:
                        exhausted = True
                    else:
                        pending[pool.submit(evaluate_candidate, candidate)] = candidate["id"]
                ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    identifier = pending.pop(future)
                    try:
                        record = future.result()
                    except BaseException as error:
                        atomic_json(output / "controller_failed.json", {"candidate": identifier,
                            "type": type(error).__name__, "error": str(error), "time": time.time()})
                        raise
                    record["plan_sha256"] = fingerprint
                    atomic_json(output / "records" / f"{identifier}.json", record)
                    done.add(identifier)
                    calls += record["evaluator_calls"]
                    unresolved += record.get("baseline_unresolved", False)
                    recovered += not record.get("baseline_unresolved", False) and len(record["baseline"].get("order_attempts", [])) > 1
                if time.monotonic()-last > 15:
                    progress("smoke" if smoke else "running")
                    print(f"Persisted {len(done)}/{len(candidates)} candidates; {calls} canonical calls", flush=True)
                    last = time.monotonic()
        phase = "complete" if len(done) == len(candidates) else "smoke_passed"
        progress(phase)
        if phase == "complete":
            atomic_json(output / "completed.json", read(output / "status.json"))
            from comparisons.search_exploration.analyze_logic5_portfolio_diagnostics import analyze
            analysis = analyze(output)
            if logger:
                for image in sorted(analysis.glob("*.png")):
                    logger.log_image(str(image), name=image.stem)
                logger.log_asset(str(analysis / "REPORT.md"))
    except BaseException:
        progress("failed")
        raise
    finally:
        if logger:
            logger.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--previous-study", type=Path, help="Preserved failed diagnostic study documented by this revision")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.campaign.resolve(), args.output.resolve(), args.workers,
                args.previous_study.resolve() if args.previous_study else None)
    if args.execute or args.smoke:
        with (args.output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(args.output.resolve(), smoke=args.smoke)
