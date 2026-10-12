"""Explicit one-reaction deletion trials, never archive replacement or rate sparsity."""

from copy import deepcopy
import math


def acceptable(parent, child, threshold, require_all_rows=False):
    bound = min(threshold, parent["loss"] + max(1e-5, .02 * parent["loss"]))
    return bool(child["simulation_complete"] and math.isfinite(child["loss"])
        and child["loss"] <= bound
        and (not require_all_rows or child.get("all_rows_correct") is True))


def prune_archive(records, *, library, fit, evaluate, threshold,
                  require_all_rows=False, max_parents=5, parent_budget=60, trial_budget=12):
    """Callbacks: fit(candidate, budget)->fit result; evaluate(candidate)->canonical result.

    Budgets include joint evaluations, failures, and one canonical verification
    per tested fit. Parent re-verification is also charged. Callback exceptions
    propagate: an infrastructure error is not evidence against a deletion.
    """
    if min(max_parents, parent_budget, trial_budget) < 1 or trial_budget < 2:
        raise ValueError("Invalid pruning budget")
    parents, seen = [], set()
    for record in sorted(records, key=lambda r: r["loss"]):
        if record["topology_hash"] in seen or not record.get("qualified", False):
            continue
        seen.add(record["topology_hash"])
        parents.append(deepcopy(record))
        if len(parents) == max_parents:
            break
    output = []
    for original in parents:
        parent = {**original, **evaluate(original)}
        used = 1
        item = {"original": original, "verified_parent": parent, "trials": [], "derived": None}
        if not acceptable(parent, parent, threshold, require_all_rows):
            item.update(status="parent_not_qualified_on_verification", evaluations=used)
            output.append(item)
            continue
        ids, rates = original["reaction_ids"], original["rates"]
        library.select(ids)
        for index, rid in enumerate(ids):
            available = min(trial_budget, parent_budget - used)
            if available < 2 or len(ids) <= 1:
                break
            child = {"reaction_ids": ids[:index] + ids[index+1:],
                "rates": rates[:index] + rates[index+1:]}
            library.select(child["reaction_ids"])
            fitted = fit(child, available - 1)
            count = fitted["evaluations"]
            if not 0 <= count <= available - 1:
                raise ValueError("Pruning fitter exceeded budget")
            used += count
            trial = {"removed_reaction_id": rid, "fit": fitted, "accepted": False}
            if fitted.get("best_rates") is not None:
                child.update(reaction_ids=fitted["reaction_ids"], rates=fitted["best_rates"])
                child.update(evaluate(child))
                used += 1
                trial["canonical"] = child
                trial["accepted"] = acceptable(parent, child, threshold, require_all_rows)
                if trial["accepted"]:
                    item["derived"] = {**child, "parent_hash": original["candidate_hash"],
                        "origin": original.get("origin"), "source": "postprocess_pruning",
                        "parameter_operator": "julia_adagrad", "removed_reaction_id": rid,
                        "not_inserted_into_search_archive": True}
            item["trials"].append(trial)
            if trial["accepted"]:
                break
        item.update(status="accepted" if item["derived"] else "no_accepted_deletion", evaluations=used)
        output.append(item)
    return {"schema": "explicit-post-search-pruning-v1", "threshold": threshold,
        "require_all_rows": require_all_rows, "max_parents": max_parents,
        "parent_budget": parent_budget, "trial_budget": trial_budget, "parents": output,
        "evaluations": sum(r["evaluations"] for r in output), "archive_modified": False}


def run_post_pruning(records, *, project, evaluator, task, build_args, output, config):
    from pathlib import Path
    from RL4CRN.parameter_optimization import JuliaOptimizer
    from RL4CRN.search_exploration.engine import atomic_json
    from RL4CRN.search_exploration.evaluation import EvaluatorPool
    from .julia_adapter import task_experiment
    from .julia_fitting import TopologyFitter

    root = Path(output) / "post_pruning"
    root.mkdir(exist_ok=False)
    atomic_json(root / "originals.json", records)
    pools = {}
    calls = 0
    try:
        def evaluate(candidate):
            nonlocal calls
            count = len(candidate["reaction_ids"])
            if count not in pools:
                args = deepcopy(build_args)
                args[1]["search"]["max_added_reactions"] = count
                pools[count] = EvaluatorPool(args, 1, config.evaluation_seconds)
            value = pools[count].evaluate(candidate)
            calls += 1
            atomic_json(root / f"canonical-{calls:03d}.json", {"candidate": candidate, "evaluation": value})
            return {k: v for k, v in value.items() if k != "task_info"}

        library, experiment = task_experiment(evaluator, task)
        with JuliaOptimizer(project=project, startup_timeout=240) as worker:
            atomic_json(root / "worker.json", worker.info)
            fitter = TopologyFitter(worker, library, experiment, (config.rate_min, config.rate_max))
            fit_calls = 0

            def fit(candidate, budget):
                nonlocal fit_calls
                fit_calls += 1
                result = fitter.fit(candidate, config.seed + fit_calls, budget,
                    timeout=config.fit_seconds, starts=1)
                atomic_json(root / f"fit-{fit_calls:03d}.json", {"candidate": candidate, "fit": result})
                return result

            result = prune_archive(records, library=library, fit=fit, evaluate=evaluate,
                threshold=config.quality_threshold,
                require_all_rows=config.require_all_rows or task.kind == "logic")
        atomic_json(root / "results.json", result)
        return result
    finally:
        for pool in pools.values():
            pool.close()
