"""Compare identical fitting jobs through old/full and new/compact worker paths."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.evaluation import EvaluatorPool, EvaluationService, build
from RL4CRN.search_exploration.refinement import fit
from RL4CRN.search_exploration.store import Store, fingerprint
from RL4CRN.search_streams.store import CompactStore
from RL4CRN.search_streams.workers import StreamPool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.jobs, args.workers) < 1:
        parser.error("Positive counts required")
    manifest = json.loads((args.reference / "manifest.json").read_text())
    config = ExplorationConfig(**manifest["config"])
    config = replace(config, workers=args.workers, evaluation_budget=args.jobs * config.fit_evaluations)
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = [json.loads(p.read_text())["proposal"] for p in sorted((args.reference / "cohorts/0").glob("*.json"))[:args.jobs]]
    if len(candidates) != args.jobs:
        raise ValueError("Not enough frozen proposals")
    task = manifest["task"]
    build_args = (task, manifest["task_config"], config.rate_min, config.rate_max)
    evaluator, _, _ = build(*build_args)
    allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
    atomic_json(args.output / "plan.json", {"reference": str(args.reference), "candidates": candidates,
        "config": config.to_dict(), "task_config": manifest["task_config"], "workers": args.workers})
    result_sets, summaries = [], []
    for mode in ("full-transfer", "compact-stream"):
        path = args.output / (mode + ".sqlite")
        identity = fingerprint({"build_args": build_args, "config": config.to_dict()})
        if mode == "full-transfer":
            store = Store(path, config.evaluation_budget, identity)
            pool = EvaluatorPool(build_args, args.workers, config.evaluation_seconds)
            service = EvaluationService(store, pool, allowed, evaluator.max_added_reactions, (config.rate_min, config.rate_max))
        else:
            store = CompactStore(path, config.evaluation_budget, identity, 0)
            pool = StreamPool(build_args, args.workers, config.evaluation_seconds, store, config.fit_seconds)
        try:
            def execute(c):
                seed = int(fingerprint({"seed": config.seed, "pairs": sorted(zip(c["reaction_ids"], c["rates"]))})[:8], 16)
                if mode == "full-transfer":
                    return fit(c, service, config, seed, include_diagnostics=False)
                return pool.run({"category": "optimization", "candidate": c, "seed": seed, "config": config.to_dict()})
            start = time.monotonic()
            with ThreadPoolExecutor(args.workers) as executor:
                results = list(executor.map(execute, candidates))
            elapsed = time.monotonic() - start
            results = [{k: v for k, v in r.items() if k != "packed_diagnostics"} for r in results]
            result_sets.append(results)
            summaries.append({"mode": mode, "elapsed_seconds_excluding_worker_startup": elapsed,
                "evaluations": store.count("expensive_evaluations"), "cache_hits": store.count("cache_hits"),
                "failures": sum(len(r["failures"]) for r in results)})
            atomic_json(args.output / (mode + ".json"), results)
        finally:
            pool.close()
            store.close()
        summaries[-1]["sqlite_bytes"] = path.stat().st_size
    equal = all(a["best"] == b["best"] and a["calls"] == b["calls"] and a["failures"] == b["failures"]
                for a, b in zip(*result_sets))
    result = {"matched_results_exact": equal, "summaries": summaries,
              "speedup": summaries[0]["elapsed_seconds_excluding_worker_startup"] / summaries[1]["elapsed_seconds_excluding_worker_startup"],
              "caveat": "One fixed workload, full-transfer first; not a repeated randomized performance benchmark"}
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    if not equal or any(s["failures"] for s in summaries):
        raise RuntimeError("Worker benchmark did not preserve successful fitting results")


if __name__ == "__main__":
    main()
