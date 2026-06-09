from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import numpy as np

from comparisons.rpa_search.src.common.evaluator import (
    build_candidate,
    candidate_summary,
    evaluate_crn,
    sampled_reaction_ids_and_rates,
)
from comparisons.rpa_search.src.common.io import (
    CANDIDATE_FIELDS,
    PROGRESS_FIELDS,
    append_csv,
    write_json,
)


_WORKER_COMPONENTS = None


def _init_worker(config: Dict[str, Any]) -> None:
    global _WORKER_COMPONENTS
    from comparisons.rpa_search.src.common.task_factory import build_components

    _WORKER_COMPONENTS = build_components(config)


def _evaluate_candidate(args):
    candidate_id, reaction_ids, rates = args
    if _WORKER_COMPONENTS is None:
        raise RuntimeError("Random-search worker was not initialized.")
    template_crn, library_components, task, _cfg = _WORKER_COMPONENTS
    library = library_components[0]
    crn = build_candidate(template_crn, library, reaction_ids, rates)
    result = evaluate_crn(crn, task)
    summary = candidate_summary(crn)
    return (
        candidate_id,
        reaction_ids,
        rates,
        result.loss,
        result.ode_simulations,
        result.scenario_count,
        result.performance,
        summary,
    )


def run_random_search(config: Dict[str, Any], run_dir, method: str, run_id: str, components) -> Dict[str, Any]:
    template_crn, library_components, task, _cfg = components
    library = library_components[0]
    search = config["search"]
    rng = np.random.default_rng(int(search.get("seed", 0)))
    budget = int(search.get("candidate_budget", 100))
    n_reactions = int(search.get("max_added_reactions", 5))
    rate_range = search.get("rate_constant_range", [0.1, 50.0])
    workers = int(search.get("random_workers", os.environ.get("RPA_RANDOM_WORKERS", 1)))
    chunk_size = int(search.get("random_chunk_size", 32))

    best_loss = float("inf")
    best_summary = None
    total_ode = 0
    total_scenarios = 0
    tic = time.time()
    start_id = 1

    progress_path = run_dir / "progress.csv"
    if progress_path.exists():
        with progress_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            start_id = int(last["candidate_evaluations"]) + 1
            total_ode = int(float(last["ode_simulations"]))
            total_scenarios = int(float(last["scenario_evaluations"]))
            best_loss = float(last["best_so_far_loss"])

    for candidate_id in range(1, start_id):
        sampled_reaction_ids_and_rates(rng, library, n_reactions, rate_range)

    def candidates():
        for candidate_id in range(start_id, budget + 1):
            reaction_ids, rates = sampled_reaction_ids_and_rates(rng, library, n_reactions, rate_range)
            yield candidate_id, reaction_ids, rates

    if start_id > budget:
        return {"best_loss": best_loss, "ode_simulations": total_ode}

    iterator = candidates()
    if workers > 1:
        results = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(config,)).map(
            _evaluate_candidate,
            iterator,
            chunksize=chunk_size,
        )
    else:
        _init_worker(config)
        results = map(_evaluate_candidate, iterator)

    for candidate_id, reaction_ids, rates, loss, ode_simulations, scenario_count, performance, summary in results:
        total_ode += ode_simulations
        total_scenarios += scenario_count

        if loss < best_loss:
            best_loss = loss
            best_summary = summary

        progress_row = {
            "method": method,
            "run_id": run_id,
            "step": candidate_id,
            "candidate_evaluations": candidate_id,
            "ode_simulations": total_ode,
            "scenario_count": scenario_count,
            "scenario_evaluations": total_scenarios,
            "loss": loss,
            "best_so_far_loss": best_loss,
            "performance": performance,
            "best_so_far_performance": -best_loss,
            "elapsed_seconds": time.time() - tic,
        }
        append_csv(run_dir / "progress.csv", progress_row, PROGRESS_FIELDS)

        candidate_row = {
            "method": method,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "candidate_evaluations": candidate_id,
            "ode_simulations": total_ode,
            "scenario_count": scenario_count,
            "scenario_evaluations": total_scenarios,
            "loss": loss,
            "best_so_far_loss": best_loss,
            "reaction_ids": reaction_ids,
            "rate_constants": rates,
        }
        append_csv(run_dir / "candidates.csv", candidate_row, CANDIDATE_FIELDS)
        if candidate_id == 1 or candidate_id == budget or candidate_id % 50 == 0:
            print(f"[{method} {run_id}] candidates={candidate_id} ode_sims={total_ode} best_loss={best_loss:.6g}", flush=True)

    if best_summary is not None:
        (run_dir / "best_network.txt").write_text("\n".join(best_summary["reactions"]), encoding="utf-8")
        write_json(run_dir / "best_network.json", best_summary)

    return {"best_loss": best_loss, "ode_simulations": total_ode}
