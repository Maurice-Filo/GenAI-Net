"""Parity-12 evolutionary replay with the same canonical evaluator and batch policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from comparisons.llm_seeded_evolution.run import build_evaluator as build_logic_evaluator
from comparisons.llm_seeded_evolution.replay import sha256
from comparisons.llm_seeded_evolution.search import Candidate, Evolution, ScoredCandidate, run_search
from comparisons.llm_seeded_evolution.parity12.replay import check_config
from comparisons.rpa_search.src.common.io import append_csv, write_json

EVALUATOR = None


def init_worker(config):
    global EVALUATOR
    check_config(config)
    # Both public builders call _build_logic_components; only the section key differs.
    translated = deepcopy(config)
    translated["benchmark"]["task"] = "logic"
    translated["logic"] = translated.pop("logic_parity")
    EVALUATOR = build_logic_evaluator(translated)
    if EVALUATOR.max_added_reactions != 12 or EVALUATOR.scenario_count != 16:
        raise ValueError("Reconstructed parity evaluator has the wrong dimensions")


def correctness(info):
    inputs = np.asarray(info["inputs"])
    targets = np.asarray(info["setpoint"]).reshape(-1)
    if (inputs.shape != (16, 4) or not np.isin(inputs, [0, 1]).all()
            or len(set(map(tuple, inputs))) != 16
            or not np.array_equal(targets, inputs.sum(axis=1).astype(int) % 2)):
        raise ValueError("Evaluator did not produce the complete parity truth table")
    ys = [np.asarray(y) for y in info["outputs"]]
    complete = (len(ys) == 16 and all(y.shape == (1, 1000) and np.isfinite(y).all()
                and not np.all(y == 1e4) for y in ys))
    if not complete:
        return {"simulation_complete": False, "correct_rows": None, "terminal_outputs": None}
    outputs = np.array([y[0, -1] for y in ys])
    return {"simulation_complete": True, "correct_rows": int(((outputs >= .5) == targets).sum()),
            "terminal_outputs": tuple(float(y) for y in outputs)}


@dataclass(frozen=True)
class ParityScore(ScoredCandidate):
    simulation_complete: bool = False
    correct_rows: int | None = None
    terminal_outputs: tuple[float, ...] | None = None


def evaluate(candidate):
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.utils.results_database import serialize_crn
    result = EVALUATOR.evaluate(LLMCandidate(list(candidate.reaction_ids), [[x] for x in candidate.rates]))
    if not result.valid or result.env is None or not math.isfinite(float(result.loss)):
        raise RuntimeError(f"Canonical candidate evaluation failed: {result.message}")
    crn = serialize_crn(result.env.state)
    return ParityScore(candidate, float(result.loss), crn["topology_hash"], crn["crn_text"],
                       **correctness(result.task_info))


def assert_parity(result, reference):
    loss = reference.get("reference_loss", reference.get("loss"))
    if result.topology_hash != reference["topology_hash"] or not math.isclose(result.loss, loss, rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("Recorded candidate loss/topology no longer reproduces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--comet-project", default="genai-net-ga-fixed-proposals")
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    if replay["schema_version"] != "parity12-fixed-v1" or not replay["all_exposed_hof_empty"]:
        raise ValueError("Expected a frozen, HOF-withheld parity replay")
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing parity run")
    args.output.mkdir(parents=True, exist_ok=True)
    config = replay["scientific_config"]
    settings = {"batch_size": 1023, "search_budget": 306900, "archive_size": 30,
                "rate_probability": .45, "topology_probability": .45,
                "log_rate_sigma": .5, "rate_range": [.1, 50.]}
    write_json(args.output / "replay.json", replay)
    write_json(args.output / "config.json", {"scientific_config": config, "evolution": settings,
        "workers": args.workers, "comet_project": args.comet_project,
        "qualification": "No borrowed loss threshold; report all-16 terminal correctness separately"})
    init_worker(config)
    if sorted(EVALUATOR.forbidden_reaction_ids) != replay["proposal_space"]["forbidden_reaction_ids"]:
        raise ValueError("Canonical reaction mask differs from source")
    allowed = [r.ID for r in EVALUATOR.library.reactions if r.ID not in EVALUATOR.forbidden_reaction_ids]
    optimizer = Evolution(seed=replay["seed"], allowed_ids=allowed, n_reactions=12,
        archive_size=30, rate_range=(.1, 50.), rate_probability=.45, topology_probability=.45, log_rate_sigma=.5)
    references = {c["proposal_id"]: c for e in replay["events"] for c in e["candidates"]}
    from comet_ml import Experiment
    experiment = Experiment(project_name=args.comet_project, auto_param_logging=False, auto_metric_logging=False,
        log_code=False, log_graph=False, log_git_patch=False, log_env_details=False, display_summary_level=0)
    if not experiment.get_key() or not getattr(experiment, "alive", False):
        experiment.end()
        raise RuntimeError("Comet experiment is not authenticated and live")
    experiment.set_name(f"parity12-fixed-insertion-seed{replay['seed']}")
    experiment.add_tags(["fixed-proposal", "mutation-only-evolution", "parity12", "llm-to-rl"])
    experiment.log_parameters({**settings, "seed": replay["seed"], "epochs": 300,
        "model_proposal_source": replay["model"], "new_model_calls": 0})
    write_json(args.output / "comet.json", {"experiment_key": experiment.get_key(), "project": args.comet_project})
    started = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,),
                               mp_context=multiprocessing.get_context("spawn"))
    def evaluate_batch(candidates):
        return list(pool.map(evaluate, candidates))
    def record(result, search_count, llm_count):
        if result.candidate.source == "llm":
            assert_parity(result, references[result.candidate.proposal_id])
        row = {"search_evaluations": search_count, "llm_evaluations": llm_count,
            "candidate_evaluations": search_count + llm_count, "loss": result.loss,
            "best_so_far_loss": optimizer.archive.ranked()[0].loss, "topology_hash": result.topology_hash,
            "source": result.candidate.source, "parent_hash": result.candidate.parent_hash,
            "proposal_id": result.candidate.proposal_id, "reaction_ids": result.candidate.reaction_ids,
            "rates": result.candidate.rates, "correct_rows": result.correct_rows,
            "simulation_complete": result.simulation_complete}
        append_csv(args.output / "candidates.csv", row, list(row))
    def checkpoint(search_count, llm_count):
        ranked = optimizer.archive.ranked()
        if not ranked:
            return
        row = {"epoch": search_count / 1023, "search_evaluations": search_count,
            "llm_evaluations": llm_count, "candidate_evaluations": search_count + llm_count,
            "best_loss": ranked[0].loss, "all16_correct_topologies": sum(r.correct_rows == 16 for r in ranked),
            "elapsed_seconds": time.monotonic() - started}
        append_csv(args.output / "progress.csv", row, list(row))
        experiment.log_metrics(row, step=search_count + llm_count)
        print(json.dumps(row), flush=True)
    try:
        n, m = run_search(optimizer, events=replay["events"], search_budget=306900, batch_size=1023,
                         evaluate_batch=evaluate_batch, record=record, checkpoint=checkpoint)
        if n != replay["search_budget"] or m != replay["llm_evaluations"]:
            raise ValueError("Final replay budget does not match paired source")
        ranked = optimizer.archive.ranked()
        write_json(args.output / "terminal_archive.json", [asdict(r) for r in ranked])
        write_json(args.output / "search_complete.json", {"search_evaluations": n, "llm_evaluations": m})
        # Source-archive replays are verification only: never inserted into GA or charged to its search.
        source_candidates = []
        for entry in replay["source_terminal_archive"]:
            added = [r for r in json.loads(entry["parameters_json"]) if r["reaction_id"] not in EVALUATOR.initial_reaction_ids]
            source_candidates.append(Candidate(tuple(r["reaction_id"] for r in added),
                                               tuple(r["parameters"][0] for r in added)))
        source_results = evaluate_batch(source_candidates)
        for result, reference in zip(source_results, replay["source_terminal_archive"]):
            assert_parity(result, reference)
        write_json(args.output / "source_archive_validation.json", [asdict(r) for r in source_results])
        summary = {"status": "complete", "seed": replay["seed"], "task": "logic_parity",
            "search_evaluations": n, "llm_evaluations": m, "candidate_evaluations": n + m,
            "validation_only_evaluations": len(source_results), "new_model_calls": 0,
            "ga_loss": ranked[0].loss, "rl_loss": min(r.loss for r in source_results),
            "ga_all16_correct": sum(r.correct_rows == 16 for r in ranked),
            "rl_all16_correct": sum(r.correct_rows == 16 for r in source_results),
            "elapsed_seconds": time.monotonic() - started,
            "replay_sha256": sha256(args.output / "replay.json"),
            "config_sha256": sha256(args.output / "config.json"),
            "terminal_archive_sha256": sha256(args.output / "terminal_archive.json"),
            "source_archive_validation_sha256": sha256(args.output / "source_archive_validation.json")}
        write_json(args.output / "completed.json", summary)
        experiment.log_other("status", "complete")
    except Exception as error:
        write_json(args.output / "failed.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        experiment.log_other("status", "failed")
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        experiment.end()


if __name__ == "__main__":
    main()
