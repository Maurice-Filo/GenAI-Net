"""Short paired parity replay with the existing GA and canonical evaluator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

from comparisons.llm_seeded_evolution.parity12 import run as canonical
from comparisons.llm_seeded_evolution.search import Candidate, Evolution, run_search
from comparisons.llm_seeded_evolution.replay import sha256
from comparisons.rpa_search.src.common.io import append_csv, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--comet-project", required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    if (replay["schema_version"] != "parity12-trial-prefix-v1" or not replay["pilot"]
            or replay["epochs"] != 100 or replay["search_budget"] != 102300):
        raise ValueError("Expected an explicitly labelled 100-epoch prefix trial")
    if any(e["after_search_evaluations"] > replay["search_budget"] for e in replay["events"]):
        raise ValueError("Future proposal present")
    args.output.mkdir(parents=True, exist_ok=False)
    config = replay["scientific_config"]
    write_json(args.output / "replay.json", replay)
    canonical.init_worker(config)
    if sorted(canonical.EVALUATOR.forbidden_reaction_ids) != replay["proposal_space"]["forbidden_reaction_ids"]:
        raise ValueError("Wrong GA mask")
    allowed = [r.ID for r in canonical.EVALUATOR.library.reactions if r.ID not in canonical.EVALUATOR.forbidden_reaction_ids]
    search = Evolution(seed=replay["seed"], allowed_ids=allowed, n_reactions=12, archive_size=30,
        rate_probability=.45, topology_probability=.45, log_rate_sigma=.5, rate_range=(.1, 50.))
    from comet_ml import Experiment
    experiment = Experiment(project_name=args.comet_project, auto_param_logging=False, auto_metric_logging=False,
        log_code=False, log_graph=False, log_git_patch=False, log_env_details=False, display_summary_level=0)
    if not experiment.get_key() or not getattr(experiment, "alive", False):
        experiment.end()
        raise RuntimeError("Comet not authenticated")
    experiment.set_name(f"parity12-ga-PILOT-100epoch-seed{replay['seed']}-20260910")
    experiment.add_tags(["pilot", "parity12", "fixed-proposal", "100epoch"])
    experiment.log_parameters({"pilot": True, "seed": replay["seed"], "epochs": 100, "batch_size": 1023,
        "search_budget": 102300, "llm_evaluations": replay["llm_evaluations"], "workers": args.workers,
        "new_model_calls": 0, "rate_mutation_probability": .45, "reaction_mutation_probability": .45,
        "random_probability": .1, "log_rate_sigma": .5})
    write_json(args.output / "comet.json", {"experiment_key": experiment.get_key(), "project": args.comet_project})
    write_json(args.output / "run_manifest.json", {"pilot": True, "epochs": 100, "replay_sha256": sha256(args.replay),
        "limits": replay["limitations"], "rate_mutation_probability": .45, "reaction_mutation_probability": .45,
        "random_probability": .1, "log_rate_sigma": .5, "initial_rate_range": [.1, 50.], "mutated_rate_bounds": None})
    references = {c["proposal_id"]: c for event in replay["events"] for c in event["candidates"]}
    start = time.monotonic()
    def record(result, n, m):
        if result.candidate.source == "llm":
            canonical.assert_parity(result, references[result.candidate.proposal_id])
        row = {"search_evaluations": n, "llm_evaluations": m, "candidate_evaluations": n + m,
            "loss": result.loss, "topology_hash": result.topology_hash, **asdict(result.candidate),
            "correct_rows": result.correct_rows, "simulation_complete": result.simulation_complete}
        append_csv(args.output / "candidates.csv", row, list(row))
    def checkpoint(n, m):
        ranked = search.archive.ranked()
        if not ranked:
            return
        rates = [v for r in ranked for v in r.candidate.rates]
        row = {"epoch": n // 1023, "search_evaluations": n, "llm_evaluations": m,
            "best_loss": ranked[0].loss, "all16_correct_topologies": sum(r.correct_rows == 16 for r in ranked),
            "hof_max_rate": max(rates), "hof_rates_outside_llm_bounds": sum(v < .001 or v > 100 for v in rates),
            "elapsed_seconds": time.monotonic() - start}
        append_csv(args.output / "progress.csv", row, list(row))
        experiment.log_metrics(row, step=n + m)
        print(json.dumps(row), flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=canonical.init_worker, initargs=(config,),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            evaluate = lambda candidates: list(pool.map(canonical.evaluate, candidates))
            n, m = run_search(search, events=replay["events"], search_budget=replay["search_budget"],
                             batch_size=1023, evaluate_batch=evaluate, record=record, checkpoint=checkpoint)
            if n != 102300 or m != replay["llm_evaluations"]:
                raise ValueError("Final budgets differ from paired checkpoint")
            ranked = search.archive.ranked()
            write_json(args.output / "terminal_archive.json", [asdict(r) for r in ranked])
            source_candidates = []
            for entry in replay["source_checkpoint_archive"]:
                added = [r for r in json.loads(entry["parameters_json"]) if r["reaction_id"] not in canonical.EVALUATOR.initial_reaction_ids]
                source_candidates.append(Candidate(tuple(r["reaction_id"] for r in added), tuple(r["parameters"][0] for r in added)))
            source_results = evaluate(source_candidates)
            for result, reference in zip(source_results, replay["source_checkpoint_archive"]):
                canonical.assert_parity(result, reference)
            write_json(args.output / "source_checkpoint_validation.json", [asdict(r) for r in source_results])
            summary = {"status": "complete", "pilot": True, "epochs": 100, "seed": replay["seed"],
                "search_evaluations": n, "llm_evaluations": m, "candidate_evaluations": n + m,
                "verification_only_evaluations": len(source_results), "new_model_calls": 0,
                "ga_best_loss": ranked[0].loss, "paired_rl_llm_best_loss": min(r.loss for r in source_results),
                "ga_all16_correct": sum(r.correct_rows == 16 for r in ranked),
                "paired_rl_llm_all16_correct": sum(r.correct_rows == 16 for r in source_results),
                "replay_sha256": sha256(args.replay), "archive_sha256": sha256(args.output / "terminal_archive.json"),
                "elapsed_seconds": time.monotonic() - start}
            write_json(args.output / "completed.json", summary)
            experiment.log_other("status", "complete")
    except Exception as error:
        write_json(args.output / "failed.json", {"type": type(error).__name__, "message": str(error)})
        experiment.log_other("status", "failed")
        raise
    finally:
        experiment.end()


if __name__ == "__main__":
    main()
