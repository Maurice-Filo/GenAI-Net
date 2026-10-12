"""Standalone legacy mutation GA, with explicit bounded rates and no LLM replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int, help="Override the reference seed for matched-seed campaigns")
    parser.add_argument("--batch-size", type=int, default=1023)
    parser.add_argument("--comet-project", default="genai-net-exploration")
    parser.add_argument("--no-comet", action="store_true", help="Keep all metrics and plots local")
    parser.add_argument("--no-evaluation-timeouts", action="store_true", help="Limit simulations by count, not elapsed time")
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1:
        parser.error("Positive worker and batch sizes required")
    if args.seed is not None and args.seed < 0:
        parser.error("Seed must be nonnegative")
    if not args.no_comet:
        from comet_ml import Experiment
    from comparisons.llm_seeded_evolution.search import ScoredCandidate
    from comparisons.search_exploration.bounded_ga import BoundedEvolution
    from comparisons.search_exploration.freeze_legacy import verify, ROOT, DEFAULT, digest
    from RL4CRN.search_exploration.archive import descriptor
    from RL4CRN.search_exploration.engine import atomic_json
    from RL4CRN.search_exploration.evaluation import build, materialize, EvaluationService
    from RL4CRN.search_exploration.store import fingerprint
    from RL4CRN.search_exploration.telemetry import Telemetry
    from RL4CRN.search_streams.database import StreamResultsDatabase
    from RL4CRN.search_streams.store import CompactStore
    from RL4CRN.search_streams.workers import StreamPool
    verify()
    reference = args.reference.resolve()
    original = json.loads((reference / "manifest.json").read_text())
    completed = json.loads((reference / "completed.json").read_text())
    if original["schema"] != "search-streams-v2" or completed["status"] != "complete":
        raise ValueError("Requires a completed corrected-stream reference")
    budget = args.budget if args.budget is not None else completed["counters"]["expensive_evaluations"]
    if budget < 1:
        raise ValueError("Positive evaluation budget required")
    config = {k: original["config"][k] for k in ("seed", "rate_min", "rate_max", "quality_threshold",
        "require_all_rows", "descriptor", "plot_every", "evaluation_seconds", "fit_seconds")}
    config.update(evaluation_budget=budget, batch_size=args.batch_size, archive_capacity=30)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_evaluation_timeouts:
        config.update(evaluation_seconds=None, fit_seconds=None)
    settings = json.loads((ROOT / "comparisons/llm_seeded_evolution/configs/default.json").read_text())
    settings["batch_size"] = args.batch_size
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_lock = (output / "run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    logger = store = pool = database = None
    batch = 0
    try:
        evaluator, _, _ = build(original["task"], original["task_config"], config["rate_min"], config["rate_max"])
        allowed = sorted(r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids)
        sources = [*sorted((ROOT / "RL4CRN/search_exploration").glob("*.py")),
            *sorted((ROOT / "RL4CRN/search_streams").glob("*.py")), Path(__file__).resolve(),
            ROOT / "comparisons/search_exploration/bounded_ga.py",
            ROOT / "comparisons/llm_seeded_evolution/search.py",
            ROOT / "comparisons/llm_seeded_evolution/configs/default.json"]
        import importlib.metadata
        science = {"schema": "standalone-ga-bounded-v1", "proposal_arm": "standalone-ga",
            "task": original["task"], "task_config": original["task_config"], "config": config,
            "reference": str(reference), "reference_identity": original["identity"],
            "harness": False, "model": None, "replay_events": [], "policy_training_enabled": False,
            "archive_policy": "legacy-top-30-lowest-loss", "parent_archive_size": 30,
            "ga_settings": settings, "rate_bound_policy": "clip after original mutation",
            "allowed_ids": allowed, "search_workers": args.workers, "optimization_workers": 0,
            "search_evaluation_reserve": budget, "proposal_safety_cap": 10 * budget,
            "termination": "actual_evaluation_budget", "batch_semantics": "frozen parents per batch",
            "new_source": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "legacy_source": digest(DEFAULT / "scientific_source.tar.gz"),
            "versions": {p: importlib.metadata.version(p) for p in original["versions"]}}
        science["identity"] = fingerprint(science)
        atomic_json(output / "manifest.json", science)
        atomic_json(output / "status.json", {"status": "initializing", "epoch": 0})
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in sources:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
            archive.add(DEFAULT / "scientific_source.tar.gz", arcname="legacy_source.tar.gz")
        if not args.no_comet:
            logger = Experiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_git_patch=False, log_graph=False,
                log_git_metadata=False, log_env_details=False, log_env_host=False,
                log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
            logger.set_name(output.parent.name + "-standalone-ga")
            logger.add_tags(["standalone-ga-bounded-v1", original["task"], "no-llm", "no-rl"])
            logger.log_parameters({**config, **settings, "physical_evaluation_target": budget,
                "GA/parent_archive_size": 30, "GA/batch_size": args.batch_size, "workers": args.workers,
                "identity": science["identity"], "no_replay": True})
            atomic_json(output / "comet.json", {"experiment_key": logger.get_key(), "project": args.comet_project})
        store = CompactStore(output / "exploration.sqlite", budget, science["identity"], budget,
            wait_timeout=None if config["evaluation_seconds"] is None else 120)
        pool = StreamPool((original["task"], original["task_config"], config["rate_min"], config["rate_max"]),
            args.workers, config["evaluation_seconds"], store, config["fit_seconds"])
        service = EvaluationService(store, pool, allowed, evaluator.max_added_reactions,
            (config["rate_min"], config["rate_max"]))
        evolution = BoundedEvolution(bounds=service.bounds, seed=config["seed"], allowed_ids=allowed,
            n_reactions=service.reaction_count, archive_size=30,
            rate_range=tuple(settings["initial_rate_range"]),
            rate_probability=settings["rate_mutation_probability"],
            topology_probability=settings["reaction_mutation_probability"], log_rate_sigma=settings["log_rate_sigma"])
        database = StreamResultsDatabase(output / "results.sqlite", run_id=output.parent.name,
            run_metadata={"method": science["schema"], "task": original["task"], "config": config,
                "physical_evaluation_target": budget, "parent_archive_capacity": 30, "harness": False})
        telemetry = Telemetry(output, logger)
        library = {r.ID: r for r in evaluator.library.reactions}
        qualified_seen, regions_seen = set(), set()
        archive_records = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while store.count("expensive_evaluations") < budget:
                count = min(args.batch_size, budget - store.count("expensive_evaluations"))
                if store.count("proposal_attempts") + count > science["proposal_safety_cap"]:
                    raise RuntimeError("Proposal safety cap reached before physical budget; no silent completion")
                candidates = [evolution.propose() for _ in range(count)]
                proposals = []
                for i, candidate in enumerate(candidates):
                    parent = archive_records.get(candidate.parent_hash)
                    proposal = {"reaction_ids": list(candidate.reaction_ids), "rates": list(candidate.rates),
                        "source": "random" if candidate.source == "random" else "ga", "origin": "random",
                        "operator": candidate.source, "parent_hash": parent["candidate_hash"] if parent else None,
                        "proposal_id": f"batch{batch}-proposal{i}", "policy_version": 0,
                        "stream": "standalone_ga", "parameter_operator": None}
                    service.validate(proposal)
                    proposals.append(proposal)
                store.count("proposal_attempts", count)
                for source in ("ga", "random"):
                    store.count("proposals/" + source, sum(p["source"] == source for p in proposals))
                results = list(executor.map(pool.run, ({"candidate": p, "category": "search"} for p in proposals)))
                for candidate, proposal, result in zip(candidates, proposals, results):
                    if result["failures"]:
                        raise RuntimeError(str(result["failures"]))
                    record = {**result["best"], **proposal}
                    record["qualified"] = bool(record["simulation_complete"] and record["loss"] < config["quality_threshold"]
                        and (not config["require_all_rows"] or record["all_rows_correct"] is True))
                    record["region"] = descriptor(record["reaction_ids"], library, evaluator.crn_template.output_labels,
                        config["descriptor"], record["topology_hash"])
                    if record["qualified"]:
                        qualified_seen.add(record["topology_hash"])
                        regions_seen.add(record["region"])
                    scored = ScoredCandidate(candidate, record["loss"], record["topology_hash"])
                    evolution.archive.add(scored)
                    admitted = evolution.archive.entries.get(scored.topology_hash) == scored
                    with store.transaction():
                        store.save_design(record)
                        store.event("archive_insertion", {**{k: record[k] for k in ("source", "origin", "parent_hash",
                            "proposal_id", "stream", "operator", "candidate_hash", "loss")},
                            "epoch": batch, "admitted": admitted, "gain": 0., "novelty": 0.})
                    if admitted:
                        archive_records[scored.topology_hash] = record
                        store.retain_diagnostics(record, result.get("packed_diagnostics"))
                        env = materialize(evaluator, record)
                        diagnostics = store.cached(record)["task_info"]
                        env.state.last_task_info = {**diagnostics, "reward": record["loss"],
                            "exploration_source": record["source"], "exploration_origin": "random"}
                        database.record_evaluation(env.state, source=record["source"], epoch=batch,
                            loss=record["loss"], valid=record["simulation_complete"], metadata=proposal)
                archive_records = {r.topology_hash: archive_records[r.topology_hash] for r in evolution.archive.ranked()}
                live = list(archive_records.values())
                metrics = {"Archive/best_loss": min(r["loss"] for r in live), "Archive/topologies": len(live),
                    "Archive/qualified_topologies": sum(r["qualified"] for r in live),
                    "Archive/qualified_regions": len({r["region"] for r in live if r["qualified"]}),
                    "Discovery/qualified_topologies": len(qualified_seen), "Discovery/qualified_regions": len(regions_seen),
                    "GA/clipped_rates": evolution.clipped_rates, "Policy/enabled": 0, "Policy/version": 0,
                    "Fit/failed_jobs_cumulative": 0, "Optimization/pending": 0,
                    "Timing/elapsed_seconds": time.monotonic() - started,
                    **{"Compute/" + k: v for k, v in store.counters().items()}}
                envs = []
                for record in live:
                    env = materialize(evaluator, record)
                    env.state.last_task_info = {**store.cached(record)["task_info"], "reward": record["loss"],
                        "exploration_source": record["source"], "exploration_origin": "random"}
                    envs.append(env)
                database.record_hof_snapshot(envs, epoch=batch, cumulative_total_evaluations=store.count("expensive_evaluations"),
                    save_plots=batch % config["plot_every"] == 0)
                database.flush()
                telemetry.metrics(batch, metrics)
                batch += 1
                status = {"status": "running", "epoch": batch, "metrics": metrics, "updated_at": time.time()}
                atomic_json(output / "status.json", status)
                atomic_json(output / "archive.json", live)
                print(json.dumps(status), flush=True)
                if batch % config["plot_every"] == 0:
                    telemetry.plot()
        telemetry.plot()
        result = {"status": "complete", "epochs": batch, "boundary_unit": "GA batch, not RL epoch",
            "elapsed_seconds": time.monotonic() - started, "metrics": metrics, "counters": store.counters(),
            "pending_optimization_jobs": 0, "termination": "actual_evaluation_budget"}
        atomic_json(output / "completed.json", result)
        atomic_json(output / "status.json", result)
        if logger:
            logger.log_other("status", "complete")
    except BaseException as error:
        atomic_json(output / "failed.json", {"type": type(error).__name__, "message": str(error)})
        atomic_json(output / "status.json", {"status": "failed", "message": str(error), "epoch": batch})
        if logger:
            logger.log_other("status", "failed")
        raise
    finally:
        if pool:
            pool.close()
        if database:
            database.close()
        if store:
            store.close()
        if logger:
            logger.end()


if __name__ == "__main__":
    main()
