"""Separate asynchronous RPA/Logic runner. No LLM calls or legacy source edits."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rpa", "logic"), default="rpa")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--exploration-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-size", type=int, default=1024)
    parser.add_argument("--search-workers", type=int, default=32)
    parser.add_argument("--optimization-workers", type=int)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--optimize-per-epoch", type=int)
    parser.add_argument("--julia-project", type=Path, help="Opt in to persistent Julia AdaGrad fitting")
    parser.add_argument("--julia-recipe", type=Path, help="Optional JuliaRecipe JSON overrides")
    parser.add_argument("--post-prune", action="store_true", help="Separate bounded deletion trials after search")
    parser.add_argument("--comet-project")
    args = parser.parse_args()
    if (args.julia_recipe or args.post_prune) and not args.julia_project:
        parser.error("Julia recipe/pruning requires --julia-project")
    args.optimization_workers = args.optimization_workers if args.optimization_workers is not None else (8 if args.julia_project else 16)
    args.queue_capacity = args.queue_capacity if args.queue_capacity is not None else (32 if args.julia_project else 512)
    args.optimize_per_epoch = args.optimize_per_epoch if args.optimize_per_epoch is not None else (32 if args.julia_project else 256)
    if min(args.embedding_size, args.search_workers, args.optimization_workers, args.optimize_per_epoch) < 1:
        parser.error("Sizes and worker counts must be positive")
    if args.queue_capacity < args.optimization_workers:
        parser.error("Queue must hold at least all active optimization jobs")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLCONFIGDIR", "/local0/tmp/mpl")
    if args.comet_project:
        from comet_ml import Experiment
    import importlib.metadata
    from RL4CRN.search_exploration.config import ExplorationConfig
    from RL4CRN.search_exploration.evaluation import build, EvaluationService
    from RL4CRN.search_exploration.engine import atomic_json
    from RL4CRN.search_exploration.session import build_session
    from RL4CRN.search_exploration.store import fingerprint
    from RL4CRN.search_streams.store import CompactStore
    from RL4CRN.search_streams.workers import StreamPool
    from RL4CRN.search_streams.engine import StreamTrainer
    from comparisons.search_exploration.freeze_legacy import verify, ROOT, DEFAULT, digest
    verify()
    cfg = ExplorationConfig(**json.loads(args.exploration_config.read_text()))
    if cfg.structural_refinement_steps:
        raise ValueError("The independent optimizer fits rates only")
    search_reserve = cfg.epochs * cfg.batch_size
    if cfg.evaluation_budget < search_reserve:
        raise ValueError("Evaluation budget must cover all direct search evaluations")
    task_config = json.loads(args.task_config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_lock = (output / "run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    logger = trainer = store = search_pool = optimization_pool = canonical_pool = None
    try:
        evaluator, task, session_cfg = build(args.task, task_config, cfg.rate_min, cfg.rate_max)
        julia_recipe = None
        if args.julia_project:
            from dataclasses import asdict
            from RL4CRN.search_streams.julia_adapter import task_experiment
            from RL4CRN.search_streams.julia_stream import JuliaRecipe, JuliaFitPool, JuliaStreamTrainer, MeasuredSearchPool, LOSS_AGREEMENT
            julia_recipe = JuliaRecipe(**(json.loads(args.julia_recipe.read_text()) if args.julia_recipe else {}))
            julia_library, julia_experiment = task_experiment(evaluator, task)
        session_cfg.policy.deep_layer_size = args.embedding_size
        sources = [*sorted((ROOT / "RL4CRN/search_exploration").glob("*.py")),
                   *sorted((ROOT / "RL4CRN/search_streams").glob("*.py")), Path(__file__).resolve()]
        if args.julia_project:
            sources += sorted((ROOT / "RL4CRN/parameter_optimization").glob("*.py"))
            sources += sorted((ROOT / "RL4CRN/parameter_optimization").glob("*.jl"))
        science = {"schema": "search-streams-v2", "task": args.task, "task_config": task_config,
            "archive_policy": "region-protected-positive-utility-v2",
            "policy_training_enabled": cfg.rl_fraction > 0,
            "proposal_arm": ("mutation-random-no-rl" if cfg.rl_fraction == 0 else
                "rl-random-no-mutation" if cfg.mutation_fraction == 0 else "rl-mutation-random"),
            "config": cfg.to_dict(), "harness": False, "model": None,
            "policy": {"width": session_cfg.policy.width, "depth": session_cfg.policy.depth,
                       "deep_layer_size": args.embedding_size},
            "search_workers": args.search_workers, "optimization_workers": args.optimization_workers,
            "queue_capacity": args.queue_capacity, "optimize_per_epoch": args.optimize_per_epoch,
            "search_evaluation_reserve": search_reserve,
            "new_source": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "legacy_source": digest(DEFAULT / "scientific_source.tar.gz"),
            "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "scikit-sundae")},
            "credit": "Immediate search rewards only; delayed fits enter archive and imitation",
            "diagnostics": "Scalar cache for all evaluations; rich diagnostics for archive admissions",
            "restart": "Explicit reconciliation required; no automatic resume/retry"}
        if julia_recipe:
            science.update(schema="search-streams-julia-v1", parameter_operator="julia_adagrad",
                julia_loss_agreement={**LOSS_AGREEMENT, "mismatch_action": julia_recipe.loss_disagreement_policy},
                julia_recipe=asdict(julia_recipe), post_prune=args.post_prune,
                julia_project=str(args.julia_project.resolve()),
                budget_unit="One canonical evaluation OR one joint loss/gradient evaluation over all scenarios",
                compute_share_unit="Sum of search evaluator seconds vs fitting worker elapsed seconds; startup excluded",
                extension_order="First eligible asynchronous completions, at most eight per submission epoch",
                config_optimizer_note="config.optimizer applies to old backend only; overridden by parameter_operator")
        identity = fingerprint(science)
        atomic_json(output / "manifest.json", {**science, "identity": identity})
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in sources:
                archive.add(path, arcname=str(path.relative_to(ROOT)))
            archive.add(DEFAULT / "scientific_source.tar.gz", arcname="legacy_source.tar.gz")
        if args.comet_project:
            logger = Experiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_git_patch=False, log_graph=False,
                log_git_metadata=False, log_env_details=False, log_env_host=False,
                log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
            logger.set_name(output.name)
            logger.add_tags(["search-streams-v2", args.task, "no-llm"])
            if cfg.rl_fraction == 0:
                logger.set_name(output.parent.name + "-no-rl")
                logger.add_tags(["no-rl", "random-replacement-ablation"])
            elif cfg.mutation_fraction == 0:
                logger.set_name(output.parent.name + "-no-mutation")
                logger.add_tags(["no-mutation", "random-replacement-ablation"])
            logger.log_parameters({**cfg.to_dict(), "task": args.task, "identity": identity,
                "NN/intermediate": args.embedding_size, "search_workers": args.search_workers,
                "optimization_workers": args.optimization_workers, "queue_capacity": args.queue_capacity,
                "optimize_per_epoch": args.optimize_per_epoch, "harness": False})
            if julia_recipe:
                logger.add_tag("julia-adagrad-stream")
                logger.log_parameters({"parameter_operator": "julia_adagrad", **{"Julia/" + k: v for k, v in asdict(julia_recipe).items()}})
            atomic_json(output / "comet.json", {"experiment_key": logger.get_key(), "project": args.comet_project})
        session_cfg.train.seed = cfg.seed
        session_cfg.train.batch_size = cfg.batch_size
        session_cfg.train.n_cpus = args.search_workers
        session_cfg.train.hall_of_fame_size = cfg.compatibility_hof_size
        session_cfg.train.forbidden_topology_m = 0
        session = build_session(session_cfg, task, device="cpu", logger=logger)
        store = CompactStore(output / "exploration.sqlite", cfg.evaluation_budget, identity, search_reserve,
            wait_timeout=None if cfg.evaluation_seconds is None else 120)
        build_args = (args.task, task_config, cfg.rate_min, cfg.rate_max)
        search_class = MeasuredSearchPool if julia_recipe else StreamPool
        search_pool = search_class(build_args, args.search_workers, cfg.evaluation_seconds, store, cfg.fit_seconds)
        if julia_recipe:
            canonical_pool = StreamPool(build_args, args.optimization_workers, cfg.evaluation_seconds, store, cfg.fit_seconds)
            optimization_pool = JuliaFitPool(project=args.julia_project, workers=args.optimization_workers,
                library=julia_library, experiment=julia_experiment, store=store, canonical_pool=canonical_pool,
                output=output, bounds=(cfg.rate_min, cfg.rate_max), timeout=cfg.fit_seconds,
                recipe=julia_recipe, search_pool=search_pool)
        else:
            optimization_pool = StreamPool(build_args, args.optimization_workers, cfg.evaluation_seconds, store, cfg.fit_seconds)
        allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
        service = EvaluationService(store, search_pool, allowed, evaluator.max_added_reactions, (cfg.rate_min, cfg.rate_max))
        trainer_class = JuliaStreamTrainer if julia_recipe else StreamTrainer
        trainer = trainer_class(session, evaluator, service, cfg, output, logger,
            search_pool=search_pool, optimization_pool=optimization_pool,
            optimization_workers=args.optimization_workers, queue_capacity=args.queue_capacity,
            optimize_per_epoch=args.optimize_per_epoch)
        if julia_recipe:
            trainer.telemetry.evaluation_label = "Evaluations (canonical + joint)"
        while trainer.state.epoch < cfg.epochs:
            trainer.step_epoch()
            print(json.dumps({"epoch": trainer.state.epoch, **trainer.last_metrics}), flush=True)
        trainer.finish()
        if args.post_prune:
            from RL4CRN.search_streams.pruning import run_post_pruning
            atomic_json(output / "status.json", {"status": "post_pruning", "epochs": trainer.state.epoch})
            pruned = run_post_pruning(trainer.discovery.ranked(), project=args.julia_project,
                evaluator=evaluator, task=task, build_args=build_args, output=output, config=cfg)
            if logger:
                logger.log_metrics({"Pruning/evaluations": pruned["evaluations"],
                    "Pruning/accepted": sum(p["derived"] is not None for p in pruned["parents"])})
        result = {"status": "complete", "epochs": trainer.state.epoch,
            "elapsed_seconds": time.monotonic() - started, "metrics": trainer.last_metrics,
            "counters": store.counters(), "pending_optimization_jobs": len(trainer.optimization_jobs),
            "termination": "epochs_and_optimizer_drained"}
        atomic_json(output / "completed.json", result)
        atomic_json(output / "status.json", result)
        if logger:
            logger.log_other("status", "complete")
    except BaseException as error:
        atomic_json(output / "failed.json", {"type": type(error).__name__, "message": str(error)})
        atomic_json(output / "status.json", {"status": "failed", "message": str(error)})
        if logger:
            logger.log_other("status", "failed")
        raise
    finally:
        if trainer:
            trainer.close_streams()
            trainer.close_results_database()
        for pool in (search_pool, optimization_pool, canonical_pool):
            if pool:
                pool.close()
        if store:
            store.close()
        if logger:
            logger.end()


if __name__ == "__main__":
    main()
