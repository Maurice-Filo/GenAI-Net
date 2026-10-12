"""Isolated Logic-5 four-arm runner using the existing asynchronous AdaGrad streams."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

PROMPT = Path(__file__).resolve().parent / "prompts/logic5-evidence-exploitation-v2.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--exploration-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--julia-project", type=Path, required=True)
    parser.add_argument("--julia-recipe", type=Path, required=True)
    parser.add_argument("--dsh-home", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--search-workers", type=int, default=6)
    parser.add_argument("--optimization-workers", type=int, default=2)
    parser.add_argument("--llm-every", type=int, default=5)
    parser.add_argument("--llm-candidates", type=int, default=10)
    parser.add_argument("--llm-timeout", type=float, default=3600.)
    parser.add_argument("--comet-project")
    parser.add_argument("--communication-mode", choices=("full", "llm-to-rl", "rl-to-llm", "none"), required=True)
    args = parser.parse_args()
    if min(args.search_workers, args.optimization_workers, args.llm_every, args.llm_candidates) < 1:
        parser.error("Worker counts and request cadence/size must be positive")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JULIA_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/local0/tmp/mpl")
    if args.comet_project:
        from comet_ml import Experiment
    from RL4CRN.search_exploration.config import ExplorationConfig
    from RL4CRN.search_exploration.evaluation import build, EvaluationService
    from RL4CRN.search_exploration.engine import atomic_json
    from RL4CRN.search_exploration.session import build_session
    from RL4CRN.search_exploration.store import fingerprint
    from RL4CRN.search_streams.store import CompactStore
    from RL4CRN.search_streams.workers import StreamPool
    from RL4CRN.search_streams.julia_adapter import task_experiment
    from RL4CRN.search_streams.julia_stream import JuliaRecipe, JuliaFitPool, MeasuredSearchPool, LOSS_AGREEMENT
    from RL4CRN.search_streams.four_arm_harness import FourArmHarnessTrainer
    from RL4CRN.llm.workspace_loss_protocol import SYSTEM_PROMPT, graph_spec
    from comparisons.search_exploration.freeze_legacy import verify, ROOT, digest
    verify()
    cfg = ExplorationConfig(**json.loads(args.exploration_config.read_text()))
    if cfg.structural_refinement_steps or (cfg.rate_min, cfg.rate_max) != (.001, 100.):
        raise ValueError("Expected rates-only fitting and production bounds")
    reserve = cfg.epochs * cfg.batch_size
    if cfg.evaluation_budget <= reserve:
        raise ValueError("Need non-search budget for both AdaGrad and LLM evaluations")
    task_config = json.loads(args.task_config.read_text())
    prompt = args.prompt.read_text()
    recipe = JuliaRecipe(**json.loads(args.julia_recipe.read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    logger = trainer = store = search = fits = canonical = None
    try:
        evaluator, task, session_cfg = build("logic", task_config, cfg.rate_min, cfg.rate_max)
        if evaluator.max_added_reactions != 5 or evaluator.forbidden_reaction_ids != frozenset({0, 1, 2, 3, 4}):
            raise ValueError("Unexpected Logic-5 reaction budget/mask")
        library, experiment = task_experiment(evaluator, task)
        session_cfg.policy.deep_layer_size = 1024
        sources = {Path(__file__).resolve(), args.prompt.resolve(), ROOT / "RL4CRN/utils/input_interface.py"}
        for folder in ("RL4CRN/search_exploration", "RL4CRN/search_streams", "RL4CRN/parameter_optimization", "RL4CRN/llm"):
            sources.update(p for p in (ROOT / folder).iterdir() if p.suffix in (".py", ".jl"))
        sources.update([ROOT / "tools/dsh_bwrap_resolved_tmp.sh", ROOT / "tools/dsh_headless_continuing_runner.mjs"])
        manifest = {"schema": "logic5-four-arm-adagrad-v1", "task": "logic", "config": cfg.to_dict(),
            "task_config": task_config, "harness": True, "model": "deepseek-v4-flash", "provider": "deepseek-official",
            "communication_mode": args.communication_mode, "withhold_initial_hof": True, "max_in_flight": 1,
            "llm_every": args.llm_every, "llm_candidates": args.llm_candidates, "llm_timeout": args.llm_timeout,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "task_prompt_sha256": digest(args.prompt),
            "role_prompts": {n.name: n.prompt_template for n in graph_spec().nodes if n.name in ("Decider", "Writer")},
            "policy": {"deep_layer_size": 1024}, "search_workers": args.search_workers,
            "optimization_workers": args.optimization_workers, "queue_capacity": 32, "optimize_per_epoch": 32,
            "parameter_operator": "julia_adagrad", "julia_recipe": asdict(recipe),
            "loss_agreement": {**LOSS_AGREEMENT, "mismatch_action": recipe.loss_disagreement_policy},
            "search_evaluation_reserve": reserve, "budget_unit": "canonical or joint loss/gradient call",
            "llm_budget": "Shares the existing non-search budget with fitting; no extra evaluations",
            "llm_fitting": ("Competes under shared admission rule" if args.communication_mode in ("full", "llm-to-rl") else "Disabled; no LLM lineage enters search"),
            "terminal_union": "Separate report only; never mutates the training archive",
            "post_prune": False, "new_source": {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
            "harness_failure_policy": "Recorded failed request; no replacement; provider balance guarded by controller",
            "julia_project": str(args.julia_project),
            "restart": "No automatic retry or resume; preserve failures for diagnosis"}
        identity = fingerprint(manifest)
        atomic_json(output / "manifest.json", {**manifest, "identity": identity})
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in sorted(sources):
                archive.add(path, arcname=str(path.relative_to(ROOT)))
        if args.comet_project:
            logger = Experiment(project_name=args.comet_project, auto_param_logging=False, auto_metric_logging=False,
                log_code=False, log_git_patch=False, log_graph=False, log_git_metadata=False,
                log_env_details=False, log_env_host=False, log_env_cpu=False, log_env_gpu=False,
                display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
            logger.set_name(output.name)
            logger.add_tags(["logic-Logic-5", "julia-adagrad-stream", "flash-" + args.communication_mode, "withhold-initial-hof"])
            logger.log_parameters({**cfg.to_dict(), "model": manifest["model"], "llm_every": args.llm_every,
                "llm_candidates": args.llm_candidates, "withhold_initial_hof": True, "identity": identity,
                "communication_mode": args.communication_mode})
            atomic_json(output / "comet.json", {"experiment_key": logger.get_key(), "project": args.comet_project})
        session_cfg.train.seed, session_cfg.train.batch_size = cfg.seed, cfg.batch_size
        session_cfg.train.n_cpus = args.search_workers
        session_cfg.train.hall_of_fame_size = cfg.compatibility_hof_size
        session_cfg.train.forbidden_topology_m = 0
        session = build_session(session_cfg, task, device="cpu", logger=logger)
        store = CompactStore(output / "exploration.sqlite", cfg.evaluation_budget, identity, reserve,
                             wait_timeout=None if cfg.evaluation_seconds is None else 120)
        build_args = ("logic", task_config, cfg.rate_min, cfg.rate_max)
        search = MeasuredSearchPool(build_args, args.search_workers, cfg.evaluation_seconds, store, cfg.fit_seconds)
        canonical = StreamPool(build_args, args.optimization_workers, cfg.evaluation_seconds, store, cfg.fit_seconds)
        fits = JuliaFitPool(project=args.julia_project, workers=args.optimization_workers, library=library,
            experiment=experiment, store=store, canonical_pool=canonical, output=output,
            bounds=(cfg.rate_min, cfg.rate_max), timeout=cfg.fit_seconds, recipe=recipe, search_pool=search)
        allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
        service = EvaluationService(store, canonical, allowed, evaluator.max_added_reactions, (cfg.rate_min, cfg.rate_max))
        trainer = FourArmHarnessTrainer(session, evaluator, service, cfg, output, logger,
            search_pool=search, optimization_pool=fits, optimization_workers=args.optimization_workers,
            queue_capacity=32, optimize_per_epoch=32)
        trainer.telemetry.evaluation_label = "Evaluations (canonical + joint)"
        trainer.attach_harness(communication_mode=args.communication_mode, dsh_home=args.dsh_home, prompt=prompt, every=args.llm_every,
            candidates=args.llm_candidates, request_timeout=args.llm_timeout)
        while trainer.state.epoch < cfg.epochs:
            trainer.step_epoch()
            print(json.dumps({"epoch": trainer.state.epoch, **trainer.last_metrics}), flush=True)
        trainer.finish()
        result = {"status": "complete", "epochs": trainer.state.epoch,
            "elapsed_seconds": time.monotonic() - started, "metrics": trainer.last_metrics,
            "counters": store.counters(), "termination": "epochs_and_all_streams_drained"}
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
        for pool in (search, fits, canonical):
            if pool:
                pool.close()
        if store:
            store.close()
        if logger:
            logger.end()


if __name__ == "__main__":
    main()



