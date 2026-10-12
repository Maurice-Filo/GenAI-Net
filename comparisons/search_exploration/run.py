"""Separate opt-in exploration runner. Never modifies or restarts legacy campaigns."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import fcntl

from comparisons.search_exploration.freeze_legacy import DEFAULT, ROOT, digest, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="logic")
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--exploration-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-cohort", action="store_true", help="With --resume, preserve and explicitly retry failed fitting jobs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-size", type=int, help="Override only the policy intermediate dimension")
    parser.add_argument("--comet-project", help="Existing Comet configuration supplies credentials; never CLI keys")
    parser.add_argument("--harness", action="store_true", help="Explicitly enable paid/local model calls")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--llm-provider", default="deepseek-official")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--dsh-home", type=Path)
    parser.add_argument("--llm-timeout", type=float, default=900.)
    parser.add_argument("--llm-every", type=int, default=20)
    parser.add_argument("--llm-candidates", type=int, default=10)
    parser.add_argument("--communication-mode", choices=("full", "llm-to-rl", "rl-to-llm", "none"), default="full")
    parser.add_argument("--task-prompt-variant", default="standard")
    args = parser.parse_args()
    if args.embedding_size is not None and args.embedding_size < 1:
        parser.error("--embedding-size must be positive")
    started = time.monotonic()
    if args.retry_failed_cohort and not args.resume:
        parser.error("--retry-failed-cohort requires --resume")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("MPLCONFIGDIR", "/local0/tmp/mpl")
    logger = None
    # Import Comet before torch, and only when requested. Credentials are never read into artifacts.
    if args.comet_project:
        from comet_ml import Experiment
    from RL4CRN.search_exploration.config import ExplorationConfig
    from RL4CRN.search_exploration.store import Store, fingerprint
    from RL4CRN.search_exploration.evaluation import build, EvaluationService, EvaluatorPool
    from RL4CRN.search_exploration.engine import ExplorationTrainer, atomic_json
    from RL4CRN.search_exploration.session import build_session
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
    verify()
    cfg = ExplorationConfig(**json.loads(args.exploration_config.read_text()))
    task_config = json.loads((args.task_config or CONFIGS[args.task]).read_text())
    output = args.output.resolve()
    if args.resume:
        if not (output / "checkpoint.pt").exists():
            raise ValueError("No durable exploration checkpoint")
    else:
        output.mkdir(parents=True, exist_ok=False)
    run_lock = (output / "run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool = store = trainer = None
    try:
        if args.comet_project:
            logger = Experiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_git_patch=False, log_graph=False,
                log_git_metadata=False, log_env_details=False, log_env_host=False,
                log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
        evaluator, task, session_cfg = build(args.task, task_config, cfg.rate_min, cfg.rate_max)
        if args.embedding_size is not None:
            session_cfg.policy.deep_layer_size = args.embedding_size
        policy_config = {"width": session_cfg.policy.width, "depth": session_cfg.policy.depth,
                         "deep_layer_size": session_cfg.policy.deep_layer_size}
        source_paths = sorted((ROOT / "RL4CRN/search_exploration").glob("*.py")) + [Path(__file__)]
        versions = {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch", "scikit-sundae")}
        scientific = {"task": args.task, "task_config": task_config,
            "legacy_source": digest(DEFAULT / "scientific_source.tar.gz"),
            "new_source": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
            "versions": versions, "bounds": [cfg.rate_min, cfg.rate_max], "policy": policy_config}
        identity = fingerprint(scientific)
        manifest = {"schema": "exploration-v1", **scientific, "identity": identity, "config": cfg.to_dict(),
            "harness": args.harness, "communication_mode": args.communication_mode,
            "withhold_initial_hof": True, "model": args.model if args.harness else None,
            "descriptor_caveat": "Pilot role counts; coverage does not establish mechanistic diversity"}
        if args.resume:
            old = json.loads((output / "manifest.json").read_text())
            if old != manifest:
                raise ValueError("Resume manifest changed")
        else:
            atomic_json(output / "manifest.json", manifest)
        if logger:
            logger.set_name(output.name)
            logger.add_tags(["exploration-v1", args.task, "pilot"])
            logger.log_parameters({**cfg.to_dict(), "task": args.task, "identity": identity,
                                   **{"NN/" + k: v for k, v in policy_config.items()}})
            atomic_json(output / "comet.json", {"experiment_key": logger.get_key(), "project": args.comet_project})
        session_cfg.train.seed = cfg.seed
        session_cfg.train.batch_size = cfg.batch_size
        session_cfg.train.n_cpus = cfg.workers
        session_cfg.train.hall_of_fame_size = cfg.compatibility_hof_size
        session_cfg.train.forbidden_topology_m = 0
        session = build_session(session_cfg, task, device=args.device, logger=logger)
        pool = EvaluatorPool((args.task, task_config, cfg.rate_min, cfg.rate_max), cfg.workers, cfg.evaluation_seconds)
        store = Store(output / "exploration.sqlite", cfg.evaluation_budget, identity)
        allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
        service = EvaluationService(store, pool, allowed, evaluator.max_added_reactions, (cfg.rate_min, cfg.rate_max))
        trainer = ExplorationTrainer(session, evaluator, service, cfg, output, logger)
        if args.resume:
            trainer.load(output / "checkpoint.pt")
            if args.retry_failed_cohort:
                trainer.retry_failed_cohort()
        if args.harness:
            if args.dsh_home is None:
                raise ValueError("--harness requires explicit --dsh-home; no keys in CLI")
            from RL4CRN.llm import HarnessLLMClient, HarnessDeciderWriterCRNGraph, get_mmc2_task_prompt_variant
            from RL4CRN.llm.workspace_loss_protocol import protocol_for_variant
            from RL4CRN.search_exploration.harness import CachedHarnessEvaluator, ExplorationHarnessClient
            client = ExplorationHarnessClient(workspace_root=output / "harness-workspaces", dsh_home=args.dsh_home,
                model=args.model, provider=args.llm_provider, openai_compatible_base_url=args.llm_base_url,
                timeout_seconds=args.llm_timeout, candidate_validation_policy="independent-members",
                harness_protocol=protocol_for_variant(args.task_prompt_variant))
            graph = HarnessDeciderWriterCRNGraph(client=client, evaluator=CachedHarnessEvaluator(evaluator, service),
                writer_retry_limit=0, max_workspace_evaluations=0, max_literature_searches=0)
            trainer.configure_llm_graph(graph, every=args.llm_every,
                task_description=get_mmc2_task_prompt_variant(args.task, variant=args.task_prompt_variant, solver="CVODE"),
                num_candidates=args.llm_candidates, withhold_initial_hof=True,
                add_to_hall_of_fame=args.communication_mode in ("full", "llm-to-rl"),
                cross_communication=args.communication_mode in ("full", "rl-to-llm"),
                jsonl_path=output / "llm_evaluations.jsonl", max_in_flight=1)
        while trainer.state.epoch < cfg.epochs:
            if store.count("expensive_evaluations") >= cfg.evaluation_budget and trainer.pending is None:
                break
            trainer.step_epoch()
            print(json.dumps({"epoch": trainer.state.epoch, **trainer.last_metrics}), flush=True)
        if trainer._llm_loop:
            # Only end-of-run draining waits for model responses; RL never waits at epoch boundaries.
            trainer.wait_for_llm_graph(timeout=args.llm_timeout + 30)
            if trainer._llm_loop["jobs"]:
                raise TimeoutError("Pending Harness round remains; reconciliation required")
            trainer.merge_isolated_llm_candidates()
        trainer.persist_hof_boundary(epoch=trainer.state.epoch, boundary="terminal_boundary",
            rl_evaluations=store.count("evaluations/fitting"), llm_tool_evaluations=0,
            llm_final_evaluations=store.count("evaluations/llm"), total_evaluations=store.count("expensive_evaluations"))
        trainer.save(output / "checkpoint.pt")
        trainer.telemetry.plot()
        summary = {"status": "complete", "epochs": trainer.state.epoch, "counters": store.counters(),
            "elapsed_seconds": time.monotonic() - started,
            "metrics": trainer.last_metrics, "termination": "epochs" if trainer.state.epoch == cfg.epochs else "evaluation_budget",
            "terminal_pool_best_loss": min((float(e.state.last_task_info["reward"]) for e in trainer.s.mult_env.hall_of_fame), default=None),
            "terminal_pool_size": len(trainer.s.mult_env.hall_of_fame),
            "training_archive_size": len(trainer.discovery.entries)}
        atomic_json(output / "completed.json", summary)
        atomic_json(output / "status.json", summary)
        if logger:
            logger.log_other("status", "complete")
    except BaseException as error:
        atomic_json(output / "failed.json", {"type": type(error).__name__, "message": str(error)})
        atomic_json(output / "status.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        if logger:
            logger.log_other("status", "failed")
        raise
    finally:
        if trainer:
            trainer.clear_llm_graph(wait=False)
            trainer.close_results_database()
        if pool:
            pool.close()
        if store:
            store.close()
        if logger:
            logger.end()


if __name__ == "__main__":
    main()
