"""Run MMC2 RL4CRN with asynchronous DeepSeek Harness proposal batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comet_ml import Experiment

from RL4CRN.llm import (
    HarnessDeciderWriterCRNGraph,
    HarnessLLMClient,
    LLMCandidateEvaluator,
    get_mmc2_task_prompt_variant,
)
from RL4CRN.llm.deepseek_direct import DirectDeepSeekClient, DirectDeepSeekCRNGenerator
from RL4CRN.utils.input_interface import make_session_and_trainer
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import BUILDERS, CONFIGS
from comparisons.llm_crn_generation.experiment_release import (
    DEFAULT_ANALYSIS_PLAN,
    DEFAULT_SENTINEL_REPORT,
    validate_experiment_release,
)
from comparisons.llm_crn_generation.prompt_approval import DEFAULT_PROMPT_APPROVAL
from comparisons.rpa_search.src.common.evaluator import candidate_summary
from comparisons.rpa_search.src.common.io import PROGRESS_FIELDS, append_csv, write_json


DEFAULT_PROJECT = "mmc2-v4-flash-hybrid"
DEFAULT_MODEL = "deepseek-v4-flash"
LLM_RATE_MIN = 0.001
LLM_RATE_MAX = 100.0


def trial_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-mmc2-v4-hybrid-cvode"


def build_hybrid_run(
    *,
    task_name: str,
    seed: int,
    n_cpus: int,
    rl_batch_size: int | None,
    logger: Experiment,
):
    config = json.loads(CONFIGS[task_name].read_text(encoding="utf-8"))
    config[task_name]["solver"] = "CVODE"
    config["search"]["seed"] = seed
    config["rl4crn"]["n_cpus"] = n_cpus
    if rl_batch_size is not None:
        config["rl4crn"]["batch_size"] = int(rl_batch_size)
    crn, library_components, task, cfg = BUILDERS[task_name](config)
    search = config["search"]
    rl = config["rl4crn"]
    cfg.train.seed = seed
    cfg.train.max_added_reactions = int(search["max_added_reactions"])
    cfg.train.epochs = int(rl["epochs"])
    cfg.train.batch_size = int(rl["batch_size"])
    cfg.train.n_cpus = n_cpus
    cfg.train.render_every = 0
    cfg.train.hall_of_fame_size = int(rl["hall_of_fame_size"])
    cfg.agent.risk_scheduler["risk"] = min(
        float(cfg.agent.risk_scheduler.get("risk", 0.95)),
        max(0.0, 1.0 - 1.0 / max(1, cfg.train.batch_size)),
    )
    cfg.policy.width = int(rl["policy_width"])
    cfg.policy.depth = int(rl["policy_depth"])
    cfg.policy.deep_layer_size = int(rl["deep_layer_size"])
    trainer = make_session_and_trainer(cfg, task, logger=logger)

    library = library_components[0]
    evaluator = LLMCandidateEvaluator.from_session(
        trainer.s,
        min_parameter_value=LLM_RATE_MIN,
        max_parameter_value=LLM_RATE_MAX,
        enforce_parameter_bounds=True,
        forbidden_reaction_ids=[library.find_zero_reaction()],
    )
    if str(getattr(crn, "solver", "")).upper() != "CVODE":
        raise RuntimeError(f"{task_name} hybrid evaluator is not using CVODE.")
    return config, task, trainer, evaluator


def _llm_candidate_evaluations(trainer) -> int:
    return sum(
        int(row.get("total_llm_candidate_evaluations", 0))
        for row in trainer.llm_graph_history()
        if not row.get("error")
    )


def _best_hof_loss(trainer, fallback: float) -> float:
    best_crn = trainer.best_crn()
    if best_crn is None:
        return float(fallback)
    return min(float(fallback), float(best_crn.last_task_info["reward"]))


def _append_progress(paths: list[Path], row: dict) -> None:
    for path in paths:
        append_csv(path, row, PROGRESS_FIELDS)


def run_one(
    *,
    task_name: str,
    seed: int,
    args: argparse.Namespace,
    trial_root: Path,
) -> None:
    run_id = f"{task_name}_full{args.total_candidate_budget}_seed{seed}_{args.run_suffix}"
    run_root = trial_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    comparison_root = args.comparison_output_root.expanduser().resolve()
    comparison_dir = comparison_root / args.method_name / run_id
    comparison_dir.mkdir(parents=True, exist_ok=True)
    completion_marker = comparison_dir / "completed.json"
    if completion_marker.exists():
        print(f"[skip] {run_id} already complete", flush=True)
        return
    experiment = Experiment(
        project_name=args.comet_project,
        auto_param_logging=False,
        auto_metric_logging=False,
        log_code=False,
        log_graph=False,
        log_git_patch=False,
        display_summary_level=0,
    )
    experiment.set_name(f"{args.trial_id}-{run_id}")
    if not experiment.get_key():
        raise RuntimeError("Comet did not return an experiment key; paper runs require Comet logging.")
    experiment.add_tags(
        [
            "genai-net-paper", "hybrid", "async-llm", "cvode", task_name,
            args.model, f"communication-{args.communication_mode}", args.generation_backend,
        ]
    )
    config, task, trainer, evaluator = build_hybrid_run(
        task_name=task_name,
        seed=seed,
        n_cpus=args.n_cpus,
        rl_batch_size=args.rl_batch_size,
        logger=experiment,
    )
    proposal_space = {
        "contract_version": 2,
        "template_reaction_ids": sorted(evaluator.initial_reaction_ids),
        "null_reaction_id": int(evaluator.library.find_zero_reaction()),
        "forbidden_reaction_ids": sorted(evaluator.forbidden_reaction_ids),
        "rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
        "rate_out_of_range_policy": "clamp-to-endpoint",
        "candidate_validation_policy": args.candidate_validation_policy,
    }
    prompt = get_mmc2_task_prompt_variant(
        task_name,
        variant=args.task_prompt_variant,
        solver="CVODE",
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    config["llm_contract_v2"] = {
        "proposal_space_contract_version": 2,
        "rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
        "out_of_range_rate_policy": "clamp-to-endpoint",
        "candidate_validation_policy": args.candidate_validation_policy,
        "workflow": (
            "two-stage-decider-writer"
            if args.generation_backend == "harness"
            else "legacy-direct-single-request"
        ),
        "provider_calls_per_round": 2 if args.generation_backend == "harness" else 1,
    }
    if args.generation_backend == "direct-nonthinking":
        if args.literature_rag_index is not None or args.max_agent_evaluations != 0:
            raise ValueError("Direct non-thinking generation does not support tools or RAG.")
        client = DirectDeepSeekClient(
            workspace_root=run_root / "direct-requests",
            dsh_home=args.dsh_home.expanduser().resolve(),
            model=args.model,
            timeout_seconds=args.llm_timeout,
            global_concurrency=args.global_llm_concurrency,
        )
        graph = DirectDeepSeekCRNGenerator(client=client, evaluator=evaluator)
    else:
        client = HarnessLLMClient(
            workspace_root=run_root / "harness-workspaces",
            dsh_home=args.dsh_home.expanduser().resolve(),
            model=args.model,
            provider=args.llm_provider,
            openai_compatible_base_url=args.llm_base_url,
            timeout_seconds=args.llm_timeout,
            global_concurrency=args.global_llm_concurrency,
            candidate_validation_policy=args.candidate_validation_policy,
        )
        graph = HarnessDeciderWriterCRNGraph(
            client=client,
            evaluator=evaluator,
            writer_retry_limit=0,
            max_workspace_evaluations=args.max_agent_evaluations,
            literature_database=args.literature_rag_index,
            max_literature_searches=args.max_literature_searches,
        )
    trainer.s.cfg.train.forbidden_topology_m = args.forbidden_topology_m
    trainer.s.cfg.train.forbidden_topology_every = args.forbidden_topology_every
    trainer.s.cfg.train.forbidden_async = args.forbidden_topology_m > 0
    trainer.s.cfg.train.forbidden_optimization_max_evaluations = (
        args.forbidden_optimization_max_evaluations
    )
    trainer.s.cfg.train.forbidden_optimization_timeout_seconds = (
        args.forbidden_optimization_timeout
    )
    trainer.configure_llm_graph(
        graph,
        every=args.llm_every,
        task_description=prompt,
        num_candidates=args.llm_candidates,
        start_epoch=0,
        add_to_hall_of_fame=args.communication_mode == "full",
        cross_communication=args.communication_mode == "full",
        withhold_initial_hof=args.withhold_initial_hof,
        jsonl_path=run_root / "llm_evaluations.jsonl",
        max_in_flight=args.max_llm_in_flight,
    )
    trainer.configure_results_database(
        run_root / "results.sqlite",
        every=1,
        run_id=run_id,
        metadata={
            "task": task_name,
            "seed": seed,
            "method": args.method_name,
            "model": args.model,
            "generation_backend": args.generation_backend,
            "thinking_mode": "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default",
            "direct_hof_context_format": (
                "rank-loss-actions-crn-v1"
                if args.generation_backend == "direct-nonthinking"
                else None
            ),
            "communication_mode": args.communication_mode,
            "task_prompt_variant": args.task_prompt_variant,
            "task_prompt_sha256": prompt_sha256,
            "withhold_initial_hof": args.withhold_initial_hof,
            "candidate_validation_policy": args.candidate_validation_policy,
            "proposal_space_contract_version": 2,
            "llm_rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
            "terminal_candidate_pooling": args.communication_mode == "none",
            "proposal_space": proposal_space,
        },
    )
    experiment.log_parameters(
        {
            "task": task_name,
            "seed": seed,
            "solver": "CVODE",
            "n_cpus": args.n_cpus,
            "rl_epochs": args.epochs,
            "rl_batch_size": trainer.s.cfg.train.batch_size,
            "rl_candidate_budget": args.epochs * trainer.s.cfg.train.batch_size,
            "total_candidate_budget_cap": args.total_candidate_budget,
            "llm_candidates_per_round": args.llm_candidates,
            "llm_model_requests_per_round": 2 if args.generation_backend == "harness" else 1,
            "llm_protocol": (
                "two-stage-decider-writer/independent-members"
                if args.generation_backend == "harness"
                else "direct-single-request/multi-candidate"
            ),
            "writer_retry_limit": 0 if args.generation_backend == "harness" else None,
            "llm_every": args.llm_every,
            "llm_max_in_flight": args.max_llm_in_flight,
            "llm_timeout_seconds": args.llm_timeout,
            "llm_global_concurrency": args.global_llm_concurrency,
            "max_agent_evaluations_per_round": args.max_agent_evaluations,
            "literature_rag_index": str(args.literature_rag_index) if args.literature_rag_index else None,
            "max_literature_searches_per_round": args.max_literature_searches,
            "forbidden_topology_m": args.forbidden_topology_m,
            "forbidden_topology_every": args.forbidden_topology_every,
            "forbidden_optimization_max_evaluations": args.forbidden_optimization_max_evaluations,
            "forbidden_optimization_timeout_seconds": args.forbidden_optimization_timeout,
            "llm_execution": "background-thread/headless-process",
            "communication_mode": args.communication_mode,
            "task_prompt_variant": args.task_prompt_variant,
            "task_prompt_sha256": prompt_sha256,
            "withhold_initial_hof": args.withhold_initial_hof,
            "candidate_validation_policy": args.candidate_validation_policy,
            "proposal_space_contract_version": 2,
            "llm_rate_min": LLM_RATE_MIN,
            "llm_rate_max": LLM_RATE_MAX,
            "template_reaction_ids": json.dumps(proposal_space["template_reaction_ids"]),
            "null_reaction_id": proposal_space["null_reaction_id"],
            "forbidden_reaction_ids": json.dumps(proposal_space["forbidden_reaction_ids"]),
            "terminal_candidate_pooling": args.communication_mode == "none",
            "model": args.model,
            "generation_backend": args.generation_backend,
            "thinking_mode": "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default",
            "llm_provider": args.llm_provider,
            "llm_base_url": args.llm_base_url,
            "method_name": args.method_name,
            "run_suffix": args.run_suffix,
        }
    )
    write_json(run_root / "config.json", config)
    write_json(comparison_dir / "config.json", config)
    write_json(
        run_root / "run_manifest.json",
        {
            "trial_id": args.trial_id,
            "run_id": run_id,
            "task": task_name,
            "seed": seed,
            "solver": "CVODE",
            "n_cpus": args.n_cpus,
            "model": args.model,
            "generation_backend": args.generation_backend,
            "thinking_mode": "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default",
            "llm_provider": args.llm_provider,
            "llm_base_url": args.llm_base_url,
            "method_name": args.method_name,
            "run_suffix": args.run_suffix,
            "llm_candidates_per_round": args.llm_candidates,
            "llm_model_requests_per_round": 2 if args.generation_backend == "harness" else 1,
            "llm_protocol": (
                "two-stage-decider-writer/independent-members"
                if args.generation_backend == "harness"
                else "direct-single-request/multi-candidate"
            ),
            "writer_retry_limit": 0 if args.generation_backend == "harness" else None,
            "llm_every": args.llm_every,
            "llm_max_in_flight": args.max_llm_in_flight,
            "communication_mode": args.communication_mode,
            "task_prompt_variant": args.task_prompt_variant,
            "task_prompt_sha256": prompt_sha256,
            "withhold_initial_hof": args.withhold_initial_hof,
            "candidate_validation_policy": args.candidate_validation_policy,
            "proposal_space_contract_version": 2,
            "llm_rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
            "proposal_space": proposal_space,
            "terminal_candidate_pooling": args.communication_mode == "none",
            "llm_timeout_seconds": args.llm_timeout,
            "max_agent_evaluations_per_round": args.max_agent_evaluations,
            "literature_rag_index": str(args.literature_rag_index) if args.literature_rag_index else None,
            "max_literature_searches_per_round": args.max_literature_searches,
            "forbidden_topology_m": args.forbidden_topology_m,
            "forbidden_topology_every": args.forbidden_topology_every,
            "forbidden_optimization_max_evaluations": args.forbidden_optimization_max_evaluations,
            "forbidden_optimization_timeout_seconds": args.forbidden_optimization_timeout,
            "total_candidate_budget_cap": args.total_candidate_budget,
            "comet_experiment_key": experiment.get_key(),
        },
    )
    experiment.log_asset(str(run_root / "run_manifest.json"))

    best_loss = float("inf")
    started = time.perf_counter()
    progress_paths = [run_root / "progress.csv", comparison_dir / "progress.csv"]
    try:
        for step in range(args.epochs):
            best, _median, rewards = trainer.step_epoch()
            best_loss = _best_hof_loss(trainer, min(best_loss, float(best)))
            rl_evaluations = (step + 1) * len(rewards)
            llm_evaluations = _llm_candidate_evaluations(trainer)
            candidate_evaluations = (
                rl_evaluations
                + llm_evaluations
                + trainer.forbidden_optimization_evaluations()
            )
            if candidate_evaluations > args.total_candidate_budget:
                raise RuntimeError(
                    f"Combined evaluation budget exceeded: {candidate_evaluations} > "
                    f"{args.total_candidate_budget}."
                )
            progress = {
                "method": args.method_name,
                "run_id": run_id,
                "step": step + 1,
                "candidate_evaluations": candidate_evaluations,
                "ode_simulations": candidate_evaluations,
                "scenario_count": len(task.u_list),
                "scenario_evaluations": candidate_evaluations * len(task.u_list),
                "loss": float(best),
                "best_so_far_loss": best_loss,
                "performance": -float(best),
                "best_so_far_performance": -best_loss,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _append_progress(progress_paths, progress)
            experiment.log_metric("benchmark/best_so_far_loss", best_loss, step=step)
            experiment.log_metric("benchmark/rl_candidate_evaluations", rl_evaluations, step=step)
            experiment.log_metric("benchmark/llm_candidate_evaluations", llm_evaluations, step=step)
            experiment.log_metric("benchmark/total_candidate_evaluations", candidate_evaluations, step=step)
            print(
                f"[{run_id}] epoch={step} rl={rl_evaluations} llm={llm_evaluations} "
                f"total={candidate_evaluations} "
                f"best={best_loss:.6g}",
                flush=True,
            )

        trainer.wait_for_llm_graph(timeout=args.llm_timeout)
        trainer.wait_for_forbidden_topologies()
        terminal_pooled_candidates = (
            trainer.merge_isolated_llm_candidates()
            if args.communication_mode == "none"
            else 0
        )
        llm_evaluations = _llm_candidate_evaluations(trainer)
        rl_evaluations = args.epochs * trainer.s.cfg.train.batch_size
        optimization_evaluations = trainer.forbidden_optimization_evaluations()
        candidate_evaluations = rl_evaluations + llm_evaluations + optimization_evaluations
        if candidate_evaluations > args.total_candidate_budget:
            raise RuntimeError(
                f"Combined evaluation budget exceeded: {candidate_evaluations} > "
                f"{args.total_candidate_budget}."
            )
        best_loss = _best_hof_loss(trainer, best_loss)
        final_progress = {
            "method": args.method_name,
            "model": args.model,
            "run_id": run_id,
            "step": args.epochs + 1,
            "candidate_evaluations": candidate_evaluations,
            "ode_simulations": candidate_evaluations,
            "scenario_count": len(task.u_list),
            "scenario_evaluations": candidate_evaluations * len(task.u_list),
            "loss": best_loss,
            "best_so_far_loss": best_loss,
            "performance": -best_loss,
            "best_so_far_performance": -best_loss,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _append_progress(progress_paths, final_progress)
        trainer._maybe_persist_hof(args.epochs, force=True)
        trainer.flush_results_database()
        best_crn = trainer.best_crn()
        if best_crn is None:
            raise RuntimeError("Hybrid run completed without a Hall-of-Fame solution.")
        for output_dir in (run_root, comparison_dir):
            (output_dir / "best_network.txt").write_text(str(best_crn), encoding="utf-8")
            write_json(output_dir / "best_network.json", candidate_summary(best_crn))
        completed = {
            "run_id": run_id,
            "task": task_name,
            "seed": seed,
            "method": args.method_name,
            "model": args.model,
            "generation_backend": args.generation_backend,
            "thinking_mode": "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default",
            "best_loss": best_loss,
            "rl_candidate_evaluations": rl_evaluations,
            "llm_candidate_evaluations": llm_evaluations,
            "forbidden_optimization_evaluations": optimization_evaluations,
            "candidate_evaluations": candidate_evaluations,
            "budget_cap": args.total_candidate_budget,
            "communication_mode": args.communication_mode,
            "task_prompt_variant": args.task_prompt_variant,
            "task_prompt_sha256": prompt_sha256,
            "withhold_initial_hof": args.withhold_initial_hof,
            "candidate_validation_policy": args.candidate_validation_policy,
            "proposal_space_contract_version": 2,
            "llm_rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
            "proposal_space": proposal_space,
            "terminal_pooled_candidates": terminal_pooled_candidates,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_root / "completed.json", completed)
        write_json(completion_marker, completed)
        experiment.log_other("status", "completed")
    except Exception:
        experiment.log_other("status", "failed")
        raise
    finally:
        trainer.clear_llm_graph(wait=True)
        trainer.close_results_database()
        experiment.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", *CONFIGS), default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-cpus", type=int, default=64)
    parser.add_argument("--rl-batch-size", type=int, default=None)
    parser.add_argument("--total-candidate-budget", type=int, default=102400)
    parser.add_argument("--llm-candidates", type=int, default=10)
    parser.add_argument("--llm-every", type=int, default=20)
    parser.add_argument("--max-llm-in-flight", type=int, default=5)
    parser.add_argument("--llm-timeout", type=float, default=900.0)
    parser.add_argument("--global-llm-concurrency", type=int, default=8)
    parser.add_argument("--max-agent-evaluations", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--task-prompt-variant",
        choices=("standard", "reported-2026", "logic-trajectory"),
        default="standard",
        help="Select a frozen task-prompt condition; standard preserves current behavior.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("harness", "direct-nonthinking"),
        default="harness",
    )
    parser.add_argument("--llm-provider", default="deepseek-official")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--communication-mode",
        choices=("full", "none"),
        default="full",
        help="Use full-duplex RL/LLM state exchange or isolate both streams until terminal pooling.",
    )
    parser.add_argument(
        "--withhold-initial-hof",
        action="store_true",
        help="Send an empty HOF only with request zero, while retaining full communication thereafter.",
    )
    parser.add_argument(
        "--candidate-validation-policy",
        choices=("independent-members", "atomic-batch"),
        default="independent-members",
        help="Validate Writer members independently (paper default) or atomically.",
    )
    parser.add_argument(
        "--recover-valid-llm-candidates",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--literature-rag-index", type=Path, default=None)
    parser.add_argument("--max-literature-searches", type=int, default=2)
    parser.add_argument("--forbidden-topology-m", type=int, default=0)
    parser.add_argument("--forbidden-topology-every", type=int, default=5)
    parser.add_argument("--forbidden-optimization-max-evaluations", type=int, default=50)
    parser.add_argument("--forbidden-optimization-timeout", type=float, default=120.0)
    parser.add_argument("--method-name", default="genai_net_llm")
    parser.add_argument("--run-suffix", default="cvode_llm")
    parser.add_argument("--comet-project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--prompt-approval-file", type=Path, default=DEFAULT_PROMPT_APPROVAL
    )
    parser.add_argument(
        "--analysis-plan-file", type=Path, default=DEFAULT_ANALYSIS_PLAN
    )
    parser.add_argument(
        "--sentinel-report-file", type=Path, default=DEFAULT_SENTINEL_REPORT
    )
    parser.add_argument("--campaign-stage", choices=("sentinel", "paper"), default="paper")
    parser.add_argument("--trial-id", default=trial_id_now())
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/crn-runs/hybrid-trials",
    )
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/.dsh-home",
    )
    parser.add_argument(
        "--comparison-output-root",
        type=Path,
        default=REPO_ROOT / "comparisons/rpa_search/data/raw",
    )
    args = parser.parse_args()
    if min(
        args.runs,
        args.epochs,
        args.n_cpus,
        args.llm_candidates,
        args.llm_every,
        args.max_llm_in_flight,
        args.global_llm_concurrency,
        args.total_candidate_budget,
    ) <= 0:
        parser.error("run, epoch, CPU, candidate, and cadence values must be positive")
    if args.max_agent_evaluations < 0:
        parser.error("max agent evaluations must be non-negative")
    if args.max_literature_searches < 0 or args.forbidden_topology_m < 0:
        parser.error("literature and exclusion counts must be non-negative")
    if args.seed_start < 0:
        parser.error("seed start must be non-negative")
    if args.rl_batch_size is not None and args.rl_batch_size <= 0:
        parser.error("RL batch size must be positive")
    if args.recover_valid_llm_candidates:
        args.candidate_validation_policy = "independent-members"
    for name in ("method_name", "run_suffix"):
        value = getattr(args, name)
        if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
            parser.error(f"{name.replace('_', '-')} must contain only letters, numbers, underscores, or hyphens")
    return args


def main() -> None:
    args = parse_args()
    experiment_release = validate_experiment_release(
        stage=args.campaign_stage,
        prompt_approval_path=args.prompt_approval_file,
        analysis_plan_path=args.analysis_plan_file,
        sentinel_report_path=args.sentinel_report_file,
    )
    tasks = tuple(CONFIGS) if args.task == "all" else (args.task,)
    trial_root = args.workspace_root.expanduser().resolve() / args.trial_id
    trial_root.mkdir(parents=True, exist_ok=False)
    write_json(
        trial_root / "trial_manifest.json",
        {
            "trial_id": args.trial_id,
            "tasks": tasks,
            "runs": args.runs,
            "epochs": args.epochs,
            "n_cpus": args.n_cpus,
            "rl_batch_size": args.rl_batch_size,
            "total_candidate_budget_cap": args.total_candidate_budget,
            "solver": "CVODE",
            "model": args.model,
            "task_prompt_variant": args.task_prompt_variant,
            "generation_backend": args.generation_backend,
            "thinking_mode": "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default",
            "llm_provider": args.llm_provider,
            "llm_base_url": args.llm_base_url,
            "method_name": args.method_name,
            "run_suffix": args.run_suffix,
            "llm_candidates_per_round": args.llm_candidates,
            "llm_model_requests_per_round": 2 if args.generation_backend == "harness" else 1,
            "llm_protocol": (
                "two-stage-decider-writer/independent-members"
                if args.generation_backend == "harness"
                else "direct-single-request/multi-candidate"
            ),
            "writer_retry_limit": 0 if args.generation_backend == "harness" else None,
            "llm_every": args.llm_every,
            "llm_max_in_flight": args.max_llm_in_flight,
            "llm_global_concurrency": args.global_llm_concurrency,
            "withhold_initial_hof": args.withhold_initial_hof,
            "candidate_validation_policy": args.candidate_validation_policy,
            "proposal_space_contract_version": 2,
            "llm_rate_bounds": [LLM_RATE_MIN, LLM_RATE_MAX],
            "max_agent_evaluations_per_round": args.max_agent_evaluations,
            "outer_run_parallelism": 1,
            "reason": "outer concurrency is managed by the paper campaign launcher",
            "experiment_release": experiment_release,
        },
    )
    for seed in range(args.seed_start, args.seed_start + args.runs):
        for task_name in tasks:
            run_one(task_name=task_name, seed=seed, args=args, trial_root=trial_root)


if __name__ == "__main__":
    main()
