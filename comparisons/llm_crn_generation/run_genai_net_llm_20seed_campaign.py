#!/usr/bin/env python3
"""Launch the 20-seed Logic/RPA GenAI-Net-LLM paper campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from comet_ml import Experiment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
from comparisons.llm_crn_generation.experiment_release import (
    DEFAULT_ANALYSIS_PLAN,
    DEFAULT_SENTINEL_REPORT,
    validate_experiment_release,
)
from comparisons.llm_crn_generation.prompt_approval import DEFAULT_PROMPT_APPROVAL
from RL4CRN.llm.benchmark_prompts import get_mmc2_task_prompt_variant


RUNNER = ROOT / "comparisons/llm_crn_generation/run_mmc2_harness_hybrid.py"
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"


def campaign_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-genai-net-llm-20seed"


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id(task: str, seed: int, run_suffix: str, budget: int) -> str:
    return f"{task}_full{budget}_seed{seed}_{run_suffix}"


def is_complete(args: argparse.Namespace, task: str, seed: int) -> bool:
    return (
        RAW_ROOT
        / args.method_name
        / run_id(task, seed, args.run_suffix, args.total_candidate_budget)
        / "completed.json"
    ).exists()


def command(args: argparse.Namespace, campaign_root: Path, task: str, seed: int) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(RUNNER),
        "--task",
        task,
        "--runs",
        "1",
        "--seed-start",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--n-cpus",
        str(args.cpus_per_run),
        "--rl-batch-size",
        str(args.rl_batch_size),
        "--total-candidate-budget",
        str(args.total_candidate_budget),
        "--llm-candidates",
        str(args.llm_candidates),
        "--llm-every",
        str(args.llm_every),
        "--max-llm-in-flight",
        str(args.max_llm_in_flight),
        "--llm-timeout",
        str(args.llm_timeout),
        "--global-llm-concurrency",
        str(args.global_llm_concurrency),
        "--max-agent-evaluations",
        str(args.max_agent_evaluations),
        "--model",
        args.model,
        "--task-prompt-variant",
        args.task_prompt_variant,
        "--generation-backend",
        args.generation_backend,
        "--llm-provider",
        args.llm_provider,
        "--communication-mode",
        args.communication_mode,
        "--method-name",
        args.method_name,
        "--run-suffix",
        args.run_suffix,
        "--dsh-home",
        str(args.dsh_home),
        "--comet-project",
        args.comet_project,
        "--trial-id",
        f"{args.campaign_id}-{task}-seed{seed}",
        "--workspace-root",
        str(campaign_root / "runs"),
        "--comparison-output-root",
        str(RAW_ROOT),
        "--prompt-approval-file",
        str(args.prompt_approval_file),
        "--analysis-plan-file",
        str(args.analysis_plan_file),
        "--sentinel-report-file",
        str(args.sentinel_report_file),
        "--campaign-stage",
        args.campaign_stage,
    ]


def configured_command(
    args: argparse.Namespace, campaign_root: Path, task: str, seed: int
) -> list[str]:
    cmd = command(args, campaign_root, task, seed)
    if args.withhold_initial_hof:
        cmd.append("--withhold-initial-hof")
    cmd.extend(["--candidate-validation-policy", args.candidate_validation_policy])
    if args.llm_base_url:
        cmd.extend(["--llm-base-url", args.llm_base_url])
    if args.literature_rag_index:
        cmd.extend(["--literature-rag-index", str(args.literature_rag_index)])
    if args.forbidden_topology_m:
        cmd.extend(
            [
                "--forbidden-topology-m",
                str(args.forbidden_topology_m),
                "--forbidden-topology-every",
                str(args.forbidden_topology_every),
                "--forbidden-optimization-max-evaluations",
                str(args.forbidden_optimization_max_evaluations),
                "--forbidden-optimization-timeout",
                str(args.forbidden_optimization_timeout),
            ]
        )
    return cmd


def comet_smoke(project: str, campaign_id: str) -> str:
    """Require one authenticated Comet event before any scientific worker starts."""

    experiment = Experiment(
        project_name=project,
        auto_param_logging=False,
        auto_metric_logging=False,
        log_code=False,
        log_graph=False,
        log_git_patch=False,
        display_summary_level=0,
    )
    experiment.set_name(f"{campaign_id}-preflight")
    experiment.add_tags(["genai-net-paper", "contract-v2", "preflight"])
    key = experiment.get_key()
    if not key:
        experiment.end()
        raise RuntimeError("Comet preflight did not return an experiment key.")
    experiment.log_other("campaign_id", campaign_id)
    experiment.log_other("status", "preflight-ok")
    experiment.log_metric("preflight/comet_authenticated", 1, step=0)
    experiment.end()
    return str(key)


def write_status(path: Path, *, pending: list[tuple[str, int]], active: dict, done: list, failed: list) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pending": [{"task": task, "seed": seed} for task, seed in pending],
        "active": [
            {"task": task, "seed": seed, "pid": job["process"].pid}
            for (task, seed), job in active.items()
        ],
        "completed": done,
        "failed": failed,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def terminate_process_groups(active: dict, *, grace_seconds: float = 10.0) -> None:
    """Terminate workers and every Harness subprocess in their process groups."""

    running = [job for job in active.values() if job["process"].poll() is None]
    for job in running:
        try:
            os.killpg(job["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    while running and time.monotonic() < deadline:
        running = [job for job in running if job["process"].poll() is None]
        if running:
            time.sleep(0.1)

    for job in running:
        try:
            os.killpg(job["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for job in active.values():
        if not job["log_file"].closed:
            job["log_file"].close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", default=campaign_id_now())
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(CONFIGS),
        default=("logic", "rpa"),
    )
    parser.add_argument("--max-parallel", type=int, default=32)
    parser.add_argument("--cpus-per-run", type=int, default=4)
    parser.add_argument(
        "--rl-gpu",
        default=None,
        help="GPU UUID or CUDA index exposed to RL workers.",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rl-batch-size", type=int, default=1023)
    parser.add_argument("--total-candidate-budget", type=int, default=102400)
    parser.add_argument("--llm-candidates", type=int, default=10)
    parser.add_argument("--llm-every", type=int, default=20)
    parser.add_argument("--max-llm-in-flight", type=int, default=5)
    parser.add_argument("--max-agent-evaluations", type=int, default=0)
    parser.add_argument("--llm-timeout", type=float, default=3600.0)
    parser.add_argument("--global-llm-concurrency", type=int, default=8)
    parser.add_argument("--comet-project", default="genai-net-llm-paper")
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
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--task-prompt-variant",
        choices=("standard", "reported-2026", "logic-trajectory"),
        default="standard",
    )
    parser.add_argument(
        "--generation-backend",
        choices=("harness", "direct-nonthinking"),
        default="harness",
    )
    parser.add_argument("--llm-provider", default="deepseek-official")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--communication-mode", choices=("full", "none"), default="full"
    )
    parser.add_argument("--withhold-initial-hof", action="store_true")
    parser.add_argument(
        "--candidate-validation-policy",
        choices=("independent-members", "atomic-batch"),
        default="independent-members",
    )
    parser.add_argument("--recover-valid-llm-candidates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--literature-rag-index", type=Path, default=None)
    parser.add_argument("--forbidden-topology-m", type=int, default=0)
    parser.add_argument("--forbidden-topology-every", type=int, default=5)
    parser.add_argument("--forbidden-optimization-max-evaluations", type=int, default=50)
    parser.add_argument("--forbidden-optimization-timeout", type=float, default=120.0)
    parser.add_argument("--method-name", default="genai_net_llm")
    parser.add_argument("--run-suffix", default="cvode_llm")
    parser.add_argument("--skip-postprocessing", action="store_true")
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/.dsh-home",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns",
    )
    args = parser.parse_args()
    experiment_release = validate_experiment_release(
        stage=args.campaign_stage,
        prompt_approval_path=args.prompt_approval_file,
        analysis_plan_path=args.analysis_plan_file,
        sentinel_report_path=args.sentinel_report_file,
    )
    positive = (
        args.max_parallel,
        args.cpus_per_run,
        args.seeds,
        args.epochs,
        args.rl_batch_size,
        args.total_candidate_budget,
        args.llm_candidates,
        args.llm_every,
        args.max_llm_in_flight,
        args.global_llm_concurrency,
    )
    if min(positive) <= 0 or args.seed_start < 0:
        parser.error("parallelism, seeds, epochs, budgets, and CPUs must be positive")
    if args.rl_gpu is not None:
        numeric_gpu = args.rl_gpu.isdigit()
        uuid_gpu = args.rl_gpu.startswith("GPU-") and len(args.rl_gpu) > 4
        if not (numeric_gpu or uuid_gpu):
            parser.error("rl-gpu must be a non-negative CUDA index or GPU UUID")
    if args.max_agent_evaluations < 0 or args.forbidden_topology_m < 0:
        parser.error("evaluation and exclusion counts must be non-negative")
    if args.recover_valid_llm_candidates:
        args.candidate_validation_policy = "independent-members"
    for name in ("method_name", "run_suffix"):
        value = getattr(args, name)
        if not value or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in value
        ):
            parser.error(
                f"{name.replace('_', '-')} must contain only letters, numbers, underscores, or hyphens"
            )

    campaign_root = args.workspace_root.expanduser().resolve() / args.campaign_id
    log_root = campaign_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    jobs = [(task, seed) for seed in seeds for task in args.tasks]
    pending = [job for job in jobs if not is_complete(args, *job)]
    done = [
        {"task": task, "seed": seed, "status": "already_complete"}
        for task, seed in jobs
        if is_complete(args, task, seed)
    ]
    failed: list[dict] = []
    active: dict[tuple[str, int], dict] = {}
    prompt_hashes = {
        task: hashlib.sha256(
            get_mmc2_task_prompt_variant(
                task,
                variant=args.task_prompt_variant,
                solver="CVODE",
            ).encode("utf-8")
        ).hexdigest()
        for task in args.tasks
    }
    comet_preflight_key = comet_smoke(args.comet_project, args.campaign_id)
    manifest = {
        "campaign_id": args.campaign_id,
        "tasks": list(args.tasks),
        "seeds": seeds,
        "solver": "CVODE",
        "model": args.model,
        "task_prompt_variant": args.task_prompt_variant,
        "task_prompt_sha256_by_task": prompt_hashes,
        "generation_backend": args.generation_backend,
        "thinking_mode": (
            "disabled" if args.generation_backend == "direct-nonthinking" else "provider-default"
        ),
        "direct_hof_context_format": (
            "rank-loss-actions-crn-v1"
            if args.generation_backend == "direct-nonthinking"
            else None
        ),
        "llm_provider": args.llm_provider,
        "communication_mode": args.communication_mode,
        "withhold_initial_hof": args.withhold_initial_hof,
        "candidate_validation_policy": args.candidate_validation_policy,
        "proposal_space_contract_version": 2,
        "llm_rate_bounds": [0.001, 100.0],
        "terminal_candidate_pooling": args.communication_mode == "none",
        "llm_base_url": args.llm_base_url,
        "method_name": args.method_name,
        "run_suffix": args.run_suffix,
        "max_parallel": args.max_parallel,
        "cpus_per_run": args.cpus_per_run,
        "rl_gpu_assignment": args.rl_gpu,
        "maximum_cvode_worker_slots": args.max_parallel * args.cpus_per_run,
        "rl_epochs": args.epochs,
        "rl_batch_size": args.rl_batch_size,
        "maximum_rl_evaluations": args.epochs * args.rl_batch_size,
        "llm_rounds": (args.epochs - 1) // args.llm_every + 1,
        "llm_model_calls_per_round": 2 if args.generation_backend == "harness" else 1,
        "writer_retry_limit": 0 if args.generation_backend == "harness" else None,
        "maximum_llm_model_calls": (
            ((args.epochs - 1) // args.llm_every + 1)
            * (2 if args.generation_backend == "harness" else 1)
        ),
        "llm_candidates_per_round": args.llm_candidates,
        "llm_timeout_seconds": args.llm_timeout,
        "llm_global_concurrency": args.global_llm_concurrency,
        "maximum_llm_evaluations": (
            ((args.epochs - 1) // args.llm_every + 1) * args.llm_candidates
        ),
        "agent_simulation_probes_per_round": args.max_agent_evaluations,
        "total_evaluation_budget_cap": args.total_candidate_budget,
        "literature_rag_index": str(args.literature_rag_index) if args.literature_rag_index else None,
        "literature_rag_index_sha256": file_sha256(args.literature_rag_index),
        "forbidden_topology_m": args.forbidden_topology_m,
        "forbidden_optimization_max_evaluations": args.forbidden_optimization_max_evaluations,
        "forbidden_optimization_timeout_seconds": args.forbidden_optimization_timeout,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comet_required": True,
        "comet_preflight_experiment_key": comet_preflight_key,
        "experiment_release": experiment_release,
    }
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    env = os.environ.copy()
    if args.rl_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.rl_gpu)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt_campaign(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signum in previous_handlers:
        signal.signal(signum, interrupt_campaign)
    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                task, seed = pending.pop(0)
                log_path = log_root / f"{task}_seed{seed}.log"
                log_file = log_path.open("w", encoding="utf-8")
                cmd = configured_command(args, campaign_root, task, seed)
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[(task, seed)] = {
                    "process": process,
                    "log_file": log_file,
                    "log": str(log_path),
                }
                print(f"[start] {task} seed={seed} pid={process.pid}", flush=True)

            for key, job in list(active.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["log_file"].close()
                task, seed = key
                record = {
                    "task": task,
                    "seed": seed,
                    "returncode": returncode,
                    "log": job["log"],
                }
                if returncode == 0 and is_complete(args, task, seed):
                    done.append(record)
                    print(f"[done] {task} seed={seed}", flush=True)
                else:
                    failed.append(record)
                    print(
                        f"[failed] {task} seed={seed} rc={returncode} log={job['log']}",
                        flush=True,
                    )
                del active[key]

            write_status(
                campaign_root / "status.json",
                pending=pending,
                active=active,
                done=done,
                failed=failed,
            )
            if pending or active:
                time.sleep(5.0)
    finally:
        terminate_process_groups(active)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if failed:
        raise SystemExit(f"Campaign finished with {len(failed)} failed runs; see status.json.")
    if args.skip_postprocessing:
        print(f"Campaign complete: {campaign_root}", flush=True)
        return
    figure_tag = "" if args.method_name == "genai_net_llm" else f"_{args.method_name}"
    if args.method_name == "genai_net_llm":
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "comparisons/rpa_search/scripts/plot_comparisons_with_ga_llm.py"),
                "--n-seeds",
                "20",
                "--output",
                str(ROOT / "comparisons_with_GA.pdf"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "comparisons/rpa_search/scripts/plot_genai_llm_diversity.py"),
                "--n-seeds",
                "20",
                "--output",
                str(ROOT / "comparisons/rpa_search/figures/genai_net_llm_diversity_20seed.pdf"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "comparisons/rpa_search/scripts/plot_llm_rl_prevalence.py"),
            "--campaign-root",
            str(campaign_root),
            "--n-seeds",
            "20",
            "--run-suffix",
            args.run_suffix,
            "--candidate-budget",
            str(args.total_candidate_budget),
            "--output",
            str(ROOT / f"comparisons/rpa_search/figures/llm_rl_prevalence{figure_tag}_20seed.pdf"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "comparisons/rpa_search/scripts/plot_llm_segment_analysis.py"),
            "--campaign-root",
            str(campaign_root),
            "--trace-root",
            str(args.dsh_home.expanduser().resolve() / "sessions"),
            "--n-seeds",
            "20",
            "--run-suffix",
            args.run_suffix,
            "--candidate-budget",
            str(args.total_candidate_budget),
            "--performance-output",
            str(ROOT / f"comparisons/rpa_search/figures/llm_segment_performance{figure_tag}_20seed.pdf"),
            "--resource-output",
            str(ROOT / f"comparisons/rpa_search/figures/llm_resource_cost{figure_tag}_20seed.pdf"),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(f"Campaign complete: {campaign_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Campaign interrupted; active worker process groups were terminated.", file=sys.stderr)
        raise SystemExit(130)
