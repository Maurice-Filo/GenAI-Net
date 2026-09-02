"""Run a small, auditable MMC2 DeepSeek Harness benchmark with Comet logging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RL4CRN.llm import HarnessCRNGenerator, HarnessLLMClient, get_mmc2_task_prompt
from RL4CRN.llm.sil_bridge import audit_sil_bridge
from RL4CRN.utils.hall_of_fame import HallOfFame
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import (
    BUILDERS,
    CONFIGS,
    build_evaluator,
)


DEFAULT_PROJECT = "mmc2-v4-flash-harness"
DEFAULT_MODEL = "deepseek-v4-flash"
CSV_FIELDS = (
    "trial_id",
    "task",
    "run_index",
    "proposal_step",
    "valid",
    "loss",
    "best_so_far_loss",
    "message",
    "workspace",
    "duration_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_trial_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-mmc2-v4-flash-cvode"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def completed_steps(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["task"], int(row["run_index"]), int(row["proposal_step"]))
            for row in csv.DictReader(handle)
        }


def start_comet_experiment(*, project: str, trial_id: str, task: str, run_index: int) -> Any:
    from comet_ml import Experiment

    experiment = Experiment(
        project_name=project,
        auto_param_logging=False,
        auto_metric_logging=False,
        auto_output_logging="simple",
        log_code=False,
        log_graph=False,
        log_git_metadata=True,
        log_git_patch=False,
        display_summary=False,
    )
    experiment.set_name(f"{trial_id}-{task}-run-{run_index:02d}")
    experiment.add_tags(["mmc2", "deepseek-v4-flash", "harness", "cvode", task, "trial"])
    experiment.log_other("trial_id", trial_id)
    experiment.log_other("task", task)
    experiment.log_other("run_index", run_index)
    return experiment


def log_progress(experiment: Any, record: dict[str, Any]) -> None:
    step = int(record["proposal_step"])
    experiment.log_metric("benchmark/valid", int(bool(record["valid"])), step=step)
    experiment.log_metric("benchmark/candidate_evaluations", step, step=step)
    experiment.log_metric("benchmark/duration_seconds", record["duration_seconds"], step=step)
    if record["loss"] != "":
        experiment.log_metric("benchmark/loss", float(record["loss"]), step=step)
    if record["best_so_far_loss"] != "":
        experiment.log_metric(
            "benchmark/best_so_far_loss", float(record["best_so_far_loss"]), step=step
        )
    experiment.log_asset_data(
        json.dumps(record, indent=2, sort_keys=True),
        name=f"proposal_{step:03d}.json",
        step=step,
    )


def run_replicate(
    *,
    task: str,
    run_index: int,
    proposals: int,
    trial_id: str,
    trial_root: Path,
    dsh_home: Path,
    comet_project: str,
    model: str,
    solver: str,
    summary_path: Path,
    sil_audit: dict[str, Any],
) -> None:
    replicate_root = trial_root / task / f"run-{run_index:02d}"
    workspaces = replicate_root / "workspaces"
    replicate_root.mkdir(parents=True, exist_ok=True)
    evaluator = build_evaluator(task, solver=solver)
    actual_solver = str(getattr(evaluator.crn_template, "solver", "")).upper()
    if actual_solver != solver:
        raise RuntimeError(f"Evaluator uses {actual_solver!r}, expected {solver!r}.")

    task_prompt = get_mmc2_task_prompt(task, solver=solver)
    client = HarnessLLMClient(
        workspace_root=workspaces,
        dsh_home=dsh_home,
        model=model,
    )
    generator = HarnessCRNGenerator(client=client, evaluator=evaluator)
    hall_of_fame = HallOfFame(max_size=proposals)
    experiment = start_comet_experiment(
        project=comet_project,
        trial_id=trial_id,
        task=task,
        run_index=run_index,
    )
    experiment.log_parameters(
        {
            "task": task,
            "run_index": run_index,
            "proposal_budget": proposals,
            "candidates_per_proposal": 1,
            "solver": solver,
            "model": model,
            "provider": client.provider,
            "rate_min": evaluator.min_parameter_value,
            "rate_max": evaluator.max_parameter_value,
            "reaction_budget": evaluator.max_added_reactions,
        }
    )
    experiment.log_asset_data(task_prompt, name="TASK.md")
    experiment.log_asset_data(
        json.dumps(sil_audit, indent=2, sort_keys=True),
        name="sil_bridge_audit.json",
    )
    experiment.log_other("sil_bridge_verified", True)
    experiment.log_other("sil_active_in_llm_only_trial", False)
    write_json(
        replicate_root / "manifest.json",
        {
            "created_at": utc_now(),
            "trial_id": trial_id,
            "task": task,
            "run_index": run_index,
            "proposal_budget": proposals,
            "solver": solver,
            "model": model,
            "provider": client.provider,
            "comet_project": comet_project,
            "comet_experiment_key": experiment.get_key(),
            "workspace_root": str(workspaces),
        },
    )

    best_loss = math.inf
    try:
        for proposal_step in range(1, proposals + 1):
            started = time.perf_counter()
            valid = False
            loss: float | str = ""
            message = ""
            workspace = ""
            try:
                result = generator.run_round(
                    task_description=task_prompt,
                    num_candidates=1,
                    hall_of_fame_iter=hall_of_fame,
                    add_to_hall_of_fame=hall_of_fame,
                    jsonl_path=replicate_root / "evaluations.jsonl",
                )
                evaluation = result.evaluations[0]
                valid = bool(evaluation.valid)
                message = evaluation.message
                if valid and evaluation.loss is not None:
                    loss = float(evaluation.loss)
                    best_loss = min(best_loss, loss)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
            if client.last_workspace is not None:
                workspace = str(client.last_workspace.path)

            record = {
                "trial_id": trial_id,
                "task": task,
                "run_index": run_index,
                "proposal_step": proposal_step,
                "valid": valid,
                "loss": loss,
                "best_so_far_loss": "" if math.isinf(best_loss) else best_loss,
                "message": message,
                "workspace": workspace,
                "duration_seconds": time.perf_counter() - started,
            }
            append_csv(summary_path, record)
            log_progress(experiment, record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        experiment.log_other("completed_at", utc_now())
        experiment.end()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=10)
    parser.add_argument("--task", choices=("all", "logic", "rpa"), default="all")
    parser.add_argument("--solver", choices=("CVODE",), default="CVODE")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--comet-project", default=DEFAULT_PROJECT)
    parser.add_argument("--trial-id", default=make_trial_id())
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/crn-runs/benchmark-trials",
    )
    parser.add_argument(
        "--dsh-home",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/.dsh-home",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.runs < 1 or args.proposals < 1:
        parser.error("--runs and --proposals must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = ("logic", "rpa") if args.task == "all" else (args.task,)
    sil_audits = {}
    for task in tasks:
        evaluator = build_evaluator(task, solver=args.solver)
        actual_solver = str(getattr(evaluator.crn_template, "solver", "")).upper()
        if actual_solver != args.solver:
            raise RuntimeError(f"{task} evaluator uses {actual_solver!r}, expected {args.solver!r}.")
        config = json.loads(CONFIGS[task].read_text(encoding="utf-8"))
        config[task]["solver"] = args.solver
        sil_audits[task] = audit_sil_bridge(
            task_name=task,
            config=config,
            build_components=BUILDERS[task],
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "tasks": tasks,
                    "solver": args.solver,
                    "sil_audits": sil_audits,
                },
                indent=2,
            )
        )
        return

    trial_root = args.workspace_root.expanduser().resolve() / args.trial_id
    trial_root.mkdir(parents=True, exist_ok=False)
    summary_path = trial_root / "progress.csv"
    write_json(
        trial_root / "trial_manifest.json",
        {
            "created_at": utc_now(),
            "trial_id": args.trial_id,
            "tasks": tasks,
            "runs": args.runs,
            "proposals_per_run": args.proposals,
            "solver": args.solver,
            "model": args.model,
            "comet_project": args.comet_project,
            "execution_order": "run-major, task-interleaved",
            "sil_mode": "bridge-verified; inactive in LLM-only comparison",
            "sil_audits": sil_audits,
        },
    )
    for run_index in range(args.runs):
        for task in tasks:
            run_replicate(
                task=task,
                run_index=run_index,
                proposals=args.proposals,
                trial_id=args.trial_id,
                trial_root=trial_root,
                dsh_home=args.dsh_home.expanduser().resolve(),
                comet_project=args.comet_project,
                model=args.model,
                solver=args.solver,
                summary_path=summary_path,
                sil_audit=sil_audits[task],
            )


if __name__ == "__main__":
    main()
