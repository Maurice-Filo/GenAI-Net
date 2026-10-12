#!/usr/bin/env python3
"""Fail-closed audit for one completed contract-v2 campaign block."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from comparisons.llm_crn_generation.audit_contract_v2_sentinel import (
    audit_run,
    file_sha256,
    read_json,
)


def audit_campaign_block(
    campaign_root: Path,
    *,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    epochs: int,
    rl_batch_size: int,
    budget: int,
    method_name: str,
    run_suffix: str,
    communication_mode: str,
    model: str = "deepseek-v4-flash",
    task_prompt_variant: str | None = None,
) -> dict:
    errors: list[str] = []
    status = read_json(campaign_root / "status.json")
    manifest = read_json(campaign_root / "campaign_manifest.json")
    expected_jobs = {(task, seed) for seed in seeds for task in tasks}
    completed_jobs = {
        (str(row.get("task")), int(row.get("seed", -1)))
        for row in status.get("completed", [])
    }
    if status.get("active") or status.get("pending") or status.get("failed"):
        errors.append("campaign status is not cleanly complete")
    if completed_jobs != expected_jobs:
        errors.append("campaign completion set does not match the expected task/seed grid")
    checks = {
        "tasks": list(tasks),
        "seeds": list(seeds),
        "rl_epochs": epochs,
        "rl_batch_size": rl_batch_size,
        "total_evaluation_budget_cap": budget,
        "method_name": method_name,
        "run_suffix": run_suffix,
        "communication_mode": communication_mode,
        "proposal_space_contract_version": 2,
        "candidate_validation_policy": "independent-members",
        "literature_searches_per_round": 0,
        "agent_simulation_probes_per_round": 0,
    }
    if task_prompt_variant is not None:
        from RL4CRN.llm.workspace_loss_protocol import protocol_metadata

        checks.update(task_prompt_variant=task_prompt_variant, **protocol_metadata(task_prompt_variant))
    if communication_mode in {"llm-to-rl", "rl-to-llm"}:
        checks.update(
            {
                "expose_state_to_llm": communication_mode == "rl-to-llm",
                "insert_llm_into_rl": communication_mode == "llm-to-rl",
                "terminal_candidate_pooling": communication_mode == "rl-to-llm",
            }
        )
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            errors.append(
                f"campaign manifest {field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if manifest.get("model") != model:
        errors.append(
            f"campaign model={manifest.get('model')!r}, expected {model!r}"
        )
    if not bool(manifest.get("withhold_initial_hof")):
        errors.append("request-0 HOF withholding is disabled")

    request_epochs = list(range(0, epochs, int(manifest.get("llm_every", 20))))
    runs = []
    for seed in seeds:
        for task in tasks:
            run_id = f"{task}_full{budget}_seed{seed}_{run_suffix}"
            run_root = campaign_root / "runs" / f"{campaign_root.name}-{task}-seed{seed}" / run_id
            if not run_root.is_dir():
                errors.append(f"missing run root: {run_root}")
                continue
            report = audit_run(
                run_root,
                task=task,
                seed=seed,
                epochs=epochs,
                expected_request_epochs=request_epochs,
                expected_budget=budget,
                expected_rl_evaluations=epochs * rl_batch_size,
                communication_mode=communication_mode,
            )
            runs.append(report)
            if task_prompt_variant is not None:
                run_manifest = read_json(run_root / "run_manifest.json")
                for field in ("task_prompt_variant", *protocol_metadata(task_prompt_variant)):
                    if run_manifest.get(field) != checks[field]:
                        report["errors"].append(f"run {field} does not match requested protocol")
                from RL4CRN.llm.workspace_loss_protocol import TASK_VARIANTS, audit_workspaces

                if task_prompt_variant in TASK_VARIANTS:
                    report["errors"].extend(audit_workspaces(run_root, task_prompt_variant))
                if report["errors"]:
                    report["status"] = "fail"
            errors.extend(
                f"{task}/seed{seed}: {message}" for message in report["errors"]
            )

    return {
        "status": "pass" if not errors else "fail",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_root.name,
        "campaign_root": str(campaign_root),
        "campaign_manifest_sha256": file_sha256(campaign_root / "campaign_manifest.json"),
        "expected": {
            "tasks": list(tasks),
            "seeds": list(seeds),
            "epochs": epochs,
            "rl_batch_size": rl_batch_size,
            "budget": budget,
            "communication_mode": communication_mode,
        },
        "completion": {
            "expected_runs": len(expected_jobs),
            "audited_runs": len(runs),
            "passing_runs": sum(row["status"] == "pass" for row in runs),
        },
        "runs": runs,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rl-batch-size", type=int, default=1023)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument(
        "--communication-mode",
        choices=("full", "llm-to-rl", "rl-to-llm", "none"),
        required=True,
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--task-prompt-variant", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_campaign_block(
        args.campaign_root.expanduser().resolve(),
        tasks=tuple(args.tasks),
        seeds=tuple(range(args.seed_start, args.seed_start + args.seeds)),
        epochs=args.epochs,
        rl_batch_size=args.rl_batch_size,
        budget=args.budget,
        method_name=args.method_name,
        run_suffix=args.run_suffix,
        communication_mode=args.communication_mode,
        model=args.model,
        task_prompt_variant=args.task_prompt_variant,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
