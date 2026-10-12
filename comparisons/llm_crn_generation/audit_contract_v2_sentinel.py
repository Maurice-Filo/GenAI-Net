#!/usr/bin/env python3
"""Audit a completed contract-v2 sentinel and write the paper release report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
DEFAULT_OUTPUT = GENERATED / "contract_v2_sentinel_report.json"
PROMPT_REVIEW = GENERATED / "CONTRACT_V2_PROMPT_REVIEW.json"
ANALYSIS_PLAN = GENERATED / "analysis_plan_v2.json"
TASKS = (
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
)
REQUIRED_WORKSPACE_FILES = (
    "DECIDER_DESIGNS.md",
    "WRITER_PAYLOAD.json",
    "FINAL_RESPONSE.json",
    "response_member_validation.json",
    "response_parameter_clamping.json",
    "response_validation_summary.json",
    "run_manifest.json",
    "run_status.json",
    "CONTEXT/HALL_OF_FAME.json",
    "CONTEXT/SEARCH_STATE.json",
    "CONTEXT/SIL_STATUS.md",
)
FAILED_WORKSPACE_AUDIT_FILES = (
    "run_manifest.json",
    "run_status.json",
    "CONTEXT/HALL_OF_FAME.json",
    "CONTEXT/SEARCH_STATE.json",
    "CONTEXT/SIL_STATUS.md",
)
MODEL_OUTCOME_FAILURE_PREFIXES = (
    "DeepSeek Harness returned an empty response.",
    "Harness returned invalid JSON:",
    "Harness returned no independently valid CRN candidates.",
)


def classify_llm_failure(error: str) -> str:
    """Separate observable model outcomes from invalid infrastructure failures."""

    message = str(error).strip()
    if message.startswith(MODEL_OUTCOME_FAILURE_PREFIXES):
        return "model_outcome"
    return "infrastructure"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}.")
    return value


def read_only_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def audit_run(
    run_root: Path,
    *,
    task: str,
    seed: int,
    epochs: int,
    expected_request_epochs: list[int],
    expected_budget: int,
    expected_rl_evaluations: int,
    communication_mode: str = "full",
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = read_json(run_root / "run_manifest.json")
    completed = read_json(run_root / "completed.json")
    database_path = run_root / "results.sqlite"
    with read_only_database(database_path) as connection:
        schema_version = int(
            connection.execute("SELECT version FROM schema_info").fetchone()[0]
        )
        llm_runs = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM llm_runs ORDER BY launched_epoch"
            )
        ]
        llm_failures = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM llm_failures ORDER BY launched_epoch"
            )
        ]
        candidates = [
            dict(row)
            for row in connection.execute(
                "SELECT candidate_json, valid FROM llm_candidates"
            )
        ]
        snapshots = [
            dict(row)
            for row in connection.execute(
                """SELECT epoch, boundary, cumulative_rl_evaluations,
                          cumulative_llm_tool_evaluations,
                          cumulative_llm_final_evaluations,
                          cumulative_total_evaluations
                   FROM hof_snapshots ORDER BY epoch, boundary"""
            )
        ]

    if schema_version != 4:
        errors.append(f"results schema is {schema_version}, expected 4")
    if manifest.get("task") != task or int(manifest.get("seed", -1)) != seed:
        errors.append("run manifest task/seed mismatch")
    if int(manifest.get("proposal_space_contract_version", -1)) != 2:
        errors.append("run manifest is not proposal-space contract v2")
    if manifest.get("candidate_validation_policy") != "independent-members":
        errors.append("member validation is not independent")
    expected_exposure = communication_mode in {"full", "rl-to-llm"}
    expected_insertion = communication_mode in {"full", "llm-to-rl"}
    if communication_mode in {"llm-to-rl", "rl-to-llm"}:
        if manifest.get("expose_state_to_llm") is not expected_exposure:
            errors.append("run manifest has the wrong outward state-exposure flag")
        if manifest.get("insert_llm_into_rl") is not expected_insertion:
            errors.append("run manifest has the wrong return-path insertion flag")
        if manifest.get("terminal_candidate_pooling") is not (not expected_insertion):
            errors.append("run manifest has the wrong terminal pooling flag")
    if int(manifest.get("max_agent_evaluations_per_round", -1)) != 0:
        errors.append("workspace simulation probes were enabled")
    if int(manifest.get("max_literature_searches_per_round", -1)) != 0:
        errors.append("literature searches were enabled")
    if not bool(manifest.get("withhold_initial_hof")):
        errors.append("request-0 HOF was not configured as withheld")
    if completed.get("budget_cap") != expected_budget:
        errors.append("completed budget cap does not equal the campaign budget")
    if completed.get("rl_candidate_evaluations") != expected_rl_evaluations:
        errors.append("completed RL evaluator count does not match the frozen schedule")
    if int(completed.get("candidate_evaluations", expected_budget + 1)) > expected_budget:
        errors.append("completed evaluator count exceeds the campaign budget")
    if completed.get("candidate_evaluations") != (
        completed.get("rl_candidate_evaluations", 0)
        + completed.get("llm_tool_evaluations", 0)
        + completed.get("llm_final_evaluations", 0)
        + completed.get("forbidden_optimization_evaluations", 0)
    ):
        errors.append("completed evaluator accounting does not reconcile")
    failure_details = [
        {
            "launched_epoch": int(row["launched_epoch"]),
            "completed_epoch": int(row["completed_epoch"]),
            "classification": classify_llm_failure(row["error"]),
            "error": str(row["error"]),
            "provider_call_count": int(
                json.loads(row["response_validation_json"] or "{}").get(
                    "provider_call_count", 0
                )
                or 0
            ),
        }
        for row in llm_failures
    ]
    infrastructure_failures = [
        row for row in failure_details if row["classification"] == "infrastructure"
    ]
    if infrastructure_failures:
        summary = "; ".join(
            f"epoch {row['launched_epoch']}: {row['error']}"
            for row in infrastructure_failures
        )
        errors.append(
            f"database contains {len(infrastructure_failures)} infrastructure LLM "
            f"failure(s): {summary}"
        )
    attempted_request_epochs = sorted(
        [int(row["launched_epoch"]) for row in llm_runs]
        + [int(row["launched_epoch"]) for row in llm_failures]
    )
    if attempted_request_epochs != expected_request_epochs:
        errors.append("LLM request epochs do not match the frozen schedule")
    if any(int(row["provider_call_count"] or 0) != 2 for row in llm_runs):
        errors.append("a Decide-then-Write round did not use exactly two provider calls")

    forbidden = set(completed["proposal_space"]["forbidden_reaction_ids"])
    forbidden_violations = []
    for row in candidates:
        if not bool(row["valid"]):
            continue
        reaction_ids = set(json.loads(row["candidate_json"])["reaction_ids"])
        if reaction_ids & forbidden:
            forbidden_violations.append(sorted(reaction_ids & forbidden))
    if forbidden_violations:
        errors.append(
            f"{len(forbidden_violations)} accepted candidates reuse forbidden IDs"
        )

    boundaries = {str(row["boundary"]): row for row in snapshots}
    for boundary, expected_epoch in (
        ("training_boundary", epochs),
        ("terminal_boundary", epochs + 1),
    ):
        row = boundaries.get(boundary)
        if row is None:
            errors.append(f"missing {boundary}")
            continue
        if int(row["epoch"]) != expected_epoch:
            errors.append(f"{boundary} has the wrong epoch")
        counted = (
            int(row["cumulative_rl_evaluations"])
            + int(row["cumulative_llm_tool_evaluations"])
            + int(row["cumulative_llm_final_evaluations"])
        )
        if counted != int(row["cumulative_total_evaluations"]):
            errors.append(f"{boundary} evaluator counts do not reconcile")
    terminal = boundaries.get("terminal_boundary")
    if terminal and int(terminal["cumulative_total_evaluations"]) != int(
        completed.get("candidate_evaluations", -1)
    ):
        errors.append("terminal boundary count does not equal the completed count")

    workspaces = sorted(
        (run_root / "harness-workspaces").glob("*"),
        key=lambda path: read_json(path / "run_manifest.json")["created_at"],
    )
    if len(workspaces) != len(expected_request_epochs):
        errors.append("workspace count does not equal scheduled LLM rounds")
    failures_by_epoch = {
        int(row["launched_epoch"]): row for row in failure_details
    }
    for workspace in workspaces:
        search_state_path = workspace / "CONTEXT/SEARCH_STATE.json"
        try:
            search_state = read_json(search_state_path)
            workspace_epoch = int(
                search_state.get(
                    "rl_epoch_at_snapshot", search_state.get("rl_epoch", -1)
                )
            )
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            workspace_epoch = -1
        failure = failures_by_epoch.get(workspace_epoch)
        required_files = (
            FAILED_WORKSPACE_AUDIT_FILES if failure is not None else REQUIRED_WORKSPACE_FILES
        )
        missing = [name for name in required_files if not (workspace / name).is_file()]
        if missing:
            errors.append(f"{workspace.name} lacks {', '.join(missing)}")
            continue
        try:
            status = read_json(workspace / "run_status.json").get("status")
        except (json.JSONDecodeError, ValueError):
            status = "unreadable"
        expected_status = "failed" if failure is not None else "completed"
        if status != expected_status:
            errors.append(
                f"{workspace.name} status is {status!r}, expected {expected_status!r}"
            )
    if workspaces:
        hof_entries = [
            read_json(workspace / "CONTEXT/HALL_OF_FAME.json").get("entries")
            for workspace in workspaces
        ]
        if hof_entries[0] != []:
            errors.append("request-0 workspace contains a non-empty HOF")
        exposes_state = communication_mode in {"full", "rl-to-llm"}
        if exposes_state and any(entries == [] for entries in hof_entries[1:]):
            errors.append("a later state-exposed workspace received an empty HOF")
        if not exposes_state and any(entries != [] for entries in hof_entries):
            errors.append("a state-withheld workspace received HOF entries")
        sil_text = [
            (workspace / "CONTEXT/SIL_STATUS.md").read_text(encoding="utf-8")
            for workspace in workspaces
        ]
        if exposes_state and not any(
            "enabled=True" in text for text in sil_text[1:]
        ):
            errors.append("later state-exposed workspaces lack completed SIL state")
        if not exposes_state and any(
            "withheld" not in text.lower() for text in sil_text
        ):
            errors.append("a state-withheld workspace exposes SIL state")

    return {
        "task": task,
        "seed": seed,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "schema_version": schema_version,
        "request_epochs": attempted_request_epochs,
        "successful_request_epochs": [int(row["launched_epoch"]) for row in llm_runs],
        "failed_request_epochs": [int(row["launched_epoch"]) for row in llm_failures],
        "provider_calls": (
            sum(int(row["provider_call_count"] or 0) for row in llm_runs)
            + sum(int(row["provider_call_count"]) for row in failure_details)
        ),
        "returned_candidates": sum(int(row["returned"] or 0) for row in llm_runs),
        "accepted_candidates": sum(int(row["accepted"] or 0) for row in llm_runs),
        "rejected_candidates": sum(int(row["rejected"] or 0) for row in llm_runs),
        "clamped_parameters": sum(
            int(row["clamped_parameters"] or 0) for row in llm_runs
        ),
        "llm_failures": len(llm_failures),
        "model_outcome_failures": len(llm_failures) - len(infrastructure_failures),
        "infrastructure_failures": len(infrastructure_failures),
        "failure_details": failure_details,
        "forbidden_id_violations": len(forbidden_violations),
        "boundaries": boundaries,
        "workspaces": len(workspaces),
        "run_root": str(run_root),
        "database_sha256": file_sha256(database_path),
        "completed_sha256": file_sha256(run_root / "completed.json"),
    }


def audit_campaign(
    campaign_root: Path,
    *,
    method_name: str,
    run_suffix: str,
    seed: int,
    epochs: int,
    budget: int,
) -> dict[str, Any]:
    status = read_json(campaign_root / "status.json")
    manifest = read_json(campaign_root / "campaign_manifest.json")
    errors: list[str] = []
    completed_jobs = status.get("completed", [])
    if status.get("active") or status.get("pending") or status.get("failed"):
        errors.append("campaign status is not cleanly complete")
    if {row.get("task") for row in completed_jobs} != set(TASKS):
        errors.append("campaign completion does not cover all eight tasks")
    if manifest.get("tasks") != list(TASKS):
        errors.append("campaign manifest task order does not match the frozen task set")
    if manifest.get("model") != "deepseek-v4-flash":
        errors.append("sentinel did not use the frozen DeepSeek V4 Flash identifier")
    if manifest.get("communication_mode") != "full":
        errors.append("sentinel communication mode is not full duplex")
    if manifest.get("method_name") != method_name:
        errors.append("sentinel method name does not match the audited method")
    if manifest.get("run_suffix") != run_suffix:
        errors.append("sentinel run suffix does not match the audited suffix")
    if int(manifest.get("literature_searches_per_round", -1)) != 0:
        errors.append("campaign did not freeze literature searches to zero")

    expected_request_epochs = list(range(0, epochs, 20))
    run_reports = []
    for task in TASKS:
        run_id = f"{task}_full{budget}_seed{seed}_{run_suffix}"
        run_root = campaign_root / "runs" / f"{campaign_root.name}-{task}-seed{seed}" / run_id
        if not run_root.is_dir():
            errors.append(f"missing run root for {task}: {run_root}")
            continue
        report = audit_run(
            run_root,
            task=task,
            seed=seed,
            epochs=epochs,
            expected_request_epochs=expected_request_epochs,
            expected_budget=budget,
            expected_rl_evaluations=epochs * 1023,
            communication_mode="full",
        )
        run_reports.append(report)
        errors.extend(f"{task}: {message}" for message in report["errors"])

    return {
        "status": "pass" if not errors else "fail",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_root.name,
        "campaign_root": str(campaign_root),
        "campaign_manifest_sha256": file_sha256(campaign_root / "campaign_manifest.json"),
        "prompt_review_sha256": file_sha256(PROMPT_REVIEW),
        "analysis_plan_sha256": file_sha256(ANALYSIS_PLAN),
        "completion": {
            "tasks": len(run_reports),
            "completed_runs": sum(row["status"] == "pass" for row in run_reports),
            "failed_runs": sum(row["status"] != "pass" for row in run_reports),
        },
        "aggregate": {
            "provider_calls": sum(row["provider_calls"] for row in run_reports),
            "returned_candidates": sum(row["returned_candidates"] for row in run_reports),
            "accepted_candidates": sum(row["accepted_candidates"] for row in run_reports),
            "rejected_candidates": sum(row["rejected_candidates"] for row in run_reports),
            "clamped_parameters": sum(row["clamped_parameters"] for row in run_reports),
            "llm_failures": sum(row["llm_failures"] for row in run_reports),
            "model_outcome_failures": sum(
                row["model_outcome_failures"] for row in run_reports
            ),
            "infrastructure_failures": sum(
                row["infrastructure_failures"] for row in run_reports
            ),
            "forbidden_id_violations": sum(
                row["forbidden_id_violations"] for row in run_reports
            ),
        },
        "runs": run_reports,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--run-suffix", required=True)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--budget", type=int, default=21503)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = audit_campaign(
        args.campaign_root.expanduser().resolve(),
        method_name=args.method_name,
        run_suffix=args.run_suffix,
        seed=args.seed,
        epochs=args.epochs,
        budget=args.budget,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
