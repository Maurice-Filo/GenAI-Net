#!/usr/bin/env python3
"""Read-only consistency and mechanism audit for the frozen paper experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "comparisons/rpa_search/data/raw"
PAPER = Path(__file__).resolve().parent
CAMPAIGN_BASE = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
)
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
FIXED_REACTIONS = {
    "rpa": 2,
    "logic": 4,
    "classifier": 0,
    "dose_hill": 1,
    "dose_ultrasensitive": 1,
    "dose_biphasic": 1,
    "oscillator_mean": 0,
    "oscillator_frequency": 1,
}
HYBRID = {
    "rpa": (
        "genai_net_llm_flash_rpa_context_free100",
        "rpa_full307200_seed{seed}_cvode_flash_rpa_context_free100",
        "flash-rpa-initial-hof-withheld-100epoch-20seed",
    ),
    "logic": (
        "genai_net_llm_flash_logic_initial_context_free100",
        "logic_full102400_seed{seed}_cvode_llm_flash_logic_initial_context_free100",
        "flash-logic-initial-hof-withheld-100epoch-20seed",
    ),
}
BREADTH_METHOD = "genai_net_llm_flash_breadth_initial_context_free20"
BREADTH_SUFFIX = "cvode_llm_flash_breadth_initial_context_free20"
BREADTH_CAMPAIGN = "flash-breadth-initial-hof-withheld-100epoch-20seed"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_progress(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hybrid_paths(task: str, seed: int, campaign_base: Path) -> tuple[Path, Path]:
    if task in HYBRID:
        method, pattern, campaign = HYBRID[task]
        run_id = pattern.format(seed=seed)
    else:
        method = BREADTH_METHOD
        run_id = f"{task}_full102400_seed{seed}_{BREADTH_SUFFIX}"
        campaign = BREADTH_CAMPAIGN
    artifact = RAW / method / run_id
    matches = sorted((campaign_base / campaign / "runs").glob(f"*/{run_id}/results.sqlite"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one database for {run_id}, found {len(matches)}")
    return artifact, matches[0]


def control_path(task: str, seed: int) -> Path:
    if task in {"rpa", "logic"}:
        return RAW / "rl4crn" / f"{task}_full102400_seed{seed}_cvode"
    return RAW / "rl4crn_breadth" / f"{task}_full102400_seed{seed}_cvode_rl_only_breadth"


def campaign_name(task: str) -> str:
    return HYBRID[task][2] if task in HYBRID else BREADTH_CAMPAIGN


def failed_batch_reasons(path: Path) -> list[str]:
    if not path.is_file():
        return []
    reasons = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "LLM graph failed after" not in line:
            continue
        if "parameters must be within" in line:
            reasons.append("parameter_outside_llm_schema")
        elif "exactly duplicates an earlier candidate" in line:
            reasons.append("duplicate_within_batch")
        else:
            reasons.append("other")
    return reasons


def close(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=atol))


def database_diagnostics(
    path: Path,
    *,
    fixed_reaction_count: int = 0,
    fixed_reaction_ids: set[int] | frozenset[int] = frozenset(),
) -> dict:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        runs = connection.execute(
            """SELECT llm_run_id, launched_epoch, completed_epoch, requested,
                      produced, valid_count, elapsed_seconds
                 FROM llm_runs ORDER BY launched_epoch"""
        ).fetchall()
        candidates = connection.execute(
            """SELECT r.llm_run_id, r.launched_epoch, r.completed_epoch,
                      c.candidate_index, c.topology_hash, e.parameters_json,
                      c.valid, c.loss
                 FROM llm_candidates c
                 JOIN llm_runs r ON r.llm_run_id = c.llm_run_id
                 LEFT JOIN evaluations e
                   ON e.source = 'llm'
                  AND e.topology_hash = c.topology_hash
                  AND json_extract(e.metadata_json, '$.llm_run_id') = c.llm_run_id
                  AND json_extract(e.metadata_json, '$.candidate_index') = c.candidate_index
                ORDER BY r.launched_epoch, c.candidate_index"""
        ).fetchall()
        snapshots = connection.execute(
            """SELECT h.epoch, e.rank, e.topology_hash, e.parameters_json, e.loss
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE e.loss IS NOT NULL
                ORDER BY h.epoch, e.rank"""
        ).fetchall()

    by_epoch: dict[int, list[tuple]] = defaultdict(list)
    for row in snapshots:
        by_epoch[int(row[0])].append(row)
    llm_identifiers = {
        (str(row[4]), str(row[5]))
        for row in candidates
        if row[4] is not None and row[5] is not None and bool(row[6])
    }
    hof_after: dict[int, set[tuple[str, str]]] = {}
    for epoch in by_epoch:
        hof_after[epoch] = {
            (str(row[2]), str(row[3]))
            for snapshot_epoch, rows in by_epoch.items()
            if snapshot_epoch >= epoch
            for row in rows
        }

    request_rows = []
    for llm_run_id, launched, completed, requested, produced, valid_count, elapsed in runs:
        current = [row for row in candidates if row[0] == llm_run_id]
        valid = [row for row in current if bool(row[6]) and row[7] is not None]
        issue_snapshot = by_epoch.get(int(launched), [])
        issue_best = min((float(row[4]) for row in issue_snapshot), default=float("nan"))
        issue_topologies = {str(row[2]) for row in issue_snapshot}
        issue_best_by_topology: dict[str, float] = {}
        for row in issue_snapshot:
            topology = str(row[2])
            issue_best_by_topology[topology] = min(
                issue_best_by_topology.get(topology, float("inf")), float(row[4])
            )
        request_best = min((float(row[7]) for row in valid), default=float("nan"))
        inserted = sum(
            (str(row[4]), str(row[5])) in hof_after.get(int(completed), set()) for row in valid
        )
        request_rows.append(
            {
                "launched_epoch": int(launched),
                "completed_epoch": int(completed),
                "merge_lag_epochs": int(completed) - int(launched),
                "requested": int(requested),
                "produced": int(produced),
                "valid": int(valid_count),
                "issue_hof_best": issue_best,
                "request_best": request_best,
                "best_over_issue_hof": request_best / issue_best,
                "candidates_beating_issue_hof": sum(float(row[7]) < issue_best for row in valid),
                "candidates_ever_in_hof": inserted,
                "unique_topologies": len({str(row[4]) for row in valid}),
                "hof_topology_refinements": sum(str(row[4]) in issue_topologies for row in valid),
                "refinements_improving_same_topology": sum(
                    str(row[4]) in issue_best_by_topology
                    and float(row[7]) < issue_best_by_topology[str(row[4])]
                    for row in valid
                ),
                "elapsed_seconds": float(elapsed),
            }
        )

    final_epoch = max(by_epoch)
    final = min(by_epoch[final_epoch], key=lambda row: float(row[4]))
    final_identifier = (str(final[2]), str(final[3]))
    archive = [row for rows in by_epoch.values() for row in rows]

    def violates_initial_mask(parameters_json: object) -> bool:
        reactions = json.loads(str(parameters_json))
        added = reactions[int(fixed_reaction_count) :]
        return any(
            int(reaction["reaction_id"]) in fixed_reaction_ids
            for reaction in added
            if reaction.get("reaction_id") is not None
        )

    valid_candidates = [
        row
        for row in candidates
        if row[4] is not None and row[5] is not None and bool(row[6])
    ]
    final_snapshot = by_epoch[final_epoch]
    rl_origin_best = min(
        (float(row[4]) for row in archive if (str(row[2]), str(row[3])) not in llm_identifiers),
        default=float("nan"),
    )
    return {
        "requests": request_rows,
        "snapshot_count": len(by_epoch),
        "snapshot_min_epoch": min(by_epoch),
        "snapshot_max_epoch": final_epoch,
        "final_hof_best": float(final[4]),
        "final_exact_llm": final_identifier in llm_identifiers,
        "rl_origin_archive_best": rl_origin_best,
        "llm_initial_mask_violations": sum(
            violates_initial_mask(row[5]) for row in valid_candidates
        ),
        "final_hof_initial_mask_violations": sum(
            violates_initial_mask(row[3]) for row in final_snapshot
        ),
        "final_rank_one_initial_mask_violation": int(
            violates_initial_mask(final[3])
        ),
    }


def audit(campaign_base: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    stale_budget_labels: list[str] = []
    missing_backend_metadata: set[str] = set()
    missing_thinking_metadata: set[str] = set()
    run_rows: list[dict] = []
    manifests = {
        name: read_json(campaign_base / name / "campaign_manifest.json")
        for name in {campaign_name(task) for task in TASKS}
    }
    for task in TASKS:
        for seed in range(20):
            hybrid, database = hybrid_paths(task, seed, campaign_base)
            control = control_path(task, seed)
            campaign = campaign_name(task)
            manifest = manifests[campaign]
            required = [
                hybrid / "completed.json",
                hybrid / "config.json",
                hybrid / "progress.csv",
                hybrid / "best_network.json",
                control / "config.json",
                control / "progress.csv",
                control / "best_network.json",
                database,
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                errors.extend(f"missing: {path}" for path in missing)
                continue

            completed = read_json(hybrid / "completed.json")
            hybrid_config = read_json(hybrid / "config.json")
            control_config = read_json(control / "config.json")
            hybrid_progress = read_progress(hybrid / "progress.csv")
            control_progress = read_progress(control / "progress.csv")
            hybrid_network = read_json(hybrid / "best_network.json")
            control_network = read_json(control / "best_network.json")
            fixed_count = FIXED_REACTIONS[task]
            fixed_ids = frozenset(
                int(value)
                for value in hybrid_network.get("reaction_ids", [])[:fixed_count]
            )
            db = database_diagnostics(
                database,
                fixed_reaction_count=fixed_count,
                fixed_reaction_ids=fixed_ids,
            )

            label = f"{task}/seed{seed}"
            expected = {
                "method": completed.get("method"),
                "model": completed.get("model"),
                "generation_backend": manifest.get("generation_backend"),
                "thinking_mode": manifest.get("thinking_mode"),
                "communication_mode": completed.get("communication_mode"),
                "withhold_initial_hof": completed.get("withhold_initial_hof"),
            }
            if manifest.get("model") != "deepseek-v4-flash":
                errors.append(f"{label}: campaign model is not deepseek-v4-flash")
            if "generation_backend" not in manifest:
                missing_backend_metadata.add(campaign)
            elif manifest["generation_backend"] != "harness":
                errors.append(f"{label}: campaign backend is not Harness")
            if "thinking_mode" not in manifest:
                missing_thinking_metadata.add(campaign)
            elif manifest["thinking_mode"] != "provider-default":
                errors.append(f"{label}: campaign did not use provider-default inference")
            if ("task" in completed and completed["task"] != task) or int(
                completed.get("seed", -1)
            ) != seed:
                errors.append(f"{label}: completion task/seed mismatch")
            if completed.get("communication_mode") != "full" or not completed.get(
                "withhold_initial_hof"
            ):
                errors.append(f"{label}: not the selected full/context-free policy")
            if int(completed.get("rl_candidate_evaluations", -1)) != 102300:
                errors.append(f"{label}: hybrid RL evaluation count is not 102300")
            llm_evaluations = int(completed.get("llm_candidate_evaluations", -1))
            if llm_evaluations < 0 or llm_evaluations > 50 or llm_evaluations % 10:
                errors.append(f"{label}: invalid hybrid LLM evaluation count")
            if int(completed.get("candidate_evaluations", -1)) != 102300 + llm_evaluations:
                errors.append(f"{label}: inconsistent total hybrid evaluation count")
            if int(completed.get("budget_cap", -1)) != 102400:
                stale_budget_labels.append(label)
            if not close(float(completed["best_loss"]), float(hybrid_progress[-1]["best_so_far_loss"])):
                errors.append(f"{label}: completion/progress hybrid loss mismatch")
            if not close(float(completed["best_loss"]), db["final_hof_best"]):
                errors.append(f"{label}: completion/database final HOF mismatch")
            if db["snapshot_count"] != 101 or db["snapshot_min_epoch"] != 0 or db[
                "snapshot_max_epoch"
            ] != 100:
                errors.append(f"{label}: incomplete HOF snapshot series")
            if db["llm_initial_mask_violations"]:
                errors.append(
                    f"{label}: {db['llm_initial_mask_violations']} accepted LLM candidates "
                    "reuse an initially masked template reaction ID"
                )

            rl = hybrid_config.get("rl4crn", {})
            if int(rl.get("epochs", -1)) != 100 or int(rl.get("batch_size", -1)) != 1023:
                errors.append(f"{label}: hybrid RL epoch/batch mismatch")
            if int(hybrid_config.get("search", {}).get("seed", -1)) != seed:
                errors.append(f"{label}: hybrid config seed mismatch")
            if str(hybrid_config.get(task, {}).get("solver", "")).upper() != "CVODE":
                errors.append(f"{label}: hybrid solver is not CVODE")

            control_final = float(control_progress[-1]["best_so_far_loss"])
            if task not in {"rpa", "logic"}:
                control_completed = read_json(control / "completed.json")
                if int(control_completed.get("ode_simulations", -1)) != 102400:
                    errors.append(f"{label}: control evaluation count is not 102400")
                if not close(control_final, float(control_completed["best_loss"])):
                    errors.append(f"{label}: control completion/progress loss mismatch")
            control_rl = control_config.get("rl4crn", {})
            if int(control_rl.get("epochs", -1)) != 100 or int(
                control_rl.get("batch_size", -1)
            ) != 1024:
                errors.append(f"{label}: control RL epoch/batch mismatch")
            if int(control_config.get("search", {}).get("seed", -1)) != seed:
                errors.append(f"{label}: control config seed mismatch")
            if str(control_config.get(task, {}).get("solver", "")).upper() != "CVODE":
                errors.append(f"{label}: control solver is not CVODE")

            added = int(hybrid_config.get("search", {}).get("max_added_reactions", -1))
            out_of_llm_bounds = {}
            minimum_added_rate = {}
            for method, network in (("hybrid", hybrid_network), ("control", control_network)):
                reaction_ids = [int(value) for value in network.get("reaction_ids", [])]
                parameters = [float(value) for value in network.get("rate_constants", [])]
                if len(reaction_ids) != FIXED_REACTIONS[task] + added:
                    errors.append(f"{label}: {method} final reaction count mismatch")
                selected = reaction_ids[FIXED_REACTIONS[task] :]
                if len(selected) != len(set(selected)):
                    errors.append(f"{label}: {method} duplicate selected reaction IDs")
                selected_parameters = parameters[FIXED_REACTIONS[task] :]
                out_of_llm_bounds[method] = int(
                    any(not 0.1 <= value <= 50.0 for value in selected_parameters)
                )
                minimum_added_rate[method] = min(selected_parameters)

            request_rows = db["requests"]
            launched = [row["launched_epoch"] for row in request_rows]
            if any(epoch not in {0, 20, 40, 60, 80} for epoch in launched) or len(
                launched
            ) != len(set(launched)):
                errors.append(f"{label}: invalid served-request schedule")
            if any(row["requested"] != 10 or row["produced"] != 10 for row in request_rows):
                errors.append(f"{label}: request did not produce the requested batch of 10")
            failure_reasons = failed_batch_reasons(
                campaign_base / campaign / "logs" / f"{task}_seed{seed}.log"
            )
            if len(failure_reasons) != 5 - len(request_rows):
                errors.append(
                    f"{label}: database/log failed-batch count mismatch "
                    f"({5 - len(request_rows)} versus {len(failure_reasons)})"
                )
            run_rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "hybrid_loss": float(completed["best_loss"]),
                    "control_loss": control_final,
                    "final_exact_llm": int(db["final_exact_llm"]),
                    "rl_origin_archive_best": float(db["rl_origin_archive_best"]),
                    "llm_initial_mask_violations": db["llm_initial_mask_violations"],
                    "final_hof_initial_mask_violations": db[
                        "final_hof_initial_mask_violations"
                    ],
                    "final_rank_one_initial_mask_violation": db[
                        "final_rank_one_initial_mask_violation"
                    ],
                    "served_requests": len(request_rows),
                    "failed_batches": 5 - len(request_rows),
                    "failed_batch_reasons": failure_reasons,
                    "hybrid_final_outside_llm_bounds": out_of_llm_bounds["hybrid"],
                    "control_final_outside_llm_bounds": out_of_llm_bounds["control"],
                    "hybrid_minimum_added_rate": minimum_added_rate["hybrid"],
                    "control_minimum_added_rate": minimum_added_rate["control"],
                    "metadata": expected,
                    "requests": request_rows,
                }
            )

    if stale_budget_labels:
        warnings.append(
            f"{len(stale_budget_labels)} RPA artifacts retain the legacy budget_cap=307200 "
            "label although their effective schedule executed at most 102350 candidates"
        )
    if missing_backend_metadata:
        warnings.append(
            "legacy campaign manifest omits generation_backend: "
            + ", ".join(sorted(missing_backend_metadata))
            + "; Harness execution is evidenced by per-request workspaces/process records"
        )
    if missing_thinking_metadata:
        warnings.append(
            "legacy campaign manifest omits thinking_mode: "
            + ", ".join(sorted(missing_thinking_metadata))
            + "; no fixed reasoning-mode claim is supportable"
        )

    task_summary = []
    for task in TASKS:
        runs = [row for row in run_rows if row["task"] == task]
        requests = [request for row in runs for request in row["requests"]]
        failure_counts = Counter(
            reason for row in runs for reason in row["failed_batch_reasons"]
        )
        task_summary.append(
            {
                "task": task,
                "runs": len(runs),
                "requests": len(requests),
                "failed_batches": sum(row["failed_batches"] for row in runs),
                "failed_batch_reasons": dict(sorted(failure_counts.items())),
                "candidates_requested": sum(row["requested"] for row in requests),
                "candidates_produced": sum(row["produced"] for row in requests),
                "valid_candidates": sum(row["valid"] for row in requests),
                "median_merge_lag_epochs": float(
                    np.median([row["merge_lag_epochs"] for row in requests])
                ),
                "requests_beating_issue_hof": sum(
                    row["best_over_issue_hof"] < 1 for row in requests
                ),
                "median_request_best_over_issue_hof": float(
                    np.median([row["best_over_issue_hof"] for row in requests])
                ),
                "candidates_beating_issue_hof": sum(
                    row["candidates_beating_issue_hof"] for row in requests
                ),
                "candidates_ever_in_hof": sum(row["candidates_ever_in_hof"] for row in requests),
                "mean_unique_topologies_per_batch": float(
                    np.mean([row["unique_topologies"] for row in requests])
                ),
                "hof_topology_refinements": sum(
                    row["hof_topology_refinements"] for row in requests
                ),
                "refinements_improving_same_topology": sum(
                    row["refinements_improving_same_topology"] for row in requests
                ),
                "runs_final_exact_llm": sum(row["final_exact_llm"] for row in runs),
                "llm_initial_mask_violations": sum(
                    row["llm_initial_mask_violations"] for row in runs
                ),
                "final_hof_initial_mask_violations": sum(
                    row["final_hof_initial_mask_violations"] for row in runs
                ),
                "runs_rank_one_initial_mask_violation": sum(
                    row["final_rank_one_initial_mask_violation"] for row in runs
                ),
                "hybrid_finals_outside_llm_bounds": sum(
                    row["hybrid_final_outside_llm_bounds"] for row in runs
                ),
                "control_finals_outside_llm_bounds": sum(
                    row["control_final_outside_llm_bounds"] for row in runs
                ),
                "minimum_hybrid_final_added_rate": min(
                    row["hybrid_minimum_added_rate"] for row in runs
                ),
                "minimum_control_final_added_rate": min(
                    row["control_minimum_added_rate"] for row in runs
                ),
                "hybrid_wins": sum(row["hybrid_loss"] < row["control_loss"] for row in runs),
                "median_hybrid_loss": float(np.median([row["hybrid_loss"] for row in runs])),
                "median_control_loss": float(np.median([row["control_loss"] for row in runs])),
                "median_rl_origin_archive_best": float(
                    np.median([row["rl_origin_archive_best"] for row in runs])
                ),
            }
        )
    return {"errors": errors, "warnings": warnings, "tasks": task_summary, "runs": run_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-base", type=Path, default=CAMPAIGN_BASE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER / "generated/paper_experiment_audit.json",
    )
    args = parser.parse_args()
    result = audit(args.campaign_base.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {"errors": result["errors"], "warnings": result["warnings"], "tasks": result["tasks"]}
    print(json.dumps(compact, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
