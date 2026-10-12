"""Read-only export of audited, RL-context-free proposal streams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_from_audits(task: str, seed: int) -> Path:
    paths = ([GENERATED / "contract_v2_none_seed0_audit.json",
              GENERATED / "contract_v2_none_seeds1to19_audit.json"] if task == "classifier" else
             [GENERATED / f"contract_v2_local_qwen_{task}_no_communication_audit.json"])
    matches = []
    for path in paths:
        audit = json.loads(path.read_text())
        for run in audit["runs"]:
            if run["task"] == task and run["seed"] == seed and run["status"] == "pass":
                matches.append((path, run))
    if len(matches) != 1:
        raise ValueError(f"Expected one audited {task} No communication run for seed {seed}; found {len(matches)}.")
    _, run = matches[0]
    source = Path(run["run_root"])
    if sha256(source / "results.sqlite") != run["database_sha256"]:
        raise ValueError("Source database no longer matches its completed-run audit.")
    return source


def check_source_mode(manifest: dict, source_mode: str = "none") -> None:
    if source_mode not in ("none", "llm-to-rl"):
        raise ValueError("Only No communication or explicit insertion-only replay is supported.")
    insertion = source_mode == "llm-to-rl"
    if (manifest.get("communication_mode") != source_mode
            or manifest.get("expose_state_to_llm", False) is not False
            or manifest.get("insert_llm_into_rl", insertion) is not insertion):
        raise ValueError("Replay requires the selected No communication/insertion-only branch without outward conditioning.")


def export_replay(source: Path, *, source_mode: str = "none") -> dict:
    source = source.resolve()
    manifest = json.loads((source / "run_manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())
    # Preserve the none-only default; insertion sources require explicit opt-in.
    check_source_mode(manifest, source_mode)
    if manifest.get("max_agent_evaluations_per_round", 0) or manifest.get("forbidden_topology_m", 0):
        raise ValueError("Tool simulations and optimization-exclusion are not supported in this replay baseline.")
    task, seed = manifest["task"], int(manifest["seed"])
    if task not in ("logic", "rpa", "classifier") or config[task]["solver"].upper() != "CVODE":
        raise ValueError("This baseline supports canonical Logic, RPA, and Classifier CVODE tasks.")
    batch_size = int(config["rl4crn"]["batch_size"])
    epochs = int(config["rl4crn"]["epochs"])
    wal = source / "results.sqlite-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Refusing an immutable read of a database with a nonempty WAL.")
    connection = sqlite3.connect(f"{(source / 'results.sqlite').as_uri()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    events = []
    try:
        for row in connection.execute("SELECT * FROM llm_runs ORDER BY completed_epoch, launched_epoch"):
            if not 0 <= row["completed_epoch"] < epochs:
                raise ValueError("Source contains a round outside the training boundary; explicit terminal handling is required.")
            members = []
            for member in connection.execute(
                "SELECT * FROM llm_candidates WHERE llm_run_id=? ORDER BY candidate_index", (row["llm_run_id"],)
            ):
                if not member["valid"]:
                    raise ValueError("A stored evaluated candidate is invalid; inspect its evaluation accounting before replay.")
                candidate = json.loads(member["candidate_json"])
                if any(len(p) != 1 for p in candidate["parameter_values"]):
                    raise ValueError("Baseline requires one mass-action rate per added reaction.")
                members.append({
                    "reaction_ids": candidate["reaction_ids"],
                    "rates": [p[0] for p in candidate["parameter_values"]],
                    "proposal_id": f"{row['llm_run_id']}:{member['candidate_index']}",
                    "reference_loss": member["loss"], "topology_hash": member["topology_hash"],
                })
            if len(members) != row["accepted"]:
                raise ValueError("Accepted proposal count disagrees with candidate records.")
            events.append({
                "launched_epoch": row["launched_epoch"], "completed_epoch": row["completed_epoch"],
                # Trainer merges AFTER evaluating batch completed_epoch, BEFORE its policy update.
                "after_search_evaluations": (row["completed_epoch"] + 1) * batch_size,
                "requested": row["requested"], "returned": row["returned"],
                "accepted": row["accepted"], "rejected": row["rejected"], "candidates": members,
            })
        failures = [dict(r) for r in connection.execute("SELECT * FROM llm_failures")]
        failure_count = len(failures)
        failed_events = []
        for failure in failures:
            # A graph exception stores null counts; final evaluator accounting
            # below must still reconcile exactly with the successful records.
            if failure.get("accepted") not in (None, 0):
                raise ValueError("Failed source round contains accepted proposals requiring an accounting audit")
            failed_events.append({key: failure[key] for key in (
                "launched_epoch", "completed_epoch", "requested", "returned", "accepted", "rejected")})
        terminal = connection.execute(
            "SELECT * FROM hof_snapshots WHERE boundary='terminal_boundary' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if terminal is None or terminal["cumulative_rl_evaluations"] != epochs * batch_size:
            raise ValueError("Source is missing a complete terminal snapshot.")
        if sum(len(e["candidates"]) for e in events) != terminal["cumulative_llm_final_evaluations"]:
            raise ValueError("Replay count disagrees with source evaluation budget.")
        if terminal["cumulative_llm_tool_evaluations"]:
            raise ValueError("Source contains extra tool evaluations.")
        entries = [dict(r) for r in connection.execute(
            "SELECT topology_hash,loss FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank",
            (terminal["snapshot_id"],))]
    finally:
        connection.close()
    return {
        "schema_version": 1, "task": task, "seed": seed,
        "source_run": str(source), "source_database_sha256": sha256(source / "results.sqlite"),
        "source_config_sha256": sha256(source / "config.json"),
        "source_manifest_sha256": sha256(source / "run_manifest.json"),
        "model": manifest["model"], "communication_mode": manifest["communication_mode"],
        "expose_state_to_llm": False, "insert_llm_into_rl": source_mode == "llm-to-rl",
        "proposal_space": manifest["proposal_space"],
        "scientific_config": {"benchmark": {"task": task}, task: config[task],
                              "search": config["search"], "rl4crn": config["rl4crn"]},
        "search_budget": epochs * batch_size, "source_rl_batch_size": batch_size,
        "llm_evaluations": sum(len(e["candidates"]) for e in events),
        "source_failed_rounds": failure_count, "failed_events": failed_events, "events": events,
        "source_terminal_archive": entries,
    }
