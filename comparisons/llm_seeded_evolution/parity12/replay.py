"""Freeze completed parity-12 insertion sources, including terminal-only replies."""

import json
import math
from pathlib import Path
import sqlite3

from comparisons.llm_seeded_evolution.replay import check_source_mode, sha256


def availability(completed_epoch, epochs, batch_size):
    if completed_epoch is None or completed_epoch < 0:
        raise ValueError("Missing or negative proposal completion epoch")
    return min(completed_epoch + 1, epochs) * batch_size


def check_config(config):
    expected = {"formula": "parity_4", "n_inputs": 4, "n_support_species": 2,
                "include_dilution": False, "solver": "CVODE", "rtol": 1e-8,
                "atol": 1e-8, "t_f": 100.0, "n_t": 1000,
                "ic": ["constant", 0.01], "weights": "transient", "library_order": 2}
    if any(config["logic_parity"].get(k) != v for k, v in expected.items()):
        raise ValueError("Parity evaluator configuration differs from the scheduled task")
    if (config["search"]["max_added_reactions"] != 12
            or config["rl4crn"]["epochs"] != 300
            or config["rl4crn"]["batch_size"] != 1023
            or config["rl4crn"]["hall_of_fame_size"] != 30):
        raise ValueError("Expected twelve added reactions, 300 x 1023 evaluations, and top-30 archive")


def export(source, expected_hash):
    source = Path(source).resolve()
    database = source / "results.sqlite"
    wal = source / "results.sqlite-wal"
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source still has an uncheckpointed WAL")
    if sha256(database) != expected_hash:
        raise ValueError("Source database differs from passing audit")
    manifest = json.loads((source / "run_manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())
    check_source_mode(manifest, "llm-to-rl")
    check_config(config)
    if (manifest["task"] != "logic_parity" or manifest["model"] != "deepseek-v4-flash"
            or manifest.get("max_agent_evaluations_per_round", 0)
            or manifest.get("forbidden_topology_m", 0)):
        raise ValueError("Wrong task/model or unmatched extra evaluations")
    contexts = []
    for workspace in (source / "harness-workspaces").iterdir():
        m = json.loads((workspace / "run_manifest.json").read_text())
        for name in ("CONTEXT/SEARCH_STATE.json", "CONTEXT/HALL_OF_FAME.json"):
            if sha256(workspace / name) != m["workspace_file_sha256"][name]:
                raise ValueError(f"Changed saved context: {workspace / name}")
        state = json.loads((workspace / "CONTEXT/SEARCH_STATE.json").read_text())
        hof = json.loads((workspace / "CONTEXT/HALL_OF_FAME.json").read_text())
        if hof["entries"]:
            raise ValueError("Insertion-only source exposed HOF entries")
        contexts.append(state["rl_epoch_at_snapshot"])
    if sorted(contexts) != list(range(0, 300, 20)):
        raise ValueError("Incomplete saved request-context schedule")
    events, failures = [], []
    with sqlite3.connect(f"{database.as_uri()}?mode=ro&immutable=1", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute("SELECT * FROM llm_runs ORDER BY completed_epoch, created_at"):
            members = []
            for member in connection.execute("SELECT * FROM llm_candidates WHERE llm_run_id=? ORDER BY candidate_index",
                                             (row["llm_run_id"],)):
                if not member["valid"]:
                    raise ValueError("Invalid evaluated source member requires accounting audit")
                candidate = json.loads(member["candidate_json"])
                ids, parameters = candidate["reaction_ids"], candidate["parameter_values"]
                if (len(ids) != 12 or len(set(ids)) != 12 or len(parameters) != 12
                        or set(ids) & set(manifest["proposal_space"]["forbidden_reaction_ids"])
                        or any(len(p) != 1 or not math.isfinite(p[0]) or p[0] <= 0 for p in parameters)):
                    raise ValueError("Source proposal violates the twelve-reaction mass-action contract")
                members.append({"reaction_ids": candidate["reaction_ids"],
                    "rates": [p[0] for p in candidate["parameter_values"]],
                    "proposal_id": f"{row['llm_run_id']}:{member['candidate_index']}",
                    "reference_loss": member["loss"], "topology_hash": member["topology_hash"]})
            if len(members) != row["accepted"]:
                raise ValueError("Source accepted count differs from evaluated candidates")
            events.append({"launched_epoch": row["launched_epoch"], "completed_epoch": row["completed_epoch"],
                "after_search_evaluations": availability(row["completed_epoch"], 300, 1023),
                "terminal_only": row["completed_epoch"] >= 300,
                "requested": row["requested"], "returned": row["returned"],
                "accepted": row["accepted"], "rejected": row["rejected"], "candidates": members})
        for row in connection.execute("SELECT * FROM llm_failures"):
            if row["accepted"] not in (None, 0):
                raise ValueError("Failed source round has accepted members requiring audit")
            failures.append({k: row[k] for k in ("launched_epoch", "completed_epoch", "requested",
                                               "returned", "accepted", "rejected")})
        if sorted(e["launched_epoch"] for e in events + failures) != list(range(0, 300, 20)):
            raise ValueError("Fifteen source request outcomes are not fully accounted for")
        terminal = connection.execute("SELECT * FROM hof_snapshots WHERE boundary='terminal_boundary' ORDER BY created_at DESC LIMIT 1").fetchone()
        count = sum(len(e["candidates"]) for e in events)
        if (terminal is None or terminal["cumulative_rl_evaluations"] != 306900
                or terminal["cumulative_llm_final_evaluations"] != count
                or terminal["cumulative_llm_tool_evaluations"]):
            raise ValueError("Source terminal budgets do not reconcile")
        archive = [dict(r) for r in connection.execute(
            "SELECT topology_hash,loss,parameters_json,task_info_json FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank",
            (terminal["snapshot_id"],))]
        if len(archive) != 30 or len({r["topology_hash"] for r in archive}) != 30:
            raise ValueError("Expected thirty distinct source terminal archive entries")
    return {"schema_version": "parity12-fixed-v1", "task": "logic_parity", "seed": manifest["seed"],
        "model": manifest["model"], "communication_mode": "llm-to-rl", "expose_state_to_llm": False,
        "insert_llm_into_rl": True, "source_run": str(source), "source_database_sha256": expected_hash,
        "source_config_sha256": sha256(source / "config.json"), "source_manifest_sha256": sha256(source / "run_manifest.json"),
        "scientific_config": {"benchmark": {"task": "logic_parity"}, "logic_parity": config["logic_parity"],
                              "search": config["search"], "rl4crn": config["rl4crn"]},
        "proposal_space": manifest["proposal_space"], "events": events, "failed_events": failures,
        "all_exposed_hof_empty": True, "search_budget": 306900, "source_rl_batch_size": 1023,
        "llm_evaluations": count, "source_terminal_archive": archive}
