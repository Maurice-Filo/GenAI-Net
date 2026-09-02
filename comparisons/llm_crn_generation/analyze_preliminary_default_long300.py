#!/usr/bin/env python3
"""Create the gated five-seed checkpoint for the clean long-300 campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
)
FLASH_ROOT = CAMPAIGN_ROOT / "flash-default-initial-hof-withheld-long300-20seed"
LOCAL_ROOT = (
    CAMPAIGN_ROOT / "local-qwen35-9b-rpa-default100-recovered-20seed-r3"
)
NO_COMM_CSV = (
    ROOT
    / "paper/iclr2027_genai_net_llm/figures/communication_ablation_over_time_20seed.csv"
)
RL_ROOT = ROOT / "comparisons/rpa_search/data/raw/rl4crn"
TASKS = ("rpa", "logic")
SEEDS = tuple(range(5))
SNAPSHOT_CUTOFF = 100
REQUEST_LAUNCH_CUTOFF = 100

FLASH_SUFFIX = "cvode_llm_flash_default_long300"
LOCAL_SUFFIX = "cvode_llm_qwen35_9b_rpa_default100_recovered"

LAUNCH_RE = re.compile(r"^\[epoch (\d+)\] launched LLM graph")
MERGE_RE = re.compile(
    r"^\[epoch (\d+)\] merged background LLM proposal launched at epoch (\d+)"
    r" \| requested=(\d+) \| valid=(\d+)"
)
FAIL_RE = re.compile(r"^\[epoch (\d+)\] LLM graph failed after epoch (\d+):")


def connect_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    try:
        connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        # SQLite opens lazily, so force a schema read while fallback is possible.
        connection.execute("PRAGMA schema_version").fetchone()
        return connection
    except sqlite3.OperationalError:
        if "connection" in locals():
            connection.close()
        # Restricted monitors cannot create SQLite's read-only WAL bookkeeping.
        # Immutable mode reads the latest checkpointed state without sidecar access.
        return sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)


def database_path(
    campaign_root: Path, task: str, seed: int, suffix: str, budget: int
) -> Path:
    run_id = f"{task}_full{budget}_seed{seed}_{suffix}"
    matches = sorted((campaign_root / "runs").glob(f"*/{run_id}/results.sqlite"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one database for {run_id}, found {len(matches)}")
    return matches[0]


def snapshot_state(path: Path, snapshot: int | None = None) -> dict[str, float | int]:
    with connect_readonly(path) as connection:
        maximum = connection.execute("SELECT MAX(epoch) FROM hof_snapshots").fetchone()[0]
        if maximum is None:
            return {"max_snapshot": -1, "snapshot": -1, "best_loss": float("nan")}
        selected = int(maximum if snapshot is None else snapshot)
        row = connection.execute(
            """SELECT MIN(e.loss)
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE h.epoch = ? AND e.loss IS NOT NULL""",
            (selected,),
        ).fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"Snapshot {selected} has no valid HOF entries in {path}")
    return {
        "max_snapshot": int(maximum),
        "snapshot": selected,
        "best_loss": float(row[0]),
    }


def request_counts(log_path: Path, launch_cutoff: int | None = None) -> dict[str, int]:
    launched: list[int] = []
    merged: list[tuple[int, int, int]] = []
    failed: list[int] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := LAUNCH_RE.match(line):
            launched.append(int(match.group(1)))
        elif match := MERGE_RE.match(line):
            merged.append(
                (int(match.group(2)), int(match.group(3)), int(match.group(4)))
            )
        elif match := FAIL_RE.match(line):
            failed.append(int(match.group(2)))
    if launch_cutoff is not None:
        launched = [epoch for epoch in launched if epoch < launch_cutoff]
        merged = [row for row in merged if row[0] < launch_cutoff]
        failed = [epoch for epoch in failed if epoch < launch_cutoff]
    return {
        "launched": len(launched),
        "served_and_merged": len(merged),
        "failed": len(failed),
        "pending": max(0, len(launched) - len(merged) - len(failed)),
        "requested_candidates": sum(row[1] for row in merged),
        "valid_candidates": sum(row[2] for row in merged),
    }


def rl_control_loss(task: str, seed: int) -> float:
    path = RL_ROOT / f"{task}_full102400_seed{seed}_cvode" / "progress.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    endpoints = [row for row in rows if int(row["step"]) == 100]
    if len(endpoints) != 1:
        raise RuntimeError(f"Expected one step-100 endpoint in {path}")
    return float(endpoints[0]["best_so_far_loss"])


def no_communication_losses() -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    with NO_COMM_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] == "independent_pool"
                and int(row["epoch"]) == SNAPSHOT_CUTOFF
                and int(row["seed"]) in SEEDS
            ):
                result[(row["task"], int(row["seed"]))] = float(row["best_loss"])
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    if set(result) != expected:
        raise RuntimeError("The audited no-communication CSV lacks matched checkpoint rows")
    return result


def campaign_failures(root: Path) -> list[dict]:
    path = root / "status.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("failed", []))


def flash_readiness() -> tuple[bool, list[dict]]:
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            path = database_path(FLASH_ROOT, task, seed, FLASH_SUFFIX, 307200)
            state = snapshot_state(path)
            rows.append({"task": task, "seed": seed, **state})
    return all(row["max_snapshot"] >= SNAPSHOT_CUTOFF for row in rows), rows


def flash_checkpoint() -> dict:
    controls = no_communication_losses()
    rows = []
    request_totals = {
        task: {
            "launched": 0,
            "served_and_merged": 0,
            "failed": 0,
            "pending": 0,
            "requested_candidates": 0,
            "valid_candidates": 0,
        }
        for task in TASKS
    }
    for task in TASKS:
        for seed in SEEDS:
            database = database_path(FLASH_ROOT, task, seed, FLASH_SUFFIX, 307200)
            hybrid = float(snapshot_state(database, SNAPSHOT_CUTOFF)["best_loss"])
            rl = rl_control_loss(task, seed)
            isolated = controls[(task, seed)]
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "default_full_duplex": hybrid,
                    "rl_only": rl,
                    "no_communication_pool": isolated,
                    "beats_rl_only": hybrid < rl,
                    "beats_no_communication": hybrid < isolated,
                }
            )
            counts = request_counts(
                FLASH_ROOT / "logs" / f"{task}_seed{seed}.log",
                REQUEST_LAUNCH_CUTOFF,
            )
            for key, value in counts.items():
                request_totals[task][key] += value
    summaries = {}
    for task in TASKS:
        subset = [row for row in rows if row["task"] == task]
        summaries[task] = {
            "n": len(subset),
            "median_default_full_duplex": median(
                row["default_full_duplex"] for row in subset
            ),
            "median_rl_only": median(row["rl_only"] for row in subset),
            "median_no_communication_pool": median(
                row["no_communication_pool"] for row in subset
            ),
            "wins_vs_rl_only": sum(row["beats_rl_only"] for row in subset),
            "wins_vs_no_communication": sum(
                row["beats_no_communication"] for row in subset
            ),
            "requests": request_totals[task],
        }
    return {"rows": rows, "summaries": summaries}


def candidate_identifiers(connection: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = connection.execute(
        """SELECT c.topology_hash, e.parameters_json
             FROM llm_candidates c
             JOIN evaluations e
               ON e.source = 'llm'
              AND e.topology_hash = c.topology_hash
              AND json_extract(e.metadata_json, '$.llm_run_id') = c.llm_run_id
              AND json_extract(e.metadata_json, '$.candidate_index') = c.candidate_index
            WHERE c.valid = 1 AND c.loss IS NOT NULL"""
    ).fetchall()
    return {(str(topology), str(parameters)) for topology, parameters in rows}


def local_run_summary(path: Path, seed: int) -> dict:
    state = snapshot_state(path)
    with connect_readonly(path) as connection:
        llm_ids = candidate_identifiers(connection)
        hof_ids = {
            (str(topology), str(parameters))
            for topology, parameters in connection.execute(
                "SELECT topology_hash, parameters_json FROM hof_snapshot_entries"
            )
        }
        best_llm = connection.execute(
            "SELECT MIN(loss) FROM llm_candidates WHERE valid = 1 AND loss IS NOT NULL"
        ).fetchone()[0]
        latest_rank_one = connection.execute(
            """SELECT e.topology_hash, e.parameters_json
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE h.epoch = ?
                ORDER BY e.rank LIMIT 1""",
            (state["max_snapshot"],),
        ).fetchone()
    counts = request_counts(LOCAL_ROOT / "logs" / f"rpa_seed{seed}.log")
    rank_one_id = (
        (str(latest_rank_one[0]), str(latest_rank_one[1])) if latest_rank_one else None
    )
    return {
        "seed": seed,
        **state,
        **counts,
        "unique_valid_llm_candidates": len(llm_ids),
        "unique_llm_candidates_entering_hof": len(llm_ids & hof_ids),
        "best_direct_llm_loss": float(best_llm) if best_llm is not None else None,
        "current_rank_one_is_direct_llm": rank_one_id in llm_ids,
    }


def local_checkpoint() -> dict:
    rows = []
    pattern = re.compile(r"rpa_full102400_seed(\d+)_" + re.escape(LOCAL_SUFFIX))
    for path in sorted((LOCAL_ROOT / "runs").glob("*/rpa_full102400_seed*/results.sqlite")):
        match = pattern.search(str(path))
        if match:
            rows.append(local_run_summary(path, int(match.group(1))))
    status = json.loads((LOCAL_ROOT / "status.json").read_text(encoding="utf-8"))
    return {
        "completed": len(status.get("completed", [])),
        "active": len(status.get("active", [])),
        "pending": len(status.get("pending", [])),
        "failures": list(status.get("failed", [])),
        "runs_started": rows,
    }


def format_loss(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def write_markdown(payload: dict, output: Path) -> None:
    lines = [
        "# Interim 100-Epoch Checkpoint",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "> Preliminary five-seed evidence from active campaigns. These are not terminal results.",
        "",
        "## Flash default full duplex",
        "",
        "| Task | Default median | RL-only median | No-com median | Wins vs RL | Wins vs no-com |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = payload["flash"]["summaries"][task]
        lines.append(
            f"| {task.upper()} | {format_loss(row['median_default_full_duplex'])} | "
            f"{format_loss(row['median_rl_only'])} | "
            f"{format_loss(row['median_no_communication_pool'])} | "
            f"{row['wins_vs_rl_only']}/{row['n']} | "
            f"{row['wins_vs_no_communication']}/{row['n']} |"
        )
    lines.extend(
        [
            "",
            "Request accounting includes only the five scheduled calls launched before epoch 100.",
            "",
            "| Task | Launched | Served/merged | Failed | Pending | Valid candidates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        row = payload["flash"]["summaries"][task]["requests"]
        lines.append(
            f"| {task.upper()} | {row['launched']} | {row['served_and_merged']} | "
            f"{row['failed']} | {row['pending']} | {row['valid_candidates']} |"
        )
    lines.extend(
        [
            "",
            "## Local Qwen3.5-9B",
            "",
            f"Completed: {payload['local']['completed']}; active: {payload['local']['active']}; "
            f"pending: {payload['local']['pending']}; failures: {len(payload['local']['failures'])}.",
            "",
            "| Seed | Snapshot | Current loss | Valid proposals | Entered HOF | Best direct LLM | LLM rank one |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in payload["local"]["runs_started"]:
        lines.append(
            f"| {row['seed']} | {row['max_snapshot']} | {format_loss(row['best_loss'])} | "
            f"{row['valid_candidates']} | {row['unique_llm_candidates_entering_hof']} | "
            f"{format_loss(row['best_direct_llm_loss'])} | "
            f"{'yes' if row['current_rank_one_is_direct_llm'] else 'no'} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "comparisons/llm_crn_generation/PRELIMINARY_DEFAULT_LONG300_CHECKPOINT.json",
    )
    args = parser.parse_args()
    ready, readiness = flash_readiness()
    if not ready:
        print(
            json.dumps(
                {
                    "ready": False,
                    "required_snapshot": SNAPSHOT_CUTOFF,
                    "flash": readiness,
                    "flash_failures": campaign_failures(FLASH_ROOT),
                    "local": local_checkpoint(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    payload = {
        "ready": True,
        "interim": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_cutoff": SNAPSHOT_CUTOFF,
        "eligible_request_launch_epochs": [0, 20, 40, 60, 80],
        "flash_failures": campaign_failures(FLASH_ROOT),
        "flash": flash_checkpoint(),
        "local": local_checkpoint(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(payload, output.with_suffix(".md"))
    print(output)
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
