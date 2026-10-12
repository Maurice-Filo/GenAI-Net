#!/usr/bin/env python3
"""Audit adopted Flash sentinels, then replace the paused queue supervisor."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from comparisons.llm_crn_generation.run_flash_half_duplex_queue import (
    CAMPAIGN_ROOT,
    MODES,
    PYTHON,
    ROOT,
    TASKS,
    audit_command,
    audit_path,
    campaign_id,
    environment,
    method_name,
    run_suffix,
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sentinel_state(task: str, mode: str) -> str:
    status_path = CAMPAIGN_ROOT / campaign_id(task, mode, "sentinel") / "status.json"
    status = read_json(status_path)
    if status.get("failed"):
        return "failed"
    if status.get("active") or status.get("pending"):
        return "running"
    completed = status.get("completed", [])
    if len(completed) == 1 and int(completed[0].get("returncode", 1)) == 0:
        return "complete"
    return "waiting"


def scientific_completion_exists(task: str, mode: str) -> bool:
    run_id = f"{task}_full102400_seed0_{run_suffix(mode)}"
    path = (
        ROOT
        / "comparisons/rpa_search/data/raw"
        / method_name(mode)
        / run_id
        / "completed.json"
    )
    return path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paused-supervisor-pid", type=int, required=True)
    parser.add_argument("--held-launcher-pid", type=int, required=True)
    parser.add_argument("--held-task", choices=TASKS, required=True)
    parser.add_argument("--held-mode", choices=MODES, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    while True:
        states: dict[str, str] = {
            f"{task}/{mode}": sentinel_state(task, mode)
            for task in TASKS
            for mode in MODES
        }
        held_key = f"{args.held_task}/{args.held_mode}"
        if scientific_completion_exists(args.held_task, args.held_mode):
            states[held_key] = "scientific-complete-launcher-held"
        print(f"[sentinel-handoff] {states}", flush=True)
        failed = [name for name, state in states.items() if state == "failed"]
        if failed:
            raise RuntimeError("Sentinel campaign failed: " + ", ".join(failed))
        if all(
            state == "complete" or (
                name == held_key and state == "scientific-complete-launcher-held"
            )
            for name, state in states.items()
        ):
            break
        time.sleep(args.poll_seconds)

    try:
        os.kill(args.paused_supervisor_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os.kill(args.held_launcher_pid, signal.SIGCONT)
    held_deadline = time.monotonic() + 120.0
    while sentinel_state(args.held_task, args.held_mode) != "complete":
        if time.monotonic() >= held_deadline:
            raise TimeoutError("Held sentinel launcher did not finalize its status.")
        time.sleep(1.0)

    for task in TASKS:
        for mode in MODES:
            completed = subprocess.run(
                audit_command(task, mode, "sentinel"),
                cwd=ROOT,
                env=environment(),
                check=False,
            )
            if completed.returncode or read_json(audit_path(task, mode, "sentinel")).get(
                "status"
            ) != "pass":
                raise RuntimeError(f"Sentinel audit failed: {task}/{mode}")
            print(f"[sentinel-audit-pass] {task}/{mode}", flush=True)

    print("[handoff] starting Logic-first Flash main queue", flush=True)
    queue = ROOT / "comparisons/llm_crn_generation/run_flash_half_duplex_queue.py"
    os.execve(
        str(PYTHON),
        [
            str(PYTHON),
            str(queue),
            "--skip-local-dependency",
            "--skip-sentinels",
        ],
        environment(),
    )


if __name__ == "__main__":
    main()
