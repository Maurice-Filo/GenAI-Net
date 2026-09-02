#!/usr/bin/env python3
"""Launch matched RL-only breadth runs with resumable campaign status."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.paper_breadth_tasks import DETERMINISTIC_TASKS

RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
RUNNER = ROOT / "comparisons/llm_crn_generation/run_breadth_rl_only.py"


def run_id(task: str, seed: int, budget: int, suffix: str) -> str:
    return f"{task}_full{budget}_seed{seed}_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--tasks", nargs="+", choices=DETERMINISTIC_TASKS, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--candidate-budget", type=int, default=102400)
    parser.add_argument("--max-parallel", type=int, default=15)
    parser.add_argument(
        "--global-slots",
        type=int,
        default=15,
        help="Cross-campaign cap for concurrent deterministic RL runs.",
    )
    parser.add_argument("--cpus-per-run", type=int, default=4)
    parser.add_argument("--rl-gpu", default=None)
    parser.add_argument("--method-name", default="rl4crn_breadth")
    parser.add_argument("--run-suffix", default="cvode_rl_only_breadth")
    parser.add_argument("--comet-project", default="genai-net-v4-flash-paper")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns",
    )
    args = parser.parse_args()
    if args.seed_start < 0 or min(
        args.seeds,
        args.epochs,
        args.batch_size,
        args.candidate_budget,
        args.max_parallel,
        args.global_slots,
        args.cpus_per_run,
    ) <= 0:
        parser.error("seed start must be non-negative and numeric settings must be positive")
    if args.epochs * args.batch_size != args.candidate_budget:
        parser.error("epochs * batch-size must exactly equal candidate-budget")

    campaign_root = args.workspace_root.expanduser().resolve() / args.campaign_id
    log_root = campaign_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    lock_root = RAW_ROOT / ".campaign_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    jobs = [(task, seed) for seed in seeds for task in args.tasks]

    def complete(task: str, seed: int) -> bool:
        return (
            RAW_ROOT
            / args.method_name
            / run_id(task, seed, args.candidate_budget, args.run_suffix)
            / "completed.json"
        ).is_file()

    pending = [job for job in jobs if not complete(*job)]
    done = [
        {"task": task, "seed": seed, "status": "already_complete"}
        for task, seed in jobs
        if complete(task, seed)
    ]
    failed: list[dict] = []
    active: dict[tuple[str, int], dict] = {}
    manifest = {
        "campaign_id": args.campaign_id,
        "candidate_budget": args.candidate_budget,
        "comet_project": args.comet_project,
        "cpus_per_run": args.cpus_per_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "method": args.method_name,
        "mode": "rl_only",
        "rl_batch_size": args.batch_size,
        "rl_gpu_assignment": args.rl_gpu,
        "run_suffix": args.run_suffix,
        "seeds": list(seeds),
        "solver": "CVODE",
        "tasks": args.tasks,
    }
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    if args.rl_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.rl_gpu
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    def write_status() -> None:
        payload = {
            "active": [
                {"task": task, "seed": seed, "pid": job["process"].pid}
                for (task, seed), job in active.items()
            ],
            "completed": done,
            "failed": failed,
            "pending": [{"task": task, "seed": seed} for task, seed in pending],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (campaign_root / "status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def stop_workers() -> None:
        for job in active.values():
            if job["process"].poll() is None:
                try:
                    os.killpg(job["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if not job["log_file"].closed:
                job["log_file"].close()
            job["run_lock"].close()
            job["slot_lock"].close()

    def try_lock(path: Path):
        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    def try_global_slot():
        for index in range(args.global_slots):
            handle = try_lock(lock_root / f"deterministic-rl-slot-{index}.lock")
            if handle is not None:
                return handle
        return None

    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                task, seed = pending.pop(0)
                if complete(task, seed):
                    done.append({"task": task, "seed": seed, "status": "already_complete"})
                    continue
                lock_name = run_id(task, seed, args.candidate_budget, args.run_suffix)
                run_lock = try_lock(lock_root / f"{args.method_name}-{lock_name}.lock")
                if run_lock is None:
                    pending.append((task, seed))
                    break
                slot_lock = try_global_slot()
                if slot_lock is None:
                    run_lock.close()
                    pending.insert(0, (task, seed))
                    break
                log_path = log_root / f"{task}_seed{seed}.log"
                log_file = log_path.open("w", encoding="utf-8")
                cmd = [
                    str(ROOT / ".venv/bin/python"),
                    str(RUNNER),
                    "--task", task,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--candidate-budget", str(args.candidate_budget),
                    "--n-cpus", str(args.cpus_per_run),
                    "--method-name", args.method_name,
                    "--run-suffix", args.run_suffix,
                    "--comet-project", args.comet_project,
                    "--trial-id", f"{args.campaign_id}-{task}-seed{seed}",
                    "--output-root", str(RAW_ROOT),
                ]
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
                    "run_lock": run_lock,
                    "slot_lock": slot_lock,
                }
                print(f"[start] {task} seed={seed} pid={process.pid}", flush=True)

            for key, job in list(active.items()):
                returncode = job["process"].poll()
                if returncode is None:
                    continue
                job["log_file"].close()
                job["run_lock"].close()
                job["slot_lock"].close()
                task, seed = key
                record = {
                    "task": task,
                    "seed": seed,
                    "returncode": returncode,
                    "log": job["log"],
                }
                (done if returncode == 0 and complete(task, seed) else failed).append(record)
                print(f"[{'done' if record in done else 'failed'}] {task} seed={seed}", flush=True)
                del active[key]
            write_status()
            if pending or active:
                time.sleep(5)
    finally:
        stop_workers()

    if failed:
        raise SystemExit(f"Campaign finished with {len(failed)} failed runs")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
