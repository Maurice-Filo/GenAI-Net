"""Explicit balance-outage replacement; preserve complete runs and all old attempts."""

import argparse
import collections
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import psutil

from comparisons.search_exploration.logic5_four_arm_campaign import (
    ROOT, read, digest, verify_inputs, balance_status, request_audit,
)
from comparisons.search_exploration.recover_rpa_directional_balance import replace_output
from RL4CRN.search_exploration.engine import atomic_json


def reconcile(plan, progress, root, spill):
    if progress["status"] != "held" or progress["active"]:
        raise ValueError("Original campaign must be stopped and held")
    runtime = {j["name"]: j for j in progress["jobs"]}
    if set(runtime) != {j["name"] for j in plan["jobs"]}:
        raise ValueError("Job set changed")
    jobs = []
    for source in plan["jobs"]:
        old = runtime[source["name"]]
        if any(source[k] != old[k] for k in ("command", "output", "seed", "arm")):
            raise ValueError("Scientific job identity changed")
        status = old["status"]
        if status == "failed" and not (old.get("outage_stop_requested") and old.get("returncode") == -15):
            raise ValueError("Failure was not an outage interruption; explicit diagnosis required")
        if status not in ("complete", "failed", "queued"):
            raise ValueError(f"Unsupported prior status: {status}")
        output = Path(source["output"]) if status == "complete" else (
            root / "runs" / (source["name"] + "-balance-recovery") if status == "failed"
            else spill / "runs" / source["name"])
        jobs.append({**source, "output": str(output), "command": replace_output(source["command"], output),
            "status": "complete" if status == "complete" else "queued", "retained": status == "complete",
            "original_status": status, "original_output": source["output"],
            "recovery_kind": {"complete": "retained", "failed": "restart_epoch_zero", "queued": "first_attempt"}[status]})
    return jobs


def assert_stopped(original):
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if process.pid != os.getpid() and any(a == str(original) or a.startswith(str(original) + "/") for a in args):
            raise RuntimeError(f"Original campaign process still active: {process.pid}")


def prepare(original, root, spill):
    plan = read(original / "plan.json")
    if digest(original / "plan.json") != (original / "plan.sha256").read_text().strip():
        raise ValueError("Original plan changed")
    verify_inputs(plan)
    assert_stopped(original)
    if root.exists() or spill.exists() or (original / "balance_recovery.json").exists():
        raise ValueError("Recovery already exists; no automatic retry")
    if not balance_status()["is_available"]:
        raise RuntimeError("Provider balance unavailable")
    progress = read(original / "progress-concurrent.json")
    jobs = reconcile(plan, progress, root, spill)
    completed_hashes = {}
    for job in jobs:
        old = Path(job["original_output"])
        if job["retained"]:
            completion = read(old / "completed.json") or {}
            manifest = read(old / "manifest.json") or {}
            audit = request_audit(old)
            if completion.get("status") != "complete" or audit["balance_affected"]:
                raise ValueError(f"Retained endpoint is not clean: {old}")
            if manifest.get("communication_mode") != job["arm"] or manifest.get("config", {}).get("seed") != job["seed"]:
                raise ValueError(f"Retained endpoint identity mismatch: {old}")
            completed_hashes[str(old / "completed.json")] = digest(old / "completed.json")
            job["original_request_audit"] = audit
        elif job["original_status"] == "queued" and old.exists():
            raise ValueError(f"Queued job already has artifacts: {old}")
    for path in (root, spill):
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < 100 * 2**30:
            raise OSError(f"Require 100 GiB free on each destination: {parent}")
    root.mkdir(parents=True, exist_ok=False)
    spill.mkdir(parents=True, exist_ok=False)
    for path in (root / "runs", root / "logs", spill / "runs"):
        path.mkdir()
    recovery = {**plan, "id": root.name, "created": time.time(), "jobs": jobs, "concurrency": 12,
        "original_root": str(original), "spill_root": str(spill),
        "recovery_policy": "Retain 36 completed runs. Restart every outage-interrupted attempt from epoch zero. First attempts for queued seeds. No best-attempt selection; no automatic retries.",
        "cost_caveat": "Interrupted attempt costs are additional to new budgets and must remain in total campaign costs.",
        "source_hashes": {**plan["source_hashes"], str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__))},
        "input_hashes": {**plan["input_hashes"], **completed_hashes,
                         str(original / "plan.json"): digest(original / "plan.json")}}
    atomic_json(root / "original-progress.json", progress)
    atomic_json(root / "plan.json", recovery)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n")
    shutil.copy2(__file__, root / "recovery_controller.py")
    atomic_json(root / "progress.json", {"status": "prepared", "active": 0, "jobs": jobs, "updated": time.time()})
    atomic_json(original / "balance_recovery.json", {"recovery_root": str(root), "spill_root": str(spill),
        "policy": recovery["recovery_policy"], "created": time.time()})
    return recovery


def execute(root):
    plan = read(root / "plan.json")
    if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Recovery plan changed")
    verify_inputs(plan)
    assert_stopped(Path(plan["original_root"]))
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "started.json").exists():
        raise ValueError("Already started; explicit reconciliation required")
    balance = balance_status()
    if not balance["is_available"]:
        raise RuntimeError("Provider balance unavailable")
    atomic_json(root / "balance_initial.json", balance)
    atomic_json(root / "started.json", {"pid": os.getpid(), "time": time.time()})
    jobs, active, hold = plan["jobs"], {}, False
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    next_balance, balance_verified = 0, True
    while True:
        outage = False
        for name, (process, log, job) in list(active.items()):
            audit = request_audit(Path(job["output"]))
            outage = outage or audit["balance_affected"]
            if process.poll() is None:
                continue
            log.close()
            completed = read(Path(job["output"]) / "completed.json") or {}
            ok = process.returncode == 0 and completed.get("status") == "complete" and not audit["balance_affected"]
            job.update(status="complete" if ok else "failed", returncode=process.returncode,
                       finished=time.time(), request_audit=audit)
            hold = hold or not ok
            del active[name]
        if time.monotonic() >= next_balance:
            try:
                balance = balance_status()
                balance_verified = True
                atomic_json(root / "balance.json", balance)
                outage = outage or not balance["is_available"]
            except Exception as error:
                balance_verified = False
                atomic_json(root / "balance_check_error.json", {"type": type(error).__name__, "time": time.time()})
            next_balance = time.monotonic() + 60
        if outage:
            hold = True
            atomic_json(root / "provider_outage.json", {"reason": "insufficient balance", "time": time.time()})
            for process, _, job in active.values():
                if process.poll() is None and not job.get("outage_stop_requested"):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    job["outage_stop_requested"] = True
        hold = hold or (root / "HOLD").exists()
        for job in jobs:
            if hold or not balance_verified or len(active) >= plan["concurrency"]:
                break
            if job["status"] != "queued":
                continue
            try:
                verify_inputs(plan)
                if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
                    raise ValueError("Frozen recovery plan changed")
                for volume in (root, Path(plan["spill_root"])):
                    if shutil.disk_usage(volume).free < 30 * 2**30:
                        raise OSError(f"Fewer than 30 GiB free: {volume}")
                if Path(job["output"]).exists():
                    raise ValueError("Queued output already exists")
            except (ValueError, OSError) as error:
                hold = True
                atomic_json(root / "dispatch_blocked.json", {"reason": str(error), "time": time.time()})
                break
            log = (root / "logs" / f"{job['name']}.log").open("x")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status="running", pid=process.pid, started=time.time())
            active[job["name"]] = (process, log, job)
            print(f"Started {job['name']} pid={process.pid}", flush=True)
        counts = collections.Counter(j["status"] for j in jobs)
        state = "held" if hold else "running" if active else "waiting_for_balance" if counts["queued"] else "complete"
        atomic_json(root / "progress.json", {"status": state, "updated": time.time(), "controller_pid": os.getpid(),
            "active": len(active), "completed": counts["complete"], "retained": sum(j["retained"] for j in jobs),
            "failed": counts["failed"], "queued": counts["queued"],
            "jobs": [{**j, "run_status": read(Path(j["output"]) / "status.json") or {}} for j in jobs]})
        if not active and (hold or not counts["queued"]):
            return
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spill", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.original.resolve(), args.root.resolve(), args.spill.resolve())
    if args.run:
        execute(args.root.resolve())
