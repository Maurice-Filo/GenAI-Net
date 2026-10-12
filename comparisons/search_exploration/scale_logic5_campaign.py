"""Resource-only scaling; original parent keeps tracking and reaping its children."""

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
    ROOT, balance_status, digest, read, request_audit, verify_inputs,
)
from comparisons.search_exploration.scale_rpa_statistical import AdoptedProcess
from RL4CRN.search_exploration.engine import atomic_json


def reconcile(plan, progress):
    if progress["status"] != "held" or progress.get("failed", 0):
        raise ValueError("Original controller must acknowledge HOLD without failures")
    runtime = {j["name"]: j for j in progress["jobs"]}
    if set(runtime) != {j["name"] for j in plan["jobs"]}:
        raise ValueError("Job identity mismatch")
    jobs = []
    for original in plan["jobs"]:
        current = runtime[original["name"]]
        for field in ("command", "output", "seed", "arm"):
            if current[field] != original[field]:
                raise ValueError(f"Job scientific identity changed: {field}")
        if current["status"] not in ("running", "queued", "complete"):
            raise ValueError("Unexpected job state")
        jobs.append({k: v for k, v in current.items() if k != "run_status"})
    return jobs


def live_identity(pid, command):
    process = psutil.Process(pid)
    if process.cmdline() != command or process.status() == psutil.STATUS_ZOMBIE:
        raise ValueError(f"Process identity mismatch: {pid}")
    return {"pid": pid, "created": process.create_time()}


def donor_alive(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.create_time() == identity["created"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def prepare(root, concurrency):
    if concurrency != 12:
        raise ValueError("Only the approved twelve-job limit is supported")
    plan = read(root / "plan.json")
    if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Plan changed")
    verify_inputs(plan)
    if (root / "amendments/concurrency12").exists() or (root / "HOLD").exists():
        raise ValueError("Existing amendment or HOLD requires explicit reconciliation")
    donor = psutil.Process(read(root / "started.json")["pid"])
    if "comparisons.search_exploration.logic5_four_arm_campaign" not in donor.cmdline():
        raise ValueError("Unexpected original controller")
    if str(root) not in donor.cmdline():
        raise ValueError("Original controller belongs to a different campaign")
    identity = {"pid": donor.pid, "created": donor.create_time()}
    balance = balance_status()
    if not balance["is_available"] or shutil.disk_usage(root).free < 30 * 2**30:
        raise RuntimeError("Balance/disk preflight failed")
    folder = root / "amendments/concurrency12"
    folder.mkdir(parents=True, exist_ok=False)
    # HOLD stops dispatch only: active experiments and original exit tracking continue.
    atomic_json(root / "HOLD", {"reason": "Queue ownership transferred to concurrency12 controller"})
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        progress = read(root / "progress.json")
        if progress["status"] == "held":
            break
        if not donor_alive(identity):
            raise RuntimeError("Original controller died before HOLD acknowledgement")
        time.sleep(2)
    else:
        raise RuntimeError("No HOLD acknowledgement; no jobs launched; HOLD retained")
    jobs = reconcile(plan, progress)
    adopted = {}
    for job in jobs:
        if job["status"] == "running":
            adopted[job["name"]] = live_identity(job["pid"], job["command"])
        elif job["status"] == "queued":
            if Path(job["output"]).exists() or (root / "logs" / f"{job['name']}.log").exists():
                raise ValueError(f"Queued job already has artifacts: {job['name']}")
    amendment = {"time": time.time(), "concurrency_before": plan["concurrency"],
        "effective_concurrency": concurrency, "donor_controller": identity, "adopted": adopted,
        "plan_sha256": digest(root / "plan.json"), "controller_sha256": digest(Path(__file__)),
        "adoption_source_sha256": digest(ROOT / "comparisons/search_exploration/scale_rpa_statistical.py"),
        "progress_file": "progress-concurrent.json", "hold_file": "HOLD_CONCURRENT",
        "scientific_settings_changed": False,
        "caveat": "Additional contention may affect asynchronous fit/LLM arrival and insertion epochs.",
        "jobs": jobs}
    atomic_json(folder / "amendment.json", amendment)
    atomic_json(folder / "donor-progress.json", progress)
    shutil.copy2(__file__, folder / "controller.py")
    return plan, amendment


def execute(root, plan, amendment):
    jobs, active, hold = amendment["jobs"], {}, False
    for job in jobs:
        if job["status"] == "running":
            active[job["name"]] = (AdoptedProcess(root / "progress.json", job), None, job)
    atomic_json(root / "amendments/concurrency12/started.json", {"pid": os.getpid(), "time": time.time()})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    next_balance, balance_verified = 0, False
    while True:
        outage = False
        for name, (process, log, job) in list(active.items()):
            audit = request_audit(Path(job["output"]))
            outage = outage or audit["balance_affected"]
            code = process.poll()
            if code is None:
                if getattr(process, "adopted", False) and not donor_alive(amendment["donor_controller"]):
                    hold = True
                    atomic_json(root / "concurrency_tracking_error.json", {
                        "reason": "Original parent unavailable: adopted exit status unknown", "time": time.time()})
                continue
            if log is not None:
                log.close()
            ok = code == 0 and read(Path(job["output"]) / "completed.json") and not audit["balance_affected"]
            job.update(status="complete" if ok else "failed", returncode=code,
                       finished=time.time(), request_audit=audit)
            hold = hold or not ok
            del active[name]
        if time.monotonic() >= next_balance:
            try:
                balance = balance_status()
                balance_verified = True
                atomic_json(root / "balance-concurrent.json", balance)
                outage = outage or not balance["is_available"]
            except Exception as error:
                balance_verified = False
                atomic_json(root / "balance_concurrent_error.json", {"type": type(error).__name__, "time": time.time()})
            next_balance = time.monotonic() + 60
        if outage:
            hold = True
            atomic_json(root / "provider_outage.json", {"reason": "insufficient balance", "time": time.time()})
            for process, _, job in active.values():
                if process.poll() is None and not job.get("outage_stop_requested"):
                    try:
                        if getattr(process, "adopted", False):
                            expected = amendment["adopted"][job["name"]]
                            if live_identity(process.pid, job["command"]) != expected:
                                raise ValueError("Adopted identity changed")
                        os.killpg(process.pid, signal.SIGTERM)
                    except psutil.NoSuchProcess:
                        pass
                    except ProcessLookupError:
                        pass
                    job["outage_stop_requested"] = True
        hold = hold or (root / "HOLD_CONCURRENT").exists() or (root / "provider_outage.json").exists()
        for job in jobs:
            if hold or not balance_verified or len(active) >= amendment["effective_concurrency"]:
                break
            if job["status"] != "queued":
                continue
            try:
                if digest(root / "plan.json") != amendment["plan_sha256"]:
                    raise ValueError("Frozen plan changed")
                verify_inputs(plan)
                if shutil.disk_usage(root).free < 30 * 2**30:
                    raise OSError("Fewer than 30 GiB free")
                if Path(job["output"]).exists():
                    raise ValueError("Queued output already exists")
            except (ValueError, OSError) as error:
                hold = True
                atomic_json(root / "dispatch_concurrent_blocked.json", {"reason": str(error), "time": time.time()})
                break
            log = (root / "logs" / f"{job['name']}.log").open("x")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status="running", pid=process.pid, started=time.time())
            active[job["name"]] = (process, log, job)
            print(f"Started {job['name']} pid={process.pid}", flush=True)
        counts = collections.Counter(j["status"] for j in jobs)
        state = "held" if hold else "running" if active else "waiting_for_balance" if counts["queued"] else "complete"
        atomic_json(root / "progress-concurrent.json", {"status": state, "updated": time.time(),
            "controller_pid": os.getpid(), "effective_concurrency": amendment["effective_concurrency"],
            "active": len(active), "completed": counts["complete"], "failed": counts["failed"], "queued": counts["queued"],
            "jobs": [{**j, "run_status": read(Path(j["output"]) / "status.json") or {}} for j in jobs]})
        if not active and (hold or not counts["queued"]):
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "concurrency.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, amendment = prepare(root, args.concurrency)
        execute(root, plan, amendment)


if __name__ == "__main__":
    main()
