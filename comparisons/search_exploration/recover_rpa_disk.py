"""Explicit recovery of the September 16 disk-full interruption, after relocation."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest, verify
from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json

INTERRUPTED = {"ga-seed09", "ga-seed10", "julia-seed10", "evolution-seed10"}


def restart_job(job, root, counters, database_check):
    """Preserve the failed attempt and change only its output destination."""
    job = deepcopy(job)
    if job["status"] not in ("running", "failed", "interrupted"):
        raise ValueError("Only interrupted jobs can restart")
    output = root / "runs" / (job["name"] + "-disk-recovery")
    log_file = job["name"] + "-disk-recovery.log"
    if output.exists() or (root / "logs" / log_file).exists():
        raise ValueError("Recovery attempt already exists")
    saved = job.get("run_status", {}).get("metrics", {}).get("Compute/expensive_evaluations", 0)
    charged = max(int(counters.get("expensive_evaluations", 0)), int(saved))
    job.setdefault("previous_attempts", []).append({
        "output": job["output"], "log_file": job["log_file"],
        "status": "interrupted-disk-full", "charged_evaluations": charged,
        "charged_evaluations_are_lower_bound": True, "counters": counters,
        "database_check": database_check,
        "reason": "Disk full; no exact asynchronous resume. Same seed, fresh attempt; no best-attempt selection.",
    })
    job["command"][job["command"].index("--output") + 1] = str(output)
    job.update(output=str(output), log_file=log_file, status="queued")
    for key in ("pid", "started", "finished", "returncode", "run_status",
                "dispatch_worker_limit", "dispatch_lane_limits"):
        job.pop(key, None)
    return job


def inspect_database(folder):
    with sqlite3.connect(f"file:{folder / 'exploration.sqlite'}?mode=ro", uri=True) as db:
        result = [row[0] for row in db.execute("PRAGMA integrity_check")]
        counters = dict(db.execute("SELECT key,n FROM visits "
            "WHERE key NOT LIKE 'region:%' AND key NOT LIKE 'topology:%'"))
    return result, counters


def prepare(root):
    verify()
    plan = read_status(root / "active_plan.json")
    if plan.get("disk_recovery_amendment"):
        raise ValueError("Disk recovery already prepared")
    if digest(root / "active_plan.json") != (root / "active_plan.sha256").read_text().strip():
        raise ValueError("Prior plan hash mismatch")
    original = ROOT / "comparisons/search_exploration/reports" / plan["id"]
    if not original.is_symlink() or original.resolve() != root:
        raise ValueError("Verified relocation with original-path symlink required")
    if shutil.disk_usage(root).free < 250 * 2**30:
        raise OSError("At least 250 GiB free required at destination")
    for path, expected in plan["input_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    changed = {p for p, h in plan["source_hashes"].items() if digest(ROOT / p) != h}
    if changed - {"comparisons/search_exploration/analyze_rpa_statistical.py"}:
        raise ValueError(f"Unexpected source changes: {changed}")
    progress = read_status(root / plan["progress_file"])
    interrupted = {j["name"] for j in progress["jobs"] if j["status"] != "complete" and j["status"] != "queued"}
    if interrupted != INTERRUPTED:
        raise ValueError(f"Unexpected interrupted jobs: {interrupted}")
    if sum(j["status"] == "complete" for j in progress["jobs"]) != 26:
        raise ValueError("Expected 26 completed runs")
    from comparisons.search_exploration.analyze_rpa_statistical import endpoint
    checks, recovered = {}, []
    for job in progress["jobs"]:
        folder = Path(job.get("output", job["command"][job["command"].index("--output") + 1]))
        if job["status"] == "complete":
            result, _ = inspect_database(folder)
            row = endpoint(root, job, plan, job)
            if result != ["ok"] or row["status"] != "complete" or row["audit_error"]:
                raise ValueError(f"Completed result invalid: {job['name']}: {result}, {row}")
            checks[job["name"]] = {"integrity_check": result, "endpoint": row}
            print("Validated", job["name"], flush=True)
        elif job["name"] in INTERRUPTED:
            try:
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError(f"Process still exists: {job['pid']}; inspect before recovery")
            log = (root / "logs" / job["log_file"]).read_text(errors="replace")
            if "database or disk is full" not in log and "No space left on device" not in log:
                raise ValueError(f"No disk failure evidence: {job['name']}")
            if (folder / "completed.json").exists():
                raise ValueError("Interrupted job has completion marker; review individually")
            try:
                result, counters = inspect_database(folder)
            except sqlite3.Error as error:
                result, counters = [str(error)], {}
            checks[job["name"]] = {"integrity_check": result, "counters": counters}
            job = restart_job(job, root, counters, result)
            print("Preserved disk failure", job["name"], flush=True)
        elif job["status"] == "queued":
            if folder.exists():
                raise ValueError(f"Queued output exists: {folder}")
            job = deepcopy(job)
            output = root / "runs" / folder.name
            job["command"][job["command"].index("--output") + 1] = str(output)
            job["output"] = str(output)
        recovered.append(job)
    amendment = root / "amendments/disk-recovery"
    amendment.mkdir(parents=True, exist_ok=False)
    for path in (root / "active_plan.json", root / "active_plan.sha256", root / plan["progress_file"]):
        shutil.copyfile(path, amendment / path.name)
    atomic_json(amendment / "database_audit.json", checks)
    document = ROOT / "comparisons/search_exploration/RPA_DISK_RECOVERY.md"
    shutil.copyfile(document, amendment / "AMENDMENT.md")
    plan.update(jobs=recovered, adopted_jobs=[], disk_recovery_amendment={
        "time": time.time(), "root": str(root), "original_alias": str(original),
        "restarted_jobs": sorted(INTERRUPTED), "completed_retained": 26,
        "checkpoint_resume": False, "scientific_settings_changed": False,
    })
    paths = {ROOT / p for p in plan["source_hashes"]} | {Path(__file__).resolve(), document}
    with tarfile.open(amendment / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    plan["source_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    atomic_json(root / "active_plan.json", plan)
    (root / "active_plan.sha256").write_text(digest(root / "active_plan.json") + "\n")
    progress = {"updated": time.time(), "phase": "prepared", "hold_reason": None,
        "active_workers": 0, "counts": {"complete": 26, "queued": 34, "running": 0, "failed": 0}, "jobs": recovered}
    atomic_json(root / plan["progress_file"], progress)
    (amendment / "prepared_progress.sha256").write_text(digest(root / plan["progress_file"]) + "\n")
    return plan


def main():
    from comparisons.search_exploration.rpa_statistical_campaign import check_frozen, execute
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    with ExitStack() as stack:
        for name in ("controller.lock", "concurrency.lock"):
            lock = stack.enter_context((root / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            prepare(root)
            return
        plan = read_status(root / "active_plan.json")
        amendment = root / "amendments/disk-recovery"
        if not plan.get("disk_recovery_amendment"):
            raise ValueError("Recovery must be prepared first")
        if digest(root / plan["progress_file"]) != (amendment / "prepared_progress.sha256").read_text().strip():
            raise ValueError("Recovery already started or progress changed")
        check_frozen(root, plan)
        with (amendment / "started.json").open("x") as stream:
            import json
            json.dump({"time": time.time(), "pid": os.getpid()}, stream)
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Controller received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        execute(root, plan)


if __name__ == "__main__":
    main()
