"""Increase dispatch concurrency by adopting, never interrupting, the active pair."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import signal
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest, verify
from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json


class AdoptedProcess:
    """The original parent reports the real exit status and reaps its own child."""
    adopted = True

    def __init__(self, donor_progress, job):
        self.donor_progress = donor_progress
        self.name, self.pid = job["name"], job["pid"]
        self.returncode = None

    def poll(self):
        progress = read_status(self.donor_progress)
        if progress is None:
            raise RuntimeError("Original controller progress unavailable")
        job = next(j for j in progress["jobs"] if j["name"] == self.name)
        if job.get("pid") != self.pid:
            raise RuntimeError("Adopted worker identity changed")
        if job["status"] in ("complete", "failed", "interrupted"):
            self.returncode = job.get("returncode", 1)
        elif job["status"] != "running":
            raise RuntimeError("Unexpected adopted job state")
        return self.returncode


def prepare(root):
    verify()
    progress = read_status(root / "progress.json")
    if progress["phase"] != "held" or not (root / "HOLD").exists():
        raise ValueError("Original controller must acknowledge HOLD before handoff")
    if progress["counts"]["failed"]:
        raise ValueError("Resolve failures before scaling")
    plan = read_status(root / "active_plan.json")
    if plan.get("concurrency_amendment"):
        raise ValueError("Concurrency amendment already prepared")
    if digest(root / "active_plan.json") != (root / "active_plan.sha256").read_text().strip():
        raise ValueError("Prior plan hash mismatch")
    for filename, expected in plan["input_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Frozen scientific input changed: {filename}")
    expected_changes = {"comparisons/search_exploration/rpa_statistical_campaign.py",
        "comparisons/search_exploration/analyze_rpa_statistical.py"}
    changes = {p for p, h in plan["source_hashes"].items() if digest(ROOT / p) != h}
    if changes - expected_changes:
        raise ValueError(f"Unexpected source edits: {changes - expected_changes}")
    folder = root / "amendments/concurrency"
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("active_plan.json", "active_plan.sha256", "progress.json"):
        shutil.copyfile(root / name, folder / name)
    runtime = {j["name"]: j for j in progress["jobs"]}
    for job in plan["jobs"]:
        job.update(runtime[job["name"]])
        if job["status"] == "queued":
            path = Path(job.get("output", root / "runs" / job["name"]))
            if path.exists():
                raise ValueError(f"Queued job already has an output: {path}")
    plan.update(worker_limit=128, lane_limits={"hybrid": 2, "ga": 2}, concurrency_amendment=time.time(),
        progress_file="progress-concurrent.json", analysis_dir="analysis-concurrent", hold_file="HOLD_CONCURRENT",
        adopted_jobs=[j["name"] for j in plan["jobs"] if j["status"] == "running"])
    paths = {ROOT / p for p in plan["source_hashes"]} | {Path(__file__).resolve()}
    plan["source_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    atomic_json(root / "active_plan.json", plan)
    (root / "active_plan.sha256").write_text(digest(root / "active_plan.json") + "\n")
    (folder / "AMENDMENT.md").write_text(
        "# Concurrency Amendment: Four Overlapping Runs\n\n"
        "User requested higher utilization on 2026-09-16. Host: 64 physical cores / 128 hardware threads; "
        "203 GiB available RAM; sampled CPU utilization 37.7% of logical capacity.\n\n"
        "Allow two hybrids (48 workers each) and two GAs (16 each): at most 128 worker slots, "
        "plus coordinators/training. No per-run worker counts, seeds, budgets, fitting recipe or "
        "scientific source files changed. Workloads may compete during busy phases, and both runtime "
        "and asynchronous fitting admission depend on load; performance equality is not assumed.\n\n"
        "The original controller acknowledged HOLD and only drains its existing jobs. The new "
        "controller adopts their PIDs and reads their true exit statuses from the original parent. "
        "It owns all queued jobs. No evaluator process was stopped, replaced or restarted.\n\n"
        "Canonical live state: progress-concurrent.json. Current report: analysis-concurrent/REPORT.md. "
        "The old progress/report continue to describe the draining controller and are historical. "
        "Create HOLD_CONCURRENT to hold new dispatch while active jobs finish.\n\n"
        "Original active plan/progress/source snapshots are retained in this folder. Errors, source "
        "changes and low disk still hold dispatch. Scientific timeouts remain disabled.\n")
    return plan


def main():
    from comparisons.search_exploration.rpa_statistical_campaign import execute
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "concurrency.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "progress-concurrent.json").exists():
            raise ValueError("Concurrent controller already started; no automatic restart")
        plan = read_status(root / "active_plan.json") if args.execute_prepared else prepare(root)
        if not plan.get("concurrency_amendment"):
            raise ValueError("Concurrency amendment required")
        if not args.prepare_only:
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"Controller received signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            execute(root, plan)


if __name__ == "__main__":
    main()
