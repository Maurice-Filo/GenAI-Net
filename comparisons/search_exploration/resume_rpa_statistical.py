"""Explicit no-scientific-deadline amendment; preserve old plans and attempts."""

import copy
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest, verify
from RL4CRN.search_exploration.engine import atomic_json


def audit_completed(path):
    completed = json.loads((path / "completed.json").read_text())
    if completed["status"] != "complete" or completed["pending_optimization_jobs"]:
        raise ValueError("Only fully drained completed runs may carry forward")
    with sqlite3.connect(f"file:{path / 'exploration.sqlite'}?mode=ro", uri=True) as db:
        if db.execute("SELECT count(*) FROM events WHERE kind='evaluation_failed'").fetchone()[0]:
            raise ValueError("Completed run contains evaluation failures; requires individual review")
        outcomes = [json.loads(r[0]) for r in db.execute("SELECT payload FROM events WHERE kind='optimization_result'")]
    if any(r.get("termination") not in ("completed", "evaluation_limit", "BudgetExhausted") for r in outcomes):
        raise ValueError("An optimizer may have been cut short; cannot silently carry forward")
    max_fit = max((r.get("elapsed_seconds", 0.) for r in outcomes), default=0.)
    for filename in (path / "julia_jobs").glob("*.json"):
        if filename.name == "workers.json":
            continue
        record = json.loads(filename.read_text())
        if record.get("fatal_error"):
            raise ValueError("Julia fatal error found")
        max_fit = max(max_fit, record.get("outcome", {}).get("job_seconds", 0.))
    manifest = json.loads((path / "manifest.json").read_text())
    if max_fit >= manifest["config"]["fit_seconds"]:
        raise ValueError("Old fitting deadline may have bound the completed run")
    return {"completed_sha256": digest(path / "completed.json"), "manifest_sha256": digest(path / "manifest.json"),
        "evaluation_failures": 0, "optimizer_jobs": len(outcomes), "max_fitting_seconds": max_fit,
        "conclusion": "No recorded evaluation failure or fitting deadline termination"}


def amended_job(root, original, runtime, folder):
    job = copy.deepcopy(original)
    job.update(runtime)
    if runtime["status"] == "complete":
        job["deadline_policy"] = "original-not-binding"
        job["completion_audit"] = audit_completed(root / "runs" / job["name"])
        return job
    if runtime["status"] not in ("queued", "failed"):
        raise ValueError("No active or unknown jobs may be amended")
    old = root / "runs" / job["name"]
    output = root / "runs" / (job["name"] + "-no-timeouts")
    if output.exists():
        raise ValueError("Refusing to overwrite a recovery attempt")
    if runtime["status"] == "failed":
        with sqlite3.connect(f"file:{old / 'exploration.sqlite'}?mode=ro", uri=True) as db:
            charged = db.execute("SELECT count(*) FROM events WHERE kind='evaluation_reserved'").fetchone()[0]
        job["previous_attempts"] = [{"output": str(old), "status": "failed", "charged_evaluations": charged,
            "failure": json.loads((old / "failed.json").read_text()), "reason": "User-authorized restart after timeout removal"}]
    command = list(original["command"])
    command[command.index("--output")+1] = str(output)
    if job["arm"] == "ga":
        command += ["--no-evaluation-timeouts"]
    else:
        cfg = json.loads(Path(command[command.index("--exploration-config")+1]).read_text())
        cfg.update(evaluation_seconds=None, fit_seconds=None)
        config_path = folder / (job["name"] + ".json")
        atomic_json(config_path, cfg)
        command[command.index("--exploration-config")+1] = str(config_path)
        job["config_path"] = str(config_path)
    for key in ("pid", "started", "finished", "returncode", "run_status"):
        job.pop(key, None)
    job.update(command=command, output=str(output), status="queued", deadline_policy="disabled",
        log_file=job["name"] + "-no-timeouts.log")
    return job


def amend(root):
    root = Path(root).resolve()
    verify()
    if (root / "active_plan.json").exists():
        raise ValueError("Amendment already exists; use --execute-prepared, not another amendment")
    plan = json.loads((root / "plan.json").read_text())
    if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Original plan hash mismatch")
    progress = json.loads((root / "progress.json").read_text())
    if progress["phase"] != "held" or progress["counts"]["running"]:
        raise ValueError("Expected held campaign with no active runs")
    for path, expected in plan["input_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Original input changed: {path}")
    folder = root / "amendments/no-timeouts"
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(root / "progress.json", folder / "progress-before.json")
    runtime = {j["name"]: j for j in progress["jobs"]}
    old_hashes = dict(plan["source_hashes"])
    plan["jobs"] = [amended_job(root, j, runtime[j["name"]], folder) for j in plan["jobs"]]
    plan.update(plan_file="active_plan.json", amendment="no-scientific-wall-clock-cutoffs", amended_at=time.time(),
        original_plan_sha256=digest(root / "plan.json"),
        failure_policy="Original timeout preserved; one explicitly authorized seed-2 retry; future errors hold dispatch")
    paths = {ROOT / p for p in old_hashes} | {Path(__file__).resolve()}
    plan["source_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    plan["changed_sources"] = {k: [old_hashes.get(k), v] for k, v in plan["source_hashes"].items() if old_hashes.get(k) != v}
    for path in folder.glob("*.json"):
        plan["input_hashes"][str(path)] = digest(path)
    # Completed inputs are never rewritten; anchor the retained run identities.
    for job in plan["jobs"]:
        if job["status"] == "complete":
            for name in ("manifest.json", "completed.json"):
                path = root / "runs" / job["name"] / name
                plan["input_hashes"][str(path)] = digest(path)
    atomic_json(root / "active_plan.json", plan)
    (root / "active_plan.sha256").write_text(digest(root / "active_plan.json") + "\n")
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    retained = sum(j["status"] == "complete" for j in plan["jobs"])
    overhead = sum(a["charged_evaluations"] for j in plan["jobs"] for a in j.get("previous_attempts", []))
    text = ["# Amendment: Remove Scientific Wall-Clock Cutoffs", "",
        "Authorized by the user on 2026-09-16 after GA seed 2 exceeded its 30-second evaluation limit.", "",
        "The original plan, source snapshot, progress snapshot, logs and every run remain preserved. "
        "active_plan.json supersedes dispatch only; original plan.json remains unchanged.", "",
        "Remaining evaluations and fitting jobs have no elapsed-time cutoff (JSON null / Python None). "
        "Worker/cache waits also no longer cap evaluation time. Startup and process-termination health checks "
        "remain separate from scientific budgets. Solver numerical limits, error reporting, rate bounds, "
        "evaluation-count ceilings, epoch counts and fitting-call limits are unchanged. Julia's existing "
        "20% measured fitting-time admission heuristic is unchanged; it schedules work but cannot terminate "
        "an evaluation. Therefore this remains a common-ceiling, not equal actual work, comparison.", "",
        f"{retained} completed runs are retained after checking evaluation/fitting failure records. "
        "No recorded cutoff bound those runs. The old deadline policy remains labeled in the CSV. "
        "GA seed 2 restarts from its original seed in a separate output folder because there is no "
        f"resumable GA checkpoint. Its {overhead} previously charged evaluations remain reported as overhead, "
        "not hidden or subtracted from its new 214793-attempt budget. This is not a fixed total campaign "
        "compute budget including aborted attempts. No bad completed result is replaced.", "",
        "This is a transparent mid-campaign amendment, not an unchanged preregistration. All 20 matched "
        "seeds, planned endpoints/tests, source hashing, error holds, and local-only logging remain.", "",
        "## Retained Run Audit", ""]
    for job in plan["jobs"]:
        if job["status"] == "complete":
            text.append(f"- {job['name']}: {json.dumps(job['completion_audit'], sort_keys=True)}")
    (folder / "AMENDMENT.md").write_text("\n".join(text) + "\n")
    return plan
