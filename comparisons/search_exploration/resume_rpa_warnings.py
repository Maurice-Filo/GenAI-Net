"""Explicitly recover the held cohort with canonical-authoritative warning policy."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest, verify
from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json


def prepare(root):
    verify()
    plan = read_status(root / "active_plan.json")
    if plan.get("loss_disagreement_amendment"):
        raise ValueError("Warning-policy amendment already prepared")
    if digest(root / "active_plan.json") != (root / "active_plan.sha256").read_text().strip():
        raise ValueError("Prior plan hash mismatch")
    progress_path = root / plan["progress_file"]
    progress = read_status(progress_path)
    if progress["phase"] != "held" or progress["counts"]["running"]:
        raise ValueError("Recovery requires a held queue with no active jobs")
    failures = [j for j in progress["jobs"] if j["status"] == "failed"]
    if len(failures) != 1 or failures[0]["name"] != "julia-seed06":
        raise ValueError("Unexpected failures; requires individual review")
    failed_path = Path(failures[0]["output"])
    failure = read_status(failed_path / "failed.json")
    if "Julia/canonical loss mismatch" not in failure.get("message", ""):
        raise ValueError("Not the authorized recoverable disagreement")
    for filename, expected in plan["input_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Frozen input changed: {filename}")
    expected_changes = {"RL4CRN/search_streams/julia_stream.py",
        "comparisons/search_exploration/run_streams.py", "comparisons/search_exploration/analyze_rpa_statistical.py"}
    changed = {p for p, h in plan["source_hashes"].items() if digest(ROOT / p) != h}
    if changed - expected_changes:
        raise ValueError(f"Unexpected source edits: {changed - expected_changes}")
    folder = root / "amendments/canonical-warnings"
    folder.mkdir(parents=True, exist_ok=False)
    for path in (root / "active_plan.json", root / "active_plan.sha256", progress_path):
        shutil.copyfile(path, folder / path.name)
    recipe = read_status(root / "configs/julia_recipe.json")
    recipe["loss_disagreement_policy"] = "warn"
    recipe_path = folder / "julia_recipe.json"
    atomic_json(recipe_path, recipe)
    with sqlite3.connect(f"file:{failed_path / 'exploration.sqlite'}?mode=ro", uri=True) as db:
        keys = ("expensive_evaluations", "evaluation_attempts", "evaluations/search", "evaluations/optimization",
            "julia/reserved", "julia/joint_evaluations", "julia/unknown_evaluations_upper_bound", "julia/scenario_solves")
        counters = dict(db.execute("SELECT key,n FROM visits WHERE key IN (" + ",".join("?" for _ in keys) + ")", keys))
    overhead = counters["expensive_evaluations"]
    runtime = {j["name"]: j for j in progress["jobs"]}
    for job in plan["jobs"]:
        job.update(runtime[job["name"]])
        if job["status"] == "failed":
            job.setdefault("previous_attempts", []).append({"output": str(failed_path), "status": "failed",
                "charged_evaluations": overhead, "counters": counters, "failure": failure,
                "reason": "User-authorized warning-policy restart, no resumable asynchronous checkpoint"})
            output = root / "runs" / (job["name"] + "-canonical-warnings")
            if output.exists():
                raise ValueError("Recovery output already exists")
            job["command"][job["command"].index("--output")+1] = str(output)
            job.update(output=str(output), log_file=job["name"] + "-canonical-warnings.log", status="queued")
            for key in ("pid", "started", "finished", "returncode", "run_status"):
                job.pop(key, None)
        if job["status"] == "queued" and job["arm"] == "julia":
            job["command"][job["command"].index("--julia-recipe")+1] = str(recipe_path)
            job.update(recipe_path=str(recipe_path), loss_disagreement_policy="warn")
    plan.update(loss_disagreement_amendment=time.time(), adopted_jobs=[])
    plan["input_hashes"][str(recipe_path)] = digest(recipe_path)
    paths = {ROOT / p for p in plan["source_hashes"]} | {Path(__file__).resolve()}
    plan["source_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    atomic_json(root / "active_plan.json", plan)
    (root / "active_plan.sha256").write_text(digest(root / "active_plan.json") + "\n")
    (folder / "AMENDMENT.md").write_text(
        "# Canonical-Authoritative Score Disagreements\n\n"
        "User explicitly requested continuation without stopping on finite Julia/canonical score "
        "disagreements. Remaining Julia runs use loss_disagreement_policy=warn. Every discrepancy "
        "retains both losses, absolute/relative differences and severity. The canonical evaluator "
        "remains authoritative for archive scores, qualification and learning. The optimizer still "
        "uses its own Julia objective internally; numerical fidelity is not claimed from this policy. "
        "Nonfinite/negative scores, invalid simulations and genuine process/program errors are not "
        "converted into accepted solutions. Finite disagreement alone cannot abort a run.\n\n"
        "The original fatal seed-6 attempt remains untouched. Asynchronous state cannot be safely "
        "resumed from its saved checkpoint, so only that seed restarts in a distinct folder with its "
        f"unchanged full attempt ceiling. Prior charged work ({overhead} attempts) is reported as "
        "additional recovery overhead. Eighteen completed outcomes, including poor outcomes, are "
        "retained. No best-attempt selection. Four-run concurrency and disabled scientific timeouts "
        "remain. This is an explicitly amended cohort, not an unchanged preregistration.\n")
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
        plan = read_status(root / "active_plan.json") if args.execute_prepared else prepare(root)
        if not plan.get("loss_disagreement_amendment"):
            raise ValueError("Explicit warning amendment required")
        if not args.prepare_only:
            old_progress = root / "amendments/canonical-warnings" / plan["progress_file"]
            if (root / plan["progress_file"]).read_bytes() != old_progress.read_bytes():
                raise ValueError("Recovery already started; no automatic retry")
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"Controller received signal {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            execute(root, plan)


if __name__ == "__main__":
    main()
