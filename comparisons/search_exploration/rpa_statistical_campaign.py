"""Frozen three-workflow RPA comparison, with bounded concurrent worker lanes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, DEFAULT, digest, verify
from comparisons.search_exploration.nn_size_quality import read_status
from RL4CRN.search_exploration.engine import atomic_json

REFERENCE = Path("/local1/rossin/search-exploration/rpa-streams-v2-100epoch-20260914/run")
SEEDS = list(range(1, 21))
BUDGET = 214793
ARMS = ("julia", "evolution", "ga")


def configuration(reference, seed, arm):
    if reference["schema"] != "search-streams-v2" or reference["task"] != "rpa":
        raise ValueError("Expected corrected RPA reference")
    cfg = dict(reference["config"])
    if (cfg["epochs"], cfg["batch_size"], reference["policy"]["deep_layer_size"]) != (100, 1023, 1024):
        raise ValueError("Unexpected development settings")
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("Unplanned seed or arm")
    cfg.update(seed=seed, evaluation_budget=BUDGET, fit_evaluations=24 if arm == "julia" else 16)
    return cfg


def jobs_for(root, project):
    jobs = []
    for seed in SEEDS:
        # Counterbalance which hybrid runs first within each seed block.
        arms = ("julia", "evolution", "ga") if seed % 2 else ("evolution", "julia", "ga")
        for arm in arms:
            name = f"{arm}-seed{seed:02d}"
            command = [str(ROOT / ".venv/bin/python"), "-m"]
            output = str(root / "runs" / name)
            if arm == "ga":
                command += ["comparisons.search_exploration.run_standalone_ga", "--reference", str(REFERENCE),
                    "--output", output, "--seed", str(seed), "--budget", str(BUDGET),
                    "--batch-size", "1023", "--workers", "16", "--no-comet"]
            else:
                command += ["comparisons.search_exploration.run_streams", "--task", "rpa",
                    "--task-config", str(root / "configs/task.json"),
                    "--exploration-config", str(root / "configs" / (name + ".json")),
                    "--output", output, "--embedding-size", "1024", "--search-workers", "32",
                    "--optimization-workers", "8" if arm == "julia" else "16",
                    "--queue-capacity", "32" if arm == "julia" else "512",
                    "--optimize-per-epoch", "32" if arm == "julia" else "256"]
                if arm == "julia":
                    command += ["--julia-project", str(project), "--julia-recipe", str(root / "configs/julia_recipe.json")]
            jobs.append({"name": name, "arm": arm, "seed": seed, "command": command,
                "lane": "ga" if arm == "ga" else "hybrid", "workers": 16 if arm == "ga" else 48,
                "status": "queued"})
    return jobs


def dispatchable(jobs, active, worker_limit=64, lane_limits=None):
    from collections import Counter
    lane_limits = lane_limits or {"hybrid": 1, "ga": 1}
    lanes = Counter(j["lane"] for j in active)
    workers = sum(j["workers"] for j in active)
    selected = []
    for job in jobs:
        if (job["status"] != "queued" or lanes[job["lane"]] >= lane_limits[job["lane"]]
                or workers + job["workers"] > worker_limit):
            continue
        selected.append(job)
        lanes[job["lane"]] += 1
        workers += job["workers"]
    return selected


def prepare(root, project):
    from dataclasses import asdict
    from RL4CRN.search_streams.julia_stream import JuliaRecipe
    verify()
    reference = json.loads((REFERENCE / "manifest.json").read_text())
    if read_status(REFERENCE / "completed.json")["status"] != "complete":
        raise ValueError("Reference must be complete")
    if shutil.disk_usage(ROOT).free < 250 * 2**30:
        raise OSError("Reserve at least 250 GiB for this 60-run campaign")
    for filename in ("Project.toml", "Manifest.toml", "src/CRNOptim.jl"):
        if not (project / filename).is_file():
            raise ValueError(f"Missing Julia project input: {filename}")
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "runs", "logs", "analysis"):
        (root / folder).mkdir()
    atomic_json(root / "configs/task.json", reference["task_config"])
    atomic_json(root / "configs/julia_recipe.json", asdict(JuliaRecipe()))
    for seed in SEEDS:
        for arm in ("julia", "evolution"):
            atomic_json(root / "configs" / f"{arm}-seed{seed:02d}.json", configuration(reference, seed, arm))
    sources = set()
    for folder in ("RL4CRN/search_exploration", "RL4CRN/search_streams", "RL4CRN/parameter_optimization",
                   "comparisons/search_exploration"):
        sources.update(p for p in (ROOT / folder).iterdir() if p.suffix in (".py", ".jl"))
    sources.add(ROOT / "comparisons/search_exploration/RPA_STATISTICAL_PLAN.md")
    external = [REFERENCE / "manifest.json", REFERENCE / "completed.json",
        *(project / f for f in ("Project.toml", "Manifest.toml", "src/CRNOptim.jl"))]
    plan = {"schema": "rpa-workflows-20seed-v1", "id": root.name, "created": time.time(),
        "seeds": SEEDS, "arms": list(ARMS), "attempt_ceiling": BUDGET, "search_epochs": 100,
        "search_batch": 1023, "threshold": reference["config"]["quality_threshold"],
        "primary": "unique qualifying discovered topology hashes across all recorded designs, not live archive size",
        "secondary": "rank-one endpoint canonical loss", "alpha": .05,
        "tests": "two-sided paired Wilcoxon, Pratt zeros, exhaustive sign permutations, four-test Holm family",
        "bootstrap": {"resamples": 10000, "seed": 20260915, "coverage": .95, "unit": "paired seed"},
        "contrasts": [["julia", "evolution"], ["julia", "ga"]],
        "failure_policy": "Hold new dispatch; drain active runs; preserve all failures; no automatic retry or exclusion",
        "inference_gate": "All 60 jobs valid and complete; otherwise descriptive partial results only",
        "interpretation": "End-to-end workflow comparison, not isolated optimizer or equal compute experiment",
        "comet_project": None, "llm_calls": 0, "gpu": False, "post_pruning": False,
        "worker_limit": 64, "jobs": jobs_for(root, project),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
        "input_hashes": {str(p): digest(p) for p in [*sorted((root / "configs").glob("*.json")), *external]},
        "legacy_sha256": digest(DEFAULT / "scientific_source.tar.gz")}
    atomic_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n")
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(sources):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(DEFAULT / "scientific_source.tar.gz", arcname="legacy_scientific_source.tar.gz")
        for index, path in enumerate(external):
            archive.add(path, arcname=f"external/{index}-{path.name}")
    shutil.copyfile(ROOT / "comparisons/search_exploration/RPA_STATISTICAL_PLAN.md", root / "PROTOCOL.md")
    return plan


def check_frozen(root, plan):
    plan_file = plan.get("plan_file", "plan.json")
    if digest(root / plan_file) != (root / Path(plan_file).with_suffix(".sha256")).read_text().strip():
        raise ValueError("Campaign plan changed")
    verify()
    for path, expected in {**{str(ROOT / p): h for p, h in plan["source_hashes"].items()},
                           **plan["input_hashes"]}.items():
        if not Path(path).exists() or digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def execute(root, plan):
    jobs, active = plan["jobs"], {}
    progress_path = root / plan.get("progress_file", "progress.json")
    if plan.get("adopted_jobs"):
        import io
        from comparisons.search_exploration.scale_rpa_statistical import AdoptedProcess
        for job in jobs:
            if job["name"] in plan["adopted_jobs"]:
                active[job["name"]] = (AdoptedProcess(root / "progress.json", job), io.StringIO(), job)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    holding, last_analysis = None, 0
    try:
        while True:
            for name, (process, log, job) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                completed = read_status(Path(job.get("output", root / "runs" / name)) / "completed.json")
                job.update(status="complete" if process.returncode == 0 and completed and
                    completed.get("status") == "complete" else "failed", returncode=process.returncode, finished=time.time())
                if job["status"] == "failed":
                    holding = f"Failed run: {name}; queued runs held for diagnosis"
                del active[name]
            if (root / plan.get("hold_file", "HOLD")).exists():
                holding = holding or "User HOLD file; active runs drain"
            if shutil.disk_usage(root).free < 50 * 2**30:
                holding = holding or "Low disk space; dispatch held"
            if not holding:
                try:
                    check_frozen(root, plan)
                except ValueError as error:
                    holding = str(error)
            if not holding:
                for job in dispatchable(jobs, [entry[2] for entry in active.values()],
                        plan.get("worker_limit", 64), plan.get("lane_limits")):
                    log = (root / "logs" / job.get("log_file", job["name"] + ".log")).open("x")
                    try:
                        process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    job.update(status="running", pid=process.pid, started=time.time(),
                        dispatch_worker_limit=plan.get("worker_limit", 64), dispatch_lane_limits=plan.get("lane_limits"))
                    active[job["name"]] = (process, log, job)
            for job in jobs:
                job["run_status"] = read_status(Path(job.get("output", root / "runs" / job["name"])) / "status.json")
            done = not active and (holding or all(j["status"] != "queued" for j in jobs))
            progress = {"updated": time.time(), "phase": "held" if holding else "complete" if done else "running",
                "hold_reason": holding, "active_workers": sum(j[2]["workers"] for j in active.values()),
                "counts": {s: sum(j["status"] == s for j in jobs) for s in ("queued", "running", "complete", "failed")},
                "jobs": jobs}
            atomic_json(progress_path, progress)
            if done or time.monotonic() - last_analysis >= 300:
                try:
                    from comparisons.search_exploration.analyze_rpa_statistical import analyze
                    analyze(root)
                except Exception as error:
                    atomic_json(root / "analysis_failed.json", {"time": time.time(), "message": str(error)})
                last_analysis = time.monotonic()
            if done:
                break
            time.sleep(15)
    except BaseException as error:
        atomic_json(root / "controller_failed.json", {"type": type(error).__name__, "message": str(error)})
        for process, _, _ in active.values():
            if not getattr(process, "adopted", False) and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log, job in active.values():
            if getattr(process, "adopted", False):
                log.close()
                continue
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            job.update(status="interrupted", returncode=process.returncode)
        atomic_json(progress_path, {"phase": "failed", "error": str(error), "jobs": jobs, "updated": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    parser.add_argument("--amend-no-timeouts", action="store_true", help="Prepare the explicitly requested no-deadline recovery")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.amend_no_timeouts:
        from comparisons.search_exploration.resume_rpa_statistical import amend
        plan = amend(root)
    elif args.execute_prepared:
        plan = json.loads((root / ("active_plan.json" if (root / "active_plan.json").exists() else "plan.json")).read_text())
    else:
        if args.project is None:
            parser.error("--project is required to prepare")
        plan = prepare(root, args.project.resolve())
    if args.prepare_only:
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "progress.json").exists() and not (plan.get("plan_file") == "active_plan.json" and
                (root / "progress.json").read_bytes() == (root / "amendments/no-timeouts/progress-before.json").read_bytes()):
            raise ValueError("Already started; explicit reconciliation required, no automatic restart")
        execute(root, plan)


if __name__ == "__main__":
    main()
