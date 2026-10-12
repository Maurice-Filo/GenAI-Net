"""Bounded, matched NN-size search-quality pilot using the opt-in exploration runner."""

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, DEFAULT, digest, verify


def task_settings(task, seed):
    base = json.loads((ROOT / "comparisons/search_exploration/configs/pilot.json").read_text())
    base.update(seed=seed, epochs=20, batch_size=32, evaluation_budget=10240,
                fit_evaluations=16, fit_seconds=60., evaluation_seconds=30., workers=4,
                optimizer="evolution", rate_min=.001, rate_max=100., plot_every=5)
    if task == "rpa":
        base.update(quality_threshold=.016491168958935352, require_all_rows=False)
    elif task != "logic":
        raise ValueError("Pilot supports Logic and RPA")
    return base


def jobs_for(root, seeds, comet_project):
    jobs = []
    for seed in seeds:
        for task in ("logic", "rpa"):
            sizes = (1024, 10240) if seed % 2 == 0 else (10240, 1024)
            for size in sizes:
                name = f"{task}-nn{size}-seed{seed}"
                config = root / "configs" / f"{task}-seed{seed}.json"
                command = [str(ROOT / ".venv/bin/python"), "-m", "comparisons.search_exploration.run",
                           "--task", task, "--exploration-config", str(config), "--embedding-size", str(size),
                           "--device", "cpu", "--output", str(root / "runs" / name)]
                if comet_project:
                    command += ["--comet-project", comet_project]
                jobs.append({"name": name, "task": task, "seed": seed, "embedding": size,
                             "command": command, "status": "queued"})
    return jobs


def prepare(root, seeds, concurrency, comet_project):
    from RL4CRN.search_exploration.config import ExplorationConfig
    from RL4CRN.search_exploration.engine import atomic_json
    verify()
    ancestor = root.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 100*2**30:
        raise OSError("Quality pilot requires at least 100 GiB free for retained trajectories")
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("configs", "runs", "logs", "analysis"):
        (root / folder).mkdir()
    for seed in seeds:
        for task in ("logic", "rpa"):
            config = ExplorationConfig(**task_settings(task, seed))
            atomic_json(root / "configs" / f"{task}-seed{seed}.json", config.to_dict())
    sources = [*sorted((ROOT / "RL4CRN/search_exploration").glob("*.py")),
               ROOT / "comparisons/search_exploration/run.py", Path(__file__).resolve(),
               ROOT / "comparisons/search_exploration/analyze_nn_size_quality.py"]
    jobs = jobs_for(root, seeds, comet_project)
    plan = {"id": root.name, "created": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
            "embeddings": [1024, 10240], "tasks": ["logic", "rpa"], "concurrency": concurrency,
            "comet_project": comet_project, "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "legacy_sha256": digest(DEFAULT / "scientific_source.tar.gz"), "jobs": jobs,
            "interpretation": "Exploratory size ablation, not a frozen primary paper cohort.",
            "budget": "Equal 20-epoch schedules and 10240 physical-evaluation caps; report actual cache-adjusted consumption.",
            "primary": ["best canonical loss", "qualifying archived topologies"],
            "secondary": ["successful descriptor regions", "convergence versus actual evaluations", "elapsed time"],
            "failure_policy": "No automatic retries. Hold queued jobs after a failure; let active jobs finish."}
    atomic_json(root / "plan.json", plan)
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(DEFAULT / "scientific_source.tar.gz", arcname="legacy_scientific_source.tar.gz")
    return plan


def read_status(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def execute(root, plan):
    from RL4CRN.search_exploration.engine import atomic_json
    jobs = plan["jobs"]
    active = {}
    started = time.time()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    # Use the same configured credential source as the approved smoke upload.
    env.pop("COMET_API_KEY", None)
    hold = False
    try:
        while True:
            for name, (process, log, job) in list(active.items()):
                if process.poll() is None:
                    continue
                log.close()
                job.update(returncode=process.returncode, finished=time.time())
                completed = read_status(root / "runs" / name / "completed.json")
                job["status"] = "complete" if process.returncode == 0 and completed else "failed"
                if job["status"] == "failed":
                    hold = True
                del active[name]
            hold = hold or (root / "HOLD").exists()
            for job in jobs:
                if hold or len(active) >= plan["concurrency"]:
                    break
                if job["status"] != "queued":
                    continue
                for path, expected in plan["source_hashes"].items():
                    if digest(ROOT / path) != expected:
                        raise ValueError(f"Frozen pilot source changed: {path}")
                log = (root / "logs" / (job["name"] + ".log")).open("w")
                try:
                    process = subprocess.Popen(job["command"], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                job.update(status="running", pid=process.pid, started=time.time())
                active[job["name"]] = (process, log, job)
            for job in jobs:
                job["run_status"] = read_status(root / "runs" / job["name"] / "status.json")
            progress = {"updated": time.time(), "elapsed_seconds": time.time()-started,
                        "holding": hold, "counts": {s: sum(j["status"] == s for j in jobs)
                            for s in ("queued", "running", "complete", "failed")}, "jobs": jobs}
            atomic_json(root / "progress.json", progress)
            print(json.dumps(progress["counts"]), flush=True)
            if not active and (hold or all(j["status"] != "queued" for j in jobs)):
                break
            time.sleep(10)
    except BaseException as error:
        atomic_json(root / "controller_failed.json", {"type": type(error).__name__, "message": str(error)})
        for process, log, job in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log, job in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
        raise
    finally:
        from comparisons.search_exploration.analyze_nn_size_quality import analyze
        try:
            analyze(root)
        except Exception as error:
            atomic_json(root / "analysis_failed.json", {"type": type(error).__name__, "message": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--comet-project", default="genai-net-exploration")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-prepared", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.concurrency < 1 or len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error("Invalid concurrency or seeds")
    if args.execute_prepared:
        plan = json.loads((root / "plan.json").read_text())
        if (root / "progress.json").exists():
            raise ValueError("Pilot already started; explicit reconciliation required")
    else:
        plan = prepare(root, args.seeds, args.concurrency, args.comet_project)
    if not args.prepare_only:
        with (root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (root / "progress.json").exists():
                raise ValueError("Pilot already started; explicit reconciliation required")
            execute(root, plan)


if __name__ == "__main__":
    main()
