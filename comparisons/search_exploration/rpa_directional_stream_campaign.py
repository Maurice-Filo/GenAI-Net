"""Frozen directional RPA cohorts plus a reused, post-hoc independent-union baseline."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest

FULL_ROOT = Path("/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1")
MODES = ("llm-to-rl", "rl-to-llm")
REFERENCE = Path("/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/runs/julia-seed20-no-timeouts")


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def prepare(root, *, concurrency=10, comet_project=None):
    from RL4CRN.search_exploration.engine import atomic_json
    from comparisons.search_exploration.check_rpa_flash_harness import PROMPT
    reference = read(REFERENCE / "manifest.json")
    if not read(REFERENCE / "completed.json"):
        raise ValueError("Completed no-LLM reference required")
    if not 1 <= concurrency <= 10:
        raise ValueError("At most ten total seeds, each with six search and two fit workers")
    if shutil.disk_usage(root.parent).free < 100 * 2**30:
        raise OSError("Require at least 100 GiB free")
    root.mkdir(exist_ok=False)
    for folder in ("configs", "runs", "logs"):
        (root / folder).mkdir()
    atomic_json(root / "configs/task.json", reference["task_config"])
    atomic_json(root / "configs/julia_recipe.json", reference["julia_recipe"])
    jobs = []
    order = [(seed, mode) for seed in range(1, 21) for mode in (MODES if seed % 2 else MODES[::-1])]
    for seed, mode in order:
        name = f"flash-adagrad-rpa-{mode}-seed{seed:02d}"
        config = {**reference["config"], "seed": seed, "workers": 6,
                  "fit_seconds": None, "evaluation_seconds": None}
        atomic_json(root / "configs" / f"{name}.json", config)
        command = [str(ROOT / ".venv/bin/python"), "-m", "comparisons.search_exploration.run_directional_streams", "--communication-mode", mode,
            "--task-config", str(root / "configs/task.json"), "--exploration-config", str(root / "configs" / f"{name}.json"),
            "--output", str(root / "runs" / name), "--julia-project", reference["julia_project"],
            "--julia-recipe", str(root / "configs/julia_recipe.json"), "--dsh-home",
            str(Path.home() / "ai-workspaces/deepseek-test/.dsh-home"),
            "--search-workers", "6", "--optimization-workers", "2", "--llm-every", "5",
            "--llm-candidates", "10"]
        if comet_project:
            command += ["--comet-project", comet_project]
        jobs.append({"name": name, "seed": seed, "arm": mode, "command": command, "status": "queued"})
    sources = {Path(__file__).resolve(), PROMPT.resolve(), ROOT / "comparisons/search_exploration/run_directional_streams.py",
               ROOT / "comparisons/search_exploration/analyze_rpa_directional_streams.py",
               ROOT / "RL4CRN/utils/input_interface.py", ROOT / "RL4CRN/utils/results_database.py",
               ROOT / "tools/dsh_bwrap_resolved_tmp.sh", ROOT / "tools/dsh_headless_continuing_runner.mjs"}
    for folder in ("RL4CRN/search_exploration", "RL4CRN/search_streams", "RL4CRN/parameter_optimization", "RL4CRN/llm"):
        sources.update(p for p in (ROOT / folder).iterdir() if p.suffix in (".py", ".jl"))
    plan = {"schema": "rpa-flash-adagrad-directional-20seed-v1", "created": time.time(), "id": root.name,
        "seeds": list(range(1, 21)), "epochs": 100, "batch_size": 1023, "evaluation_budget": 214793,
        "concurrency": concurrency, "numerical_workers_per_seed": 8, "max_numerical_workers": 8 * concurrency,
        "resident_canonical_workers_per_seed": 2,
        "cpu_note": "Canonical verification normally replaces a Julia solve; LLM evaluation and coordinators add transient overhead.",
        "model": "deepseek-v4-flash", "communication": list(MODES), "full_duplex_root": str(FULL_ROOT),
        "independent_union": "Post-hoc existing no-LLM control + direct insertion-only LLM proposals; not a matched uncoupled run",
        "union_cost": "Charge old control actual calls plus every recorded direct LLM evaluation conservatively; no fitting descendants", "withhold_initial_hof": True,
        "llm_every": 5, "llm_candidates": 10, "max_in_flight_per_seed": 1,
        "max_scheduled_rounds_per_seed": 20, "max_role_invocations_total": 1600,
        "model_call_note": "Role invocations include native-tool turns and up to two continuations, not single API calls.",
        "scientific_timeouts": None, "harness_role_timeout_seconds": 900,
        "comet_project": comet_project, "reference": str(REFERENCE), "jobs": jobs,
        "comet_credentials": "Existing local Comet configuration; inherited COMET_API_KEY removed",
        "comparison_caveat": "Same scientific settings/seeds and total cap; worker allocation and asynchronous completion schedules differ. Report actual evaluations, fits and LLM insertion times.",
        "failure_policy": "No automatic retry. Failed numerical/infrastructure jobs hold new dispatch; active jobs drain. LLM payload failures remain recorded outcomes.",
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)},
        "input_hashes": {str(p): digest(p) for p in sorted((root / "configs").glob("*.json"))}}
    atomic_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n")
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(sources):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        archive.add(REFERENCE / "manifest.json", arcname="reference/manifest.json")
    atomic_json(root / "progress.json", {"status": "prepared", "jobs": jobs, "updated": time.time()})
    return plan


def external_active(root):
    progress = read(root / "progress.json")
    if progress is None:
        raise RuntimeError("Cannot verify existing full-duplex campaign occupancy")
    if progress["status"] == "complete":
        return 0
    count = 0
    for job in progress["jobs"]:
        if job["status"] == "running":
            try:
                os.kill(int(job["pid"]), 0)
            except ProcessLookupError:
                continue
            count += 1
    return count


def execute(root, readiness):
    from RL4CRN.search_exploration.engine import atomic_json
    gate = read(readiness)
    if not gate or gate.get("status") != "completed" or gate.get("valid_evaluations", 0) < 1:
        raise ValueError("Live Flash readiness must pass before dispatch")
    if gate.get("role_invocations") != 2 or gate.get("initial_hof_entries") != 0:
        raise ValueError("Readiness must verify both roles and withholding")
    plan = read(root / "plan.json")
    if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Plan changed")
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "started.json").exists():
        raise ValueError("Already started; explicit reconciliation required")
    atomic_json(root / "started.json", {"pid": os.getpid(), "time": time.time(), "readiness": str(readiness)})
    jobs, active = plan["jobs"], {}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "JULIA_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    hold = False
    while True:
        analysis_dirty = False
        for name, (process, log, job) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            job.update(returncode=process.returncode, finished=time.time())
            job["status"] = "complete" if process.returncode == 0 and read(root / "runs" / name / "completed.json") else "failed"
            if job["status"] == "failed":
                hold = True
            analysis_dirty = True
            del active[name]
        hold = hold or (root / "HOLD").exists()
        for job in jobs:
            if hold or len(active) + external_active(FULL_ROOT) >= plan["concurrency"]:
                break
            if job["status"] != "queued":
                continue
            for path, expected in {**{str(ROOT / p): h for p, h in plan["source_hashes"].items()}, **plan["input_hashes"]}.items():
                if digest(Path(path)) != expected:
                    hold = True
                    atomic_json(root / "dispatch_blocked.json", {"reason": "frozen input changed", "path": path})
                    break
            if hold:
                break
            log = (root / "logs" / f"{job['name']}.log").open("x")
            process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job.update(status="running", pid=process.pid, started=time.time())
            active[job["name"]] = (process, log, job)
            print(f"Started {job['name']} pid={process.pid}", flush=True)
        snapshots = []
        for job in jobs:
            status = read(root / "runs" / job["name"] / "status.json") or {}
            snapshots.append({**job, "run_status": status})
        queued = sum(j["status"] == "queued" for j in jobs)
        state = "held" if hold else "running" if active else "waiting_for_slots" if queued else "complete"
        atomic_json(root / "progress.json", {"status": state, "updated": time.time(),
            "active": len(active), "completed": sum(j["status"] == "complete" for j in jobs),
            "failed": sum(j["status"] == "failed" for j in jobs), "jobs": snapshots})
        if analysis_dirty:
            from comparisons.search_exploration.analyze_rpa_directional_streams import analyze
            try:
                analyze(root)
            except Exception as error:
                atomic_json(root / "analysis_failed.json", {"error": str(error), "time": time.time()})
        if not active and (hold or not queued):
            from comparisons.search_exploration.analyze_rpa_directional_streams import analyze
            analyze(root)
            return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--comet-project")
    parser.add_argument("--readiness", type=Path)
    args = parser.parse_args()
    if args.run and args.readiness is None:
        parser.error("--run requires --readiness")
    if args.prepare:
        prepare(args.root, concurrency=args.concurrency, comet_project=args.comet_project)
    if args.run:
        execute(args.root, args.readiness)


if __name__ == "__main__":
    main()
