"""Frozen parity-12, 20-seed, four-arm AdaGrad/Harness campaign."""

import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

from comparisons.search_exploration.freeze_legacy import ROOT, digest
from comparisons.search_exploration.rpa_directional_stream_campaign import read, REFERENCE
from comparisons.search_exploration.recover_rpa_directional_balance import balance_status, request_audit

MODES = ("full", "llm-to-rl", "rl-to-llm", "none")
PROMPT = ROOT / "comparisons/search_exploration/prompts/parity12-evidence-exploitation-v2.md"
TASK = ROOT / "comparisons/rpa_search/configs/logic_parity_12reactions_300epoch.json"


def configuration(reference, seed):
    task = read(TASK)
    task["benchmark"].update(name="parity12_four_arm_100epoch", min_ode_simulations=102300)
    task["search"].update(seed=seed, rate_constant_range=[.001, 100.], candidate_budget=214793)
    task["rl4crn"].update(epochs=100, batch_size=1023, n_cpus=6, deep_layer_size=1024)
    config = {**reference["config"], "seed": seed, "workers": 6,
              "fit_seconds": None, "evaluation_seconds": None, "quality_threshold": .05,
              "require_all_rows": False}
    if (config["epochs"], config["batch_size"], config["evaluation_budget"]) != (100, 1023, 214793):
        raise ValueError("Reference no longer matches approved RPA settings")
    return task, config


def sources():
    paths = {Path(__file__).resolve(), PROMPT, TASK,
        ROOT / "comparisons/search_exploration/run_parity12_four_arm.py",
        ROOT / "comparisons/search_exploration/preflight_parity12_four_arm.py",
        ROOT / "comparisons/search_exploration/recover_rpa_directional_balance.py",
        ROOT / "RL4CRN/utils/input_interface.py", ROOT / "RL4CRN/utils/results_database.py",
        ROOT / "tools/dsh_bwrap_resolved_tmp.sh", ROOT / "tools/dsh_headless_continuing_runner.mjs"}
    for folder in ("RL4CRN/search_exploration", "RL4CRN/search_streams", "RL4CRN/parameter_optimization", "RL4CRN/llm"):
        paths.update(p for p in (ROOT / folder).iterdir() if p.suffix in (".py", ".jl"))
    return paths


def verify_inputs(plan):
    for path, expected in plan["source_hashes"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for path, expected in plan["input_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen campaign input changed: {path}")


def prepare(root, preflight, concurrency=10):
    from RL4CRN.search_exploration.engine import atomic_json
    ready = read(preflight)
    if not ready or ready.get("status") != "passed" or ready["prompt_sha256"] != digest(PROMPT):
        raise ValueError("Parity canonical/Julia preflight must pass with this prompt")
    if not 1 <= concurrency <= 10:
        raise ValueError("Supported concurrency is one to ten jobs")
    if shutil.disk_usage(root.parent).free < 100 * 2**30:
        raise OSError("Require at least 100 GiB free")
    reference = read(REFERENCE / "manifest.json")
    root.mkdir(exist_ok=False)
    for folder in ("configs", "runs", "logs"):
        (root / folder).mkdir()
    shutil.copy2(PROMPT, root / "TASK.md")
    shutil.copy2(preflight, root / "preflight.json")
    atomic_json(root / "configs/julia_recipe.json", reference["julia_recipe"])
    jobs = []
    for seed in range(1, 21):
        task, config = configuration(reference, seed)
        atomic_json(root / "configs" / f"task-seed{seed:02d}.json", task)
        atomic_json(root / "configs" / f"search-seed{seed:02d}.json", config)
        # Rotate launch order to avoid always giving the same arm the first provider slots.
        offset = (seed - 1) % len(MODES)
        for mode in MODES[offset:] + MODES[:offset]:
            name = f"parity12-flash-{mode}-seed{seed:02d}"
            output = root / "runs" / name
            command = [str(ROOT / ".venv/bin/python"), "-m", "comparisons.search_exploration.run_parity12_four_arm",
                "--communication-mode", mode, "--task-config", str(root / "configs" / f"task-seed{seed:02d}.json"),
                "--exploration-config", str(root / "configs" / f"search-seed{seed:02d}.json"),
                "--output", str(output), "--julia-project", reference["julia_project"],
                "--julia-recipe", str(root / "configs/julia_recipe.json"), "--prompt", str(PROMPT),
                "--dsh-home", str(Path.home() / "ai-workspaces/deepseek-test/.dsh-home"),
                "--search-workers", "6", "--optimization-workers", "2", "--llm-every", "5",
                "--llm-candidates", "10", "--llm-timeout", "900", "--comet-project", "genai-net-exploration"]
            jobs.append({"name": name, "seed": seed, "arm": mode, "output": str(output),
                         "command": command, "status": "queued"})
    paths = sources()
    inputs = [*(root / "configs").glob("*.json"), root / "TASK.md", root / "preflight.json"]
    plan = {"schema": "parity12-four-arm-100epoch-20seed-v1", "id": root.name,
        "created": time.time(), "seeds": list(range(1, 21)), "arms": list(MODES),
        "epochs": 100, "batch_size": 1023, "evaluation_budget": 214793,
        "search_evaluation_reserve": 102300, "concurrency": concurrency,
        "search_workers_per_seed": 6, "fit_workers_per_seed": 2,
        "canonical_verification_workers_per_seed": 2, "model": "deepseek-official/deepseek-v4-flash",
        "withhold_initial_hof": True, "llm_every": 5, "llm_candidates": 10,
        "max_in_flight_per_seed": 1, "global_harness_concurrency": 8,
        "max_scheduled_rounds_per_seed": 20, "max_role_invocations": 3200,
        "scientific_timeouts": None, "harness_role_timeout_seconds": 900,
        "quality_threshold": .05, "quality_threshold_note": "Operational weighted trajectory-L1 threshold chosen after the pilot, before this cohort; not a classification guarantee. Report threshold curves and all-row correctness separately.",
        "self_feedback": "Enabled in every arm; only cross-source channels vary",
        "none": "Fresh uncoupled RL/AdaGrad and LLM streams, terminal topology union; not reused pilot data",
        "comparison_caveat": "Matched seeds, epochs and numerical caps; realized calls, fits, provider queue times and insertion epochs can differ. No fixed proposal-bank causal control.",
        "failure_policy": "No retries; invalid/empty/timed-out requests remain outcomes. Numerical failures hold dispatch. Provider-balance outage stops active attempts and holds dispatch; affected endpoints are not complete.",
        "comet_project": "genai-net-exploration", "reference": str(REFERENCE), "jobs": jobs,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)},
        "input_hashes": {str(p): digest(p) for p in sorted(inputs)}}
    atomic_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n")
    with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    atomic_json(root / "progress.json", {"status": "prepared", "jobs": jobs, "updated": time.time()})
    return plan


def execute(root):
    from RL4CRN.search_exploration.engine import atomic_json
    plan = read(root / "plan.json")
    if digest(root / "plan.json") != (root / "plan.sha256").read_text().strip():
        raise ValueError("Plan changed")
    verify_inputs(plan)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "started.json").exists():
        raise ValueError("Already started; explicit reconciliation required")
    balance = balance_status()
    atomic_json(root / "balance_initial.json", balance)
    if not balance["is_available"]:
        raise RuntimeError("Provider balance unavailable; no dispatch")
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
            completed = read(Path(job["output"]) / "completed.json")
            ok = process.returncode == 0 and completed and not audit["balance_affected"]
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
                if shutil.disk_usage(root).free < 30 * 2**30:
                    raise OSError("Fewer than 30 GiB free; holding new dispatch")
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
        snapshots = [{**j, "run_status": read(Path(j["output"]) / "status.json") or {}} for j in jobs]
        counts = collections.Counter(j["status"] for j in jobs)
        state = "held" if hold else "running" if active else "waiting_for_balance" if counts["queued"] else "complete"
        atomic_json(root / "progress.json", {"status": state, "updated": time.time(), "jobs": snapshots,
            "active": len(active), "completed": counts["complete"], "failed": counts["failed"], "queued": counts["queued"]})
        if not active and (hold or not counts["queued"]):
            return
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    if args.prepare:
        if args.preflight is None:
            parser.error("--prepare requires --preflight")
        prepare(args.root, args.preflight, args.concurrency)
    if args.run:
        execute(args.root)
