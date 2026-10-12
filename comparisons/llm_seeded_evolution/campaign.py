"""Frozen insertion-stream Logic comparison; bounded, CPU-only parallel workers."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from comparisons.llm_seeded_evolution import METHOD
from comparisons.llm_seeded_evolution.replay import GENERATED, ROOT, export_replay, sha256
from comparisons.rpa_search.src.common.io import write_json as write_replay

CAMPAIGN = "logic-insertion-fixed-b1023-20260909"
DEFAULT_ROOT = ROOT / "comparisons/llm_seeded_evolution/campaigns" / CAMPAIGN


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def context_check(source):
    contexts = []
    for workspace in sorted((source / "harness-workspaces").iterdir()):
        manifest = json.loads((workspace / "run_manifest.json").read_text())
        for name in ("CONTEXT/SEARCH_STATE.json", "CONTEXT/HALL_OF_FAME.json"):
            if sha256(workspace / name) != manifest["workspace_file_sha256"][name]:
                raise ValueError(f"Changed request context: {workspace / name}")
        state = json.loads((workspace / "CONTEXT/SEARCH_STATE.json").read_text())
        hof = json.loads((workspace / "CONTEXT/HALL_OF_FAME.json").read_text())
        if hof["entries"]:
            raise ValueError("Insertion-only source exposed an HOF")
        contexts.append(state["rl_epoch_at_snapshot"])
    if sorted(contexts) != [0, 20, 40, 60, 80]:
        raise ValueError(f"Unexpected request schedule: {contexts}")
    return {"request_epochs": sorted(contexts), "all_exposed_hof_empty": True}


def prepare(root, workers, parallel, local_only=False):
    if root.exists():
        raise FileExistsError(f"Campaign already exists: {root}; use --run to continue its queue")
    with (GENERATED / "llm_proposal_diversity_endpoints.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["cohort"] in ("local_logic", "flash_logic")
                and r["arm"] == "llm_to_rl"]
    expected = {(model, seed) for model in ("local_logic", "flash_logic") for seed in range(20)}
    if len(rows) != 40 or {(r["cohort"], int(r["seed"])) for r in rows} != expected:
        raise ValueError("Need exactly 20 matched insertion-only seeds per Logic model")
    exports = []
    for row in sorted(rows, key=lambda r: (int(r["seed"]), r["cohort"] == "flash_logic")):
        database = Path(row["database"])
        if sha256(database) != row["database_sha256"]:
            raise ValueError(f"Changed audited source: {database}")
        replay = export_replay(database.parent, source_mode="llm-to-rl")
        if replay["source_rl_batch_size"] != 1023 or replay["search_budget"] != 102300:
            raise ValueError("Source must have 100 batches of 1023 evaluations")
        request_epochs = [e["launched_epoch"] for e in replay["events"] + replay["failed_events"]]
        if sorted(request_epochs) != [0, 20, 40, 60, 80]:
            raise ValueError("Source must account for all five scheduled request outcomes")
        exports.append((row, replay, context_check(database.parent)))
    root.mkdir(parents=True)
    (root / "replays").mkdir()
    (root / "logs").mkdir()
    settings = json.loads((Path(__file__).parent / "configs/default.json").read_text())
    settings.update(protocol_version=2, batch_size=1023)
    write_json(root / "settings.json", settings)
    jobs = []
    for row, replay, context in exports:
        cohort, seed = row["cohort"], int(row["seed"])
        run_id = f"logic_search102300_seed{seed}_{cohort}_insertion_fixed_b1023_v2"
        replay_path = root / "replays" / f"{cohort}_seed{seed}.json"
        write_replay(replay_path, replay)
        jobs.append({"cohort": cohort, "task": "logic", "seed": seed, "run_id": run_id,
            "replay": str(replay_path), "replay_sha256": sha256(replay_path),
            "source_database_sha256": row["database_sha256"], "context_audit": context,
            "output": str(root / "runs" / METHOD / run_id), "log": str(root / "logs" / f"{run_id}.log")})
    plan = {"campaign_id": CAMPAIGN, "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "logic", "epochs": 100, "batch_size": 1023, "workers": workers,
        "max_parallel": parallel, "cpu_nice": 15, "new_model_calls": 0,
        "settings_sha256": sha256(root / "settings.json"),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p)
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "comet_project": None if local_only else "genai-net-llm-paper", "jobs": jobs,
        "logging_mode": "local_only" if local_only else "comet_and_local",
        "comparison": "GA versus recorded insertion-only RL, identical LLM candidates and completion-epoch boundaries",
        "algorithm": "mutation-only elitist evolutionary search; no crossover or learned policy",
        "limitations": ["Same proposals and evaluation-count availability, not matched wall time",
            "Different random initial populations and optimizer-specific initialization priors",
            "Conditions on recorded LLM self-feedback; no new feedback loop to the model"]}
    write_json(root / "plan.json", plan)
    print(json.dumps({"prepared": len(jobs), "root": str(root), "workers": workers,
                      "max_parallel": parallel}, indent=2))


def command(root, plan, job):
    return [sys.executable, "-m", "comparisons.llm_seeded_evolution.run", "--task", "logic",
            "--seed", str(job["seed"]), "--source-mode", "llm-to-rl", "--replay", job["replay"],
            "--settings", str(root / "settings.json"), "--batch-size", "1023",
            "--workers", str(plan["workers"]), "--run-id", job["run_id"],
            "--output-root", str(root / "runs"), "--campaign-id", plan["campaign_id"],
            *(["--comet-project", plan["comet_project"]] if plan["comet_project"] else [])]


def progress(job):
    path = Path(job["output"]) / "progress.csv"
    if not path.exists():
        return {}
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not rows[-1].get("search_evaluations"):
        return {}
    row = rows[-1]
    return {"epoch_equivalent": int(row["search_evaluations"]) / 1023,
            "best_loss": float(row["best_so_far_loss"]),
            "qualified_topologies": int(row["qualified_topologies"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "replayed_proposals": int(row["llm_evaluations"])}


def run(root):
    plan = json.loads((root / "plan.json").read_text())
    if sha256(root / "settings.json") != plan["settings_sha256"]:
        raise ValueError("Frozen search settings changed")
    for path, expected in plan["source_hashes"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen optimizer code changed: {path}")
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, completed, failed, active = [], [], [], []
    for job in plan["jobs"]:
        if sha256(Path(job["replay"])) != job["replay_sha256"]:
            raise ValueError(f"Changed replay: {job['run_id']}")
        output = Path(job["output"])
        done = output / "completed.json"
        if done.exists():
            result = json.loads(done.read_text())
            if result["status"] != "complete" or result["search_evaluations"] != 102300:
                raise ValueError(f"Invalid completion: {done}")
            completed.append(job)
        elif output.exists() and any(output.iterdir()):
            failed.append({**job, "reason": "Existing incomplete output; not automatically rerun"})
        else:
            pending.append(job)
    if plan["comet_project"]:
        from comparisons.llm_crn_generation.run_genai_net_llm_20seed_campaign import comet_smoke
        comet_smoke(plan["comet_project"], plan["campaign_id"])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", MPLCONFIGDIR="/local0/tmp/mpl", PYTHONUNBUFFERED="1",
               COMET_LOG_ENV_DETAILS="false", COMET_LOG_ENV_CPU="false", COMET_LOG_ENV_GPU="false")
    while pending or active:
        while pending and len(active) < plan["max_parallel"] and not failed:
            job = pending.pop(0)
            log = Path(job["log"]).open("x")
            cmd = command(root, plan, job)
            if plan["cpu_nice"]:
                cmd = ["nice", "-n", str(plan["cpu_nice"]), *cmd]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((job, process, log))
            print(f"Started {job['run_id']} pid={process.pid}", flush=True)
        for job, process, log in list(active):
            code = process.poll()
            if code is not None:
                log.close()
                active.remove((job, process, log))
                done = Path(job["output"]) / "completed.json"
                if code == 0 and done.exists():
                    completed.append(job)
                    from comparisons.llm_seeded_evolution.fixed_stream_report import report
                    report(root)
                else:
                    failed.append({**job, "exit_code": code})
                print(f"Finished {job['run_id']} exit={code}", flush=True)
        write_json(root / "status.json", {"campaign_id": plan["campaign_id"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "halted_for_audit" if failed else "running" if pending or active else "complete",
            "active": [{**job, "pid": proc.pid, **progress(job)} for job, proc, _ in active],
            "pending": pending, "completed": completed, "failed": failed})
        if failed and not active:
            raise RuntimeError("Queue stopped after failure; inspect outputs before restarting")
        if active or pending:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--local-only", action="store_true", help="Prepare with local artifacts only; no external export")
    parser.add_argument("--run", action="store_true", help="Run an already prepared, frozen queue")
    args = parser.parse_args()
    if args.workers < 1 or args.max_parallel < 1:
        parser.error("Resource counts must be positive")
    if args.run:
        run(args.root.resolve())
    else:
        prepare(args.root.resolve(), args.workers, args.max_parallel, args.local_only)


if __name__ == "__main__":
    main()
