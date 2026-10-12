"""Prepare audited 100-epoch parity prefixes and run two isolated GA seeds."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from comparisons.llm_seeded_evolution.replay import ROOT, GENERATED, sha256
from comparisons.llm_seeded_evolution.parity12.replay import export
from comparisons.llm_seeded_evolution.campaign import write_json

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "campaigns/parity12-ga-trial-100epoch-seeds0to1-20260910"


def prefix_events(events, budget):
    return [event for event in events if event["after_search_evaluations"] <= budget]


def prepare(root):
    from comparisons.llm_crn_generation.audit_contract_v2_sentinel import audit_run
    from RL4CRN.llm.workspace_loss_protocol import audit_workspaces
    if root.exists():
        raise FileExistsError("Trial already exists")
    root.mkdir(parents=True)
    schedule = json.loads((GENERATED / "parity12_communication_300epoch_20260909_schedule.json").read_text())
    source_campaign = Path(schedule["campaigns"]["llm-to-rl"]["audit_command"][2])
    jobs = []
    for seed in (0, 1):
        source = next((source_campaign / "runs").glob(f"*seed{seed}/*/completed.json")).parent
        audit = audit_run(source, task="logic_parity", seed=seed, epochs=300,
            expected_request_epochs=list(range(0, 300, 20)), expected_budget=307200,
            expected_rl_evaluations=306900, communication_mode="llm-to-rl")
        workspace_errors = audit_workspaces(source, "parity-evidence-exploitation-12r-v1")
        audit["workspace_errors"] = workspace_errors
        if workspace_errors:
            audit["status"] = "fail"
        write_json(root / f"source_audit_seed{seed}.json", audit)
        if audit["status"] != "pass":
            raise ValueError(f"Source audit failed for seed {seed}: {audit['errors']} {workspace_errors}")
        replay = export(source, audit["database_sha256"])
        budget = 100 * 1023
        events = prefix_events(replay["events"], budget)
        with sqlite3.connect((source / "results.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True) as c:
            c.row_factory = sqlite3.Row
            snap = c.execute("SELECT * FROM hof_snapshots WHERE boundary='epoch' AND cumulative_rl_evaluations=?", (budget,)).fetchone()
            count = sum(len(event["candidates"]) for event in events)
            if snap is None or snap["cumulative_llm_final_evaluations"] != count or snap["cumulative_llm_tool_evaluations"]:
                raise ValueError("Prefix budget does not match the saved RL checkpoint")
            archive = [dict(r) for r in c.execute("SELECT topology_hash,loss,parameters_json FROM hof_snapshot_entries WHERE snapshot_id=? ORDER BY rank", (snap["snapshot_id"],))]
        if len(archive) != 30:
            raise ValueError("Incomplete prefix archive")
        replay.pop("source_terminal_archive")
        replay.update(schema_version="parity12-trial-prefix-v1", pilot=True, epochs=100,
            source_total_search_budget=306900, search_budget=budget, events=events, llm_evaluations=count,
            failed_events=[e for e in replay["failed_events"] if e["completed_epoch"] is not None and e["completed_epoch"] < 100],
            source_checkpoint_archive=archive, source_checkpoint_epoch=99,
            limitations=["RL permits null reaction while GA and direct LLM proposals exclude it",
                         "Two-seed 100-epoch prefix trial; not the planned 300-epoch cohort",
                         "GA rates are positive and unbounded, as in the ongoing Logic GA baseline"])
        path = root / f"replay_seed{seed}.json"
        write_json(path, replay)
        jobs.append({"seed": seed, "replay": str(path), "replay_sha256": sha256(path),
                     "output": str(root / "runs" / f"seed{seed}"), "search_budget": budget, "llm_evaluations": count})
    files = [*HERE.glob("*.py"), *HERE.parent.glob("*.py"), *HERE.parent.parent.glob("*.py"),
             *ROOT.glob("RL4CRN/**/*.py"), *ROOT.glob("comparisons/rpa_search/src/common/*.py")]
    plan = {"campaign_id": root.name, "pilot": True, "epochs": 100, "seeds": [0, 1],
        "max_parallel": 2, "workers_per_run": 8, "cpu_nice": 0, "new_model_calls": 0,
        "comet_project": "genai-net-ga-fixed-proposals", "jobs": jobs,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "prepared", "pending": jobs, "active": [], "completed": [], "failed": []})
    print(json.dumps({k: v for k, v in plan.items() if k != "source_hashes"}, indent=2))


def launch(root):
    import fcntl
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "plan.json").read_text())
    state = json.loads((root / "status.json").read_text())
    if state["status"] != "prepared":
        raise ValueError("Trial is not a fresh prepared run; no automatic restart")
    for path, expected in plan["source_hashes"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen code changed: {path}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLCONFIGDIR="/local0/tmp/mpl", PYTHONUNBUFFERED="1",
               COMET_LOG_ENV_DETAILS="false", COMET_LOG_ENV_CPU="false", COMET_LOG_ENV_GPU="false")
    env.pop("COMET_API_KEY", None)
    from comparisons.llm_crn_generation.run_genai_net_llm_20seed_campaign import comet_smoke
    comet_smoke(plan["comet_project"], plan["campaign_id"])
    active = []
    state.update(status="running", pending=[])
    for job in plan["jobs"]:
        if sha256(Path(job["replay"])) != job["replay_sha256"]:
            raise ValueError("Replay changed")
        log = (root / f"seed{job['seed']}.log").open("x")
        command = [sys.executable, "-m", "comparisons.llm_seeded_evolution.parity12.trial.run",
                   "--replay", job["replay"], "--output", job["output"],
                   "--workers", str(plan["workers_per_run"]), "--comet-project", plan["comet_project"]]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active.append((job, proc, log))
    while active:
        for job, proc, log in list(active):
            code = proc.poll()
            if code is not None:
                log.close()
                active.remove((job, proc, log))
                finished = Path(job["output"]) / "completed.json"
                state["completed" if code == 0 and finished.exists() else "failed"].append({**job, "exit_code": code})
        state.update(active=[{**job, "pid": proc.pid} for job, proc, _ in active], updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root / "status.json", state)
        if active:
            time.sleep(10)
    state["status"] = "complete" if not state["failed"] else "failed_requires_audit"
    write_json(root / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    (launch if args.launch else prepare)(args.root.resolve())
