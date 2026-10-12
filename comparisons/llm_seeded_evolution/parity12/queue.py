"""Wait for Logic GA and audited parity sources, then run twenty paired GA seeds."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from comparisons.llm_seeded_evolution.campaign import write_json
from comparisons.llm_seeded_evolution.replay import ROOT, GENERATED, sha256
from comparisons.llm_seeded_evolution.parity12.replay import export
from comparisons.rpa_search.src.common.io import write_json as write_replay

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "campaigns/flash-parity12-insertion-fixed-300epoch-20260909"
LOGIC_ROOT = HERE.parent / "campaigns/logic-insertion-fixed-b1023-20260909-local"


def complete_grid(status, count, task=None):
    if any(status.get(k) for k in ("active", "pending", "failed")):
        return False
    done = status.get("completed", [])
    if len(done) != count:
        return False
    if task:
        return {(r.get("task"), r.get("seed")) for r in done} == {(task, s) for s in range(count)}
    return {(r.get("cohort"), r.get("seed")) for r in done} == {
        (model, s) for model in ("local_logic", "flash_logic") for s in range(20)}


def dependencies(plan):
    def read(path):
        return json.loads(Path(path).read_text()) if Path(path).exists() else {}
    logic = read(Path(plan["logic_dependency"]) / "status.json")
    parity = read(Path(plan["source_campaign"]) / "status.json")
    return {"logic_ga_complete": complete_grid(logic, 40),
            "parity_insertion_complete": complete_grid(parity, 20, "logic_parity"),
            "logic_completed": len(logic.get("completed", [])),
            "parity_completed": len(parity.get("completed", [])),
            "dependency_failed_runs": len(logic.get("failed", [])) + len(parity.get("failed", []))}


def prepare(root):
    if root.exists():
        raise FileExistsError("Parity follow-up already exists; no overwrite")
    schedule_path = GENERATED / "parity12_communication_300epoch_20260909_schedule.json"
    schedule = json.loads(schedule_path.read_text())
    audit = list(schedule["campaigns"]["llm-to-rl"]["audit_command"])
    source = Path(audit[2])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    audit[audit.index("--output") + 1] = str(root / "source_audit.json")
    files = [*HERE.glob("*.py"), *HERE.parent.glob("*.py"),
             *ROOT.glob("RL4CRN/**/*.py"), *ROOT.glob("comparisons/rpa_search/src/common/*.py"),
             ROOT / "comparisons/llm_crn_generation/paper_breadth_tasks.py",
             Path(audit[1])]
    plan = {"campaign_id": root.name, "task": "logic_parity", "seeds": list(range(20)),
        "model_proposal_source": "deepseek-v4-flash", "source_mode": "llm-to-rl",
        "added_reactions": 12, "fixed_reactions": 4, "epochs": 300, "batch_size": 1023,
        "search_budget": 306900, "llm_budget": "Actual per-seed source evaluations; no replacement for missing proposals",
        "new_model_calls": 0, "max_parallel": 4, "workers": 4, "cpu_nice": 0,
        "comet_project": "genai-net-ga-fixed-proposals", "logic_dependency": str(LOGIC_ROOT),
        "source_campaign": str(source), "source_audit_command": audit,
        "source_schedule_sha256": sha256(schedule_path),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)},
        "metrics": {"primary": "paired canonical endpoint loss",
            "diagnostic": "All-16 terminal binary-correct topology counts within the loss-ranked top-30 archive",
            "binary_rule": "OUT(t=100) >= 0.5; complete finite simulations required",
            "loss_qualification_threshold": None},
        "comparison": "Same source LLM candidates/rates and completion-epoch availability; equal search counts",
        "validation_only": "Reevaluate the 30 source terminal designs after GA search; excluded from search budget and never fed to GA",
        "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(root / "plan.json", plan)
    write_json(root / "status.json", {"status": "waiting_for_dependencies", "dependencies": dependencies(plan),
        "pending": [{"task": "logic_parity", "seed": s} for s in range(20)],
        "active": [], "completed": [], "failed": [], "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"root": str(root), "dependencies": dependencies(plan)}, indent=2))


def command(root, plan, job):
    return [sys.executable, "-m", "comparisons.llm_seeded_evolution.parity12.run",
            "--replay", job["replay"], "--output", job["output"],
            "--workers", str(plan["workers"]), "--comet-project", plan["comet_project"]]


def watch(root, once=False):
    plan = json.loads((root / "plan.json").read_text())
    if once:
        print(json.dumps(dependencies(plan), indent=2))
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((root / "status.json").read_text())
    if state["status"] != "waiting_for_dependencies":
        raise ValueError("Follow-up already launched or halted; inspect artifacts before any recovery")
    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "status.json", state)
    try:
        while True:
            state["dependencies"] = dependencies(plan)
            save()
            if state["dependencies"]["logic_ga_complete"] and state["dependencies"]["parity_insertion_complete"]:
                break
            time.sleep(60)
        for path, expected in plan["source_hashes"].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Scheduled implementation changed: {path}")
        from comparisons.llm_seeded_evolution.fixed_stream_report import collect
        logic_plan = json.loads((Path(plan["logic_dependency"]) / "plan.json").read_text())
        if len(collect(logic_plan)) != 40:
            raise ValueError("Logic completion artifacts do not verify")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", MPLCONFIGDIR="/local0/tmp/mpl", PYTHONUNBUFFERED="1",
                   COMET_LOG_ENV_DETAILS="false", COMET_LOG_ENV_CPU="false", COMET_LOG_ENV_GPU="false")
        env.pop("COMET_API_KEY", None)
        state["status"] = "auditing_source"
        save()
        with (root / "logs/source_audit.log").open("x") as log:
            subprocess.run(plan["source_audit_command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = json.loads((root / "source_audit.json").read_text())
        if audit["status"] != "pass" or len(audit["runs"]) != 20:
            raise ValueError("Source campaign did not pass its audit")
        (root / "replays").mkdir()
        jobs = []
        for row in sorted(audit["runs"], key=lambda r: r["seed"]):
            if row["status"] != "pass":
                raise ValueError("Source seed failed its audit")
            replay = export(Path(row["run_root"]), row["database_sha256"])
            path = root / "replays" / f"seed{row['seed']}.json"
            write_replay(path, replay)
            jobs.append({"task": "logic_parity", "seed": row["seed"], "replay": str(path),
                "replay_sha256": sha256(path), "output": str(root / "runs" / f"parity12_seed{row['seed']}"),
                "candidate_budget": replay["search_budget"] + replay["llm_evaluations"]})
        write_json(root / "replay_manifest.json", {"source_audit_sha256": sha256(root / "source_audit.json"), "jobs": jobs})
        from comparisons.llm_crn_generation.run_genai_net_llm_20seed_campaign import comet_smoke
        comet_smoke(plan["comet_project"], plan["campaign_id"])
        pending, active = list(jobs), []
        state.update(status="running", completed=[], failed=[])
        while pending or active:
            while pending and len(active) < plan["max_parallel"] and not state["failed"]:
                job = pending.pop(0)
                log = (root / "logs" / f"seed{job['seed']}.log").open("x")
                proc = subprocess.Popen(command(root, plan, job), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((job, proc, log))
            for job, proc, log in list(active):
                code = proc.poll()
                if code is not None:
                    active.remove((job, proc, log))
                    log.close()
                    done = Path(job["output"]) / "completed.json"
                    if code == 0 and done.exists():
                        state["completed"].append(job)
                    else:
                        state["failed"].append({**job, "exit_code": code})
            state.update(pending=pending, active=[{**job, "pid": proc.pid} for job, proc, _ in active])
            save()
            if state["failed"] and not active:
                raise RuntimeError("Parity queue halted for failure audit; no automatic reruns")
            if pending or active:
                time.sleep(15)
        state["status"] = "complete"
        save()
    except Exception as error:
        state.update(status="halted_for_audit", error_type=type(error).__name__, error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--check-once", action="store_true")
    args = parser.parse_args()
    if args.watch or args.check_once:
        watch(args.root.resolve(), args.check_once)
    else:
        prepare(args.root.resolve())


if __name__ == "__main__":
    main()
