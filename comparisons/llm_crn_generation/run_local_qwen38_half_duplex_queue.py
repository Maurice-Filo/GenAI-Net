#!/usr/bin/env python3
"""Queue and audit the two local-Qwen half-duplex communication arms."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa import (
    probe_tool_call,
    require_qwen38_model,
)
from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa_logic_campaign import (
    CAMPAIGN_BASE,
    DSH_HOME,
    ENDPOINT,
    PYTHON,
    RL_GPU,
    environment,
)


GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
SCHEDULE_PATH = GENERATED / "contract_v2_local_qwen_half_duplex_schedule.json"
LOG_ROOT = GENERATED / "local_qwen_half_duplex_logs"
PREREQUISITE_AUDITS = {
    "rpa": GENERATED / "contract_v2_local_qwen_rpa_no_communication_audit.json",
    "logic": GENERATED / "contract_v2_local_qwen_logic_no_communication_audit.json",
}
TASKS = ("rpa", "logic")
MODES = ("llm-to-rl", "rl-to-llm")
MODE_SLUG = {"llm-to-rl": "llm_to_rl", "rl-to-llm": "rl_to_llm"}
DATE_TAG = "20260904"
STATUS_LOCK = threading.Lock()
TASK_STATUS = {task: "queued" for task in TASKS}


def campaign_id(task: str, mode: str, phase: str) -> str:
    seed_label = "sentinel-seed0" if phase == "sentinel" else "seeds1to19"
    return f"qwen38-local-v2-half-{task}-{mode}-{seed_label}-{DATE_TAG}"


def method_name(mode: str) -> str:
    return f"genai_net_llm_qwen38_local_contract_v2_{MODE_SLUG[mode]}"


def run_suffix(mode: str) -> str:
    return f"cvode_llm_qwen38_local_contract_v2_{MODE_SLUG[mode]}"


def audit_path(task: str, mode: str, phase: str) -> Path:
    return GENERATED / (
        f"contract_v2_local_qwen_half_{task}_{MODE_SLUG[mode]}_{phase}_audit.json"
    )


def campaign_command(model: str, task: str, mode: str, phase: str) -> list[str]:
    sentinel = phase == "sentinel"
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"),
        "--campaign-id",
        campaign_id(task, mode, phase),
        "--tasks",
        task,
        "--max-parallel",
        "1" if sentinel else "4",
        "--cpus-per-run",
        "4",
        "--rl-gpu",
        RL_GPU,
        "--seeds",
        "1" if sentinel else "19",
        "--seed-start",
        "0" if sentinel else "1",
        "--epochs",
        "100",
        "--rl-batch-size",
        "1023",
        "--total-candidate-budget",
        "102400",
        "--llm-candidates",
        "10",
        "--llm-every",
        "20",
        "--max-llm-in-flight",
        "5",
        "--max-agent-evaluations",
        "0",
        "--max-literature-searches",
        "0",
        "--forbidden-topology-m",
        "0",
        "--llm-timeout",
        "3600",
        "--global-llm-concurrency",
        "8",
        "--comet-project",
        "genai-net-local-llm-paper",
        "--campaign-stage",
        "sentinel" if sentinel else "paper",
        "--model",
        model,
        "--task-prompt-variant",
        "standard",
        "--generation-backend",
        "harness",
        "--llm-provider",
        "local-qwen38-18081",
        "--llm-base-url",
        ENDPOINT,
        "--local-llm-context-window",
        "262144",
        "--local-llm-max-output-tokens",
        "8192",
        "--communication-mode",
        mode,
        "--withhold-initial-hof",
        "--candidate-validation-policy",
        "independent-members",
        "--method-name",
        method_name(mode),
        "--run-suffix",
        run_suffix(mode),
        "--dsh-home",
        str(DSH_HOME),
        "--workspace-root",
        str(CAMPAIGN_BASE),
        "--skip-postprocessing",
    ]


def audit_command(model: str, task: str, mode: str, phase: str) -> list[str]:
    sentinel = phase == "sentinel"
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"),
        str(CAMPAIGN_BASE / campaign_id(task, mode, phase)),
        "--tasks",
        task,
        "--seed-start",
        "0" if sentinel else "1",
        "--seeds",
        "1" if sentinel else "19",
        "--epochs",
        "100",
        "--rl-batch-size",
        "1023",
        "--budget",
        "102400",
        "--method-name",
        method_name(mode),
        "--run-suffix",
        run_suffix(mode),
        "--communication-mode",
        mode,
        "--model",
        model,
        "--output",
        str(audit_path(task, mode, phase)),
    ]


def read_passing_audit(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "pass"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def write_schedule(status: str, **extra: object) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_endpoint": ENDPOINT,
        "tasks": list(TASKS),
        "task_status": dict(TASK_STATUS),
        "modes": list(MODES),
        "sentinel_seeds": [0],
        "main_seeds": list(range(1, 20)),
        "request_0_hof": "withheld-empty",
        "later_state_exposure": {
            "llm-to-rl": False,
            "rl-to-llm": True,
        },
        "pre_boundary_llm_insertion": {
            "llm-to-rl": True,
            "rl-to-llm": False,
        },
        "main_cvode_worker_slots_per_task": 32,
        "main_maximum_cvode_worker_slots": 64,
        "global_harness_concurrency": 8,
        "prerequisite_audits": {
            task: str(path) for task, path in PREREQUISITE_AUDITS.items()
        },
        **extra,
    }
    SCHEDULE_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def update_task_status(task: str, status: str, **extra: object) -> None:
    with STATUS_LOCK:
        TASK_STATUS[task] = status
        write_schedule("running", **extra)


def wait_for_prerequisite(task: str) -> None:
    path = PREREQUISITE_AUDITS[task]
    update_task_status(task, "waiting_for_no_communication_audit")
    last_notice = 0.0
    while not read_passing_audit(path):
        now = time.monotonic()
        if now - last_notice >= 300 or last_notice == 0:
            print(f"[wait] {task} prerequisite audit not yet passing: {path.name}", flush=True)
            last_notice = now
        time.sleep(30)


def run_phase(model: str, task: str, phase: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    update_task_status(task, f"running_{phase}", model=model)
    active: list[tuple[str, subprocess.Popen[bytes], object, Path]] = []
    for mode in MODES:
        log_path = LOG_ROOT / f"{campaign_id(task, mode, phase)}.log"
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            campaign_command(model, task, mode, phase),
            cwd=ROOT,
            env=environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        active.append((mode, process, log_handle, log_path))
        print(f"[start] {task} {phase} {mode} pid={process.pid} log={log_path}", flush=True)

    failures: list[str] = []
    for mode, process, log_handle, log_path in active:
        returncode = process.wait()
        log_handle.close()
        if returncode:
            failures.append(f"{mode}: rc={returncode}, log={log_path}")
    if failures:
        update_task_status(task, f"failed_{phase}", failures=failures, model=model)
        raise RuntimeError("; ".join(failures))

    audit_failures: list[str] = []
    for mode in MODES:
        completed = subprocess.run(
            audit_command(model, task, mode, phase),
            cwd=ROOT,
            env=environment(),
            check=False,
        )
        if completed.returncode:
            audit_failures.append(f"{mode}: audit rc={completed.returncode}")
            print(f"[audit-fail] {task} {phase} {mode}", flush=True)
        else:
            print(f"[audit-pass] {task} {phase} {mode}", flush=True)
    if audit_failures:
        update_task_status(
            task,
            f"failed_{phase}_audit",
            failures=audit_failures,
            model=model,
        )
        raise RuntimeError("; ".join(audit_failures))


def run_task_pipeline(task: str) -> None:
    wait_for_prerequisite(task)
    model = require_qwen38_model(ENDPOINT)
    probe_tool_call(ENDPOINT, model)
    update_task_status(task, "model_preflight_passed", model=model)
    run_phase(model, task, "sentinel")
    run_phase(model, task, "main")
    update_task_status(task, "complete", model=model)


def main() -> None:
    write_schedule("running")
    failures = []
    with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        futures = {task: executor.submit(run_task_pipeline, task) for task in TASKS}
        for task, future in futures.items():
            try:
                future.result()
            except Exception as error:
                failures.append(f"{task}: {error}")
    if failures:
        write_schedule("failed", failures=failures)
        raise RuntimeError("; ".join(failures))
    write_schedule("complete")
    print(f"[complete] local half-duplex campaign: {SCHEDULE_PATH}", flush=True)


if __name__ == "__main__":
    main()
