#!/usr/bin/env python3
"""Queue and audit the two DeepSeek Flash half-duplex communication arms."""

from __future__ import annotations

import argparse
import json
import os
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


PYTHON = ROOT / ".venv/bin/python"
LAUNCHER = ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
AUDITOR = ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"
PREFLIGHT = ROOT / "comparisons/llm_crn_generation/preflight_contract_v2.py"
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
CAMPAIGN_ROOT = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
DSH_HOME = Path.home() / "ai-workspaces/deepseek-test/.dsh-home"
LOCAL_SCHEDULE = GENERATED / "contract_v2_local_qwen_half_duplex_schedule.json"
SCHEDULE_PATH = GENERATED / "contract_v2_flash_half_duplex_schedule.json"
LOG_ROOT = GENERATED / "flash_half_duplex_logs"
PRIMARY_AUDITS = (
    GENERATED / "contract_v2_full_seed0_audit.json",
    GENERATED / "contract_v2_full_seeds1to19_audit.json",
    GENERATED / "contract_v2_none_seed0_audit.json",
    GENERATED / "contract_v2_none_seeds1to19_audit.json",
)
RL_GPU = "GPU-bee386ad-79df-5a1c-b5aa-7b2e140dd6ce"
MODEL = "deepseek-v4-flash"
PROVIDER = "deepseek-official"
TASKS = ("rpa", "logic")
MAIN_TASK_ORDER = ("logic", "rpa")
MODES = ("llm-to-rl", "rl-to-llm")
MODE_SLUG = {"llm-to-rl": "llm_to_rl", "rl-to-llm": "rl_to_llm"}
DATE_TAG = "20260904"
STATUS_LOCK = threading.Lock()
TASK_STATUS = {task: "queued" for task in TASKS}


def environment() -> dict[str, str]:
    env = os.environ.copy()
    # Let Comet use its persisted credential instead of a stale tmux override.
    env.pop("COMET_API_KEY", None)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "MPLCONFIGDIR": "/local0/tmp/mpl",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def campaign_id(task: str, mode: str, phase: str) -> str:
    seed_label = "sentinel-seed0" if phase == "sentinel" else "seeds1to19"
    return f"flash-v2-half-{task}-{mode}-{seed_label}-{DATE_TAG}"


def method_name(mode: str) -> str:
    return f"genai_net_llm_flash_contract_v2_{MODE_SLUG[mode]}"


def run_suffix(mode: str) -> str:
    return f"cvode_llm_flash_contract_v2_{MODE_SLUG[mode]}"


def audit_path(task: str, mode: str, phase: str) -> Path:
    return GENERATED / f"contract_v2_flash_half_{task}_{MODE_SLUG[mode]}_{phase}_audit.json"


def campaign_command(task: str, mode: str, phase: str) -> list[str]:
    sentinel = phase == "sentinel"
    return [
        str(PYTHON),
        str(LAUNCHER),
        "--campaign-id",
        campaign_id(task, mode, phase),
        "--tasks",
        task,
        "--max-parallel",
        "1" if sentinel else "10",
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
        "genai-net-llm-paper",
        "--campaign-stage",
        "sentinel" if sentinel else "paper",
        "--model",
        MODEL,
        "--task-prompt-variant",
        "standard",
        "--generation-backend",
        "harness",
        "--llm-provider",
        PROVIDER,
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
        str(CAMPAIGN_ROOT),
        "--skip-postprocessing",
    ]


def audit_command(task: str, mode: str, phase: str) -> list[str]:
    sentinel = phase == "sentinel"
    return [
        str(PYTHON),
        str(AUDITOR),
        str(CAMPAIGN_ROOT / campaign_id(task, mode, phase)),
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
        MODEL,
        "--output",
        str(audit_path(task, mode, phase)),
    ]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_schedule(status: str, **extra: object) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dependency": str(LOCAL_SCHEDULE),
        "model": MODEL,
        "provider": PROVIDER,
        "tasks": list(TASKS),
        "task_status": dict(TASK_STATUS),
        "modes": list(MODES),
        "sentinel_seeds": [0],
        "main_seeds": list(range(1, 20)),
        "request_0_hof": "withheld-empty",
        "later_state_exposure": {"llm-to-rl": False, "rl-to-llm": True},
        "pre_boundary_llm_insertion": {"llm-to-rl": True, "rl-to-llm": False},
        "main_maximum_cvode_worker_slots": 80,
        "main_concurrent_seed_jobs": 20,
        "main_task_order": list(MAIN_TASK_ORDER),
        "global_harness_concurrency": 8,
        "primary_control_audits": [str(path) for path in PRIMARY_AUDITS],
        **extra,
    }
    SCHEDULE_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def update_task_status(task: str, status: str, **extra: object) -> None:
    with STATUS_LOCK:
        TASK_STATUS[task] = status
        write_schedule("running", **extra)


def wait_for_local_half_duplex() -> None:
    write_schedule("waiting-for-local-half-duplex")
    print(f"[wait] local half-duplex dependency: {LOCAL_SCHEDULE}", flush=True)
    while True:
        record = read_json(LOCAL_SCHEDULE)
        status = record.get("status")
        if status == "complete":
            print("[ready] local half-duplex campaign is audited complete", flush=True)
            return
        if status == "failed":
            write_schedule("blocked-by-local-half-duplex", dependency_status=status)
            raise RuntimeError("Local half-duplex campaign failed; Flash queue remains closed.")
        time.sleep(30)


def verify_primary_controls() -> None:
    failures = [str(path) for path in PRIMARY_AUDITS if read_json(path).get("status") != "pass"]
    if failures:
        raise RuntimeError("Flash full/no-communication controls are not audited: " + ", ".join(failures))


def run_phase(task: str, phase: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    update_task_status(task, f"running_{phase}")
    active: list[tuple[str, subprocess.Popen[bytes], object, Path]] = []
    for mode in MODES:
        log_path = LOG_ROOT / f"{campaign_id(task, mode, phase)}.log"
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            campaign_command(task, mode, phase),
            cwd=ROOT,
            env=environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        active.append((mode, process, log_handle, log_path))
        print(f"[start] {task} {phase} {mode} pid={process.pid} log={log_path}", flush=True)

    failures = []
    for mode, process, log_handle, log_path in active:
        returncode = process.wait()
        log_handle.close()
        if returncode:
            failures.append(f"{mode}: rc={returncode}, log={log_path}")
    if failures:
        update_task_status(task, f"failed_{phase}", failures=failures)
        raise RuntimeError("; ".join(failures))

    audit_failures: list[str] = []
    for mode in MODES:
        completed = subprocess.run(
            audit_command(task, mode, phase),
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
        update_task_status(task, f"failed_{phase}_audit", failures=audit_failures)
        raise RuntimeError("; ".join(audit_failures))


def verify_sentinel_audits() -> None:
    failures = [
        str(audit_path(task, mode, "sentinel"))
        for task in TASKS
        for mode in MODES
        if read_json(audit_path(task, mode, "sentinel")).get("status") != "pass"
    ]
    if failures:
        raise RuntimeError("Flash directional sentinels are not audited: " + ", ".join(failures))


def main() -> None:
    global TASKS, MAIN_TASK_ORDER, TASK_STATUS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-local-dependency",
        action="store_true",
        help=(
            "Start after verifying controls even when the operationally prior local "
            "campaign needs infrastructure-only recovery."
        ),
    )
    parser.add_argument(
        "--skip-sentinels",
        action="store_true",
        help="Require existing passing sentinel audits and proceed directly to Logic main runs.",
    )
    parser.add_argument(
        "--logic-only",
        action="store_true",
        help="Run only the Logic directional sentinels and cohort.",
    )
    args = parser.parse_args()
    if args.logic_only:
        TASKS = ("logic",)
        MAIN_TASK_ORDER = ("logic",)
        TASK_STATUS = {"logic": "queued"}
    write_schedule("queued")
    if args.skip_local_dependency:
        dependency_status = read_json(LOCAL_SCHEDULE).get("status", "unknown")
        write_schedule(
            "local-dependency-overridden",
            dependency_status=dependency_status,
            dependency_override_reason=(
                "User-authorized Flash launch after local execution completed; "
                "four ENOSPC arm-seed records await targeted recovery."
            ),
        )
        print(
            f"[override] local half-duplex status is {dependency_status}; "
            "continuing with the independent Flash campaign",
            flush=True,
        )
    else:
        wait_for_local_half_duplex()
    verify_primary_controls()
    subprocess.run([str(PYTHON), str(PREFLIGHT)], cwd=ROOT, env=environment(), check=True)

    failures: list[str] = []
    if args.skip_sentinels:
        verify_sentinel_audits()
        for task in TASKS:
            TASK_STATUS[task] = "sentinel_audited"
        write_schedule("sentinels-complete")
    else:
        with ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
            futures = {task: executor.submit(run_phase, task, "sentinel") for task in TASKS}
            for task, future in futures.items():
                try:
                    future.result()
                    TASK_STATUS[task] = "sentinel_audited"
                except Exception as error:
                    failures.append(f"{task}: {error}")
    if failures:
        write_schedule("failed", failures=failures)
        raise RuntimeError("; ".join(failures))

    for task in MAIN_TASK_ORDER:
        run_phase(task, "main")
        update_task_status(task, "complete")
    write_schedule("complete")
    print(f"[complete] Flash half-duplex campaign: {SCHEDULE_PATH}", flush=True)


if __name__ == "__main__":
    main()
