#!/usr/bin/env python3
"""Run the gated contract-v2 sentinel and Cohort B paper campaigns."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
LAUNCHER = ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
SENTINEL_AUDIT = ROOT / "comparisons/llm_crn_generation/audit_contract_v2_sentinel.py"
CAMPAIGN_AUDIT = ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"
STATIC_PREFLIGHT = ROOT / "comparisons/llm_crn_generation/preflight_contract_v2.py"
ANALYZER = ROOT / "paper/iclr2027_genai_net_llm/analyze_contract_v2_primary.py"
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
CAMPAIGN_ROOT = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
LOCAL_STATUS = CAMPAIGN_ROOT / "qwen38-local-v2-rpa-seed9000-100epoch-20260902-r8/status.json"
RL_GPU = "GPU-bee386ad-79df-5a1c-b5aa-7b2e140dd6ce"
TASKS = (
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
)
COMMUNICATION_TASKS = ("rpa", "logic", "classifier")
SENTINEL_ID = "flash-v2-sentinel-8task-seed9000-21epoch-20260902-r5"
FULL_PILOT_ID = "flash-v2-primary-full-100epoch-seed0-20260902"
NONE_PILOT_ID = "flash-v2-communication-none-100epoch-seed0-20260902"
FULL_MAIN_ID = "flash-v2-primary-full-100epoch-seeds1to19-20260902"
NONE_MAIN_ID = "flash-v2-communication-none-100epoch-seeds1to19-20260902"
FULL_METHOD = "genai_net_llm_flash_contract_v2_full"
FULL_SUFFIX = "cvode_llm_flash_contract_v2_full"
NONE_METHOD = "genai_net_llm_flash_contract_v2_no_communication"
NONE_SUFFIX = "cvode_llm_flash_contract_v2_no_communication"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    # The long-lived tmux server can retain a stale override after ~/.comet.config
    # is refreshed. Let Comet read the persisted credential for this campaign.
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


def run(command: Sequence[str], *, log: Path | None = None) -> None:
    print("[queue]", " ".join(map(str, command)), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=environment(), check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_parallel(commands: Sequence[tuple[Sequence[str], Path]]) -> None:
    processes = []
    for command, log in commands:
        print("[queue]", " ".join(map(str, command)), flush=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, handle, log))
    failures = []
    for process, handle, log in processes:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append(f"{log} (exit {returncode})")
    if failures:
        raise RuntimeError("Parallel campaign block failed: " + ", ".join(failures))


def wait_for_local_validation() -> None:
    print(f"[queue] waiting for local validation: {LOCAL_STATUS}", flush=True)
    while True:
        if LOCAL_STATUS.is_file():
            status = json.loads(LOCAL_STATUS.read_text(encoding="utf-8"))
            if status.get("failed"):
                raise RuntimeError("Local Qwen validation failed; paper queue remains closed.")
            if not status.get("active") and not status.get("pending"):
                if status.get("completed"):
                    print("[queue] local validation completed", flush=True)
                    return
                raise RuntimeError("Local Qwen validation stopped without a completion record.")
        time.sleep(15)


def launcher_command(
    *,
    campaign_id: str,
    tasks: Sequence[str],
    seed_start: int,
    seeds: int,
    epochs: int,
    budget: int,
    max_parallel: int,
    communication_mode: str,
    method_name: str,
    run_suffix: str,
    stage: str,
) -> list[str]:
    return [
        str(PYTHON),
        str(LAUNCHER),
        "--campaign-id",
        campaign_id,
        "--tasks",
        *tasks,
        "--max-parallel",
        str(max_parallel),
        "--cpus-per-run",
        "4",
        "--rl-gpu",
        RL_GPU,
        "--seeds",
        str(seeds),
        "--seed-start",
        str(seed_start),
        "--epochs",
        str(epochs),
        "--rl-batch-size",
        "1023",
        "--total-candidate-budget",
        str(budget),
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
        "--llm-timeout",
        "3600",
        "--global-llm-concurrency",
        "8",
        "--comet-project",
        "genai-net-llm-paper",
        "--campaign-stage",
        stage,
        "--model",
        "deepseek-v4-flash",
        "--task-prompt-variant",
        "standard",
        "--generation-backend",
        "harness",
        "--llm-provider",
        "deepseek-official",
        "--communication-mode",
        communication_mode,
        "--withhold-initial-hof",
        "--candidate-validation-policy",
        "independent-members",
        "--method-name",
        method_name,
        "--run-suffix",
        run_suffix,
        "--skip-postprocessing",
    ]


def campaign_audit_command(
    campaign_id: str,
    *,
    tasks: Sequence[str],
    seed_start: int,
    seeds: int,
    method_name: str,
    run_suffix: str,
    communication_mode: str,
    output: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str(CAMPAIGN_AUDIT),
        str(CAMPAIGN_ROOT / campaign_id),
        "--tasks",
        *tasks,
        "--seed-start",
        str(seed_start),
        "--seeds",
        str(seeds),
        "--epochs",
        "100",
        "--rl-batch-size",
        "1023",
        "--budget",
        "102400",
        "--method-name",
        method_name,
        "--run-suffix",
        run_suffix,
        "--communication-mode",
        communication_mode,
        "--output",
        str(output),
    ]


def write_registry(audits: Sequence[Path]) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in audits]
    if any(record.get("status") != "pass" for record in records):
        raise RuntimeError("Cannot freeze a primary registry from a failing campaign audit.")
    registry = {
        "schema_version": 2,
        "status": "audited-complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_space_contract_version": 2,
        "full_duplex_method": FULL_METHOD,
        "full_duplex_run_suffix": FULL_SUFFIX,
        "no_communication_method": NONE_METHOD,
        "no_communication_run_suffix": NONE_SUFFIX,
        "full_duplex_campaigns": [FULL_PILOT_ID, FULL_MAIN_ID],
        "no_communication_campaigns": [NONE_PILOT_ID, NONE_MAIN_ID],
        "tasks": list(TASKS),
        "communication_tasks": list(COMMUNICATION_TASKS),
        "seeds": list(range(20)),
        "epochs": 100,
        "candidate_budget": 102400,
        "audit_files": [str(path.relative_to(ROOT)) for path in audits],
    }
    path = GENERATED / "contract_v2_primary_registry.json"
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[queue] froze {path}", flush=True)


def main() -> None:
    wait_for_local_validation()
    run([str(PYTHON), str(STATIC_PREFLIGHT)])

    sentinel_command = launcher_command(
        campaign_id=SENTINEL_ID,
        tasks=TASKS,
        seed_start=9000,
        seeds=1,
        epochs=21,
        budget=21503,
        max_parallel=8,
        communication_mode="full",
        method_name="genai_net_llm_flash_contract_v2_sentinel_r5",
        run_suffix="cvode_llm_flash_contract_v2_sentinel_r5",
        stage="sentinel",
    )
    run(sentinel_command, log=CAMPAIGN_ROOT / SENTINEL_ID / "queue.log")
    run(
        [
            str(PYTHON),
            str(SENTINEL_AUDIT),
            str(CAMPAIGN_ROOT / SENTINEL_ID),
            "--method-name",
            "genai_net_llm_flash_contract_v2_sentinel_r5",
            "--run-suffix",
            "cvode_llm_flash_contract_v2_sentinel_r5",
            "--output",
            str(GENERATED / "contract_v2_sentinel_report.json"),
        ]
    )

    run_parallel(
        (
            (
                launcher_command(
                    campaign_id=FULL_PILOT_ID,
                    tasks=TASKS,
                    seed_start=0,
                    seeds=1,
                    epochs=100,
                    budget=102400,
                    max_parallel=8,
                    communication_mode="full",
                    method_name=FULL_METHOD,
                    run_suffix=FULL_SUFFIX,
                    stage="paper",
                ),
                CAMPAIGN_ROOT / FULL_PILOT_ID / "queue.log",
            ),
            (
                launcher_command(
                    campaign_id=NONE_PILOT_ID,
                    tasks=COMMUNICATION_TASKS,
                    seed_start=0,
                    seeds=1,
                    epochs=100,
                    budget=102400,
                    max_parallel=3,
                    communication_mode="none",
                    method_name=NONE_METHOD,
                    run_suffix=NONE_SUFFIX,
                    stage="paper",
                ),
                CAMPAIGN_ROOT / NONE_PILOT_ID / "queue.log",
            ),
        )
    )

    audit_paths = (
        GENERATED / "contract_v2_full_seed0_audit.json",
        GENERATED / "contract_v2_none_seed0_audit.json",
        GENERATED / "contract_v2_full_seeds1to19_audit.json",
        GENERATED / "contract_v2_none_seeds1to19_audit.json",
    )
    run(
        campaign_audit_command(
            FULL_PILOT_ID,
            tasks=TASKS,
            seed_start=0,
            seeds=1,
            method_name=FULL_METHOD,
            run_suffix=FULL_SUFFIX,
            communication_mode="full",
            output=audit_paths[0],
        )
    )
    run(
        campaign_audit_command(
            NONE_PILOT_ID,
            tasks=COMMUNICATION_TASKS,
            seed_start=0,
            seeds=1,
            method_name=NONE_METHOD,
            run_suffix=NONE_SUFFIX,
            communication_mode="none",
            output=audit_paths[1],
        )
    )

    run_parallel(
        (
            (
                launcher_command(
                    campaign_id=FULL_MAIN_ID,
                    tasks=TASKS,
                    seed_start=1,
                    seeds=19,
                    epochs=100,
                    budget=102400,
                    max_parallel=10,
                    communication_mode="full",
                    method_name=FULL_METHOD,
                    run_suffix=FULL_SUFFIX,
                    stage="paper",
                ),
                CAMPAIGN_ROOT / FULL_MAIN_ID / "queue.log",
            ),
            (
                launcher_command(
                    campaign_id=NONE_MAIN_ID,
                    tasks=COMMUNICATION_TASKS,
                    seed_start=1,
                    seeds=19,
                    epochs=100,
                    budget=102400,
                    max_parallel=5,
                    communication_mode="none",
                    method_name=NONE_METHOD,
                    run_suffix=NONE_SUFFIX,
                    stage="paper",
                ),
                CAMPAIGN_ROOT / NONE_MAIN_ID / "queue.log",
            ),
        )
    )
    run(
        campaign_audit_command(
            FULL_MAIN_ID,
            tasks=TASKS,
            seed_start=1,
            seeds=19,
            method_name=FULL_METHOD,
            run_suffix=FULL_SUFFIX,
            communication_mode="full",
            output=audit_paths[2],
        )
    )
    run(
        campaign_audit_command(
            NONE_MAIN_ID,
            tasks=COMMUNICATION_TASKS,
            seed_start=1,
            seeds=19,
            method_name=NONE_METHOD,
            run_suffix=NONE_SUFFIX,
            communication_mode="none",
            output=audit_paths[3],
        )
    )
    write_registry(audit_paths)
    run([str(PYTHON), str(ANALYZER)])
    print("[queue] Cohort B and its frozen analysis are complete", flush=True)


if __name__ == "__main__":
    main()
