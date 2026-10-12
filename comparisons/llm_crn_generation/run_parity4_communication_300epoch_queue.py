#!/usr/bin/env python3
"""Run the four-arm, 20-seed, 300-epoch parity communication experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL4CRN.llm.benchmark_prompts import get_mmc2_task_prompt, get_mmc2_task_prompt_variant
from RL4CRN.llm import workspace_loss_protocol as loss_protocol
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import (
    BUILDERS,
    CONFIGS,
    build_evaluator,
)
from comparisons.rpa_search.src.common.logic_task import build_logic_fn


PYTHON = ROOT / ".venv/bin/python"
CAMPAIGN_LAUNCHER = (
    ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"
)
AUDITOR = ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
CAMPAIGN_ROOT = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
DSH_HOME = Path.home() / "ai-workspaces/deepseek-test/.dsh-home"
SCHEDULE = GENERATED / "parity4_communication_300epoch_schedule.json"
LOG_ROOT = GENERATED / "parity4_communication_300epoch_logs"

TASK = "logic_parity"
MODES = ("full", "llm-to-rl", "rl-to-llm", "none")
MODE_SLUG = {mode: mode.replace("-", "_") for mode in MODES}
MODEL = "deepseek-v4-flash"
PROVIDER = "deepseek-official"
EPOCHS = 300
SEEDS = 20
SEED_START = 0
RL_BATCH_SIZE = 1023
BUDGET = 307200
MAX_PARALLEL_PER_ARM = 5
CPUS_PER_RUN = 4
GLOBAL_LLM_CONCURRENCY = 8
RL_GPU = "GPU-bee386ad-79df-5a1c-b5aa-7b2e140dd6ce"
DATE_TAG = "20260908"


def campaign_id(mode: str, protocol: str = "legacy") -> str:
    version = "v1" if protocol == "legacy" else "v2-workspace-loss"
    return f"flash-parity4-{version}-{mode}-300epoch-seeds0to19-{DATE_TAG}"


def method_name(mode: str, protocol: str = "legacy") -> str:
    version = "v1" if protocol == "legacy" else "v2_workspace_loss"
    return f"genai_net_llm_flash_parity4_{version}_{MODE_SLUG[mode]}"


def run_suffix(mode: str, protocol: str = "legacy") -> str:
    version = "v1" if protocol == "legacy" else "v2_workspace_loss"
    return f"cvode_llm_flash_parity4_{version}_{MODE_SLUG[mode]}"


def audit_path(mode: str, protocol: str = "legacy") -> Path:
    suffix = "" if protocol == "legacy" else "_workspace_loss_v1"
    return GENERATED / f"parity4_communication_300epoch{suffix}_{MODE_SLUG[mode]}_audit.json"


def schedule_path(protocol: str = "legacy") -> Path:
    if protocol == "legacy":
        return SCHEDULE
    return GENERATED / "parity4_communication_300epoch_workspace_loss_v1_schedule.json"


def environment() -> dict[str, str]:
    env = os.environ.copy()
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


def validate_task_definition(protocol: str = "legacy") -> dict:
    config = json.loads(CONFIGS[TASK].read_text(encoding="utf-8"))
    logic = config[TASK]
    expected = {
        "formula": "parity_4",
        "n_inputs": 4,
        "n_support_species": 2,
        "include_dilution": False,
        "solver": "CVODE",
        "n_t": 1000,
        "t_f": 100.0,
        "weights": "transient",
        "library_order": 2,
    }
    for field, value in expected.items():
        if logic.get(field) != value:
            raise RuntimeError(
                f"Parity task {field}={logic.get(field)!r}, expected {value!r}."
            )
    if config["search"].get("max_added_reactions") != 8:
        raise RuntimeError("Parity task must select exactly eight reactions.")
    if config["rl4crn"].get("epochs") != EPOCHS:
        raise RuntimeError("Parity task must run for 300 epochs.")

    parity = build_logic_fn("parity_4")
    truth_table = []
    for row in range(16):
        inputs = [(row >> index) & 1 for index in range(4)]
        target = int(parity(inputs))
        if target != sum(inputs) % 2:
            raise RuntimeError(f"Parity truth-table mismatch for {inputs}.")
        truth_table.append({"inputs": inputs, "target": target})

    _, _, task, _ = BUILDERS[TASK](config)
    expected_species = {"X_1", "X_2", "X_3", "X_4", "S_1", "S_2", "OUT"}
    if set(task.species_labels) != expected_species:
        raise RuntimeError(
            "Parity search species do not match the four inputs, two support "
            f"species, and one output: {task.species_labels!r}."
        )

    evaluator = build_evaluator(TASK, solver="CVODE")
    if evaluator.max_added_reactions != 8:
        raise RuntimeError("Parity evaluator reaction budget is not eight.")
    prompt = (get_mmc2_task_prompt(TASK, solver="CVODE") if protocol == "legacy" else
              get_mmc2_task_prompt_variant(TASK, variant=loss_protocol.TASK_VARIANT, solver="CVODE"))
    required_fragments = (
        "u_1 XOR u_2 XOR u_3 XOR u_4",
        "S_1 and S_2",
        "exactly eight distinct reactions",
        "transient-weighted mean absolute error",
        ("Do not restrict proposals to Hall-of-Fame derivatives" if protocol == "legacy" else
         "six evidence-guided refinements and four structurally different exploratory designs"),
    )
    missing = [fragment for fragment in required_fragments if fragment not in prompt]
    if missing:
        raise RuntimeError(f"Parity prompt is missing required text: {missing}")
    protocol_checks = {}
    if protocol != "legacy":
        from comparisons.llm_crn_generation.run_mmc2_harness_hybrid import LLM_RATE_MIN, LLM_RATE_MAX
        from RL4CRN.llm.harness_client import build_crn_output_contract

        # The smoke helper reads RL configuration bounds; production widens only LLM rates.
        evaluator.min_parameter_value = LLM_RATE_MIN
        evaluator.max_parameter_value = LLM_RATE_MAX
        contract = build_crn_output_contract(evaluator, num_candidates=10, task_description=prompt)
        allowed = {r["id"] for r in contract["reaction_library"]}
        if (allowed.intersection(contract["forbidden_reaction_ids"])
                or allowed.intersection(contract["template_reaction_ids"])
                or contract["null_reaction_id"] in allowed
                or contract["rules"]["allowed_parameter_range"] != [0.001, 100.0]):
            raise RuntimeError("Parity production proposal-space preflight failed")
        protocol_checks = {
            "production_llm_rate_bounds": contract["rules"]["allowed_parameter_range"],
            "allowed_reaction_count": len(allowed),
            "masked_reaction_ids": contract["forbidden_reaction_ids"],
            "design_constraints_example": loss_protocol.design_constraints(contract, "No exclusions supplied."),
        }
    return {
        **protocol_checks,
        "task": TASK,
        "config": str(CONFIGS[TASK]),
        "builder": BUILDERS[TASK].__name__,
        "truth_table": truth_table,
        "species": list(task.species_labels),
        "reaction_budget": evaluator.max_added_reactions,
        "prompt": prompt,
        "task_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def campaign_command(mode: str, protocol: str = "legacy") -> list[str]:
    return [
        str(PYTHON),
        str(CAMPAIGN_LAUNCHER),
        "--campaign-id",
        campaign_id(mode, protocol),
        "--tasks",
        TASK,
        "--max-parallel",
        str(MAX_PARALLEL_PER_ARM),
        "--cpus-per-run",
        str(CPUS_PER_RUN),
        "--rl-gpu",
        RL_GPU,
        "--seeds",
        str(SEEDS),
        "--seed-start",
        str(SEED_START),
        "--epochs",
        str(EPOCHS),
        "--rl-batch-size",
        str(RL_BATCH_SIZE),
        "--total-candidate-budget",
        str(BUDGET),
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
        str(GLOBAL_LLM_CONCURRENCY),
        "--comet-project",
        "genai-net-llm-parity",
        "--campaign-stage",
        "paper",
        "--model",
        MODEL,
        "--task-prompt-variant",
        "standard" if protocol == "legacy" else loss_protocol.TASK_VARIANT,
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
        method_name(mode, protocol),
        "--run-suffix",
        run_suffix(mode, protocol),
        "--dsh-home",
        str(DSH_HOME),
        "--workspace-root",
        str(CAMPAIGN_ROOT),
        "--skip-postprocessing",
    ]


def audit_command(mode: str, protocol: str = "legacy") -> list[str]:
    command = [
        str(PYTHON),
        str(AUDITOR),
        str(CAMPAIGN_ROOT / campaign_id(mode, protocol)),
        "--tasks",
        TASK,
        "--seed-start",
        str(SEED_START),
        "--seeds",
        str(SEEDS),
        "--epochs",
        str(EPOCHS),
        "--rl-batch-size",
        str(RL_BATCH_SIZE),
        "--budget",
        str(BUDGET),
        "--method-name",
        method_name(mode, protocol),
        "--run-suffix",
        run_suffix(mode, protocol),
        "--communication-mode",
        mode,
        "--model",
        MODEL,
        "--output",
        str(audit_path(mode, protocol)),
    ]
    if protocol != "legacy":
        command += ["--task-prompt-variant", loss_protocol.TASK_VARIANT]
    return command


def write_schedule(status: str, task_definition: dict, *, protocol: str = "legacy", **extra: object) -> None:
    record = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "task_definition": task_definition,
        "model": MODEL,
        "provider": PROVIDER,
        "harness_protocol": protocol,
        "task_prompt_variant": "standard" if protocol == "legacy" else loss_protocol.TASK_VARIANT,
        "communication_modes": list(MODES),
        "seeds": list(range(SEED_START, SEED_START + SEEDS)),
        "epochs": EPOCHS,
        "evaluation_budget": BUDGET,
        "concurrent_arms": 4,
        "maximum_concurrent_seed_jobs": len(MODES) * MAX_PARALLEL_PER_ARM,
        "maximum_cvode_worker_slots": (
            len(MODES) * MAX_PARALLEL_PER_ARM * CPUS_PER_RUN
        ),
        "global_harness_concurrency": GLOBAL_LLM_CONCURRENCY,
        "campaigns": {
            mode: {
                "campaign_id": campaign_id(mode, protocol),
                "method_name": method_name(mode, protocol),
                "run_suffix": run_suffix(mode, protocol),
                "audit": str(audit_path(mode, protocol)),
                "command": campaign_command(mode, protocol),
            }
            for mode in MODES
        },
        **extra,
    }
    schedule_path(protocol).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_protocol_review(definition: dict) -> None:
    """Freeze review artifacts separately from the manuscript's historical catalog."""
    payload = loss_protocol.review_payload()
    texts, hashes = payload["texts"], payload["sha256"]
    if texts["TASK.md"] != definition["prompt"]:
        raise RuntimeError("Parity task and frozen review differ")
    path = loss_protocol.REVIEW_PATH
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen parity prompt review differs; use a new protocol version.")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    lines = ["# Parity workspace-loss-v1 protocol", "", "Separate planned replacement; historical prompts and runs remain unchanged."]
    for name, text in texts.items():
        lines += ["", f"## {name}", "", f"SHA-256: `{hashes[name]}`", "", "```text", text.rstrip(), "```"]
    (GENERATED / "PARITY4_WORKSPACE_LOSS_V1_PROMPTS.md").write_text("\n".join(lines)+"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("legacy", loss_protocol.PROTOCOL), default="legacy")
    parser.add_argument("--plan-only", action="store_true", help="Validate and freeze the plan without starting experiments.")
    args = parser.parse_args()
    protocol = args.protocol
    GENERATED.mkdir(parents=True, exist_ok=True)
    log_root = LOG_ROOT if protocol == "legacy" else GENERATED / "parity4_workspace_loss_v1_logs"
    definition = validate_task_definition(protocol)
    if protocol != "legacy":
        if any((CAMPAIGN_ROOT / campaign_id(mode, protocol)).exists() for mode in MODES):
            raise SystemExit("Replacement parity output already exists; refusing to overwrite or relaunch.")
        freeze_protocol_review(definition)
        if not args.plan_only:
            loss_protocol.validate_protocol_release(loss_protocol.TASK_VARIANT)
    write_schedule("planned" if args.plan_only else "validated", definition, protocol=protocol)
    if args.plan_only:
        print(f"[planned, not launched] {schedule_path(protocol)}")
        return
    log_root.mkdir(parents=True, exist_ok=True)

    active = {}
    arm_status = {mode: "queued" for mode in MODES}
    for mode in MODES:
        log_path = log_root / f"{campaign_id(mode, protocol)}.log"
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            campaign_command(mode, protocol),
            cwd=ROOT,
            env=environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        active[mode] = (process, log_handle, log_path)
        arm_status[mode] = "running"
        print(f"[start] {mode} pid={process.pid} log={log_path}", flush=True)
    write_schedule("running", definition, protocol=protocol, arm_status=arm_status)

    while active:
        for mode, (process, log_handle, log_path) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle.close()
            arm_status[mode] = "complete" if returncode == 0 else f"failed:{returncode}"
            del active[mode]
            write_schedule("running", definition, protocol=protocol, arm_status=arm_status)
            print(f"[finish] {mode} rc={returncode} log={log_path}", flush=True)
        if active:
            time.sleep(10)

    failed = [mode for mode, status in arm_status.items() if status != "complete"]
    if failed:
        write_schedule("failed", definition, protocol=protocol, arm_status=arm_status, failed_arms=failed)
        raise SystemExit(f"Parity campaigns failed: {failed}")

    audit_status = {}
    for mode in MODES:
        completed = subprocess.run(audit_command(mode, protocol), cwd=ROOT, env=environment())
        audit_status[mode] = "pass" if completed.returncode == 0 else "fail"
    final_status = "complete" if set(audit_status.values()) == {"pass"} else "audit-failed"
    write_schedule(final_status, definition, protocol=protocol, arm_status=arm_status, audit_status=audit_status)
    if final_status != "complete":
        raise SystemExit("One or more parity campaign audits failed.")


if __name__ == "__main__":
    main()
