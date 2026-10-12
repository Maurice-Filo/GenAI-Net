#!/usr/bin/env python3
"""Run and audit the matched local-Qwen RPA no-communication cohort."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    ROOT,
    environment,
)


CAMPAIGN_ID = "qwen38-local-v2-rpa-no-communication-100epoch-20seed-20260903"
METHOD = "genai_net_llm_qwen38_local_contract_v2_no_communication"
RUN_SUFFIX = "cvode_llm_qwen38_local_contract_v2_no_communication"
CAMPAIGN_ROOT = CAMPAIGN_BASE / CAMPAIGN_ID
AUDIT_PATH = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/"
    "contract_v2_local_qwen_rpa_no_communication_audit.json"
)


def build_campaign_command(
    model: str,
    *,
    campaign_id: str,
    task: str,
    method: str,
    run_suffix: str,
) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"),
        "--campaign-id",
        campaign_id,
        "--tasks",
        task,
        "--max-parallel",
        "8",
        "--cpus-per-run",
        "4",
        "--rl-gpu",
        RL_GPU,
        "--seeds",
        "20",
        "--seed-start",
        "0",
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
        "--llm-timeout",
        "3600",
        "--global-llm-concurrency",
        "8",
        "--comet-project",
        "genai-net-local-llm-paper",
        "--campaign-stage",
        "paper",
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
        "none",
        "--withhold-initial-hof",
        "--candidate-validation-policy",
        "independent-members",
        "--method-name",
        method,
        "--run-suffix",
        run_suffix,
        "--dsh-home",
        str(DSH_HOME),
        "--workspace-root",
        str(CAMPAIGN_BASE),
        "--skip-postprocessing",
    ]


def campaign_command(model: str) -> list[str]:
    return build_campaign_command(
        model,
        campaign_id=CAMPAIGN_ID,
        task="rpa",
        method=METHOD,
        run_suffix=RUN_SUFFIX,
    )


def build_audit_command(
    model: str,
    *,
    campaign_root,
    task: str,
    method: str,
    run_suffix: str,
    audit_path,
) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"),
        str(campaign_root),
        "--tasks",
        task,
        "--seed-start",
        "0",
        "--seeds",
        "20",
        "--epochs",
        "100",
        "--rl-batch-size",
        "1023",
        "--budget",
        "102400",
        "--method-name",
        method,
        "--run-suffix",
        run_suffix,
        "--communication-mode",
        "none",
        "--model",
        model,
        "--output",
        str(audit_path),
    ]


def audit_command(model: str) -> list[str]:
    return build_audit_command(
        model,
        campaign_root=CAMPAIGN_ROOT,
        task="rpa",
        method=METHOD,
        run_suffix=RUN_SUFFIX,
        audit_path=AUDIT_PATH,
    )


def write_preflight(model: str) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "model": model,
        "endpoint": ENDPOINT,
        "tasks": ["rpa"],
        "seeds": list(range(20)),
        "epochs": 100,
        "max_parallel": 8,
        "maximum_cvode_worker_slots": 32,
        "global_harness_concurrency": 8,
        "rl_gpu": RL_GPU,
        "communication_mode": "none",
        "request_0_hof": "withheld-empty",
        "later_hof_and_sil": "disabled",
        "candidate_validation_policy": "independent-members",
        "proposal_space_contract_version": 2,
        "matched_full_duplex_campaign": (
            "qwen38-local-v2-rpa-logic-100epoch-20seed-20260903"
        ),
    }
    (CAMPAIGN_ROOT / "local_campaign_preflight.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    model = require_qwen38_model(ENDPOINT)
    probe_tool_call(ENDPOINT, model)
    write_preflight(model)
    print(
        f"[ready] model={model}; campaign={CAMPAIGN_ID}; "
        "communication=none; request-0 HOF=withheld; parallel_runs=8",
        flush=True,
    )
    subprocess.run(campaign_command(model), cwd=ROOT, env=environment(), check=True)
    subprocess.run(audit_command(model), cwd=ROOT, env=environment(), check=True)
    print(f"[complete] campaign and audit passed: {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
