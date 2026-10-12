#!/usr/bin/env python3
"""Run and audit the 20-seed local-Qwen RPA/Logic contract-v2 cohort."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa import (
    probe_tool_call,
    require_qwen38_model,
)


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
CAMPAIGN_ID = "qwen38-local-v2-rpa-logic-100epoch-20seed-20260903"
METHOD = "genai_net_llm_qwen38_local_contract_v2_full"
RUN_SUFFIX = "cvode_llm_qwen38_local_contract_v2_full"
ENDPOINT = "http://127.0.0.1:18082/v1"
RL_GPU = "GPU-e28480e9-6ce2-60d6-887a-7c2a0cbd1e19"
DSH_HOME = Path.home() / "ai-workspaces/deepseek-test/.dsh-home"
CAMPAIGN_BASE = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
CAMPAIGN_ROOT = CAMPAIGN_BASE / CAMPAIGN_ID
AUDIT_PATH = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/contract_v2_local_qwen_rpa_logic_audit.json"
)


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


def campaign_command(model: str) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/run_genai_net_llm_20seed_campaign.py"),
        "--campaign-id",
        CAMPAIGN_ID,
        "--tasks",
        "rpa",
        "logic",
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
        "full",
        "--withhold-initial-hof",
        "--candidate-validation-policy",
        "independent-members",
        "--method-name",
        METHOD,
        "--run-suffix",
        RUN_SUFFIX,
        "--dsh-home",
        str(DSH_HOME),
        "--workspace-root",
        str(CAMPAIGN_BASE),
        "--skip-postprocessing",
    ]


def audit_command(model: str) -> list[str]:
    return [
        str(PYTHON),
        str(ROOT / "comparisons/llm_crn_generation/audit_contract_v2_campaign.py"),
        str(CAMPAIGN_ROOT),
        "--tasks",
        "rpa",
        "logic",
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
        METHOD,
        "--run-suffix",
        RUN_SUFFIX,
        "--communication-mode",
        "full",
        "--model",
        model,
        "--output",
        str(AUDIT_PATH),
    ]


def write_preflight(model: str) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "model": model,
        "endpoint": ENDPOINT,
        "tasks": ["rpa", "logic"],
        "seeds": list(range(20)),
        "epochs": 100,
        "max_parallel": 8,
        "maximum_cvode_worker_slots": 32,
        "global_harness_concurrency": 8,
        "rl_gpu": RL_GPU,
        "communication_mode": "full",
        "request_0_hof": "withheld-empty",
        "later_hof_and_sil": "enabled",
        "candidate_validation_policy": "independent-members",
        "proposal_space_contract_version": 2,
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
        "request-0 HOF=withheld; parallel_runs=8",
        flush=True,
    )
    subprocess.run(campaign_command(model), cwd=ROOT, env=environment(), check=True)
    subprocess.run(audit_command(model), cwd=ROOT, env=environment(), check=True)
    print(f"[complete] campaign and audit passed: {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
