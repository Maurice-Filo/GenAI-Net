#!/usr/bin/env python3
"""Run and audit the matched local-Qwen Logic no-communication cohort."""

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
    ENDPOINT,
    ROOT,
    environment,
)
from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa_no_communication_campaign import (
    build_audit_command,
    build_campaign_command,
)


CAMPAIGN_ID = "qwen38-local-v2-logic-no-communication-100epoch-20seed-20260903"
METHOD = "genai_net_llm_qwen38_local_contract_v2_logic_no_communication"
RUN_SUFFIX = "cvode_llm_qwen38_local_contract_v2_logic_no_communication"
CAMPAIGN_ROOT = CAMPAIGN_BASE / CAMPAIGN_ID
AUDIT_PATH = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/"
    "contract_v2_local_qwen_logic_no_communication_audit.json"
)


def campaign_command(model: str) -> list[str]:
    return build_campaign_command(
        model,
        campaign_id=CAMPAIGN_ID,
        task="logic",
        method=METHOD,
        run_suffix=RUN_SUFFIX,
    )


def audit_command(model: str) -> list[str]:
    return build_audit_command(
        model,
        campaign_root=CAMPAIGN_ROOT,
        task="logic",
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
        "tasks": ["logic"],
        "seeds": list(range(20)),
        "epochs": 100,
        "max_parallel": 8,
        "maximum_cvode_worker_slots": 32,
        "global_harness_concurrency": 8,
        "communication_mode": "none",
        "request_0_hof": "withheld-empty",
        "later_hof_and_sil": "disabled",
        "candidate_validation_policy": "independent-members",
        "proposal_space_contract_version": 2,
        "matched_full_duplex_campaign": (
            "qwen38-local-v2-rpa-logic-100epoch-20seed-20260903"
        ),
        "prerequisite_audit": (
            "contract_v2_local_qwen_rpa_no_communication_audit.json"
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
