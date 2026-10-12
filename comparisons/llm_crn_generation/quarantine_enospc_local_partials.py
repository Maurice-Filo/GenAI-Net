#!/usr/bin/env python3
"""Quarantine incomplete local-Qwen runs interrupted by storage exhaustion."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = Path.home() / "ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
RAW = ROOT / "comparisons/rpa_search/data/raw"
QUARANTINE = (
    Path.home() / "ai-workspaces/deepseek-test/crn-runs/quarantine/enospc-20260904"
)
SPECS = (
    {
        "campaign": "qwen38-local-v2-logic-no-communication-100epoch-20seed-20260903",
        "task": "logic",
        "method": "genai_net_llm_qwen38_local_contract_v2_logic_no_communication",
        "suffix": "cvode_llm_qwen38_local_contract_v2_logic_no_communication",
        "seeds": range(20),
    },
    {
        "campaign": "qwen38-local-v2-half-rpa-llm-to-rl-seeds1to19-20260904",
        "task": "rpa",
        "method": "genai_net_llm_qwen38_local_contract_v2_llm_to_rl",
        "suffix": "cvode_llm_qwen38_local_contract_v2_llm_to_rl",
        "seeds": range(1, 20),
    },
    {
        "campaign": "qwen38-local-v2-half-rpa-rl-to-llm-seeds1to19-20260904",
        "task": "rpa",
        "method": "genai_net_llm_qwen38_local_contract_v2_rl_to_llm",
        "suffix": "cvode_llm_qwen38_local_contract_v2_rl_to_llm",
        "seeds": range(1, 20),
    },
)


def move(source: Path, destination: Path, *, apply: bool, moves: list[dict]) -> None:
    if not source.exists():
        return
    if destination.exists():
        raise RuntimeError(f"Quarantine destination already exists: {destination}")
    moves.append({"source": str(source), "destination": str(destination)})
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    moves: list[dict] = []
    retained: list[dict] = []

    for spec in SPECS:
        campaign = CAMPAIGNS / spec["campaign"]
        metadata_destination = QUARANTINE / "metadata" / spec["campaign"]
        if args.apply:
            metadata_destination.mkdir(parents=True, exist_ok=True)
            for name in ("campaign_manifest.json", "status.json"):
                source = campaign / name
                if source.exists():
                    shutil.copy2(source, metadata_destination / name)

        for seed in spec["seeds"]:
            run_id = f"{spec['task']}_full102400_seed{seed}_{spec['suffix']}"
            raw_run = RAW / spec["method"] / run_id
            if (raw_run / "completed.json").exists():
                retained.append(
                    {"campaign": spec["campaign"], "task": spec["task"], "seed": seed}
                )
                continue
            trial = campaign / "runs" / f"{spec['campaign']}-{spec['task']}-seed{seed}"
            log = campaign / "logs" / f"{spec['task']}_seed{seed}.log"
            move(
                trial,
                QUARANTINE / "campaigns" / spec["campaign"] / "runs" / trial.name,
                apply=args.apply,
                moves=moves,
            )
            move(
                raw_run,
                QUARANTINE / "raw" / spec["method"] / run_id,
                apply=args.apply,
                moves=moves,
            )
            move(
                log,
                QUARANTINE / "campaigns" / spec["campaign"] / "logs" / log.name,
                apply=args.apply,
                moves=moves,
            )

    report = {
        "status": "applied" if args.apply else "dry-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "ENOSPC interrupted these runs before a completed.json marker was written.",
        "quarantine_root": str(QUARANTINE),
        "retained_completed": retained,
        "moves": moves,
    }
    if args.apply:
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        (QUARANTINE / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
