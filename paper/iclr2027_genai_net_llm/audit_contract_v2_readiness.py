#!/usr/bin/env python3
"""Fail-closed readiness audit for the contract-v2 manuscript build."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.prompt_approval import validate_prompt_approval
from comparisons.llm_crn_generation.experiment_release import validate_analysis_plan


def audit_readiness(paper: Path = PAPER) -> Dict[str, Any]:
    errors: List[str] = []
    preflight_path = paper / "generated/contract_v2_preflight/preflight_report.json"
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        preflight = {}
        errors.append(f"static preflight report missing or invalid: {exc}")
    if preflight:
        if preflight.get("status") != "pass":
            errors.append("static proposal-space preflight is not passing")
        if len(preflight.get("tasks", ())) != 8:
            errors.append("static proposal-space preflight does not cover eight tasks")
        if int(preflight.get("model_calls_made", -1)) != 0:
            errors.append("static preflight unexpectedly records model calls")

    try:
        approval = validate_prompt_approval(
            paper / "generated/CONTRACT_V2_PROMPT_APPROVAL.json",
            review_path=paper / "generated/CONTRACT_V2_PROMPT_REVIEW.json",
        )
    except RuntimeError as exc:
        approval = None
        errors.append(str(exc))

    try:
        analysis_plan = validate_analysis_plan(paper / "generated/analysis_plan_v2.json")
    except RuntimeError as exc:
        analysis_plan = None
        errors.append(str(exc))

    primary_registry = paper / "generated/contract_v2_primary_registry.json"
    if not primary_registry.is_file():
        errors.append(
            "no audited contract-v2 primary registry exists; numerical manuscript build remains blocked"
        )

    active_sources = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]
    stale_markers = (
        "0.5382",
        "0.5804",
        "779/800",
        "Within one model request",
        "lists the executed contracts",
    )
    for source in active_sources:
        text = source.read_text(encoding="utf-8")
        for marker in stale_markers:
            if marker in text:
                errors.append(f"stale pre-v2 marker {marker!r} in {source.name}")

    return {
        "status": "pass" if not errors else "blocked",
        "static_preflight_status": preflight.get("status"),
        "author_approval": approval,
        "analysis_plan": analysis_plan,
        "primary_registry": str(primary_registry),
        "errors": errors,
    }


def main() -> None:
    result = audit_readiness()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
