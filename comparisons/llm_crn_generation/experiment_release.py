"""Fail-closed release gates for sentinel and primary contract-v2 campaigns."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from comparisons.llm_crn_generation.prompt_approval import (
    DEFAULT_PROMPT_APPROVAL,
    DEFAULT_PROMPT_REVIEW,
    file_sha256,
    validate_prompt_approval,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER_GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"
DEFAULT_ANALYSIS_PLAN = PAPER_GENERATED / "analysis_plan_v2.json"
DEFAULT_STATIC_PREFLIGHT = (
    PAPER_GENERATED / "contract_v2_preflight/preflight_report.json"
)
DEFAULT_SENTINEL_REPORT = PAPER_GENERATED / "contract_v2_sentinel_report.json"
DETERMINISTIC_TASKS = {
    "rpa",
    "logic",
    "classifier",
    "dose_hill",
    "dose_ultrasensitive",
    "dose_biphasic",
    "oscillator_mean",
    "oscillator_frequency",
}


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def validate_analysis_plan(path: str | Path = DEFAULT_ANALYSIS_PLAN) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    plan = _read_json(path, "Frozen analysis plan")
    if plan.get("status") != "frozen":
        raise RuntimeError("Analysis plan status must be exactly 'frozen'.")
    thresholds = plan.get("quality_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != DETERMINISTIC_TASKS:
        raise RuntimeError("Analysis plan must freeze one quality threshold for every task.")
    for task, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Analysis threshold for {task} must be numeric.")
        if not math.isfinite(float(value)):
            raise RuntimeError(f"Analysis threshold for {task} must be finite.")
    for field in ("frozen_by", "frozen_at", "failed_run_policy", "tie_policy"):
        if not str(plan.get(field, "")).strip():
            raise RuntimeError(f"Analysis plan field {field!r} must be non-empty.")
    return {**plan, "path": str(path), "sha256": file_sha256(path)}


def validate_experiment_release(
    *,
    stage: str,
    prompt_approval_path: str | Path = DEFAULT_PROMPT_APPROVAL,
    analysis_plan_path: str | Path = DEFAULT_ANALYSIS_PLAN,
    static_preflight_path: str | Path = DEFAULT_STATIC_PREFLIGHT,
    sentinel_report_path: str | Path = DEFAULT_SENTINEL_REPORT,
) -> Dict[str, Any]:
    """Validate immutable prerequisites without performing a model or network call."""

    normalized_stage = str(stage).strip().lower()
    if normalized_stage not in {"sentinel", "paper"}:
        raise ValueError("stage must be 'sentinel' or 'paper'.")
    approval = validate_prompt_approval(prompt_approval_path)
    analysis = validate_analysis_plan(analysis_plan_path)
    static_path = Path(static_preflight_path).expanduser().resolve()
    static = _read_json(static_path, "Static contract-v2 preflight")
    if static.get("status") != "pass" or len(static.get("tasks", ())) != 8:
        raise RuntimeError("Static contract-v2 preflight is not passing for eight tasks.")
    if int(static.get("model_calls_made", -1)) != 0:
        raise RuntimeError("Static contract-v2 preflight unexpectedly records model calls.")

    sentinel = None
    if normalized_stage == "paper":
        sentinel_path = Path(sentinel_report_path).expanduser().resolve()
        sentinel = _read_json(sentinel_path, "Contract-v2 sentinel report")
        if sentinel.get("status") != "pass":
            raise RuntimeError("Contract-v2 sentinel report is not passing.")
        if sentinel.get("prompt_review_sha256") != file_sha256(DEFAULT_PROMPT_REVIEW):
            raise RuntimeError("Sentinel report does not match the current prompt review.")
        if sentinel.get("analysis_plan_sha256") != analysis["sha256"]:
            raise RuntimeError("Sentinel report does not match the frozen analysis plan.")

    return {
        "stage": normalized_stage,
        "prompt_approval": approval,
        "analysis_plan": analysis,
        "static_preflight": {"path": str(static_path), "sha256": file_sha256(static_path)},
        "sentinel_report": sentinel,
    }
