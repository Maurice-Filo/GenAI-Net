#!/usr/bin/env python3
"""Audit RL-only controls and freeze the contract-v2 analysis plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "comparisons/rpa_search/data/raw"
DEFAULT_OUTPUT = (
    ROOT / "paper/iclr2027_genai_net_llm/generated/analysis_plan_v2.json"
)
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
SEEDS = tuple(range(20))
EPOCHS = 100
BATCH_SIZE = 1024
CANDIDATE_BUDGET = EPOCHS * BATCH_SIZE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def control_path(task: str, seed: int) -> Path:
    if task in {"rpa", "logic"}:
        return RAW / "rl4crn" / f"{task}_full102400_seed{seed}_cvode"
    return (
        RAW
        / "rl4crn_breadth"
        / f"{task}_full102400_seed{seed}_cvode_rl_only_breadth"
    )


def audit_control(task: str, seed: int) -> dict:
    path = control_path(task, seed)
    config_path = path / "config.json"
    progress_path = path / "progress.csv"
    if not config_path.is_file() or not progress_path.is_file():
        raise RuntimeError(f"Missing RL-only control artifacts: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with progress_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EPOCHS:
        raise RuntimeError(f"Expected {EPOCHS} progress rows in {path}; found {len(rows)}")
    steps = [int(float(row["step"])) for row in rows]
    if steps != list(range(1, EPOCHS + 1)):
        raise RuntimeError(f"Noncanonical epoch sequence in {path}")
    final = rows[-1]
    if int(float(final["candidate_evaluations"])) != CANDIDATE_BUDGET:
        raise RuntimeError(f"Candidate budget mismatch in {path}")
    endpoint = float(final["best_so_far_loss"])
    if not math.isfinite(endpoint):
        raise RuntimeError(f"Non-finite endpoint in {path}")
    if str(config[task]["solver"]).upper() != "CVODE":
        raise RuntimeError(f"Non-CVODE control in {path}")
    if int(config["rl4crn"]["epochs"]) != EPOCHS:
        raise RuntimeError(f"Epoch mismatch in {path}")
    if int(config["rl4crn"]["batch_size"]) != BATCH_SIZE:
        raise RuntimeError(f"Batch-size mismatch in {path}")
    if int(config["search"]["candidate_budget"]) != CANDIDATE_BUDGET:
        raise RuntimeError(f"Configured budget mismatch in {path}")
    if int(config["search"]["seed"]) != seed:
        raise RuntimeError(f"Seed mismatch in {path}")

    completed_path = path / "completed.json"
    completed_hash = None
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("task", task),
            ("seed", seed),
            ("epochs", EPOCHS),
            ("candidate_budget", CANDIDATE_BUDGET),
            ("solver", "CVODE"),
        ):
            if completed.get(key) != expected:
                raise RuntimeError(f"Completion field {key!r} mismatch in {path}")
        if not math.isclose(
            float(completed["best_loss"]), endpoint, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"Completion endpoint mismatch in {path}")
        completed_hash = sha256(completed_path)

    return {
        "seed": seed,
        "endpoint_loss": endpoint,
        "artifact_path": str(path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "progress_sha256": sha256(progress_path),
        "completed_sha256": completed_hash,
    }


def build_plan(*, frozen_by: str, frozen_at: str) -> dict:
    control_audit = {}
    thresholds = {}
    for task in TASKS:
        records = [audit_control(task, seed) for seed in SEEDS]
        endpoints = np.asarray([record["endpoint_loss"] for record in records])
        q25, median, q75 = np.quantile(endpoints, (0.25, 0.5, 0.75))
        thresholds[task] = float(median)
        control_audit[task] = {
            "n": len(records),
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "controls": records,
        }
    return {
        "schema_version": 2,
        "status": "frozen",
        "primary_estimand": "Per-seed count of distinct labelled topology hashes among the final top-30 candidates with canonical loss strictly below the frozen task threshold.",
        "unit": "matched RL seed",
        "tasks": list(TASKS),
        "quality_threshold_definition": "Median final best-so-far canonical loss across the 20 independently completed RL-only controls for that task.",
        "quality_threshold_source": "RL-only controls only; no contract-v2 LLM outcome was available or inspected when frozen.",
        "quality_thresholds": thresholds,
        "threshold_inequality": "canonical_loss < threshold",
        "portfolio_top_k": 30,
        "topology_identity": "SHA-256 over sorted complete labelled structural records including implementation type and input channels; kinetic parameters and serialization order excluded.",
        "primary_family": "Eight matched full-duplex versus RL-only qualified-topology tests with Holm correction.",
        "communication_family": "RPA, Logic, and classifier matched full-duplex versus no-communication portfolio tests with Holm correction.",
        "rank_one_secondary": "All 20 paired endpoints, median and IQR, W/T/L, median paired ratio, two-sided Wilcoxon signed-rank test, and Holm correction across eight tasks.",
        "structural_distance_secondary": "Exact complete-record Jaccard distance only among threshold-qualified candidates after exact fixed-template-record subtraction.",
        "training_boundary": "After exactly the configured RL batches and before terminal waiting; only responses merged by this boundary are included.",
        "terminal_boundary": "After every scheduled request has completed, failed, or timed out.",
        "failed_run_policy": "No imputation and no silent seed deletion. A prespecified paired analysis is released only when both members of all 20 seed pairs pass the protocol audit; failures remain reported and block that task comparison until resolved under a new fingerprint.",
        "tie_policy": "A pair is a tie only when losses are equal within rtol=1e-9 and atol=1e-12. Qualified-topology count ties require exact integer equality. Wilcoxon zero differences use the Pratt convention.",
        "missing_candidate_policy": "Requested but invalid, failed, timed-out, or unevaluated LLM members contribute no candidate and remain explicit in request accounting; they are never imputed or replaced.",
        "multiplicity_policy": "Holm adjustment is applied separately to the named eight-task rank-one family, eight-task qualified-topology family, and three-task communication family.",
        "sensitivity_policy": "Threshold sweeps at RL-only q25, median, and q75 are secondary and all three are reported together; the frozen median threshold defines the primary portfolio result.",
        "control_contract": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "candidate_budget": CANDIDATE_BUDGET,
            "solver": "CVODE",
        },
        "control_audit": control_audit,
        "frozen_by": frozen_by,
        "frozen_at": frozen_at,
        "notes": "Frozen after prompt approval and before the first contract-v2 model request.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frozen-by", required=True)
    parser.add_argument(
        "--frozen-at",
        default=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    plan = build_plan(frozen_by=args.frozen_by, frozen_at=args.frozen_at)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"Refusing to overwrite an existing frozen plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
