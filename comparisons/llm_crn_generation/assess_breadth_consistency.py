#!/usr/bin/env python3
"""Apply the predeclared five-to-ten-seed breadth extension rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.paper_breadth_tasks import (
    BREADTH_TASKS,
    DETERMINISTIC_TASKS,
)


def assess_task(
    root: Path, task: str, method: str, suffix: str, candidate_budget: int
) -> dict:
    losses = []
    missing = []
    for seed in range(5):
        path = (
            root
            / method
            / f"{task}_full{candidate_budget}_seed{seed}_{suffix}"
            / "completed.json"
        )
        if not path.is_file():
            missing.append(seed)
            continue
        value = float(json.loads(path.read_text(encoding="utf-8"))["best_loss"])
        if math.isfinite(value) and value >= 0:
            losses.append(value)
        else:
            missing.append(seed)
    log_losses = np.log10(np.maximum(np.asarray(losses, dtype=float), 1e-12))
    log_range = float(np.ptp(log_losses)) if len(log_losses) else float("inf")
    log_mad = (
        float(np.median(np.abs(log_losses - np.median(log_losses))))
        if len(log_losses)
        else float("inf")
    )
    consistent = not missing and log_range <= 1.0 and log_mad <= 0.35
    return {
        "task": task,
        "losses": losses,
        "missing_seeds": missing,
        "log10_range": log_range,
        "log10_mad": log_mad,
        "consistent": consistent,
        "extend_to_ten_seeds": not consistent,
        "rule": "extend if a seed is missing, log10 range > 1.0, or log10 MAD > 0.35",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--method", default="genai_net_llm_flash_breadth")
    parser.add_argument("--run-suffix", default="cvode_llm_flash_breadth")
    parser.add_argument("--candidate-budget", type=int, default=102400)
    parser.add_argument("--tasks", nargs="+", choices=BREADTH_TASKS, default=DETERMINISTIC_TASKS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks-only", action="store_true")
    args = parser.parse_args()
    assessments = [
        assess_task(
            args.raw_root,
            task,
            args.method,
            args.run_suffix,
            args.candidate_budget,
        )
        for task in args.tasks
    ]
    payload = {
        "schema_version": 1,
        "initial_seed_count": 5,
        "extension_seed_range": [5, 9],
        "assessments": assessments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.tasks_only:
        for row in assessments:
            if row["extend_to_ten_seeds"]:
                print(row["task"])
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
