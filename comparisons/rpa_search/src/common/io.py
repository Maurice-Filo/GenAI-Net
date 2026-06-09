from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable


PROGRESS_FIELDS = [
    "method",
    "run_id",
    "step",
    "candidate_evaluations",
    "ode_simulations",
    "scenario_count",
    "scenario_evaluations",
    "loss",
    "best_so_far_loss",
    "performance",
    "best_so_far_performance",
    "elapsed_seconds",
]

CANDIDATE_FIELDS = [
    "method",
    "run_id",
    "candidate_id",
    "candidate_evaluations",
    "ode_simulations",
    "scenario_count",
    "scenario_evaluations",
    "loss",
    "best_so_far_loss",
    "reaction_ids",
    "rate_constants",
]


def ensure_run_dir(output_root: str | Path, method: str, run_id: str) -> Path:
    run_dir = Path(output_root) / method / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_csv(path: str | Path, row: Dict[str, Any], fieldnames: Iterable[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value
