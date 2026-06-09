#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.config import load_config
from comparisons.rpa_search.src.common.task_factory import build_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    _template_crn, _library_components, task, _cfg = build_components(config)
    scenario_count = len(task.u_list)
    search = config["search"]
    rl = config["rl4crn"]

    target_full_simulations = int(rl["epochs"]) * int(rl["batch_size"])
    candidate_budget = int(search["candidate_budget"])
    mcts_iterations = int(search.get("mcts_iterations", candidate_budget))

    rows = [
        ("random_search", candidate_budget),
        ("circuitree", mcts_iterations),
        ("reaction_network_evolution_jl", candidate_budget),
        ("rl4crn", target_full_simulations),
    ]

    print(f"config: {args.config}")
    print(f"task: {config.get('task', config.get('benchmark', {}).get('task', 'rpa'))}")
    print(f"scenarios per full simulation: {scenario_count}")
    for method, full_simulations in rows:
        print(
            f"{method}: full_simulations={full_simulations} "
            f"scenario_evaluations={full_simulations * scenario_count}"
        )

    full_counts = {value for _method, value in rows}
    if len(full_counts) != 1:
        raise SystemExit("Unfair budget: methods do not share the same full-simulation budget.")

    print("fairness: OK")


if __name__ == "__main__":
    main()
