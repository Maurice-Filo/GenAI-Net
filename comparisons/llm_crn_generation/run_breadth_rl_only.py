#!/usr/bin/env python3
"""Run one matched RL-only paper-breadth experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from comet_ml import Experiment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.paper_breadth_tasks import (
    DETERMINISTIC_TASKS,
    build_paper_breadth_components,
)
from comparisons.rpa_search.src.common.config import load_config, write_config
from comparisons.rpa_search.src.common.io import ensure_run_dir
from comparisons.rpa_search.src.methods.rl4crn_runner import run_rl4crn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=DETERMINISTIC_TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--candidate-budget", type=int, default=102400)
    parser.add_argument("--n-cpus", type=int, default=4)
    parser.add_argument("--method-name", default="rl4crn_breadth")
    parser.add_argument("--run-suffix", default="cvode_rl_only_breadth")
    parser.add_argument("--comet-project", default="genai-net-v4-flash-paper")
    parser.add_argument("--trial-id", default="rl-only-breadth")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "comparisons/rpa_search/data/raw",
    )
    args = parser.parse_args()

    if args.seed < 0 or min(args.epochs, args.batch_size, args.candidate_budget, args.n_cpus) <= 0:
        parser.error("seed must be non-negative and numeric settings must be positive")
    if args.epochs * args.batch_size != args.candidate_budget:
        parser.error("epochs * batch-size must exactly equal candidate-budget")

    run_id = f"{args.task}_full{args.candidate_budget}_seed{args.seed}_{args.run_suffix}"
    run_dir = ensure_run_dir(args.output_root, args.method_name, run_id)
    completed = run_dir / "completed.json"
    if completed.exists():
        print(f"Already complete: {run_dir}")
        return

    config = load_config(
        ROOT / "comparisons/llm_crn_generation/configs/paper_breadth_100epoch.json"
    )
    config["benchmark"] = {
        "task": args.task,
        "output_root": str(args.output_root),
    }
    config["search"]["seed"] = args.seed
    config["search"]["candidate_budget"] = args.candidate_budget
    config["rl4crn"]["epochs"] = args.epochs
    config["rl4crn"]["batch_size"] = args.batch_size
    config["rl4crn"]["n_cpus"] = args.n_cpus
    write_config(config, run_dir / "config.json")
    components = build_paper_breadth_components(config, args.task)
    experiment = Experiment(
        project_name=args.comet_project,
        auto_param_logging=False,
        auto_metric_logging=False,
        log_code=False,
        log_graph=False,
        log_git_patch=False,
        display_summary_level=0,
    )
    experiment.set_name(f"{args.trial_id}-{run_id}")
    experiment.add_tags(["genai-net-paper", "rl-only", "breadth", "cvode", args.task])
    experiment.log_parameters(
        {
            "candidate_budget": args.candidate_budget,
            "epochs": args.epochs,
            "method": args.method_name,
            "rl_batch_size": args.batch_size,
            "seed": args.seed,
            "solver": "CVODE",
            "task": args.task,
        }
    )
    try:
        result = run_rl4crn(
            config, run_dir, args.method_name, run_id, components, logger=experiment
        )
        experiment.log_metric("benchmark/final_best_loss", result["best_loss"])
    finally:
        experiment.end()
    completed.write_text(
        json.dumps(
            {
                **result,
                "candidate_budget": args.candidate_budget,
                "epochs": args.epochs,
                "seed": args.seed,
                "solver": "CVODE",
                "task": args.task,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
