#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
CONFIGS = {
    "rpa": ROOT / "comparisons/rpa_search/configs/rpa_100k.json",
    "logic": ROOT / "comparisons/rpa_search/configs/logic_100k.json",
}


def _run_id(task: str, method: str, seed: int) -> str:
    if method == "llm_seeded_evolution":
        family = "flash" if task == "classifier" else "local"
        return f"{task}_search102300_seed{seed}_{family}_none_replay_v1"
    if method == "rl4crn":
        return f"{task}_full102400_seed{seed}_cvode"
    if method == "reaction_network_evolution_jl":
        return f"{task}_full102400_seed{seed}"
    if method == "reaction_network_evolution_jl_constrained":
        return f"{task}_full102400_seed{seed}_constrained"
    if method == "reaction_network_evolution_jl_constrained_bounded":
        return f"{task}_full102400_seed{seed}_constrained_bounded"
    raise ValueError(f"Unknown method: {method}")


def _complete(method: str, run_id: str, min_sims: int) -> bool:
    if method == "llm_seeded_evolution":
        import json
        completed = RAW_ROOT / method / run_id / "completed.json"
        if not completed.exists():
            return False
        result = json.loads(completed.read_text())
        return (result.get("status") == "complete" and result.get("pilot") is False
                and result.get("search_evaluations") == 102300)
    progress = RAW_ROOT / method / run_id / "progress.csv"
    if not progress.exists():
        return False
    with progress.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    return float(rows[-1].get("ode_simulations", 0)) >= float(min_sims)


def _command(task: str, method: str, seed: int, run_id: str, n_cpus: int, julia: str) -> list[str]:
    if method == "llm_seeded_evolution":
        return [
            str(ROOT / ".venv/bin/python"), "-m", "comparisons.llm_seeded_evolution.run",
            "--task", task, "--seed", str(seed), "--run-id", run_id,
            "--workers", str(n_cpus),
        ]
    if task not in CONFIGS:
        raise ValueError(f"This runner supports {task} only with llm_seeded_evolution.")
    config = str(CONFIGS[task])
    if method == "rl4crn":
        return [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "comparisons/rpa_search/scripts/run_rl4crn.py"),
            "--config",
            config,
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--solver",
            "CVODE",
            "--n-cpus",
            str(n_cpus),
        ]
    if method in {
        "reaction_network_evolution_jl",
        "reaction_network_evolution_jl_constrained",
        "reaction_network_evolution_jl_constrained_bounded",
    }:
        cmd = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "comparisons/rpa_search/scripts/run_reaction_network_evolution_jl.py"),
            "--config",
            config,
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--julia",
            julia,
        ]
        if method == "reaction_network_evolution_jl_constrained":
            cmd.append("--constrain-reactions")
        if method == "reaction_network_evolution_jl_constrained_bounded":
            cmd.extend(["--constrain-reactions", "--bounded-state"])
        return cmd
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rpa", "logic", "classifier"], required=True)
    parser.add_argument(
        "--method",
        choices=[
            "llm_seeded_evolution",
            "rl4crn",
            "reaction_network_evolution_jl",
            "reaction_network_evolution_jl_constrained",
            "reaction_network_evolution_jl_constrained_bounded",
        ],
        required=True,
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19)
    parser.add_argument("--min-sims", type=int, default=102400)
    parser.add_argument("--n-cpus", type=int, default=32)
    parser.add_argument("--julia", default=str(ROOT / "comparisons/rpa_search/julia/julia-1.9.4/bin/julia"))
    args = parser.parse_args()

    for seed in range(args.seed_start, args.seed_end + 1):
        run_id = _run_id(args.task, args.method, seed)
        if _complete(args.method, run_id, args.min_sims):
            print(f"[skip] {args.method} {run_id} already complete", flush=True)
            continue

        cmd = _command(args.task, args.method, seed, run_id, args.n_cpus, args.julia)
        print(f"[run] {args.method} {run_id}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
