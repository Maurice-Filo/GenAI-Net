#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
TASKS = ("rpa", "logic")
METHOD = "reaction_network_evolution_jl_constrained_bounded"


def _run_id(task: str, seed: int) -> str:
    return f"{task}_full102400_seed{seed}_constrained_bounded"


def _sims(task: str, seed: int) -> float:
    path = RAW_ROOT / METHOD / _run_id(task, seed) / "progress.csv"
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    return float(rows[-1].get("ode_simulations", 0.0))


def _status(n_seeds: int, min_sims: int) -> tuple[int, int, list[str]]:
    total = len(TASKS) * n_seeds
    done = 0
    active = []
    for task in TASKS:
        complete = 0
        max_partial = 0.0
        for seed in range(n_seeds):
            sims = _sims(task, seed)
            if sims >= min_sims:
                done += 1
                complete += 1
            else:
                max_partial = max(max_partial, sims)
        active.append(f"{task}:{METHOD} {complete}/{n_seeds} max_partial={int(max_partial)}")
    return done, total, active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--min-sims", type=int, default=102400)
    parser.add_argument("--sleep", type=int, default=600)
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    while True:
        done, total, active = _status(args.n_seeds, args.min_sims)
        print(f"[wait] complete {done}/{total} | " + " | ".join(active), flush=True)
        if done == total:
            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "comparisons/rpa_search/scripts/plot_genai_julia_constrained_20seed.py"),
                "--n-seeds",
                str(args.n_seeds),
            ]
            if args.paper:
                cmd.append("--paper")
            subprocess.run(cmd, cwd=ROOT, check=True)
            return
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
