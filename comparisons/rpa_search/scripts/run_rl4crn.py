#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.config import load_config, write_config
from comparisons.rpa_search.src.common.io import ensure_run_dir
from comparisons.rpa_search.src.common.task_factory import build_components
from comparisons.rpa_search.src.methods.rl4crn_runner import run_rl4crn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="comparisons/rpa_search/configs/rpa_smoke.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--n-cpus", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["search"]["seed"] = int(args.seed)
    if args.solver is not None:
        task_name = config.get("task", config.get("benchmark", {}).get("task", "rpa"))
        config[task_name]["solver"] = args.solver
    if args.n_cpus is not None:
        config["rl4crn"]["n_cpus"] = int(args.n_cpus)
    method = "rl4crn"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_run_dir(config["benchmark"]["output_root"], method, run_id)
    write_config(config, run_dir / "config.json")
    components = build_components(config)
    run_rl4crn(config, run_dir, method, run_id, components)


if __name__ == "__main__":
    main()
