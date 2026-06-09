#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from comparisons.rpa_search.src.common.config import load_config, write_config
from comparisons.rpa_search.src.common.io import ensure_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="comparisons/rpa_search/configs/rpa_smoke.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--julia", default="comparisons/rpa_search/julia/julia-1.9.4/bin/julia")
    parser.add_argument("--constrain-reactions", action="store_true")
    parser.add_argument("--bounded-state", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = Path(args.config)
    if args.constrain_reactions:
        config.setdefault("rne", {})["constrain_reactions"] = True
    if args.bounded_state:
        config.setdefault("rne", {})["bounded_state"] = True
        config.setdefault("rne", {})["LARGE_NUMBER"] = 1.0e4
    if args.seed is not None or args.constrain_reactions or args.bounded_state:
        stem = Path(args.config).stem
        seed_part = f"_seed{args.seed}" if args.seed is not None else ""
        mode_part = "_constrained_bounded" if args.constrain_reactions and args.bounded_state else "_constrained" if args.constrain_reactions else ""
        if args.seed is not None:
            config["search"]["seed"] = int(args.seed)
        config_path = Path("/local0/tmp") / f"rpa_search_{stem}{seed_part}{mode_part}.json"
        write_config(config, config_path)
    method = (
        "reaction_network_evolution_jl_constrained_bounded"
        if args.constrain_reactions and args.bounded_state
        else "reaction_network_evolution_jl_constrained"
        if args.constrain_reactions
        else "reaction_network_evolution_jl"
    )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_run_dir(config["benchmark"]["output_root"], method, run_id)
    lock_path = run_dir / ".running"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        print(f"[skip] {method} {run_id} already has an active lock at {lock_path}", flush=True)
        return
    write_config(config, run_dir / "config.json")

    script = Path("comparisons/rpa_search/julia/run_rne_rpa.jl")
    env = os.environ.copy()
    env["JULIA_PROJECT"] = str(Path("comparisons/rpa_search/julia").resolve())
    env["JULIA_DEPOT_PATH"] = (
        str(Path("comparisons/rpa_search/julia/depot").resolve())
        + ":"
        + str(Path.home() / ".julia")
    )
    try:
        subprocess.run(
            [
                args.julia,
                "--project=comparisons/rpa_search/julia",
                str(script),
                str(config_path),
                str(run_dir),
                run_id,
            ],
            check=True,
            env=env,
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
