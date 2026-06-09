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
from comparisons.rpa_search.src.common.rpa_task import build_rpa_components
from comparisons.rpa_search.src.methods.mcts_search import run_mcts_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="comparisons/rpa_search/configs/rpa_smoke.json")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    method = "mcts_search"
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_run_dir(config["benchmark"]["output_root"], method, run_id)
    write_config(config, run_dir / "config.json")
    components = build_rpa_components(config)
    run_mcts_search(config, run_dir, method, run_id, components)


if __name__ == "__main__":
    main()
