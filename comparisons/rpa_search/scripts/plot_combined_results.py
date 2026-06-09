#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.config import load_config
from comparisons.rpa_search.src.common.plotting import plot_side_by_side_seed_summary


def _panel_from_config(config_path: str, title: str) -> dict:
    config = load_config(config_path)
    benchmark = config["benchmark"]
    task = config.get("task", benchmark.get("task", "rpa"))
    return {
        "raw_root": benchmark["output_root"],
        "methods": benchmark.get("plot_methods"),
        "benchmark_names": [benchmark["name"]],
        "tasks": [task],
        "run_ids": benchmark.get("plot_run_ids"),
        "method_run_ids": benchmark.get("plot_method_run_ids"),
        "min_ode_simulations": benchmark.get("min_ode_simulations"),
        "title": title,
        "show_panel_label": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpa-config", default="comparisons/rpa_search/configs/rpa_100k.json")
    parser.add_argument("--logic-config", default="comparisons/rpa_search/configs/logic_100k.json")
    parser.add_argument("--figure-dir", default="comparisons/rpa_search/figures")
    parser.add_argument("--figure-name", default="rpa_logic_102400_full_side_by_side.png")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    out = plot_side_by_side_seed_summary(
        [
            _panel_from_config(args.rpa_config, ""),
            _panel_from_config(args.logic_config, ""),
        ],
        args.figure_dir,
        figure_name=args.figure_name,
        formats=("png", "pdf", "svg") if args.paper else ("png",),
        log_x=False,
        log_y=False,
        y_limit=(0.0, 0.15),
        ylabel="Best-so-far loss",
        paper=args.paper,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
