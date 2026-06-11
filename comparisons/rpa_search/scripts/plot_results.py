#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("comparisons/rpa_search/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.config import load_config
from comparisons.rpa_search.src.common.plotting import (
    plot_best_so_far,
    plot_seed_summary,
    write_final_summary,
    write_seed_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="comparisons/rpa_search/configs/rpa_smoke.json")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out = plot_best_so_far(
        config["benchmark"]["output_root"],
        config["benchmark"]["figure_dir"],
        methods=config["benchmark"].get("plot_methods"),
        benchmark_names=[config["benchmark"]["name"]],
        tasks=[config.get("task", config["benchmark"].get("task", "rpa"))],
        run_ids=config["benchmark"].get("plot_run_ids"),
        method_run_ids=config["benchmark"].get("plot_method_run_ids"),
        min_ode_simulations=config["benchmark"].get("min_ode_simulations"),
        log_y=bool(config["benchmark"].get("log_y", True)),
        x_field=config["benchmark"].get("x_field", "ode_simulations"),
        title=config["benchmark"].get("plot_title", "Search Progress"),
        ylabel=config["benchmark"].get("plot_ylabel", "Best-so-far loss"),
        figure_name=config["benchmark"].get(
            "figure_name",
            f"{config['benchmark']['name']}_best_so_far_vs_simulations.png",
        ),
        formats=("png", "pdf", "svg") if args.paper else ("png",),
        paper=args.paper,
    )
    print(f"Wrote {out}")
    if args.paper:
        summary = write_final_summary(
            config["benchmark"]["output_root"],
            Path(config["benchmark"]["figure_dir"]) / f"{config['benchmark']['name']}_summary.csv",
            methods=config["benchmark"].get("plot_methods"),
            benchmark_names=[config["benchmark"]["name"]],
            tasks=[config.get("task", config["benchmark"].get("task", "rpa"))],
            run_ids=config["benchmark"].get("plot_run_ids"),
            method_run_ids=config["benchmark"].get("plot_method_run_ids"),
            min_ode_simulations=config["benchmark"].get("min_ode_simulations"),
        )
        print(f"Wrote {summary}")
    if args.aggregate:
        aggregate = plot_seed_summary(
            config["benchmark"]["output_root"],
            config["benchmark"]["figure_dir"],
            methods=config["benchmark"].get("plot_methods"),
            benchmark_names=[config["benchmark"]["name"]],
            tasks=[config.get("task", config["benchmark"].get("task", "rpa"))],
            run_ids=config["benchmark"].get("plot_run_ids"),
            method_run_ids=config["benchmark"].get("plot_method_run_ids"),
            min_ode_simulations=config["benchmark"].get("min_ode_simulations"),
            log_y=bool(config["benchmark"].get("log_y", True)),
            x_field=config["benchmark"].get("x_field", "ode_simulations"),
            title=config["benchmark"].get("aggregate_plot_title", config["benchmark"].get("plot_title", "Search Progress")),
            ylabel=config["benchmark"].get("plot_ylabel", "Best-so-far loss"),
            figure_name=config["benchmark"].get(
                "aggregate_figure_name",
                f"{config['benchmark']['name']}_triplicate_summary.png",
            ),
            formats=("png", "pdf", "svg") if args.paper else ("png",),
            paper=args.paper,
        )
        print(f"Wrote {aggregate}")
        seed_summary = write_seed_summary(
            config["benchmark"]["output_root"],
            Path(config["benchmark"]["figure_dir"]) / f"{config['benchmark']['name']}_triplicate_summary.csv",
            methods=config["benchmark"].get("plot_methods"),
            benchmark_names=[config["benchmark"]["name"]],
            tasks=[config.get("task", config["benchmark"].get("task", "rpa"))],
            run_ids=config["benchmark"].get("plot_run_ids"),
            method_run_ids=config["benchmark"].get("plot_method_run_ids"),
            min_ode_simulations=config["benchmark"].get("min_ode_simulations"),
        )
        print(f"Wrote {seed_summary}")


if __name__ == "__main__":
    main()
