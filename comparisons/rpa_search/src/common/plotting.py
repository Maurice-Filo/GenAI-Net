from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

METHOD_LABELS = {
    "rl4crn": "GenAI-Net (ours)",
    "genai_net_llm": "GenAI-Net-LLM (ours)",
    "circuitree": "CircuiTree",
    "reaction_network_evolution_jl": "ReactionNetworkEvolution.jl",
    "reaction_network_evolution_jl_constrained": "ReactionNetworkEvolution.jl (<=5 rxns)",
    "reaction_network_evolution_jl_constrained_bounded": "ReactionNetworkEvolution.jl (bounded to 5 rxns)",
    "random_search": "Random search",
}

METHOD_COLORS = {
    "rl4crn": "#0072B2",
    "genai_net_llm": "#009E73",
    "circuitree": "#009E73",
    "reaction_network_evolution_jl": "#D55E00",
    "reaction_network_evolution_jl_constrained": "#E69F00",
    "reaction_network_evolution_jl_constrained_bounded": "#CC79A7",
    "random_search": "#7A3E9D",
}

# Candidate provenance follows the database viewer convention. These colors
# describe the emitting source, independently of the method-level palette above.
SOURCE_COLORS = {
    "llm": "#0072B2",
    "rl": "#009E73",
}

METHOD_ORDER = [
    "rl4crn",
    "genai_net_llm",
    "reaction_network_evolution_jl",
    "reaction_network_evolution_jl_constrained",
    "reaction_network_evolution_jl_constrained_bounded",
    "circuitree",
    "random_search",
]


def collect_progress(
    raw_root: str | Path,
    methods: Iterable[str] | None = None,
    benchmark_names: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    method_run_ids: Dict[str, Iterable[str]] | None = None,
    min_ode_simulations: int | None = None,
) -> List[Dict[str, str]]:
    method_filter: Set[str] | None = set(methods) if methods is not None else None
    benchmark_filter: Set[str] | None = set(benchmark_names) if benchmark_names is not None else None
    task_filter: Set[str] | None = set(tasks) if tasks is not None else None
    run_id_filter: Set[str] | None = set(run_ids) if run_ids is not None else None
    method_run_filter = (
        {method: set(ids) for method, ids in method_run_ids.items()}
        if method_run_ids is not None
        else None
    )
    rows: List[Dict[str, str]] = []
    for path in Path(raw_root).glob("*/*/progress.csv"):
        method_name = path.parent.parent.name
        run_id = path.parent.name
        if method_run_filter is not None and run_id not in method_run_filter.get(method_name, set()):
            continue
        if run_id_filter is not None and run_id not in run_id_filter:
            continue
        if benchmark_filter is not None or task_filter is not None:
            run_config = _read_run_config(path.parent / "config.json")
            benchmark_name = run_config.get("benchmark", {}).get("name")
            task_name = run_config.get("task", run_config.get("benchmark", {}).get("task", "rpa"))
            if benchmark_filter is not None and benchmark_name not in benchmark_filter:
                continue
            if task_filter is not None and task_name not in task_filter:
                continue
        with path.open("r", encoding="utf-8") as f:
            path_rows = list(csv.DictReader(f))
        if min_ode_simulations is not None:
            if not path_rows or float(path_rows[-1]["ode_simulations"]) < min_ode_simulations:
                continue
        for row in path_rows:
            if method_filter is None or row["method"] in method_filter:
                rows.append(row)
    return rows


def plot_best_so_far(
    raw_root: str | Path,
    figure_dir: str | Path,
    methods: Iterable[str] | None = None,
    benchmark_names: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    method_run_ids: Dict[str, Iterable[str]] | None = None,
    min_ode_simulations: int | None = None,
    log_y: bool = True,
    x_field: str = "ode_simulations",
    title: str = "Search Progress",
    ylabel: str = "Best-so-far loss",
    figure_name: str = "best_so_far_vs_simulations.png",
    formats: Iterable[str] = ("png",),
    paper: bool = False,
) -> Path:
    rows = collect_progress(
        raw_root,
        methods=methods,
        benchmark_names=benchmark_names,
        tasks=tasks,
        run_ids=run_ids,
        method_run_ids=method_run_ids,
        min_ode_simulations=min_ode_simulations,
    )
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    out = figure_dir / figure_name

    if not rows:
        raise ValueError(f"No progress.csv files found under {raw_root}")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        key = f"{row['method']}:{row['run_id']}"
        grouped.setdefault(key, []).append(row)

    if paper:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figsize = (3.35, 2.45)
        marker_size = 1.6
        linewidth = 1.2
    else:
        figsize = (7, 4.5)
        marker_size = 3
        linewidth = 1.5

    fig, ax = plt.subplots(figsize=figsize)
    for key, group in sorted(grouped.items()):
        group.sort(key=lambda r: float(r.get(x_field, r["ode_simulations"])))
        x = [float(r.get(x_field, r["ode_simulations"])) for r in group]
        y = [max(float(r["best_so_far_loss"]), 1.0e-12) for r in group]
        method = group[0]["method"]
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            markersize=marker_size,
            label=label,
            color=METHOD_COLORS.get(method),
            markevery=_markevery(len(x), paper=paper),
        )

    ax.set_xlabel("Full task simulations")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    saved = []
    stem = out.with_suffix("")
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=400 if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved[0]


def write_final_summary(
    raw_root: str | Path,
    out_path: str | Path,
    methods: Iterable[str] | None = None,
    benchmark_names: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    method_run_ids: Dict[str, Iterable[str]] | None = None,
    min_ode_simulations: int | None = None,
) -> Path:
    rows = collect_progress(
        raw_root,
        methods=methods,
        benchmark_names=benchmark_names,
        tasks=tasks,
        run_ids=run_ids,
        method_run_ids=method_run_ids,
        min_ode_simulations=min_ode_simulations,
    )
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        key = f"{row['method']}:{row['run_id']}"
        grouped.setdefault(key, []).append(row)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "run_id",
        "candidate_evaluations",
        "ode_simulations",
        "scenario_count",
        "scenario_evaluations",
        "best_so_far_loss",
        "elapsed_seconds",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for _key, group in sorted(grouped.items()):
            group.sort(key=lambda r: float(r["ode_simulations"]))
            row = group[-1]
            writer.writerow({field: row.get(field, "") for field in fields})
    return out_path


def plot_seed_summary(
    raw_root: str | Path,
    figure_dir: str | Path,
    methods: Iterable[str] | None = None,
    benchmark_names: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    method_run_ids: Dict[str, Iterable[str]] | None = None,
    min_ode_simulations: int | None = None,
    log_y: bool = True,
    x_field: str = "ode_simulations",
    title: str = "Search Progress",
    ylabel: str = "Best-so-far loss",
    figure_name: str = "best_so_far_triplicate_summary.png",
    formats: Iterable[str] = ("png",),
    paper: bool = False,
) -> Path:
    rows = collect_progress(
        raw_root,
        methods=methods,
        benchmark_names=benchmark_names,
        tasks=tasks,
        run_ids=run_ids,
        method_run_ids=method_run_ids,
        min_ode_simulations=min_ode_simulations,
    )
    if not rows:
        raise ValueError(f"No progress.csv files found under {raw_root}")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['method']}:{row['run_id']}", []).append(row)

    by_method: Dict[str, List[List[Dict[str, str]]]] = {}
    for _key, group in grouped.items():
        group.sort(key=lambda r: float(r.get(x_field, r["ode_simulations"])))
        by_method.setdefault(group[0]["method"], []).append(group)

    if paper:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figsize = (3.35, 2.45)
        linewidth = 1.35
    else:
        figsize = (7, 4.5)
        linewidth = 1.8

    max_x = max(float(group[-1].get(x_field, group[-1]["ode_simulations"])) for group in grouped.values())
    grid = _summary_grid(max_x, log_y=log_y)

    fig, ax = plt.subplots(figsize=figsize)
    for method, runs in sorted(by_method.items()):
        curves = []
        for run in runs:
            x = np.array([float(r.get(x_field, r["ode_simulations"])) for r in run])
            y = np.array([max(float(r["best_so_far_loss"]), 1.0e-12) for r in run])
            curves.append(_step_sample(x, y, grid))
        arr = np.vstack(curves)
        median = np.median(arr, axis=0)
        lower = np.min(arr, axis=0)
        upper = np.max(arr, axis=0)
        color = METHOD_COLORS.get(method)
        label = f"{METHOD_LABELS.get(method, method)} median"
        ax.plot(grid, median, linewidth=linewidth, label=label, color=color)
        if len(runs) > 1:
            ax.fill_between(grid, lower, upper, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Full task simulations")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = (figure_dir / figure_name).with_suffix("")
    saved = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=400 if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved[0]


def plot_side_by_side_seed_summary(
    panels: Iterable[Mapping],
    figure_dir: str | Path,
    figure_name: str = "rpa_logic_triplicate_side_by_side.png",
    formats: Iterable[str] = ("png",),
    log_y: bool = True,
    log_x: bool = False,
    y_limit: tuple[float | None, float | None] | None = None,
    x_field: str = "ode_simulations",
    ylabel: str = "Best-so-far loss",
    paper: bool = False,
) -> Path:
    panel_specs = list(panels)
    if not panel_specs:
        raise ValueError("At least one panel specification is required")

    if paper:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 7.5,
                "axes.labelsize": 8,
                "axes.titlesize": 8.5,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.linewidth": 0.75,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        figsize = (6.8, 2.55)
        median_linewidth = 1.8
        replicate_linewidth = 0.65
    else:
        figsize = (11, 4.2)
        median_linewidth = 2.2
        replicate_linewidth = 0.9

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(panel_specs) == 1:
        axes = [axes]

    legend_handles: Dict[str, object] = {}
    for panel_index, (ax, spec) in enumerate(zip(axes, panel_specs)):
        rows = collect_progress(
            spec["raw_root"],
            methods=spec.get("methods"),
            benchmark_names=spec.get("benchmark_names"),
            tasks=spec.get("tasks"),
            run_ids=spec.get("run_ids"),
            method_run_ids=spec.get("method_run_ids"),
            min_ode_simulations=spec.get("min_ode_simulations"),
        )
        if not rows:
            raise ValueError(f"No progress.csv files found for panel {spec.get('title', panel_index)}")

        grouped: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(f"{row['method']}:{row['run_id']}", []).append(row)

        by_method: Dict[str, List[List[Dict[str, str]]]] = {}
        for group in grouped.values():
            group.sort(key=lambda r: float(r.get(x_field, r["ode_simulations"])))
            by_method.setdefault(group[0]["method"], []).append(group)

        max_x = max(float(group[-1].get(x_field, group[-1]["ode_simulations"])) for group in grouped.values())
        grid = _summary_grid(max_x, log_y=log_y)

        for method in _ordered_methods(by_method):
            runs = by_method[method]
            color = METHOD_COLORS.get(method)
            curves = []
            for run in runs:
                x = np.array([float(r.get(x_field, r["ode_simulations"])) for r in run])
                y = np.array([max(float(r["best_so_far_loss"]), 1.0e-12) for r in run])
                curves.append(_step_sample(x, y, grid))
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=replicate_linewidth,
                    alpha=0.22,
                    solid_capstyle="round",
                    zorder=1,
                )

            median = np.median(np.vstack(curves), axis=0)
            (handle,) = ax.plot(
                grid,
                median,
                color=color,
                linewidth=median_linewidth,
                label=METHOD_LABELS.get(method, method),
                solid_capstyle="round",
                zorder=3,
            )
            legend_handles.setdefault(method, handle)

        ax.set_title(spec.get("title", ""), pad=5)
        ax.set_xlabel("Full task simulations")
        if panel_index == 0:
            ax.set_ylabel(ylabel)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.grid(True, which="major", alpha=0.18, linewidth=0.55)
        ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3, pad=2)
        if spec.get("show_panel_label", True):
            ax.text(
                -0.12,
                1.04,
                chr(ord("A") + panel_index),
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                va="bottom",
            )

    method_handles = [
        legend_handles[method]
        for method in METHOD_ORDER
        if method in legend_handles
    ]
    method_handles.extend(
        handle
        for method, handle in sorted(legend_handles.items())
        if method not in METHOD_ORDER
    )
    method_labels = [handle.get_label() for handle in method_handles]
    fig.legend(
        method_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(4, len(method_handles)),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.6,
    )
    style_handles = [
        Line2D([0], [0], color="#777777", linewidth=median_linewidth, label="Median"),
        Line2D([0], [0], color="#BBBBBB", linewidth=replicate_linewidth, alpha=0.75, label="Individual runs"),
    ]
    fig.legend(
        style_handles,
        [handle.get_label() for handle in style_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.2)

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = (figure_dir / figure_name).with_suffix("")
    saved = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=600 if fmt == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved[0]


def write_seed_summary(
    raw_root: str | Path,
    out_path: str | Path,
    methods: Iterable[str] | None = None,
    benchmark_names: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    method_run_ids: Dict[str, Iterable[str]] | None = None,
    min_ode_simulations: int | None = None,
) -> Path:
    rows = collect_progress(
        raw_root,
        methods=methods,
        benchmark_names=benchmark_names,
        tasks=tasks,
        run_ids=run_ids,
        method_run_ids=method_run_ids,
        min_ode_simulations=min_ode_simulations,
    )
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['method']}:{row['run_id']}", []).append(row)

    finals: Dict[str, List[float]] = {}
    for _key, group in grouped.items():
        group.sort(key=lambda r: float(r["ode_simulations"]))
        finals.setdefault(group[-1]["method"], []).append(float(group[-1]["best_so_far_loss"]))

    fields = ["method", "n_runs", "mean_best_loss", "std_best_loss", "median_best_loss", "min_best_loss", "max_best_loss"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, values in sorted(finals.items()):
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow(
                {
                    "method": method,
                    "n_runs": len(values),
                    "mean_best_loss": statistics.mean(values),
                    "std_best_loss": std,
                    "median_best_loss": statistics.median(values),
                    "min_best_loss": min(values),
                    "max_best_loss": max(values),
                }
            )
    return out_path


def _read_run_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _markevery(n_points: int, paper: bool):
    if not paper or n_points <= 60:
        return None
    return max(1, n_points // 45)


def _summary_grid(max_x: float, log_y: bool) -> np.ndarray:
    if max_x <= 1:
        return np.array([max_x])
    if log_y:
        grid = np.unique(np.round(np.geomspace(1, max_x, 260)).astype(int))
    else:
        grid = np.linspace(1, max_x, 260)
    return grid.astype(float)


def _ordered_methods(by_method: Mapping[str, object]) -> List[str]:
    ordered = [method for method in METHOD_ORDER if method in by_method]
    ordered.extend(method for method in sorted(by_method) if method not in METHOD_ORDER)
    return ordered


def _step_sample(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    idx = np.searchsorted(x, grid, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]
