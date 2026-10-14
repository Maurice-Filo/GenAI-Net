#!/usr/bin/env python3
"""Generate submission figures and tables from audited contract-v2 artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import analyze_contract_v2_primary as primary
except ImportError:
    import analyze_contract_v2_primary as primary


PAPER = Path(__file__).resolve().parent
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
LOCAL_AUDIT = GENERATED / "contract_v2_local_qwen_rpa_logic_audit.json"
COMMUNICATION_TASKS = ("rpa", "logic", "classifier")
LOCAL_TASKS = ("rpa", "logic")
PROVENANCE_ORDER = ("direct_llm", "llm_topology_rl", "rl_native")
PROVENANCE_LABELS = {
    "direct_llm": "Direct LLM",
    "llm_topology_rl": "RL on LLM topology",
    "rl_native": "RL-native",
}
PROVENANCE_COLORS = {
    "direct_llm": "#D17B24",
    "llm_topology_rl": "#6A9F58",
    "rl_native": "#2463A5",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.4,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def _save(fig: plt.Figure, stem: str, title: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title},
    )
    fig.savefig(
        FIGURES / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        dpi=400,
    )
    plt.close(fig)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.985,
        label,
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=10,
    )


def _clean(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color="#E0E0E0", linewidth=0.55, zorder=0)


def _audit_gate(registry: dict) -> dict:
    if registry.get("status") != "audited-complete":
        raise RuntimeError("Primary registry is not audited-complete.")
    for relative in registry["audit_files"]:
        audit = primary.read_json(primary.ROOT / relative)
        if audit.get("status") != "pass":
            raise RuntimeError(f"Primary audit does not pass: {relative}")
    local = primary.read_json(LOCAL_AUDIT)
    if local.get("status") != "pass" or local["completion"] != {
        "audited_runs": 40,
        "expected_runs": 40,
        "passing_runs": 40,
    }:
        raise RuntimeError("Local-model full-duplex cohort is not completely audited.")
    return local


def _rl_emitter_trajectory(path: Path) -> list[dict]:
    with primary.connect(path) as connection:
        rows = connection.execute(
            """SELECT h.epoch, h.cumulative_rl_evaluations,
                      MIN(CASE WHEN e.emitter = 'RL' THEN e.loss END) AS best_rl_loss
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE h.boundary = 'epoch'
                GROUP BY h.snapshot_id
                ORDER BY h.epoch"""
        ).fetchall()
    result = [dict(row) for row in rows]
    if len(result) != 100 or any(row["best_rl_loss"] is None for row in result):
        raise RuntimeError(f"Expected 100 complete RL-emitter snapshots in {path}.")
    return result


def collect_rl_emitter_convergence(registry: dict) -> list[dict]:
    rows = []
    for task in COMMUNICATION_TASKS:
        by_arm: dict[str, dict[int, list[dict]]] = {}
        for arm in ("full_duplex", "no_communication"):
            by_arm[arm] = {
                seed: _rl_emitter_trajectory(primary.database_path(registry, arm, task, seed))
                for seed in range(20)
            }
        for seed in range(20):
            for full, none in zip(
                by_arm["full_duplex"][seed],
                by_arm["no_communication"][seed],
            ):
                if int(full["epoch"]) != int(none["epoch"]):
                    raise RuntimeError("Matched RL-emitter snapshots have different epochs.")
                full_loss = float(full["best_rl_loss"])
                none_loss = float(none["best_rl_loss"])
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "epoch": int(full["epoch"]),
                        "candidate_evaluations": int(full["cumulative_rl_evaluations"]),
                        "full_rl_emitter_loss": full_loss,
                        "none_rl_emitter_loss": none_loss,
                        "none_over_full": none_loss / full_loss,
                    }
                )
    return rows


def _provenance_group(value: str) -> str:
    if value == "direct_llm":
        return "direct_llm"
    if value in {
        "rl_exact_reemission_of_llm_candidate",
        "rl_parameter_refinement_of_llm_topology",
    }:
        return "llm_topology_rl"
    if value == "rl_native_topology":
        return "rl_native"
    raise RuntimeError(f"Unknown provenance class: {value}")


def collect_flash_provenance(registry: dict, plan: dict) -> list[dict]:
    rows = []
    for task in primary.TASKS:
        arms = ("full_duplex", "no_communication") if task in COMMUNICATION_TASKS else ("full_duplex",)
        for arm in arms:
            for seed in range(20):
                hof = primary.terminal_hof(primary.database_path(registry, arm, task, seed))
                threshold = float(plan["quality_thresholds"][task])
                for entry in hof:
                    rows.append(
                        {
                            "model": "DeepSeek V4 Flash",
                            "task": task,
                            "arm": arm,
                            "seed": seed,
                            "rank": int(entry["rank"]),
                            "loss": float(entry["loss"]),
                            "qualified": float(entry["loss"]) < threshold,
                            "provenance": _provenance_group(entry["provenance_class"]),
                        }
                    )
    return rows


def collect_local(
    registry: dict, local_audit: dict, plan: dict
) -> tuple[list[dict], list[dict]]:
    endpoints = []
    provenance = []
    for audited in local_audit["runs"]:
        task = str(audited["task"])
        seed = int(audited["seed"])
        run_root = Path(audited["run_root"])
        completed = primary.read_json(run_root / "completed.json")
        hof = primary.terminal_hof(run_root / "results.sqlite")
        threshold = float(plan["quality_thresholds"][task])
        local_loss = float(completed["best_loss"])
        rl_loss = primary.control_loss(plan, task, seed)
        flash_loss = float(
            primary.read_json(primary.completed_path(registry, "full_duplex", task, seed))[
                "best_loss"
            ]
        )
        endpoints.append(
            {
                "task": task,
                "seed": seed,
                "local_loss": local_loss,
                "flash_loss": flash_loss,
                "rl_loss": rl_loss,
                "rl_over_local": rl_loss / local_loss,
                "local_over_flash": local_loss / flash_loss,
            }
        )
        for entry in hof:
            provenance.append(
                {
                    "model": "Qwen3.8-27B-DFlash2",
                    "task": task,
                    "arm": "full_duplex",
                    "seed": seed,
                    "rank": int(entry["rank"]),
                    "loss": float(entry["loss"]),
                    "qualified": float(entry["loss"]) < threshold,
                    "provenance": _provenance_group(entry["provenance_class"]),
                }
            )
    if len(endpoints) != 40:
        raise RuntimeError(f"Expected 40 local endpoints, found {len(endpoints)}.")
    return endpoints, provenance


def summarize_rl_emitter(rows: list[dict]) -> list[dict]:
    summary = []
    for task in COMMUNICATION_TASKS:
        for epoch in range(100):
            subset = [row for row in rows if row["task"] == task and row["epoch"] == epoch]
            ratios = np.asarray([row["none_over_full"] for row in subset], dtype=float)
            full = np.asarray([row["full_rl_emitter_loss"] for row in subset], dtype=float)
            none = np.asarray([row["none_rl_emitter_loss"] for row in subset], dtype=float)
            relation = Counter(
                "tie"
                if np.isclose(left, right, rtol=primary.TIE_RTOL, atol=primary.TIE_ATOL)
                else "win"
                if left < right
                else "loss"
                for left, right in zip(full, none)
            )
            summary.append(
                {
                    "task": task,
                    "epoch": epoch,
                    "candidate_evaluations": int(subset[0]["candidate_evaluations"]),
                    "median_none_over_full": float(np.median(ratios)),
                    "q25_none_over_full": float(np.quantile(ratios, 0.25)),
                    "q75_none_over_full": float(np.quantile(ratios, 0.75)),
                    "full_median": float(np.median(full)),
                    "none_median": float(np.median(none)),
                    "wins": relation["win"],
                    "ties": relation["tie"],
                    "losses": relation["loss"],
                }
            )
    return summary


def summarize_provenance(rows: list[dict]) -> list[dict]:
    summary = []
    keys = sorted({(row["model"], row["task"], row["arm"]) for row in rows})
    for model, task, arm in keys:
        subset = [
            row
            for row in rows
            if row["model"] == model and row["task"] == task and row["arm"] == arm
        ]
        for scope, selected in (
            ("rank_one", [row for row in subset if row["rank"] == 0]),
            ("qualified_hof", [row for row in subset if row["qualified"]]),
        ):
            counts = Counter(row["provenance"] for row in selected)
            summary.append(
                {
                    "model": model,
                    "task": task,
                    "arm": arm,
                    "scope": scope,
                    "n": len(selected),
                    **{name: counts[name] for name in PROVENANCE_ORDER},
                }
            )
    return summary


def summarize_local(endpoints: list[dict]) -> list[dict]:
    summary = []
    for task in LOCAL_TASKS:
        rows = sorted((row for row in endpoints if row["task"] == task), key=lambda row: row["seed"])
        local = [float(row["local_loss"]) for row in rows]
        flash = [float(row["flash_loss"]) for row in rows]
        rl = [float(row["rl_loss"]) for row in rows]
        relation = Counter(primary.paired_relation(left, right) for left, right in zip(local, rl))
        summary.append(
            {
                "task": task,
                "n": len(rows),
                "local_median": float(np.median(local)),
                "flash_median": float(np.median(flash)),
                "rl_median": float(np.median(rl)),
                "median_paired_rl_over_local": float(np.median(np.asarray(rl) / local)),
                "median_paired_local_over_flash": float(np.median(np.asarray(local) / flash)),
                "wins_vs_rl": relation["win"],
                "ties_vs_rl": relation["tie"],
                "losses_vs_rl": relation["loss"],
                "wilcoxon_vs_rl_p": primary.wilcoxon_p(local, rl),
            }
        )
    return summary


def summarize_requests(insertions: list[dict]) -> list[dict]:
    summary = []
    for arm in ("full_duplex", "no_communication"):
        rows = [row for row in insertions if row["arm"] == arm]
        summary.append(
            {
                "arm": arm,
                "runs": len(rows),
                "scheduled": sum(int(row["scheduled"]) for row in rows),
                "served": sum(int(row["served"]) for row in rows),
                "successful": sum(int(row["successful"]) for row in rows),
                "failed": sum(int(row["failed"]) for row in rows),
                "accepted": sum(int(row["accepted"]) for row in rows),
                "rejected": sum(int(row["rejected"]) for row in rows),
                "before_training_boundary": sum(
                    int(row["before_training_boundary"]) for row in rows
                ),
                "terminal_only": sum(int(row["terminal_only"]) for row in rows),
                "provider_calls": sum(int(row["provider_calls"]) for row in rows),
                "median_run_latency_seconds": float(
                    np.median([float(row["median_latency_seconds"]) for row in rows])
                ),
            }
        )
    return summary


def summarize_structure(portfolios: list[dict]) -> list[dict]:
    summary = []
    for task in COMMUNICATION_TASKS:
        full_by_seed = {
            int(row["seed"]): row["qualified_mean_jaccard"]
            for row in portfolios
            if row["task"] == task and row["arm"] == "full_duplex"
        }
        none_by_seed = {
            int(row["seed"]): row["qualified_mean_jaccard"]
            for row in portfolios
            if row["task"] == task and row["arm"] == "no_communication"
        }
        seeds = [
            seed
            for seed in sorted(full_by_seed)
            if full_by_seed[seed] is not None and none_by_seed[seed] is not None
        ]
        full = [float(full_by_seed[seed]) for seed in seeds]
        none = [float(none_by_seed[seed]) for seed in seeds]
        relation = Counter(
            "tie"
            if np.isclose(left, right, rtol=primary.TIE_RTOL, atol=primary.TIE_ATOL)
            else "win"
            if left > right
            else "loss"
            for left, right in zip(full, none)
        )
        summary.append(
            {
                "task": task,
                "paired_n": len(seeds),
                "full_median_mean_jaccard": float(np.median(full)),
                "none_median_mean_jaccard": float(np.median(none)),
                "wins": relation["win"],
                "ties": relation["tie"],
                "losses": relation["loss"],
                "wilcoxon_p": primary.wilcoxon_p([-value for value in full], [-value for value in none]),
            }
        )
    return summary


def plot_main_results(
    endpoints: list[dict], portfolios: list[dict], rl_summary: list[dict]
) -> None:
    _style()
    fig = plt.figure(figsize=(7.2, 4.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 1.0), width_ratios=(1.05, 1.0))

    axis = fig.add_subplot(grid[0, 0])
    rng = np.random.default_rng(27)
    short_labels = {
        "rpa": "RPA",
        "logic": "Logic",
        "classifier": "Classifier",
        "dose_hill": "Hill",
        "dose_ultrasensitive": "Ultrasensitive",
        "dose_biphasic": "Biphasic",
        "oscillator_mean": "Osc. mean",
        "oscillator_frequency": "Osc. frequency",
    }
    for index, task in enumerate(primary.TASKS):
        values = np.asarray(
            [
                float(row["rl_over_full"])
                for row in endpoints
                if row["task"] == task and row["arm"] == "full_duplex"
            ]
        )
        axis.scatter(
            values,
            index + rng.uniform(-0.10, 0.10, len(values)),
            color=primary.COLORS["full"],
            s=12,
            alpha=0.58,
            edgecolors="none",
            zorder=2,
        )
        q25, median, q75 = np.quantile(values, (0.25, 0.5, 0.75))
        axis.plot((q25, q75), (index, index), color="#222222", linewidth=1.8, zorder=3)
        axis.scatter(median, index, marker="D", color="#222222", s=20, zorder=4)
    axis.axvline(1, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_yticks(range(len(primary.TASKS)), [short_labels[task] for task in primary.TASKS])
    axis.set_ylim(len(primary.TASKS) - 0.5, -0.5)
    axis.set_xlabel("Endpoint loss ratio: RL-only / both paths")
    axis.set_title("Rank-one quality across eight tasks", fontweight="semibold")
    axis.text(0.98, 0.04, "> 1 favors both paths", transform=axis.transAxes, ha="right", color="#555555")
    _clean(axis, grid_axis="x")
    _panel_label(axis, "A")

    axis = fig.add_subplot(grid[0, 1])
    xbase = np.arange(len(COMMUNICATION_TASKS), dtype=float)
    gain_rng = np.random.default_rng(41)
    for index, task in enumerate(COMMUNICATION_TASKS):
        full = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "full_duplex"),
            key=lambda row: row["seed"],
        )
        none = sorted(
            (row for row in portfolios if row["task"] == task and row["arm"] == "no_communication"),
            key=lambda row: row["seed"],
        )
        gains = np.asarray(
            [
                left["qualified_topologies"] - right["qualified_topologies"]
                for left, right in zip(full, none)
            ],
            dtype=float,
        )
        median = float(np.median(gains))
        bar_x = index - 0.12
        violin_x = index + 0.18
        axis.bar(
            bar_x,
            median,
            width=0.25,
            color=primary.COLORS["full"],
            alpha=0.92,
            zorder=2,
            label="Median paired gain" if index == 0 else None,
        )
        violin = axis.violinplot(
            gains,
            positions=[violin_x],
            widths=0.28,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.45,
        )
        for body in violin["bodies"]:
            body.set_facecolor(primary.COLORS["full"])
            body.set_edgecolor(primary.COLORS["full"])
            body.set_alpha(0.16)
            body.set_linewidth(0.7)
            body.set_zorder(1)
        q25, q75 = np.quantile(gains, (0.25, 0.75))
        axis.plot(
            (violin_x, violin_x),
            (q25, q75),
            color="#333333",
            linewidth=1.6,
            solid_capstyle="round",
            zorder=4,
        )
        jitter = gain_rng.uniform(-0.055, 0.055, len(gains))
        axis.scatter(
            violin_x + jitter,
            gains,
            color="#3F3F3F",
            s=9,
            alpha=0.72,
            linewidths=0,
            zorder=3,
            label="Matched seeds" if index == 0 else None,
        )
        axis.scatter(
            violin_x,
            median,
            marker="D",
            color="#222222",
            s=15,
            zorder=5,
        )
        axis.text(
            bar_x,
            median + (0.8 if median >= 0 else -0.8),
            f"{median:+.1f}",
            ha="center",
            va="bottom" if median >= 0 else "top",
            fontsize=6.2,
            fontweight="semibold",
            color="#333333",
            zorder=5,
        )
    axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
    axis.set_xticks(xbase, [primary.LABELS[task] for task in COMMUNICATION_TASKS])
    axis.set_ylabel("Gain in qualified topologies\nboth paths - independent union")
    axis.set_title("Communication expands useful portfolios", fontweight="semibold")
    axis.text(
        0.99,
        0.04,
        "> 0 favors both paths",
        transform=axis.transAxes,
        ha="right",
        color="#555555",
    )
    axis.legend(frameon=False, loc="upper left", handlelength=1.2)
    _clean(axis)
    _panel_label(axis, "B")

    axis = fig.add_subplot(grid[1, :])
    task_colors = {"rpa": "#2672B8", "logic": "#D17B24", "classifier": "#16845B"}
    for task in COMMUNICATION_TASKS:
        subset = [row for row in rl_summary if row["task"] == task]
        epoch = np.asarray([row["epoch"] for row in subset])
        median = np.asarray([row["median_none_over_full"] for row in subset])
        q25 = np.asarray([row["q25_none_over_full"] for row in subset])
        q75 = np.asarray([row["q75_none_over_full"] for row in subset])
        axis.fill_between(epoch, q25, q75, color=task_colors[task], alpha=0.11, linewidth=0)
        axis.plot(epoch, median, color=task_colors[task], linewidth=1.8, label=primary.LABELS[task])
    axis.axhline(1, color="#555555", linewidth=0.8, linestyle="--")
    for request_epoch in (20, 40, 60, 80):
        axis.axvline(request_epoch, color="#BBBBBB", linewidth=0.55, linestyle=":", zorder=0)
    axis.set_xlim(0, 99)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("RL epoch (dotted lines: scheduled LLM rounds)")
    axis.set_ylabel("Best RL-emitted loss ratio\nindependent union / both paths")
    axis.set_title("The RL-emitted frontier improves faster after communication", fontweight="semibold")
    axis.text(0.99, 0.05, "> 1 favors both paths", transform=axis.transAxes, ha="right", color="#555555")
    axis.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.055, 1.0))
    _clean(axis)
    _panel_label(axis, "C")
    _save(fig, "main_results", "Contract-v2 full-duplex results")


def _summary_lookup(rows: list[dict], **keys: str) -> dict:
    matches = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary row for {keys}, found {len(matches)}.")
    return matches[0]


def plot_provenance(provenance_summary: list[dict]) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)

    def horizontal_stack(
        axis: plt.Axes,
        rows: list[dict],
        labels: list[str],
        *,
        scale: float,
        xmax: float,
        title: str,
        panel: str,
        value_format: str,
    ) -> None:
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for category in PROVENANCE_ORDER:
            values = np.asarray([row[category] / scale for row in rows], dtype=float)
            axis.barh(
                y,
                values,
                left=left,
                color=PROVENANCE_COLORS[category],
                height=0.68,
                label=PROVENANCE_LABELS[category],
                zorder=2,
            )
            for ypos, start, value in zip(y, left, values):
                if value >= (2.0 if scale == 1 else 2.4):
                    axis.text(
                        start + value / 2,
                        ypos,
                        format(value, value_format),
                        color="white",
                        fontsize=5.8,
                        fontweight="semibold",
                        ha="center",
                        va="center",
                        zorder=3,
                    )
            left += values
        axis.set_yticks(y, labels)
        axis.set_xlim(0, xmax)
        axis.set_title(title, fontweight="semibold", pad=5)
        _clean(axis, grid_axis="x")
        _panel_label(axis, panel)

    axis = axes[0]
    labels = []
    bars = []
    for task in COMMUNICATION_TASKS:
        for arm in ("full_duplex", "no_communication"):
            labels.append(f"{primary.LABELS[task]}\n{'Full' if arm == 'full_duplex' else 'Independent'}")
            bars.append(
                _summary_lookup(
                    provenance_summary,
                    model="DeepSeek V4 Flash",
                    task=task,
                    arm=arm,
                    scope="rank_one",
                )
            )
    labels = [label.replace("\n", " | ") for label in labels]
    horizontal_stack(
        axis,
        bars,
        labels,
        scale=1,
        xmax=20.5,
        title="Flash rank-one source",
        panel="A",
        value_format=".0f",
    )
    axis.set_xlabel("Seeds (of 20)")

    axis = axes[1]
    labels = []
    bars = []
    for task in COMMUNICATION_TASKS:
        for arm in ("full_duplex", "no_communication"):
            labels.append(f"{primary.LABELS[task]}\n{'Full' if arm == 'full_duplex' else 'Independent'}")
            bars.append(
                _summary_lookup(
                    provenance_summary,
                    model="DeepSeek V4 Flash",
                    task=task,
                    arm=arm,
                    scope="qualified_hof",
                )
            )
    labels = [label.replace("\n", " | ") for label in labels]
    horizontal_stack(
        axis,
        bars,
        labels,
        scale=20,
        xmax=27,
        title="Qualified portfolio source",
        panel="B",
        value_format=".1f",
    )
    axis.set_xlabel("Mean entries per seed")

    axis = axes[2]
    labels = []
    bars = []
    for task in LOCAL_TASKS:
        for model in ("DeepSeek V4 Flash", "Qwen3.8-27B-DFlash2"):
            labels.append(f"{primary.LABELS[task]}\n{'Flash' if model.startswith('DeepSeek') else 'Local'}")
            bars.append(
                _summary_lookup(
                    provenance_summary,
                    model=model,
                    task=task,
                    arm="full_duplex",
                    scope="rank_one",
                )
            )
    labels = [label.replace("\n", " | ") for label in labels]
    horizontal_stack(
        axis,
        bars,
        labels,
        scale=1,
        xmax=20.5,
        title="Model-dependent rank one",
        panel="C",
        value_format=".0f",
    )
    axis.set_xlabel("Seeds (of 20)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    _save(fig, "provenance_and_local", "Emitter provenance and local-model portability")


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def write_endpoint_table(summary: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Matched 100-epoch rank-one results. W/T/L counts compare both communication paths enabled with RL-only; ratios greater than one favor the communication-enabled arm. $p_H$ is Holm-adjusted across eight tasks.}",
        r"\label{tab:endpoints}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & Full & RL-only & W/T/L & RL/full & $p_H$ \\",
        r"\midrule",
    ]
    for row in summary["endpoint"]:
        marker = r"$^{*}$" if float(row["holm_p"]) < 0.05 else ""
        lines.append(
            f"{primary.LABELS[row['task']]} & {_fmt(row['full_median'])} & "
            f"{_fmt(row['rl_median'])} & {row['wins']}/{row['ties']}/{row['losses']} & "
            f"{row['median_paired_rl_over_full']:.3g}{marker} & {_fmt(row['holm_p'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}\parbox{0.94\linewidth}{\scriptsize $^{*}p_H<0.05$. Machine-readable raw and adjusted values accompany the generated table.}",
            r"\end{table}",
        ]
    )
    (GENERATED / "endpoint_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_communication_table(summary: dict, rl_summary: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Hosted-model communication intervention. Portfolio columns are median qualified topology counts in the terminal top 30. The last column is the median paired uncoupled/both-paths ratio for the best RL-emitted HOF member at epoch 99.}",
        r"\label{tab:communication}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & Full & Indep. & W/T/L & $p_H$ & RL-frontier ratio \\",
        r"\midrule",
    ]
    for row in summary["communication_portfolio"]:
        final = _summary_lookup(rl_summary, task=row["task"], epoch=99)
        marker = r"$^{*}$" if float(row["holm_p"]) < 0.05 else ""
        lines.append(
            f"{primary.LABELS[row['task']]} & {row['full_median_qualified_topologies']:.3g} & "
            f"{row['none_median_qualified_topologies']:.3g} & "
            f"{row['wins']}/{row['ties']}/{row['losses']}{marker} & {_fmt(row['holm_p'])} & "
            f"{final['median_none_over_full']:.3g} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}\parbox{0.94\linewidth}{\scriptsize $^{*}p_H<0.05$ for the three-task portfolio family. Ratios greater than one indicate a better RL-emitted frontier with both communication paths enabled.}",
            r"\end{table}",
        ]
    )
    (GENERATED / "communication_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_local_table(local_summary: list[dict], provenance_summary: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Exploratory local-model endpoints with both communication paths enabled. This comparison against RL-only assesses portability but does not itself isolate a communication direction.}",
        r"\label{tab:local}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & Local & Flash & RL-only & W/T/L vs RL & $p$ \\",
        r"\midrule",
    ]
    for row in local_summary:
        lines.append(
            f"{primary.LABELS[row['task']]} & {_fmt(row['local_median'])} & "
            f"{_fmt(row['flash_median'])} & {_fmt(row['rl_median'])} & "
            f"{row['wins_vs_rl']}/{row['ties_vs_rl']}/{row['losses_vs_rl']} & "
            f"{_fmt(row['wilcoxon_vs_rl_p'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (GENERATED / "local_model_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_request_table(rows: list[dict]) -> None:
    labels = {"full_duplex": "Both paths", "no_communication": "Independent union"}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Audited request accounting. Each successful round comprises one Decider and one Writer provider call and requests ten proposals.}",
        r"\label{tab:requests}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Arm & Runs & Served & Failed & Accepted & Rejected & Before boundary \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['arm']]} & {row['runs']} & {row['served']}/{row['scheduled']} & "
            f"{row['failed']} & {row['accepted']} & {row['rejected']} & "
            f"{row['before_training_boundary']}/{row['served']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (GENERATED / "request_accounting_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_table(plan: dict) -> None:
    details = {
        "rpa": ("9 input/disturbance pairs", 5, r"$10^{-8}$"),
        "logic": ("16 truth-table rows", 5, r"$10^{-8}$"),
        "classifier": ("32 labelled initial states", 6, r"$10^{-6}$"),
        "dose_hill": ("10 input levels", 6, r"$10^{-10}$"),
        "dose_ultrasensitive": ("10 input levels", 6, r"$10^{-10}$"),
        "dose_biphasic": ("10 input levels", 6, r"$10^{-10}$"),
        "oscillator_mean": ("autonomous mean target", 6, r"$10^{-10}$"),
        "oscillator_frequency": ("3 frequency targets", 6, r"$10^{-10}$"),
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Task and scenarios & Added reactions & Tolerance & Threshold \\",
        r"\midrule",
    ]
    for task in primary.TASKS:
        scenario, reactions, tolerance = details[task]
        lines.append(
            f"{primary.LABELS[task]}: {scenario} & {reactions} & {tolerance} & "
            f"{_fmt(float(plan['quality_thresholds'][task]))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Executable task contracts. Tolerances show equal CVODE relative and absolute tolerances. The quality threshold is the frozen median RL-only endpoint loss and qualification uses a strict inequality.}",
            r"\label{tab:tasks}",
            r"\end{table}",
        ]
    )
    (GENERATED / "task_contract_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    registry = primary.read_json(primary.REGISTRY_PATH)
    plan = primary.read_json(primary.PLAN_PATH)
    local_audit = _audit_gate(registry)
    endpoints, portfolios, insertions, _ = primary.collect(registry, plan)
    summary = primary.summarize(endpoints, portfolios)
    rl_rows = collect_rl_emitter_convergence(registry)
    rl_summary = summarize_rl_emitter(rl_rows)
    flash_provenance = collect_flash_provenance(registry, plan)
    local_endpoints, local_provenance = collect_local(registry, local_audit, plan)
    provenance_summary = summarize_provenance(flash_provenance + local_provenance)
    local_summary = summarize_local(local_endpoints)
    request_summary = summarize_requests(insertions)
    structure_summary = summarize_structure(portfolios)

    primary.write_csv(GENERATED / "rl_emitter_convergence.csv", rl_rows)
    primary.write_csv(GENERATED / "local_model_endpoints.csv", local_endpoints)
    primary.write_csv(GENERATED / "provenance_summary.csv", provenance_summary)
    output = {
        "schema_version": 1,
        "evidence_gate": {
            "primary_registry_status": registry["status"],
            "local_audit_status": local_audit["status"],
            "historical_runs_used": False,
        },
        "endpoint": summary["endpoint"],
        "communication_portfolio": summary["communication_portfolio"],
        "rl_emitter_convergence": rl_summary,
        "provenance": provenance_summary,
        "local_model": local_summary,
        "request_accounting": request_summary,
        "quality_conditioned_structure": structure_summary,
        "prespecified_estimand_disclosure": (
            "The frozen eight-task RL-only qualified-topology family is not estimable because "
            "the available RL-only records contain no terminal top-30 archives. No control "
            "archive was reconstructed or imputed."
        ),
    }
    (GENERATED / "submission_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_main_results(endpoints, portfolios, rl_summary)
    plot_provenance(provenance_summary)
    write_endpoint_table(summary)
    write_communication_table(summary, rl_summary)
    write_local_table(local_summary, provenance_summary)
    write_request_table(request_summary)
    write_task_table(plan)
    print(json.dumps({"local_model": local_summary, "request_accounting": request_summary}, indent=2))


if __name__ == "__main__":
    main()
