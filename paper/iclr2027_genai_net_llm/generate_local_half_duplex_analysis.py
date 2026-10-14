#!/usr/bin/env python3
"""Generate audited local-model directional-communication artifacts."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


PAPER = Path(__file__).resolve().parent
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
TASKS = ("rpa", "logic")
ARMS = ("none", "rl_to_llm", "llm_to_rl", "full")
ARM_LABELS = {
    "none": "None",
    "rl_to_llm": r"HOF $\to$ LLM",
    "llm_to_rl": r"LLM $\to$ RL",
    "full": "Full duplex",
}
COLORS = {
    "none": "#6F6F6F",
    "rl_to_llm": "#D55E00",
    "llm_to_rl": "#0072B2",
    "full": "#009E73",
}
THRESHOLDS = {
    "rpa": 0.016491168958935352,
    "logic": 0.04271267898491677,
}
AUDITS = {
    ("rpa", "full"): ("contract_v2_local_qwen_rpa_logic_audit.json",),
    ("logic", "full"): ("contract_v2_local_qwen_rpa_logic_audit.json",),
    ("rpa", "none"): ("contract_v2_local_qwen_rpa_no_communication_audit.json",),
    ("logic", "none"): ("contract_v2_local_qwen_logic_no_communication_audit.json",),
    ("rpa", "llm_to_rl"): (
        "contract_v2_local_qwen_half_rpa_llm_to_rl_sentinel_audit.json",
        "contract_v2_local_qwen_half_rpa_llm_to_rl_main_audit.json",
    ),
    ("logic", "llm_to_rl"): (
        "contract_v2_local_qwen_half_logic_llm_to_rl_sentinel_audit.json",
        "contract_v2_local_qwen_half_logic_llm_to_rl_main_audit.json",
    ),
    ("rpa", "rl_to_llm"): (
        "contract_v2_local_qwen_half_rpa_rl_to_llm_sentinel_audit.json",
        "contract_v2_local_qwen_half_rpa_rl_to_llm_main_audit.json",
    ),
    ("logic", "rl_to_llm"): (
        "contract_v2_local_qwen_half_logic_rl_to_llm_sentinel_audit.json",
        "contract_v2_local_qwen_half_logic_rl_to_llm_main_audit.json",
    ),
}
CONTRASTS = (
    ("feedback_with_insertion", "full", "llm_to_rl", "+ RL/HOF state\n(insertion on)"),
    ("insertion_with_feedback", "full", "rl_to_llm", "+ insertion\n(RL/HOF state on)"),
    ("insertion_without_feedback", "llm_to_rl", "none", "+ insertion\n(RL/HOF state off)"),
    ("feedback_without_insertion", "rl_to_llm", "none", "+ RL/HOF state\n(insertion off)"),
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def audited_run_roots(task: str, arm: str) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    for name in AUDITS[task, arm]:
        audit = read_json(GENERATED / name)
        for run in audit["runs"]:
            if run["task"] == task and run["status"] == "pass":
                seed = int(run["seed"])
                if seed in roots:
                    raise RuntimeError(f"Duplicate audited {task}/{arm} seed {seed}.")
                roots[seed] = Path(run["run_root"])
    return roots


def terminal_record(task: str, arm: str, seed: int, run_root: Path) -> dict:
    database = run_root / "results.sqlite"
    connection = sqlite3.connect(
        f"file:{database.resolve()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        snapshot = connection.execute(
            """SELECT snapshot_id FROM hof_snapshots
               WHERE boundary = 'terminal_boundary'
               ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        if snapshot is None:
            raise RuntimeError(f"No terminal HOF in {database}.")
        rows = connection.execute(
            """SELECT topology_hash, loss, provenance_class
               FROM hof_snapshot_entries WHERE snapshot_id = ?""",
            (snapshot["snapshot_id"],),
        ).fetchall()
        accepted = connection.execute(
            "SELECT COALESCE(SUM(accepted), 0) FROM llm_runs"
        ).fetchone()[0]
        model_failures = connection.execute(
            "SELECT COUNT(*) FROM llm_failures"
        ).fetchone()[0]
    finally:
        connection.close()

    qualified = [row for row in rows if float(row["loss"]) < THRESHOLDS[task]]
    unique_qualified = {row["topology_hash"] for row in qualified}
    return {
        "task": task,
        "arm": arm,
        "seed": seed,
        "rank_one_loss": min(float(row["loss"]) for row in rows),
        "qualified_topologies": len(unique_qualified),
        "accepted_llm_candidates": int(accepted),
        "model_output_failures": int(model_failures),
        "direct_llm_qualified": len(
            {
                row["topology_hash"]
                for row in qualified
                if row["provenance_class"] == "direct_llm"
            }
        ),
        "rl_refinement_qualified": len(
            {
                row["topology_hash"]
                for row in qualified
                if row["provenance_class"]
                == "rl_parameter_refinement_of_llm_topology"
            }
        ),
        "rl_native_qualified": len(
            {
                row["topology_hash"]
                for row in qualified
                if row["provenance_class"] == "rl_native_topology"
            }
        ),
        "run_root": str(run_root),
    }


def paired_test(left: np.ndarray, right: np.ndarray, *, higher_is_better: bool) -> dict:
    delta = left - right
    p_value = (
        1.0
        if np.allclose(delta, 0.0)
        else float(
            wilcoxon(left, right, zero_method="pratt", alternative="two-sided").pvalue
        )
    )
    wins = int(np.sum(delta > 0)) if higher_is_better else int(np.sum(delta < 0))
    ties = int(np.sum(delta == 0))
    return {
        "median_delta": float(np.median(delta)),
        "wins": wins,
        "ties": ties,
        "losses": int(len(delta) - wins - ties),
        "wilcoxon_p_unadjusted": p_value,
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def distribution_panel(ax, values: list[np.ndarray], *, ylabel: str, log: bool) -> None:
    positions = np.arange(len(ARMS))
    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, arm in zip(violins["bodies"], ARMS):
        body.set_facecolor(COLORS[arm])
        body.set_edgecolor(COLORS[arm])
        body.set_alpha(0.16)
    for arm_index, (arm, data) in enumerate(zip(ARMS, values)):
        q25, median, q75 = np.quantile(data, (0.25, 0.5, 0.75))
        ax.plot([arm_index, arm_index], [q25, q75], color=COLORS[arm], linewidth=2.2)
        ax.plot(
            [arm_index - 0.13, arm_index + 0.13],
            [median, median],
            color="#202020",
            linewidth=1.7,
            zorder=5,
        )
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions, [ARM_LABELS[arm] for arm in ARMS])
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def endpoint_figure(records: list[dict], common: dict[str, list[int]]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.0), constrained_layout=True)
    rng = np.random.default_rng(904)
    for column, task in enumerate(TASKS):
        seeds = common[task]
        by_key = {
            (row["arm"], row["seed"]): row
            for row in records
            if row["task"] == task
        }
        jitter = {seed: rng.uniform(-0.065, 0.065) for seed in seeds}
        for row_index, (metric, ylabel, log) in enumerate(
            (
                ("rank_one_loss", "Rank-one loss", True),
                ("qualified_topologies", "Qualified topologies", False),
            )
        ):
            ax = axes[row_index, column]
            values = [
                np.asarray([by_key[arm, seed][metric] for seed in seeds], dtype=float)
                for arm in ARMS
            ]
            distribution_panel(ax, values, ylabel=ylabel, log=log)
            for seed in seeds:
                x = np.arange(len(ARMS), dtype=float) + jitter[seed]
                y = np.asarray([by_key[arm, seed][metric] for arm in ARMS], dtype=float)
                ax.plot(x, y, color="#777777", alpha=0.15, linewidth=0.55, zorder=1)
                for index, arm in enumerate(ARMS):
                    ax.scatter(
                        x[index],
                        y[index],
                        s=11,
                        color=COLORS[arm],
                        alpha=0.65,
                        edgecolor="white",
                        linewidth=0.25,
                        zorder=3,
                    )
            if row_index == 0:
                title = "RPA" if task == "rpa" else "Logic circuit"
                ax.set_title(f"{title} (matched n={len(seeds)})", weight="bold")
            if row_index == 1:
                ax.set_ylim(bottom=-0.8)
    fig.savefig(
        FIGURES / "local_half_duplex_endpoints.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / "local_half_duplex_endpoints.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def logic_directional_figure(records: list[dict], common: dict[str, list[int]]) -> None:
    """Render the compact Logic-only directional figure used in the main text."""

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
    seeds = common["logic"]
    by_key = {
        (row["arm"], row["seed"]): row
        for row in records
        if row["task"] == "logic"
    }
    rng = np.random.default_rng(906)
    jitter = {seed: rng.uniform(-0.065, 0.065) for seed in seeds}
    compact_labels = {
        "none": "None",
        "rl_to_llm": "RL/HOF\n$\\rightarrow$ LLM",
        "llm_to_rl": "LLM\n$\\rightarrow$ RL",
        "full": "Full\nduplex",
    }
    for panel, (ax, metric, ylabel, log) in enumerate(
        zip(
            axes,
            ("rank_one_loss", "qualified_topologies"),
            ("Rank-one loss", "Qualified topologies"),
            (True, False),
        )
    ):
        values = [
            np.asarray([by_key[arm, seed][metric] for seed in seeds], dtype=float)
            for arm in ARMS
        ]
        distribution_panel(ax, values, ylabel=ylabel, log=log)
        for seed in seeds:
            x = np.arange(len(ARMS), dtype=float) + jitter[seed]
            y = np.asarray([by_key[arm, seed][metric] for arm in ARMS], dtype=float)
            ax.plot(x, y, color="#777777", alpha=0.15, linewidth=0.55, zorder=1)
            for index, arm in enumerate(ARMS):
                ax.scatter(
                    x[index],
                    y[index],
                    s=11,
                    color=COLORS[arm],
                    alpha=0.65,
                    edgecolor="white",
                    linewidth=0.25,
                    zorder=3,
                )
        ax.set_xticks(
            np.arange(len(ARMS)), [compact_labels[arm] for arm in ARMS]
        )
        ax.set_title(
            f"{'AB'[panel]}  {'Loss' if panel == 0 else 'Qualified-topology yield'}",
            loc="left",
            weight="bold",
        )
        if metric == "qualified_topologies":
            ax.set_ylim(bottom=-0.8)
    fig.savefig(
        FIGURES / "local_logic_directional.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / "local_logic_directional.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def contrast_figure(contrast_rows: list[dict]) -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), constrained_layout=True)
    effect_colors = (
        COLORS["rl_to_llm"],
        COLORS["llm_to_rl"],
        COLORS["llm_to_rl"],
        COLORS["rl_to_llm"],
    )
    rng = np.random.default_rng(905)
    for ax, task in zip(axes, TASKS):
        task_rows = [row for row in contrast_rows if row["task"] == task]
        grouped = {
            name: np.asarray(
                [
                    row["qualified_topology_delta"]
                    for row in task_rows
                    if row["contrast"] == name
                ],
                dtype=float,
            )
            for name, *_ in CONTRASTS
        }
        positions = np.arange(len(CONTRASTS))
        violin = ax.violinplot(
            [grouped[name] for name, *_ in CONTRASTS],
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], effect_colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.16)
        for index, (name, _left, _right, _label) in enumerate(CONTRASTS):
            data = grouped[name]
            x = index + rng.uniform(-0.12, 0.12, len(data))
            ax.scatter(
                x,
                data,
                s=13,
                color=effect_colors[index],
                alpha=0.68,
                edgecolor="white",
                linewidth=0.25,
            )
            median = float(np.median(data))
            ax.plot(
                [index - 0.16, index + 0.16],
                [median, median],
                color="#202020",
                linewidth=2.0,
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(positions, [item[3] for item in CONTRASTS])
        ax.set_ylabel(r"Change in qualified topologies ($\Delta$)")
        ax.set_title("RPA" if task == "rpa" else "Logic circuit", weight="bold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(
        FIGURES / "local_half_duplex_directional_effects.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES / "local_half_duplex_directional_effects.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    roots = {
        (task, arm): audited_run_roots(task, arm) for task in TASKS for arm in ARMS
    }
    common = {
        task: sorted(set.intersection(*(set(roots[task, arm]) for arm in ARMS)))
        for task in TASKS
    }
    records = [
        terminal_record(task, arm, seed, root)
        for (task, arm), seed_roots in roots.items()
        for seed, root in sorted(seed_roots.items())
    ]
    by_key = {(row["task"], row["arm"], row["seed"]): row for row in records}
    contrast_rows = []
    summary = {
        "status": "provisional-audited-partial",
        "quality_thresholds": THRESHOLDS,
        "model_output_failure_policy": "retained as zero contribution",
        "infrastructure_failure_policy": "excluded pending targeted rerun",
        "tasks": {},
    }
    for task in TASKS:
        seeds = common[task]
        task_summary = {"complete_case_seeds": seeds, "arms": {}, "contrasts": {}}
        for arm in ARMS:
            arm_rows = [by_key[task, arm, seed] for seed in sorted(roots[task, arm])]
            task_summary["arms"][arm] = {
                "valid_n": len(arm_rows),
                "median_rank_one_loss": float(
                    np.median([row["rank_one_loss"] for row in arm_rows])
                ),
                "median_qualified_topologies": float(
                    np.median([row["qualified_topologies"] for row in arm_rows])
                ),
                "mean_qualified_topologies": float(
                    np.mean([row["qualified_topologies"] for row in arm_rows])
                ),
                "total_model_output_failures": int(
                    sum(row["model_output_failures"] for row in arm_rows)
                ),
            }
        for name, left, right, _label in CONTRASTS:
            left_q = np.asarray(
                [by_key[task, left, seed]["qualified_topologies"] for seed in seeds],
                dtype=float,
            )
            right_q = np.asarray(
                [by_key[task, right, seed]["qualified_topologies"] for seed in seeds],
                dtype=float,
            )
            left_loss = np.asarray(
                [by_key[task, left, seed]["rank_one_loss"] for seed in seeds],
                dtype=float,
            )
            right_loss = np.asarray(
                [by_key[task, right, seed]["rank_one_loss"] for seed in seeds],
                dtype=float,
            )
            task_summary["contrasts"][name] = {
                "left_arm": left,
                "right_arm": right,
                "n": len(seeds),
                "qualified_topologies": paired_test(
                    left_q, right_q, higher_is_better=True
                ),
                "rank_one_loss": paired_test(
                    left_loss, right_loss, higher_is_better=False
                ),
            }
            for index, seed in enumerate(seeds):
                contrast_rows.append(
                    {
                        "task": task,
                        "contrast": name,
                        "left_arm": left,
                        "right_arm": right,
                        "seed": seed,
                        "qualified_topology_delta": int(left_q[index] - right_q[index]),
                        "rank_one_loss_delta": float(left_loss[index] - right_loss[index]),
                    }
                )
        summary["tasks"][task] = task_summary

    with (GENERATED / "local_half_duplex_endpoints.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (GENERATED / "local_half_duplex_contrasts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast_rows[0]))
        writer.writeheader()
        writer.writerows(contrast_rows)
    (GENERATED / "local_half_duplex_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    endpoint_figure(records, common)
    logic_directional_figure(records, common)
    contrast_figure(contrast_rows)


if __name__ == "__main__":
    main()
