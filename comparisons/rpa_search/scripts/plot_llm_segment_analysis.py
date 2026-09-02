#!/usr/bin/env python3
"""Analyze LLM-only batch quality, Harness tools, tokens, cost, and latency."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[3]
TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic circuit"}
SEGMENT_EPOCHS = (0, 20, 40, 60, 80)
COLORS = {
    "read": "#0072B2",
    "image": "#CC79A7",
    "write": "#009E73",
    "shell": "#D55E00",
    "skill": "#E69F00",
    "cache": "#56B4E9",
    "input": "#0072B2",
    "output": "#009E73",
}


def run_id(task: str, seed: int, run_suffix: str, candidate_budget: int) -> str:
    return f"{task}_full{candidate_budget}_seed{seed}_{run_suffix}"


def run_directory(
    campaign_root: Path,
    task: str,
    seed: int,
    run_suffix: str,
    candidate_budget: int,
) -> Path:
    matches = sorted(
        (campaign_root / "runs").glob(
            f"*/{run_id(task, seed, run_suffix, candidate_budget)}"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run directory for {task} seed {seed}, found {len(matches)}")
    return matches[0]


def trace_metrics(trace_root: Path, campaign_name: str) -> dict[Path, dict[str, object]]:
    result: dict[Path, dict[str, object]] = {}
    traces = sorted(trace_root.glob(f"**/*{campaign_name}*/**/session.jsonl.zstd"))
    for trace in traces:
        cwd: Path | None = None
        tools: Counter[str] = Counter()
        usage: Counter[str] = Counter()
        api_turns = 0
        process = subprocess.Popen(
            ["zstdcat", str(trace)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            event = json.loads(line)
            if event.get("type") == "session":
                cwd = Path(event["cwd"]).resolve()
            elif event.get("type") == "tool/call":
                tools[str(event["data"].get("name", "unknown"))] += 1
            elif event.get("type") == "assistant/message":
                api_turns += 1
                for key, value in (event["data"].get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] += value
        if process.wait() != 0 or cwd is None:
            raise RuntimeError(f"Could not decode Harness trace {trace}")
        result[cwd] = {"tools": tools, "usage": usage, "api_turns": api_turns}
    return result


def database_metrics(database: Path) -> tuple[dict[int, dict[str, object]], set[str]]:
    uri = f"file:{database.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        final_epoch = int(connection.execute("SELECT MAX(epoch) FROM hof_snapshots").fetchone()[0])
        final_hashes = {
            str(row[0])
            for row in connection.execute(
                """SELECT e.topology_hash FROM hof_snapshot_entries e
                     JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                    WHERE h.epoch = ?""",
                (final_epoch,),
            )
        }
        result = {}
        for llm_row in connection.execute(
            """SELECT llm_run_id, launched_epoch, completed_epoch, requested, produced,
                      valid_count, elapsed_seconds
                 FROM llm_runs ORDER BY launched_epoch"""
        ):
            llm_run, launched, completed, requested, produced, valid, elapsed = llm_row
            baseline = connection.execute(
                """SELECT MIN(e.loss) FROM hof_snapshot_entries e
                     JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                    WHERE h.epoch = ?""",
                (launched,),
            ).fetchone()[0]
            candidates = connection.execute(
                """SELECT topology_hash, valid, loss FROM llm_candidates
                    WHERE llm_run_id = ?""",
                (llm_run,),
            ).fetchall()
            losses = [float(row[2]) for row in candidates if row[1] and row[2] is not None]
            hashes = {str(row[0]) for row in candidates if row[0]}
            future_hashes = {
                str(row[0])
                for row in connection.execute(
                    """SELECT DISTINCT e.topology_hash FROM hof_snapshot_entries e
                         JOIN hof_snapshots h ON h.snapshot_id = e.snapshot_id
                        WHERE h.epoch >= ?""",
                    (completed,),
                )
            }
            result[int(launched)] = {
                "completed_epoch": int(completed),
                "requested": int(requested or 0),
                "produced": int(produced),
                "valid_count": int(valid),
                "elapsed_seconds": float(elapsed or 0.0),
                "baseline_hof_loss": float(baseline),
                "batch_best_loss": min(losses) if losses else None,
                "batch_median_loss": float(np.median(losses)) if losses else None,
                "future_hof_candidates": len(hashes & future_hashes),
                "final_hof_candidates": len(hashes & final_hashes),
            }
    return result, final_hashes


def initialized_workspaces(directory: Path) -> list[Path]:
    """Return only workspaces attributable to an initialized Harness round."""

    accepted = []
    for workspace in sorted((directory / "harness-workspaces").glob("*-crn-generation-*")):
        required = (workspace / "run_manifest.json", workspace / "CONTEXT/SEARCH_STATE.json")
        missing = [path.relative_to(workspace) for path in required if not path.is_file()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            print(f"[skip] uninitialized Harness workspace {workspace}: missing {names}", file=sys.stderr)
            continue
        accepted.append(workspace)
    return accepted


def collect(
    campaign_root: Path,
    trace_root: Path,
    n_seeds: int,
    cache_price: float,
    input_price: float,
    output_price: float,
    run_suffix: str,
    candidate_budget: int = 102400,
) -> list[dict[str, object]]:
    traces = trace_metrics(trace_root, campaign_root.name)
    rows: list[dict[str, object]] = []
    for task in TASKS:
        for seed in range(n_seeds):
            directory = run_directory(
                campaign_root, task, seed, run_suffix, candidate_budget
            )
            database, _ = database_metrics(directory / "results.sqlite")
            workspaces = initialized_workspaces(directory)
            by_epoch = {}
            for workspace in workspaces:
                state = json.loads((workspace / "CONTEXT/SEARCH_STATE.json").read_text())
                epoch = int(state["rl_epoch_at_snapshot"])
                if epoch in by_epoch:
                    raise RuntimeError(
                        f"Duplicate initialized workspaces for {task} seed {seed}, epoch {epoch}"
                    )
                by_epoch[epoch] = workspace
            for segment, epoch in enumerate(SEGMENT_EPOCHS, start=1):
                workspace = by_epoch.get(epoch)
                if workspace is None:
                    raise RuntimeError(f"Missing workspace for {task} seed {seed}, epoch {epoch}")
                trace = traces.get(workspace.resolve(), {"tools": Counter(), "usage": Counter(), "api_turns": 0})
                tools: Counter[str] = trace["tools"]  # type: ignore[assignment]
                usage: Counter[str] = trace["usage"]  # type: ignore[assignment]
                db = database.get(epoch)
                status = json.loads((workspace / "run_status.json").read_text())
                tool_summary = json.loads((workspace / "tool_evaluation_summary.json").read_text())
                process_path = workspace / "calls/0001/process.json"
                process_data = json.loads(process_path.read_text()) if process_path.is_file() else {}
                wall_seconds = process_data.get("duration_seconds")
                if wall_seconds is None and db:
                    wall_seconds = db["elapsed_seconds"]
                cache_tokens = int(usage.get("cacheReadTokens", 0))
                input_tokens = int(usage.get("inputTokens", 0))
                output_tokens = int(usage.get("outputTokens", 0))
                estimated_cost = (
                    cache_tokens * cache_price
                    + input_tokens * input_price
                    + output_tokens * output_price
                ) / 1_000_000
                best = db["batch_best_loss"] if db else None
                baseline = db["baseline_hof_loss"] if db else None
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "segment": segment,
                        "launched_epoch": epoch,
                        "request_status": "accepted" if db else str(status.get("status", "failed")),
                        "requested": db["requested"] if db else 10,
                        "produced": db["produced"] if db else 0,
                        "valid_count": db["valid_count"] if db else 0,
                        "baseline_hof_loss": baseline,
                        "batch_best_loss": best,
                        "batch_median_loss": db["batch_median_loss"] if db else None,
                        "best_to_hof_ratio": float(best) / float(baseline) if best is not None else None,
                        "beat_launch_hof": int(best is not None and float(best) < float(baseline)),
                        "future_hof_candidates": db["future_hof_candidates"] if db else 0,
                        "final_hof_candidates": db["final_hof_candidates"] if db else 0,
                        "wall_seconds": float(wall_seconds) if wall_seconds is not None else None,
                        "api_turns": int(trace["api_turns"]),
                        "input_tokens_cache_miss": input_tokens,
                        "input_tokens_cache_hit": cache_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_tokens": int(usage.get("reasoningTokens", 0)),
                        "estimated_cost_usd": estimated_cost,
                        "tool_calls_total": sum(tools.values()),
                        "read_search_calls": sum(tools[name] for name in ("read", "glob", "grep")),
                        "image_inspection_calls": tools["read_image"],
                        "workspace_write_calls": sum(tools[name] for name in ("write", "edit", "str_replace_editor")),
                        "shell_calls": tools["bash"],
                        "skill_calls": tools["skill"],
                        "evaluator_calls": int(tool_summary.get("used", 0)),
                        "workspace": str(workspace),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for task in TASKS:
        for segment, epoch in enumerate(SEGMENT_EPOCHS, start=1):
            group = [row for row in rows if row["task"] == task and row["segment"] == segment]
            accepted = [row for row in group if row["request_status"] == "accepted"]
            ratios = [float(row["best_to_hof_ratio"]) for row in accepted]
            summary.append(
                {
                    "task": task,
                    "segment": segment,
                    "launched_epoch": epoch,
                    "requests": len(group),
                    "accepted_fraction": len(accepted) / len(group),
                    "candidate_valid_fraction": sum(int(row["valid_count"]) for row in group)
                    / sum(int(row["requested"]) for row in group),
                    "median_best_to_hof_ratio_accepted": float(np.median(ratios)) if ratios else None,
                    "q25_best_to_hof_ratio_accepted": float(np.percentile(ratios, 25)) if ratios else None,
                    "q75_best_to_hof_ratio_accepted": float(np.percentile(ratios, 75)) if ratios else None,
                    "beat_hof_fraction_all_requests": np.mean([int(row["beat_launch_hof"]) for row in group]),
                    "final_hof_fraction_all_requests": np.mean([int(row["final_hof_candidates"]) > 0 for row in group]),
                    "median_wall_seconds": float(
                        np.median([float(row["wall_seconds"]) for row in group if row["wall_seconds"] is not None])
                    ),
                    "total_tokens": sum(
                        int(row["input_tokens_cache_miss"])
                        + int(row["input_tokens_cache_hit"])
                        + int(row["output_tokens"])
                        for row in group
                    ),
                    "total_estimated_cost_usd": sum(float(row["estimated_cost_usd"]) for row in group),
                    "total_tool_calls": sum(int(row["tool_calls_total"]) for row in group),
                    "total_evaluator_calls": sum(int(row["evaluator_calls"]) for row in group),
                }
            )
    return summary


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(ax: plt.Axes, letter: str) -> None:
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.105, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)


def plot_performance(rows: list[dict[str, object]], output: Path, n_seeds: int) -> None:
    style()
    fig, axes = plt.subplots(
        3, 2, figsize=(7.2, 6.15), sharey="row", constrained_layout=True
    )
    tool_fields = (
        ("read_search_calls", "Read/search", COLORS["read"]),
        ("image_inspection_calls", "Inspect plot", COLORS["image"]),
        ("workspace_write_calls", "Write note/output", COLORS["write"]),
        ("shell_calls", "Shell", COLORS["shell"]),
        ("skill_calls", "Load skill", COLORS["skill"]),
    )
    for column, task in enumerate(TASKS):
        groups = [[row for row in rows if row["task"] == task and row["segment"] == segment] for segment in range(1, 6)]
        ax = axes[0, column]
        ratios = [[float(row["best_to_hof_ratio"]) for row in group if row["best_to_hof_ratio"] is not None] for group in groups]
        boxes = ax.boxplot(ratios, positions=range(1, 6), widths=0.55, showfliers=False, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor="#A8D8C8", edgecolor="#00664B", linewidth=0.8)
        for median in boxes["medians"]:
            median.set(color="#004D3A", linewidth=1.3)
        ax.axhline(1, color="#444444", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_yscale("log")
        if column == 0:
            ax.set_ylabel("Best batch loss / launch HoF loss")
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        ax.tick_params(labelbottom=False)
        finish_axis(ax, chr(ord("A") + column))

        ax = axes[1, column]
        accepted = [np.mean([row["request_status"] == "accepted" for row in group]) for group in groups]
        beat = [np.mean([int(row["beat_launch_hof"]) for row in group]) for group in groups]
        retained = [np.mean([int(row["final_hof_candidates"]) > 0 for row in group]) for group in groups]
        ax.plot(range(1, 6), accepted, marker="o", color="#666666", label="Accepted batch")
        ax.plot(range(1, 6), beat, marker="s", color="#009E73", label="Beat launch HoF")
        ax.plot(range(1, 6), retained, marker="^", color="#0072B2", label="Survived in final HoF")
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if column == 0:
            ax.set_ylabel("Fraction of requests")
        ax.tick_params(labelbottom=False)
        finish_axis(ax, chr(ord("C") + column))

        ax = axes[2, column]
        bottoms = np.zeros(5)
        for field, label, color in tool_fields:
            means = np.asarray([np.mean([int(row[field]) for row in group]) for group in groups])
            ax.bar(range(1, 6), means, bottom=bottoms, color=color, width=0.68, label=label)
            bottoms += means
        if column == 0:
            ax.set_ylabel("Mean tool calls / request")
        ax.set_xlabel("RL epoch at request")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(SEGMENT_EPOCHS)
        finish_axis(ax, chr(ord("E") + column))
    axes[1, 1].legend(frameon=False, loc="lower left", handlelength=2.5)
    handles = [Patch(color=color, label=label) for _, label, color in tool_fields]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04, metadata={"Title": "LLM segment performance and tool use"})
    fig.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_resources(rows: list[dict[str, object]], output: Path, n_seeds: int) -> None:
    style()
    fig, axes = plt.subplots(
        3, 2, figsize=(7.2, 6.35), sharex="col", constrained_layout=True
    )
    fig.get_layout_engine().set(hspace=0.08, wspace=0.08)
    token_handles = [
        Patch(color=COLORS["cache"], label="Cached input"),
        Patch(color=COLORS["input"], label="Uncached input"),
        Patch(color=COLORS["output"], label="Output"),
    ]
    for column, task in enumerate(TASKS):
        groups = [[row for row in rows if row["task"] == task and row["segment"] == segment] for segment in range(1, 6)]
        ax = axes[0, column]
        bottoms = np.zeros(5)
        for field, label, color in (
            ("input_tokens_cache_hit", "Cached input", COLORS["cache"]),
            ("input_tokens_cache_miss", "Uncached input", COLORS["input"]),
            ("output_tokens", "Output", COLORS["output"]),
        ):
            means = np.asarray([np.mean([int(row[field]) for row in group]) / 1000 for group in groups])
            ax.bar(range(1, 6), means, bottom=bottoms, color=color, width=0.68, label=label)
            bottoms += means
        ax.set_ylabel("Tokens / request ($10^3$)")
        ax.set_title(TASK_LABELS[task], fontweight="semibold")
        ax.tick_params(labelbottom=False)
        finish_axis(ax, chr(ord("A") + column))

        ax = axes[1, column]
        costs = [[100 * float(row["estimated_cost_usd"]) for row in group] for group in groups]
        boxes = ax.boxplot(costs, positions=range(1, 6), widths=0.55, showfliers=False, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor="#F0D58A", edgecolor="#9A6700", linewidth=0.8)
        for median in boxes["medians"]:
            median.set(color="#6C4900", linewidth=1.3)
        ax.set_ylabel("Estimated cost / request (US cents)")
        ax.tick_params(labelbottom=False)
        finish_axis(ax, chr(ord("C") + column))

        ax = axes[2, column]
        for segment, group in enumerate(groups, start=1):
            values = np.asarray(
                [float(row["wall_seconds"]) / 60 for row in group if row["wall_seconds"] is not None]
            )
            ax.scatter(np.full(len(values), segment), values, s=8, color="#777777", alpha=0.38)
        medians = [
            np.median([float(row["wall_seconds"]) / 60 for row in group if row["wall_seconds"] is not None])
            for group in groups
        ]
        ax.plot(range(1, 6), medians, marker="o", color="#D55E00", linewidth=1.3)
        ax.set_ylabel("Wall time / request (min)")
        ax.set_xlabel("RL epoch at request")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(SEGMENT_EPOCHS)
        finish_axis(ax, chr(ord("E") + column))
    fig.legend(
        handles=token_handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        title=f"DeepSeek-V4-Flash token accounting (n = {n_seeds} per task)",
        title_fontsize=7.2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04, metadata={"Title": "LLM token, cost, and latency analysis"})
    fig.savefig(output.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--run-suffix", default="cvode_llm")
    parser.add_argument("--candidate-budget", type=int, default=102400)
    parser.add_argument("--cache-price", type=float, default=0.0028, help="USD per 1M cached input tokens")
    parser.add_argument("--input-price", type=float, default=0.14, help="USD per 1M uncached input tokens")
    parser.add_argument("--output-price", type=float, default=0.28, help="USD per 1M output tokens")
    parser.add_argument(
        "--performance-output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/llm_segment_performance_20seed.pdf",
    )
    parser.add_argument(
        "--resource-output",
        type=Path,
        default=ROOT / "comparisons/rpa_search/figures/llm_resource_cost_20seed.pdf",
    )
    args = parser.parse_args()
    performance_output = args.performance_output.expanduser().resolve()
    resource_output = args.resource_output.expanduser().resolve()
    rows = collect(
        args.campaign_root.expanduser().resolve(),
        args.trace_root.expanduser().resolve(),
        args.n_seeds,
        args.cache_price,
        args.input_price,
        args.output_price,
        args.run_suffix,
        args.candidate_budget,
    )
    summary = summarize(rows)
    write_csv(performance_output.with_name(f"{performance_output.stem}_requests.csv"), rows)
    write_csv(performance_output.with_name(f"{performance_output.stem}_summary.csv"), summary)
    plot_performance(rows, performance_output, args.n_seeds)
    plot_resources(rows, resource_output, args.n_seeds)
    print(f"Wrote {performance_output}")
    print(f"Wrote {resource_output}")
    print(f"Requests: {len(rows)}; accepted: {sum(row['request_status'] == 'accepted' for row in rows)}")
    print(f"Estimated cost: ${sum(float(row['estimated_cost_usd']) for row in rows):.4f}")


if __name__ == "__main__":
    main()
