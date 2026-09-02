#!/usr/bin/env python3
"""Analyze the RPA Harness-agent versus direct non-thinking ablation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


CONDITIONS = ("reasoning_harness", "minimal_nonthinking")
LABELS = {
    "reasoning_harness": "Reasoning Harness agent",
    "minimal_nonthinking": "Minimal non-thinking generator",
}
COLORS = {"reasoning_harness": "#0072B2", "minimal_nonthinking": "#D55E00"}


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def completed_seeds(root: Path) -> set[int]:
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    return {
        int(row["seed"])
        for row in status.get("completed", [])
        if row.get("task") == "rpa" and int(row.get("returncode", 1)) == 0
    }


def database_path(root: Path, seed: int) -> Path:
    matches = sorted((root / "runs").glob(f"*-rpa-seed{seed}/rpa_*/results.sqlite"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one RPA database for seed {seed} under {root}, found {len(matches)}")
    return matches[0]


def read_run(path: Path, max_epoch: int) -> dict:
    with _connect(path) as connection:
        snapshots = connection.execute(
            """SELECT h.epoch, e.rank, e.topology_hash, e.loss
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE e.loss IS NOT NULL ORDER BY h.epoch, e.rank"""
        ).fetchall()
        llm_rows = connection.execute(
            """SELECT r.launched_epoch, r.completed_epoch, r.elapsed_seconds,
                      c.topology_hash, c.loss, c.valid
                 FROM llm_candidates c JOIN llm_runs r ON r.llm_run_id = c.llm_run_id
                ORDER BY r.launched_epoch, c.candidate_index"""
        ).fetchall()

    best_by_epoch = {}
    final_topology = None
    for epoch, rank, topology, loss in snapshots:
        if int(rank) == 0:
            best_by_epoch[int(epoch)] = float(loss)
            if int(epoch) <= max_epoch:
                final_topology = str(topology)
    epochs = sorted(best_by_epoch)
    convergence = []
    for epoch in range(max_epoch + 1):
        available = [value for value in epochs if value <= epoch]
        convergence.append(best_by_epoch[max(available)] if available else np.nan)

    batches = {}
    llm_topologies = set()
    valid = 0
    produced = 0
    request_timings = {}
    for launched, completed, elapsed, topology, loss, is_valid in llm_rows:
        if int(launched) > max_epoch:
            continue
        request_timings[int(launched)] = (int(completed) - int(launched), float(elapsed))
        produced += 1
        if not is_valid or loss is None:
            continue
        valid += 1
        llm_topologies.add(str(topology))
        batches.setdefault(int(launched), []).append(float(loss))
    batch_best = [min(batches[epoch]) for epoch in sorted(batches)]
    return {
        "final_loss": float(convergence[-1]),
        "convergence": convergence,
        "batch_best": batch_best,
        "first_llm_best": batch_best[0] if batch_best else np.nan,
        "all_llm_best": min(batch_best) if batch_best else np.nan,
        "final_origin": "LLM" if final_topology in llm_topologies else "RL",
        "valid": valid,
        "produced": produced,
        "unique_llm_topologies": len(llm_topologies),
        "median_request_lag_epochs": float(np.median([row[0] for row in request_timings.values()])),
        "median_request_seconds": float(np.median([row[1] for row in request_timings.values()])),
        "first_completion_epoch": min(
            (int(completed) for launched, completed, *_ in llm_rows if int(launched) == 0),
            default=np.nan,
        ),
    }


def audit_direct_context(root: Path) -> dict:
    requests = list(root.rglob("request.json"))
    hof_entries = 0
    entries_with_actions = 0
    entries_with_crn = 0
    disabled = 0
    for path in requests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        disabled += payload.get("thinking") == {"type": "disabled"}
        prompt = payload["messages"][-1]["content"]
        match = re.search(
            r"Current Hall of Fame, ranked by lower loss:\n(.*?)\n\nGenerate",
            prompt,
            flags=re.DOTALL,
        )
        if not match or match.group(1).startswith("No Hall-of-Fame"):
            continue
        entries = json.loads(match.group(1))
        hof_entries += len(entries)
        entries_with_actions += sum(bool(entry.get("actions")) for entry in entries)
        entries_with_crn += sum(bool(entry.get("crn")) for entry in entries)
    return {
        "requests": len(requests),
        "thinking_disabled_requests": disabled,
        "hof_entries": hof_entries,
        "hof_entries_with_actions": entries_with_actions,
        "hof_entries_with_crn": entries_with_crn,
    }


def bootstrap_median_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(1729)
    indices = rng.integers(0, len(left), size=(100_000, len(left)))
    samples = np.median((right - left)[indices], axis=1)
    return tuple(float(value) for value in np.percentile(samples, (2.5, 97.5)))


def collect(reasoning_root: Path, nonthinking_root: Path, max_epoch: int) -> tuple[list[dict], dict]:
    seeds = sorted(completed_seeds(reasoning_root) & completed_seeds(nonthinking_root))
    rows = []
    run_data = {}
    for seed in seeds:
        for condition, root in zip(CONDITIONS, (reasoning_root, nonthinking_root)):
            data = read_run(database_path(root, seed), max_epoch)
            run_data[(seed, condition)] = data
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "final_loss": data["final_loss"],
                    "first_llm_best": data["first_llm_best"],
                    "all_llm_best": data["all_llm_best"],
                    "final_origin": data["final_origin"],
                    "valid_candidates": data["valid"],
                    "produced_candidates": data["produced"],
                    "unique_llm_topologies": data["unique_llm_topologies"],
                    "median_request_lag_epochs": data["median_request_lag_epochs"],
                    "median_request_seconds": data["median_request_seconds"],
                }
            )
    return rows, {"seeds": seeds, "runs": run_data}


def summarize(rows: list[dict], collected: dict, context_audit: dict) -> dict:
    summary = {"n": len(collected["seeds"]), "direct_context_audit": context_audit}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "final_median": float(np.median([row["final_loss"] for row in subset])),
            "final_mean": float(np.mean([row["final_loss"] for row in subset])),
            "final_min": float(np.min([row["final_loss"] for row in subset])),
            "first_llm_median": float(np.median([row["first_llm_best"] for row in subset])),
            "all_llm_median": float(np.median([row["all_llm_best"] for row in subset])),
            "llm_origin_winners": sum(row["final_origin"] == "LLM" for row in subset),
            "valid_candidates": sum(row["valid_candidates"] for row in subset),
            "produced_candidates": sum(row["produced_candidates"] for row in subset),
            "median_unique_llm_topologies": float(
                np.median([row["unique_llm_topologies"] for row in subset])
            ),
            "median_request_lag_epochs": float(
                np.median([row["median_request_lag_epochs"] for row in subset])
            ),
            "median_request_seconds": float(
                np.median([row["median_request_seconds"] for row in subset])
            ),
        }
    lookup = {(row["seed"], row["condition"]): row for row in rows}
    reasoning = np.asarray(
        [lookup[(seed, "reasoning_harness")]["final_loss"] for seed in collected["seeds"]]
    )
    nonthinking = np.asarray(
        [lookup[(seed, "minimal_nonthinking")]["final_loss"] for seed in collected["seeds"]]
    )
    statistic = wilcoxon(reasoning, nonthinking)
    summary["paired_endpoint"] = {
        "reasoning_wins": int(np.sum(reasoning < nonthinking)),
        "nonthinking_wins": int(np.sum(nonthinking < reasoning)),
        "ties": int(np.sum(reasoning == nonthinking)),
        "median_nonthinking_over_reasoning": float(np.median(nonthinking / reasoning)),
        "median_difference_nonthinking_minus_reasoning": float(np.median(nonthinking - reasoning)),
        "median_difference_bootstrap_95ci": bootstrap_median_difference(reasoning, nonthinking),
        "wilcoxon_statistic": float(statistic.statistic),
        "wilcoxon_pvalue": float(statistic.pvalue),
    }
    for field in ("first_llm_best", "all_llm_best"):
        left = np.asarray(
            [lookup[(seed, "reasoning_harness")][field] for seed in collected["seeds"]]
        )
        right = np.asarray(
            [lookup[(seed, "minimal_nonthinking")][field] for seed in collected["seeds"]]
        )
        finite = np.isfinite(left) & np.isfinite(right)
        test = wilcoxon(left[finite], right[finite])
        summary[f"paired_{field}"] = {
            "reasoning_wins": int(np.sum(left[finite] < right[finite])),
            "median_nonthinking_over_reasoning": float(np.median(right[finite] / left[finite])),
            "wilcoxon_pvalue": float(test.pvalue),
        }
    return summary


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )


def plot(rows: list[dict], collected: dict, output: Path, max_epoch: int) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    rng = np.random.default_rng(11)
    seeds = collected["seeds"]
    runs = collected["runs"]

    ax = axes[0, 0]
    endpoint = {condition: [] for condition in CONDITIONS}
    for seed in seeds:
        for condition in CONDITIONS:
            endpoint[condition].append(runs[(seed, condition)]["final_loss"])
        ax.plot((0, 1), [endpoint[c][-1] for c in CONDITIONS], color="#BBBBBB", lw=0.65, zorder=1)
    for index, condition in enumerate(CONDITIONS):
        ax.scatter(
            np.full(len(seeds), index) + rng.normal(0, 0.025, len(seeds)),
            endpoint[condition], color=COLORS[condition], s=23, alpha=0.82, zorder=2,
        )
        ax.plot(index, np.median(endpoint[condition]), marker="_", markersize=16,
                markeredgewidth=2.2, color="#111111", zorder=3)
    ax.set_xticks((0, 1), ("Reasoning\nHarness", "Minimal\nnon-thinking"))
    ax.set_yscale("log")
    ax.set_ylabel("Final joint-HOF loss")
    ax.set_title("Paired endpoint", fontweight="semibold")

    ax = axes[0, 1]
    epochs = np.arange(max_epoch + 1)
    for condition in CONDITIONS:
        values = np.asarray([runs[(seed, condition)]["convergence"] for seed in seeds])
        median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, (25, 75), axis=0)
        ax.fill_between(epochs, q25, q75, color=COLORS[condition], alpha=0.14, linewidth=0)
        ax.plot(epochs, median, color=COLORS[condition], lw=1.8, label=LABELS[condition])
    ax.set_yscale("log")
    ax.set_xlim(0, max_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best joint-HOF loss")
    ax.set_title("Optimization trajectory", fontweight="semibold")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for condition in CONDITIONS:
        batch_rows = [runs[(seed, condition)]["batch_best"] for seed in seeds]
        batch_count = max(len(values) for values in batch_rows)
        batches = np.full((len(batch_rows), batch_count), np.nan)
        for row_index, values in enumerate(batch_rows):
            batches[row_index, : len(values)] = values
        x = np.arange(1, batch_count + 1)
        median = np.nanmedian(batches, axis=0)
        q25, q75 = np.nanpercentile(batches, (25, 75), axis=0)
        ax.fill_between(x, q25, q75, color=COLORS[condition], alpha=0.14, linewidth=0)
        ax.plot(x, median, marker="o", markersize=3.5, color=COLORS[condition], lw=1.7)
    ax.set_yscale("log")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("LLM request index")
    ax.set_ylabel("Best direct proposal loss")
    ax.set_title("Proposal quality", fontweight="semibold")

    ax = axes[1, 1]
    origins = {
        condition: [
            sum(runs[(seed, condition)]["final_origin"] == source for seed in seeds)
            for source in ("LLM", "RL")
        ]
        for condition in CONDITIONS
    }
    positions = np.arange(2)
    width = 0.34
    for index, condition in enumerate(CONDITIONS):
        ax.bar(positions + (index - 0.5) * width, origins[condition], width,
               color=COLORS[condition], label=LABELS[condition])
    ax.set_xticks(positions, ("LLM-origin", "RL-origin"))
    ax.set_ylim(0, len(seeds))
    ax.set_ylabel("Rank-one runs")
    ax.set_title("Final winning topology provenance", fontweight="semibold")

    for label, ax in zip("ABCD", axes.flat):
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontweight="bold", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = "RPA reasoning-agent versus minimal non-thinking generator ablation"
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=400 if suffix == ".png" else None,
                    facecolor="white", metadata={"Title": title} if suffix == ".pdf" else None)
    plt.close(fig)


def write_report(summary: dict, output: Path) -> None:
    reasoning = summary["reasoning_harness"]
    direct = summary["minimal_nonthinking"]
    paired = summary["paired_endpoint"]
    audit = summary["direct_context_audit"]
    first = summary["paired_first_llm_best"]
    all_batches = summary["paired_all_llm_best"]
    ci = paired["median_difference_bootstrap_95ci"]
    text = f"""# RPA Reasoning-Agent Ablation

This completed 20-seed diagnostic compares the full reasoning-enabled DeepSeek V4
Flash Harness agent with a one-request, simple-prompt generator using the provider's
explicit non-thinking mode. Lower loss is better.

## Endpoint results

| Condition | Median loss | Mean loss | Best loss | LLM-origin rank one |
|---|---:|---:|---:|---:|
| Reasoning Harness agent | {reasoning['final_median']:.6g} | {reasoning['final_mean']:.6g} | {reasoning['final_min']:.6g} | {reasoning['llm_origin_winners']}/{summary['n']} |
| Minimal non-thinking generator | {direct['final_median']:.6g} | {direct['final_mean']:.6g} | {direct['final_min']:.6g} | {direct['llm_origin_winners']}/{summary['n']} |

The reasoning Harness arm won {paired['reasoning_wins']}/{summary['n']} paired seeds.
The median endpoint ratio was {paired['median_nonthinking_over_reasoning']:.2f}x
(non-thinking / reasoning). The paired median loss difference was
{paired['median_difference_nonthinking_minus_reasoning']:.6g}, with a seed-bootstrap
95% interval [{ci[0]:.6g}, {ci[1]:.6g}]. The exact two-sided Wilcoxon signed-rank test
gave `p={paired['wilcoxon_pvalue']:.6g}`.

## Proposal mechanism

| Condition | First-batch median | Best-of-five median | Valid proposals | Median unique LLM topologies |
|---|---:|---:|---:|---:|
| Reasoning Harness agent | {reasoning['first_llm_median']:.6g} | {reasoning['all_llm_median']:.6g} | {reasoning['valid_candidates']}/{reasoning['produced_candidates']} | {reasoning['median_unique_llm_topologies']:.1f} |
| Minimal non-thinking generator | {direct['first_llm_median']:.6g} | {direct['all_llm_median']:.6g} | {direct['valid_candidates']}/{direct['produced_candidates']} | {direct['median_unique_llm_topologies']:.1f} |

The minimal generator usually satisfied the schema but did not generate competitive
RPA networks. This separates syntactic compliance from useful scientific proposal
quality. In the full Harness arm, LLM-origin topologies often supplied the winner;
in the minimal arm, RL supplied every final rank-one topology.
The reasoning arm produced the better first batch in
{first['reasoning_wins']}/{summary['n']} seeds (median ratio
{first['median_nonthinking_over_reasoning']:.1f}x; `p={first['wilcoxon_pvalue']:.6g}`)
and the better best-of-five proposal pool in
{all_batches['reasoning_wins']}/{summary['n']} seeds (median ratio
{all_batches['median_nonthinking_over_reasoning']:.1f}x;
`p={all_batches['wilcoxon_pvalue']:.6g}`). The minimal generator sampled more unique
topologies, but this did not translate into performance; diversity must therefore be
reported jointly with quality.

The median per-run request duration was {reasoning['median_request_seconds']:.1f} s
for the Harness agent and {direct['median_request_seconds']:.1f} s for the minimal
generator, corresponding to median insertion lags of
{reasoning['median_request_lag_epochs']:.1f} and
{direct['median_request_lag_epochs']:.1f} RL epochs. The Harness advantage therefore
did not arise from receiving proposals sooner; its requests were substantially slower.

![Paired endpoints, convergence, direct proposal quality, and final provenance.](../rpa_search/figures/rpa_reasoning_agent_ablation.png)

## Interpretation boundary

This is a compound agent-design ablation, not a pure latent-reasoning toggle. The
Harness arm has workspace memory, complete CRN text, SIL state, cached diagnostics,
and iterative evaluator feedback. The minimal arm makes one direct model call per
batch, has no tools, and requests no reasoning text.

The artifact audit found {audit['requests']} direct requests, all with thinking
disabled. Its {audit['hof_entries']} post-initial HOF entries contained
{audit['hof_entries_with_actions']} nonempty action lists and
{audit['hof_entries_with_crn']} complete CRN strings. Thus only losses, not network
structures, reached the direct generator after the first call. The implementation
has been corrected for future runs by including `str(state)` as a fallback CRN
representation, but this completed cohort must remain labeled as the minimal-agent
comparison. A clean reasoning-only claim requires a confirmatory rerun with matched
HOF content and all other observable context held fixed.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-root", type=Path, required=True)
    parser.add_argument("--nonthinking-root", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, collected = collect(args.reasoning_root.resolve(), args.nonthinking_root.resolve(), args.max_epoch)
    context_audit = audit_direct_context(args.nonthinking_root.resolve())
    summary = summarize(rows, collected, context_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot(rows, collected, args.output, args.max_epoch)
    write_report(summary, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
