#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path
from statistics import mean, median


RAW_ROOT = Path("comparisons/rpa_search/data/raw")
FIGURE_DIR = Path("comparisons/rpa_search/figures")
METHODS = ("rl4crn", "reaction_network_evolution_jl")
TASKS = ("rpa", "logic")


def _progress_row(method: str, task: str, seed: int) -> dict | None:
    suffix = "_cvode" if method == "rl4crn" else ""
    run_id = f"{task}_full102400_seed{seed}{suffix}"
    path = RAW_ROOT / method / run_id / "progress.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def _best_network(method: str, task: str, seed: int) -> dict | None:
    suffix = "_cvode" if method == "rl4crn" else ""
    run_id = f"{task}_full102400_seed{seed}{suffix}"
    path = RAW_ROOT / method / run_id / "best_network.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fixed_reaction_count(task: str) -> int:
    return 2 if task == "rpa" else 4


def _normalize_species_list(text: str) -> tuple[str, ...]:
    parts = [p.strip() for p in text.split("+")]
    parts = [p for p in parts if p and p not in {"∅", "emptyset"}]
    return tuple(sorted(parts))


def _normalize_reaction_string(text: str) -> str:
    text = text.split(";")[0].strip()
    text = text.replace("---->", "->").replace("-->", "->")
    text = re.sub(r"\s+", "", text)
    if "->" not in text:
        return text
    left, right = text.split("->", 1)
    lhs = "+".join(_normalize_species_list(left)) or "∅"
    rhs = "+".join(_normalize_species_list(right)) or "∅"
    return f"{lhs}->{rhs}"


def _topology(method: str, task: str, data: dict) -> frozenset[str]:
    reactions = list(data.get("reactions", []))
    if method == "rl4crn":
        reactions = reactions[_fixed_reaction_count(task):]
    return frozenset(_normalize_reaction_string(r) for r in reactions)


def _jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _load_records(min_sims: int) -> list[dict]:
    records = []
    for task in TASKS:
        for method in METHODS:
            for seed in range(20):
                progress = _progress_row(method, task, seed)
                data = _best_network(method, task, seed)
                if progress is None or data is None:
                    continue
                if float(progress.get("ode_simulations", 0.0)) < float(min_sims):
                    continue
                topo = _topology(method, task, data)
                records.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "loss": float(progress["best_so_far_loss"]),
                        "topology": topo,
                        "active_reactions": len(topo),
                    }
                )
    return records


def _select(records: list[dict], top_fraction: float, loss_threshold: float | None) -> list[tuple[str, list[dict]]]:
    out = [("all_complete", records)]
    if records:
        n_top = max(1, int(round(len(records) * float(top_fraction))))
        out.append(("top_fraction", sorted(records, key=lambda r: r["loss"])[:n_top]))
    if loss_threshold is not None:
        out.append((f"loss_below_{loss_threshold:g}", [r for r in records if r["loss"] < loss_threshold]))
    return out


def _summarize_group(task: str, method: str, subset_name: str, records: list[dict]) -> dict:
    losses = [r["loss"] for r in records]
    counts = [r["active_reactions"] for r in records]
    topologies = [r["topology"] for r in records]
    distances = [_jaccard_distance(a, b) for a, b in combinations(topologies, 2)]
    return {
        "task": task,
        "method": method,
        "subset": subset_name,
        "n_selected": len(records),
        "mean_loss": mean(losses) if losses else "",
        "median_loss": median(losses) if losses else "",
        "unique_topologies": len(set(topologies)),
        "unique_topology_fraction": len(set(topologies)) / len(topologies) if topologies else "",
        "mean_pairwise_jaccard_distance": mean(distances) if distances else "",
        "median_pairwise_jaccard_distance": median(distances) if distances else "",
        "mean_active_reactions": mean(counts) if counts else "",
        "median_active_reactions": median(counts) if counts else "",
        "min_active_reactions": min(counts) if counts else "",
        "max_active_reactions": max(counts) if counts else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sims", type=int, default=102400)
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--loss-threshold", type=float, default=0.05)
    parser.add_argument("--out-dir", default=str(FIGURE_DIR))
    args = parser.parse_args()

    records = _load_records(args.min_sims)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    frequency_rows = []
    by_task_method: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_task_method.setdefault((record["task"], record["method"]), []).append(record)

    for (task, method), group in sorted(by_task_method.items()):
        for subset_name, subset in _select(group, args.top_fraction, args.loss_threshold):
            summary_rows.append(_summarize_group(task, method, subset_name, subset))

            counts: dict[str, int] = {}
            for record in subset:
                for reaction in record["topology"]:
                    counts[reaction] = counts.get(reaction, 0) + 1
            for reaction, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                frequency_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "subset": subset_name,
                        "reaction": reaction,
                        "count": count,
                        "fraction": count / len(subset) if subset else "",
                    }
                )

    summary_path = out_dir / "rpa_logic_genai_julia_topology_diversity_summary.csv"
    fields = [
        "task",
        "method",
        "subset",
        "n_selected",
        "mean_loss",
        "median_loss",
        "unique_topologies",
        "unique_topology_fraction",
        "mean_pairwise_jaccard_distance",
        "median_pairwise_jaccard_distance",
        "mean_active_reactions",
        "median_active_reactions",
        "min_active_reactions",
        "max_active_reactions",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    freq_path = out_dir / "rpa_logic_genai_julia_topology_reaction_frequencies.csv"
    with freq_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "method", "subset", "reaction", "count", "fraction"])
        writer.writeheader()
        writer.writerows(frequency_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {freq_path}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
