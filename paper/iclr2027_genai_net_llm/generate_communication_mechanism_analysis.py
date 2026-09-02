#!/usr/bin/env python3
"""Reproduce topology-distance and source-filtered communication diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparisons.llm_crn_generation.run_mmc2_harness_smoke import BUILDERS, CONFIGS
from comparisons.rpa_search.scripts.plot_communication_ablation_over_time import (
    _through,
    database_path,
    read_hof,
    read_llm_candidates,
)
from paper.iclr2027_genai_net_llm.audit_paper_experiments import (
    CAMPAIGN_BASE,
    hybrid_paths,
)
from paper.iclr2027_genai_net_llm.generate_quality_portfolio import (
    elite_portfolios,
    rl_endpoint,
)
from RL4CRN.utils.results_database import serialize_crn


TASKS = ("rpa", "logic")
TASK_LABELS = {"rpa": "RPA", "logic": "Logic"}
METHODS = ("full_duplex", "independent_pool")
NO_COMMUNICATION_CAMPAIGN = "flash-no-communication-long300-20seed"
NO_COMMUNICATION_SUFFIX = "cvode_llm_flash_no_communication"
CANDIDATE_BUDGET = 307200
EPOCH = 100
SEEDS = tuple(range(20))


def structural_record_token(record: Mapping) -> str:
    """Return an exact, order-stable identity for one labelled reaction."""

    return json.dumps(
        dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def structural_records(structure_json: str) -> tuple[str, ...]:
    return tuple(
        sorted(structural_record_token(record) for record in json.loads(structure_json))
    )


def fixed_template_records(task: str) -> tuple[str, ...]:
    """Build the executed task and return its exact fixed structural records."""

    config = json.loads(CONFIGS[task].read_text(encoding="utf-8"))
    crn, _library_components, _task, _cfg = BUILDERS[task](config)
    return structural_records(serialize_crn(crn)["structure_json"])


def topology_records_by_hash(
    path: Path, topology_hashes: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    """Load complete structural records for the requested topology hashes."""

    requested = sorted(set(str(value) for value in topology_hashes))
    if not requested:
        return {}
    placeholders = ",".join("?" for _value in requested)
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            f"SELECT topology_hash, structure_json FROM crns "
            f"WHERE topology_hash IN ({placeholders})",
            requested,
        ).fetchall()
    result = {
        str(topology_hash): structural_records(str(structure_json))
        for topology_hash, structure_json in rows
    }
    missing = set(requested) - set(result)
    if missing:
        raise RuntimeError(
            f"Database {path} lacks {len(missing)} requested topology structures"
        )
    return result


def reaction_id(record_token: str) -> int | None:
    value = json.loads(record_token).get("reaction_id")
    return None if value is None else int(value)


def canonical_parameterization_identifier(
    identifier: tuple[str, str],
) -> tuple[str, str]:
    """Canonicalize a database identifier without changing scientific values."""

    topology_hash, parameters_json = identifier
    reactions = json.loads(parameters_json)
    canonical_reactions = [
        {str(key): value for key, value in reaction.items() if key != "index"}
        for reaction in reactions
    ]
    canonical_reactions.sort(
        key=lambda reaction: json.dumps(
            reaction, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    return (
        str(topology_hash),
        json.dumps(
            canonical_reactions,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def qualified_topology_structures(
    candidates: Iterable[dict],
    *,
    threshold: float,
    topology_records: Mapping[str, tuple[str, ...]],
    fixed_records: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    """Return exact added-reaction records per qualified labelled topology."""

    structures: dict[str, frozenset[str]] = {}
    for candidate in candidates:
        if float(candidate["loss"]) >= float(threshold):
            continue
        topology = str(candidate["topology"])
        records = topology_records[topology]
        remaining = Counter(records)
        missing = []
        for fixed_record in fixed_records:
            if remaining[fixed_record] <= 0:
                missing.append(fixed_record)
            else:
                remaining[fixed_record] -= 1
        if missing:
            raise RuntimeError(
                f"Topology {topology} omits {len(missing)} exact "
                "fixed-template records"
            )
        structures.setdefault(
            topology,
            frozenset(record for record, count in remaining.items() if count > 0),
        )
    return structures


def mean_pairwise_distance(structures: Iterable[frozenset[str]]) -> float | None:
    values = list(structures)
    if len(values) < 2:
        return None
    distances = [jaccard_distance(left, right) for left, right in combinations(values, 2)]
    return float(np.mean(distances))


def _no_communication_database(campaign_base: Path, task: str, seed: int) -> Path:
    return database_path(
        campaign_base / NO_COMMUNICATION_CAMPAIGN,
        task,
        seed,
        NO_COMMUNICATION_SUFFIX,
        CANDIDATE_BUDGET,
    )


def collect_run(campaign_base: Path, task: str, seed: int, threshold: float) -> dict:
    """Collect one matched-seed row without modifying campaign artifacts."""

    fixed_records = fixed_template_records(task)
    fixed_ids = frozenset(
        value for value in map(reaction_id, fixed_records) if value is not None
    )
    portfolios = elite_portfolios(task, seed)
    _artifact, communication_database = hybrid_paths(task, seed, campaign_base)
    isolated_database = _no_communication_database(campaign_base, task, seed)
    record_maps = {
        "full_duplex": topology_records_by_hash(
            communication_database,
            (candidate["topology"] for candidate in portfolios["full_duplex"]),
        ),
        "independent_pool": topology_records_by_hash(
            isolated_database,
            (candidate["topology"] for candidate in portfolios["independent_pool"]),
        ),
    }
    structures = {
        method: qualified_topology_structures(
            portfolios[method],
            threshold=threshold,
            topology_records=record_maps[method],
            fixed_records=fixed_records,
        )
        for method in METHODS
    }
    distances = {
        method: mean_pairwise_distance(structures[method].values()) for method in METHODS
    }

    collision_counts = {}
    for method in METHODS:
        collision_counts[method] = sum(
            any(
                reaction_id(record) in fixed_ids
                for record in qualified_topology_structures(
                    [candidate],
                    threshold=float("inf"),
                    topology_records=record_maps[method],
                    fixed_records=fixed_records,
                )[str(candidate["topology"])]
            )
            for candidate in portfolios[method]
            if float(candidate["loss"]) < threshold
        )

    communication_hof = read_hof(communication_database)
    isolated_hof = read_hof(isolated_database)
    communication_llm = read_llm_candidates(communication_database)
    isolated_llm = read_llm_candidates(isolated_database)

    communication_llm_raw = {
        candidate["identifier"]
        for completed_epoch, candidate in communication_llm
        if completed_epoch <= EPOCH
    }
    communication_llm_canonical = {
        canonical_parameterization_identifier(candidate["identifier"])
        for completed_epoch, candidate in communication_llm
        if completed_epoch <= EPOCH
    }
    communication_records = _through(communication_hof, EPOCH)
    source_order_mismatches = sum(
        (candidate["identifier"] in communication_llm_raw)
        != (
            canonical_parameterization_identifier(candidate["identifier"])
            in communication_llm_canonical
        )
        for candidate in communication_records
    )
    if source_order_mismatches:
        raise RuntimeError(
            f"{task}/seed{seed}: {source_order_mismatches} HOF records change source "
            "classification after reaction-order canonicalization"
        )

    communication_rl = [
        candidate
        for candidate in communication_records
        if candidate["identifier"] not in communication_llm_raw
    ]
    isolated_rl = _through(isolated_hof, EPOCH)
    if not communication_rl or not isolated_rl:
        raise RuntimeError(f"{task}/seed{seed}: missing source-filtered HOF records")

    isolated_llm_raw = {
        candidate["identifier"]
        for completed_epoch, candidate in isolated_llm
        if completed_epoch <= EPOCH
    }
    isolated_llm_canonical = {
        canonical_parameterization_identifier(candidate["identifier"])
        for completed_epoch, candidate in isolated_llm
        if completed_epoch <= EPOCH
    }
    isolated_preterminal = [
        candidate
        for snapshot_epoch, snapshot in isolated_hof.items()
        if snapshot_epoch <= EPOCH
        for candidate in snapshot
    ]
    raw_leakage = sum(
        candidate["identifier"] in isolated_llm_raw for candidate in isolated_preterminal
    )
    canonical_leakage = sum(
        canonical_parameterization_identifier(candidate["identifier"])
        in isolated_llm_canonical
        for candidate in isolated_preterminal
    )
    if raw_leakage or canonical_leakage:
        raise RuntimeError(
            f"{task}/seed{seed}: no-communication HOF contains direct LLM leakage "
            f"through epoch {EPOCH} (raw={raw_leakage}, canonical={canonical_leakage})"
        )

    communication_rl_qualified = {
        candidate["topology"]
        for candidate in communication_rl
        if float(candidate["loss"]) < threshold
    }
    isolated_rl_qualified = {
        candidate["topology"]
        for candidate in isolated_rl
        if float(candidate["loss"]) < threshold
    }

    return {
        "task": task,
        "seed": int(seed),
        "epoch": EPOCH,
        "loss_threshold": float(threshold),
        "fixed_reaction_ids": json.dumps(sorted(fixed_ids), separators=(",", ":")),
        "fixed_template_records": json.dumps(
            sorted(fixed_records), separators=(",", ":")
        ),
        "full_duplex_qualified_topologies": len(structures["full_duplex"]),
        "independent_pool_qualified_topologies": len(structures["independent_pool"]),
        "full_duplex_mean_jaccard": distances["full_duplex"],
        "independent_pool_mean_jaccard": distances["independent_pool"],
        "full_duplex_qualified_fixed_id_collisions": collision_counts["full_duplex"],
        "independent_pool_qualified_fixed_id_collisions": collision_counts[
            "independent_pool"
        ],
        "jaccard_pair_eligible": int(all(value is not None for value in distances.values())),
        "full_duplex_rl_emitted_best_loss": min(
            float(candidate["loss"]) for candidate in communication_rl
        ),
        "isolated_rl_best_loss": min(float(candidate["loss"]) for candidate in isolated_rl),
        "full_duplex_rl_emitted_qualified_topologies": len(communication_rl_qualified),
        "isolated_rl_qualified_topologies": len(isolated_rl_qualified),
        "full_duplex_hof_records_through_epoch": len(communication_records),
        "full_duplex_direct_llm_records_removed": len(communication_records)
        - len(communication_rl),
        "isolated_hof_records_through_epoch": len(isolated_rl),
        "source_order_classification_mismatches": source_order_mismatches,
        "no_communication_raw_llm_leakage": raw_leakage,
        "no_communication_canonical_llm_leakage": canonical_leakage,
    }


def _paired_counts(
    left: Sequence[float],
    right: Sequence[float],
    *,
    left_better: Callable[[float, float], bool],
) -> dict[str, int]:
    wins = losses = ties = 0
    for left_value, right_value in zip(left, right):
        if np.isclose(left_value, right_value, rtol=1e-12, atol=1e-15):
            ties += 1
        elif left_better(left_value, right_value):
            wins += 1
        else:
            losses += 1
    return {"full_duplex_wins": wins, "ties": ties, "independent_wins": losses}


def _wilcoxon(left: Sequence[float], right: Sequence[float]) -> float:
    if not left:
        return float("nan")
    if np.allclose(left, right, rtol=1e-12, atol=1e-15):
        return 1.0
    return float(wilcoxon(left, right, alternative="two-sided").pvalue)


def summarize(rows: list[dict]) -> dict:
    summary = {
        "definitions": {
            "topology": (
                "SHA-256 of order-sorted reaction identity, implementation type, labelled "
                "reactants/products, and input channels; kinetic parameters excluded"
            ),
            "quality_threshold": "median endpoint loss of the 20 matched RL-only controls",
            "structural_distance": (
                "within-run mean pairwise Jaccard distance over complete labelled "
                "structural reaction records among qualified unique topologies, after "
                "removing the exact fixed-template records"
            ),
            "source_filter": (
                "remove exact direct LLM (topology_hash, parameters_json) records from all "
                "HOF snapshots through epoch 100; compare remaining emitted records with "
                "the isolated RL HOF history"
            ),
            "statistical_unit": "matched seed",
            "status": "exploratory post-hoc mechanism and structural diagnostics",
        },
        "tasks": {},
    }
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        full_best = [row["full_duplex_rl_emitted_best_loss"] for row in task_rows]
        isolated_best = [row["isolated_rl_best_loss"] for row in task_rows]
        full_yield = [
            row["full_duplex_rl_emitted_qualified_topologies"] for row in task_rows
        ]
        isolated_yield = [row["isolated_rl_qualified_topologies"] for row in task_rows]

        full_distance = [
            row["full_duplex_mean_jaccard"]
            for row in task_rows
            if row["full_duplex_mean_jaccard"] is not None
        ]
        independent_distance = [
            row["independent_pool_mean_jaccard"]
            for row in task_rows
            if row["independent_pool_mean_jaccard"] is not None
        ]
        paired_distance_rows = [row for row in task_rows if row["jaccard_pair_eligible"]]
        paired_full_distance = [
            row["full_duplex_mean_jaccard"] for row in paired_distance_rows
        ]
        paired_independent_distance = [
            row["independent_pool_mean_jaccard"] for row in paired_distance_rows
        ]

        summary["tasks"][task] = {
            "n_seeds": len(task_rows),
            "loss_threshold": float(task_rows[0]["loss_threshold"]),
            "fixed_reaction_ids": json.loads(task_rows[0]["fixed_reaction_ids"]),
            "fixed_template_records": json.loads(
                task_rows[0]["fixed_template_records"]
            ),
            "structural_distance": {
                "full_duplex_median_within_run_mean": float(np.median(full_distance)),
                "independent_pool_median_within_run_mean": float(
                    np.median(independent_distance)
                ),
                "full_duplex_eligible_runs": len(full_distance),
                "independent_pool_eligible_runs": len(independent_distance),
                "paired_eligible_runs": len(paired_distance_rows),
                **_paired_counts(
                    paired_full_distance,
                    paired_independent_distance,
                    left_better=lambda left, right: left > right,
                ),
                "wilcoxon_pvalue_exploratory": _wilcoxon(
                    paired_full_distance, paired_independent_distance
                ),
            },
            "source_filtered_best_loss": {
                "full_duplex_rl_emitted_median": float(np.median(full_best)),
                "isolated_rl_median": float(np.median(isolated_best)),
                **_paired_counts(
                    full_best, isolated_best, left_better=lambda left, right: left < right
                ),
                "wilcoxon_pvalue_exploratory": _wilcoxon(full_best, isolated_best),
            },
            "source_filtered_qualified_topologies": {
                "full_duplex_rl_emitted_median": float(np.median(full_yield)),
                "isolated_rl_median": float(np.median(isolated_yield)),
                **_paired_counts(
                    full_yield, isolated_yield, left_better=lambda left, right: left > right
                ),
                "wilcoxon_pvalue_exploratory": _wilcoxon(full_yield, isolated_yield),
            },
            "checks": {
                "source_order_classification_mismatches": sum(
                    row["source_order_classification_mismatches"] for row in task_rows
                ),
                "no_communication_raw_llm_leakage": sum(
                    row["no_communication_raw_llm_leakage"] for row in task_rows
                ),
                "no_communication_canonical_llm_leakage": sum(
                    row["no_communication_canonical_llm_leakage"] for row in task_rows
                ),
                "qualified_candidates_with_fixed_id_collision": {
                    "full_duplex": sum(
                        row["full_duplex_qualified_fixed_id_collisions"]
                        for row in task_rows
                    ),
                    "independent_pool": sum(
                        row["independent_pool_qualified_fixed_id_collisions"]
                        for row in task_rows
                    ),
                },
            },
        }
    return summary


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer() and abs(value) >= 1:
        return str(int(value))
    if abs(value) < 1e-4:
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / 10**exponent
        return rf"${coefficient:.2f}\times10^{{{exponent}}}$"
    if abs(value) < 0.01:
        return f"{value:.5f}"
    return f"{value:.4f}"


def render_table(summary: dict) -> str:
    source_rows = []
    distance_rows = []
    for task in TASKS:
        label = TASK_LABELS[task]
        task_summary = summary["tasks"][task]
        for metric_key, metric_label in (
            ("source_filtered_best_loss", r"Best loss $\downarrow$"),
            (
                "source_filtered_qualified_topologies",
                r"Qualified topologies $\uparrow$",
            ),
        ):
            metric = task_summary[metric_key]
            source_rows.append(
                " & ".join(
                    [
                        label,
                        metric_label,
                        _format_number(metric["full_duplex_rl_emitted_median"]),
                        _format_number(metric["isolated_rl_median"]),
                        (
                            f"{metric['full_duplex_wins']}/{metric['ties']}/"
                            f"{metric['independent_wins']}"
                        ),
                        _format_number(metric["wilcoxon_pvalue_exploratory"]),
                    ]
                )
                + r" \\"
            )
        distance = task_summary["structural_distance"]
        distance_rows.append(
            " & ".join(
                [
                    label,
                    _format_number(distance["full_duplex_median_within_run_mean"]),
                    _format_number(distance["independent_pool_median_within_run_mean"]),
                    (
                        f"{distance['full_duplex_eligible_runs']}/"
                        f"{distance['independent_pool_eligible_runs']}"
                    ),
                    str(distance["paired_eligible_runs"]),
                    (
                        f"{distance['full_duplex_wins']}/{distance['ties']}/"
                        f"{distance['independent_wins']}"
                    ),
                    _format_number(distance["wilcoxon_pvalue_exploratory"]),
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[h]",
            r"\caption{Source-filtered communication diagnostic through epoch 100. Direct LLM returns are removed by exact topology--parameter identity. W/T/L favors the full-duplex RL-emitted stream; tests are paired and exploratory.}",
            r"\label{tab:source-filtered}",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Task & Metric & Full-duplex RL & Isolated RL & W/T/L & $p$ \\ ",
            r"\midrule",
            *source_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[h]",
            r"\caption{Structural-reaction-set separation among qualified unique topologies. Values are medians of within-run mean pairwise Jaccard distance after exact fixed-template removal. Complete records include reaction identity, type, labelled species, and input channels. Eligible gives full/independent run counts; W/T/L favors full duplex.}",
            r"\label{tab:structural-distance}",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Task & Full duplex & Independent & Eligible & Paired $n$ & W/T/L & $p$ \\ ",
            r"\midrule",
            *distance_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "communication_mechanism_per_seed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "communication_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "communication_mechanism_table.tex").write_text(
        render_table(summary), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-base", type=Path, default=CAMPAIGN_BASE)
    parser.add_argument("--output-dir", type=Path, default=PAPER / "generated")
    args = parser.parse_args()

    campaign_base = args.campaign_base.expanduser().resolve()
    rows = []
    for task in TASKS:
        threshold = float(np.median([rl_endpoint(task, seed) for seed in SEEDS]))
        for seed in SEEDS:
            rows.append(collect_run(campaign_base, task, seed, threshold))
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir.expanduser().resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
