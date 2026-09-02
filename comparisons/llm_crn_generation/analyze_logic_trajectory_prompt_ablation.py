#!/usr/bin/env python3
"""Analyze the matched Logic task-prompt conditioning sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from RL4CRN.llm.benchmark_prompts import (
    get_mmc2_task_prompt_variant,
    get_reported_mmc2_task_prompt_2026,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "comparisons/rpa_search/data/raw"
CAMPAIGNS = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
)
DEFAULT_CAMPAIGN = "flash-logic-initial-hof-withheld-100epoch-20seed"
VARIANT_CAMPAIGN = "flash-logic-trajectory-prompt-100epoch-20seed"
DEFAULT_METHOD = "genai_net_llm_flash_logic_initial_context_free100"
DEFAULT_SUFFIX = "cvode_llm_flash_logic_initial_context_free100"
VARIANT_METHOD = "genai_net_llm_flash_logic_trajectory_prompt"
VARIANT_SUFFIX = "cvode_llm_flash_logic_trajectory_prompt"
CONTROL_METHOD = "rl4crn"
CAMPAIGN_ID = VARIANT_CAMPAIGN
PROMPT_VARIANT = "logic-trajectory"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_id(seed: int, suffix: str) -> str:
    return f"logic_full102400_seed{seed}_{suffix}"


def artifact(method: str, suffix: str, seed: int) -> Path:
    return RAW / method / run_id(seed, suffix)


def database(campaign: str, suffix: str, seed: int) -> Path | None:
    identifier = run_id(seed, suffix)
    matches = sorted((CAMPAIGNS / campaign / "runs").glob(f"*/{identifier}/results.sqlite"))
    if len(matches) > 1:
        raise RuntimeError(f"Expected at most one database for {identifier}, found {len(matches)}")
    return matches[0] if matches else None


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def database_diagnostics(path: Path) -> dict:
    with _connect(path) as connection:
        requests = connection.execute(
            """SELECT llm_run_id, launched_epoch, completed_epoch, requested,
                      produced, valid_count
                 FROM llm_runs ORDER BY launched_epoch"""
        ).fetchall()
        candidates = connection.execute(
            """SELECT r.llm_run_id, c.topology_hash, e.parameters_json,
                      c.valid, c.loss
                 FROM llm_candidates c
                 JOIN llm_runs r ON r.llm_run_id = c.llm_run_id
                 LEFT JOIN evaluations e
                   ON e.source = 'llm'
                  AND e.topology_hash = c.topology_hash
                  AND json_extract(e.metadata_json, '$.llm_run_id') = c.llm_run_id
                  AND json_extract(e.metadata_json, '$.candidate_index') = c.candidate_index
                ORDER BY r.launched_epoch, c.candidate_index"""
        ).fetchall()
        snapshots = connection.execute(
            """SELECT h.epoch, e.rank, e.topology_hash, e.parameters_json, e.loss
                 FROM hof_snapshots h
                 JOIN hof_snapshot_entries e ON e.snapshot_id = h.snapshot_id
                WHERE e.loss IS NOT NULL ORDER BY h.epoch, e.rank"""
        ).fetchall()

    by_epoch: dict[int, list[tuple]] = defaultdict(list)
    for row in snapshots:
        by_epoch[int(row[0])].append(row)
    if not by_epoch:
        raise RuntimeError(f"No HOF snapshots in {path}")

    valid_by_request: dict[str, list[tuple]] = defaultdict(list)
    llm_identifiers: set[tuple[str, str]] = set()
    for llm_run_id, topology_hash, parameters_json, valid, loss in candidates:
        if not valid or loss is None or topology_hash is None or parameters_json is None:
            continue
        row = (str(topology_hash), str(parameters_json), float(loss))
        valid_by_request[str(llm_run_id)].append(row)
        llm_identifiers.add((row[0], row[1]))

    beat_hof = 0
    for llm_run_id, launched, _completed, _requested, _produced, _valid_count in requests:
        issue = by_epoch.get(int(launched), [])
        issue_best = min((float(row[4]) for row in issue), default=float("nan"))
        request_best = min(
            (row[2] for row in valid_by_request[str(llm_run_id)]),
            default=float("nan"),
        )
        beat_hof += int(np.isfinite(request_best) and request_best < issue_best)

    final_epoch = max(by_epoch)
    final = min(by_epoch[final_epoch], key=lambda row: float(row[4]))
    return {
        "endpoint_loss": float(final[4]),
        "accepted_payloads": len(requests),
        "valid_candidates": sum(int(row[5]) for row in requests),
        "requests_beating_issue_hof": beat_hof,
        "llm_rank_one": int((str(final[2]), str(final[3])) in llm_identifiers),
        "snapshot_max_epoch": final_epoch,
    }


def collect_hybrid(method: str, suffix: str, campaign: str) -> dict[int, dict]:
    rows = {}
    for seed in range(20):
        completed_path = artifact(method, suffix, seed) / "completed.json"
        db_path = database(campaign, suffix, seed)
        if not completed_path.is_file() or db_path is None:
            continue
        completed = read_json(completed_path)
        diagnostic = database_diagnostics(db_path)
        if not np.isclose(
            float(completed["best_loss"]),
            diagnostic["endpoint_loss"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(f"Completion/database endpoint mismatch for seed {seed}")
        rows[seed] = {
            "seed": seed,
            "loss": float(completed["best_loss"]),
            "database": str(db_path),
            **diagnostic,
        }
    return rows


def collect_control() -> dict[int, dict]:
    rows = {}
    for seed in range(20):
        root = RAW / CONTROL_METHOD / f"logic_full102400_seed{seed}_cvode"
        progress = root / "progress.csv"
        if not progress.is_file():
            continue
        with progress.open(encoding="utf-8") as handle:
            progress_rows = list(csv.DictReader(handle))
        if not progress_rows:
            continue
        rows[seed] = {
            "seed": seed,
            "loss": float(progress_rows[-1]["best_so_far_loss"]),
        }
    return rows


def paired_summary(
    left: dict[int, dict],
    right: dict[int, dict],
    *,
    left_name: str,
    right_name: str,
) -> dict:
    seeds = sorted(set(left) & set(right))
    left_values = np.asarray([left[seed]["loss"] for seed in seeds], dtype=float)
    right_values = np.asarray([right[seed]["loss"] for seed in seeds], dtype=float)
    ties = np.isclose(left_values, right_values, rtol=1e-10, atol=1e-12)
    wins = int(np.sum((left_values < right_values) & ~ties))
    losses = int(np.sum((left_values > right_values) & ~ties))
    if not seeds:
        p_value = None
        ratio = None
    elif np.all(ties):
        p_value = 1.0
        ratio = 1.0
    else:
        p_value = float(
            wilcoxon(left_values, right_values, alternative="two-sided").pvalue
        )
        ratio = float(np.median(right_values / left_values))
    return {
        "left": left_name,
        "right": right_name,
        "matched_seeds": seeds,
        "n": len(seeds),
        "left_wins_ties_losses": [wins, int(np.sum(ties)), losses],
        "median_paired_ratio_right_over_left": ratio,
        "two_sided_wilcoxon_p": p_value,
    }


def condition_summary(rows: dict[int, dict]) -> dict:
    values = [row["loss"] for row in rows.values()]
    result = {
        "completed": len(rows),
        "median_endpoint_loss": float(np.median(values)) if values else None,
    }
    if rows and "accepted_payloads" in next(iter(rows.values())):
        result.update(
            {
                "accepted_payloads": sum(row["accepted_payloads"] for row in rows.values()),
                "valid_candidates": sum(row["valid_candidates"] for row in rows.values()),
                "requests_beating_issue_hof": sum(
                    row["requests_beating_issue_hof"] for row in rows.values()
                ),
                "llm_rank_one_runs": sum(row["llm_rank_one"] for row in rows.values()),
            }
        )
    return result


def campaign_status() -> str:
    path = CAMPAIGNS / VARIANT_CAMPAIGN / "status.json"
    if not path.is_file():
        return "queued (campaign has not launched yet)"
    payload = read_json(path)
    completed = len([row for row in payload.get("completed", []) if row.get("task") == "logic"])
    failed = len([row for row in payload.get("failed", []) if row.get("task") == "logic"])
    active = len([row for row in payload.get("active", []) if row.get("task") == "logic"])
    pending = len([row for row in payload.get("pending", []) if row.get("task") == "logic"])
    if completed == 20 and not (failed or active or pending):
        return "complete (20/20)"
    return f"running: {completed}/20 complete, {active} active, {pending} pending, {failed} failed"


def format_number(value: float | None) -> str:
    return "pending" if value is None else f"{value:.6g}"


def interpretation(prompt_comparison: dict, default: dict[int, dict], variant: dict[int, dict]) -> str:
    if prompt_comparison["n"] < 20:
        return "Pending: the trajectory-faithful cohort is incomplete, so no directional conclusion is reported."
    default_median = float(np.median([row["loss"] for row in default.values()]))
    variant_median = float(np.median([row["loss"] for row in variant.values()]))
    p_value = float(prompt_comparison["two_sided_wilcoxon_p"])
    if p_value < 0.05 and variant_median < default_median:
        return "The trajectory-faithful prompt improves endpoint loss relative to the default prompt."
    if p_value < 0.05 and variant_median > default_median:
        return "The trajectory-faithful prompt worsens endpoint loss relative to the default prompt."
    return "The trajectory-faithful and default prompts are statistically indistinguishable at this sample size."


def write_csv(path: Path, default: dict, variant: dict, control: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "default_prompt_loss", "trajectory_prompt_loss", "rl_only_loss"),
        )
        writer.writeheader()
        for seed in range(20):
            writer.writerow(
                {
                    "seed": seed,
                    "default_prompt_loss": default.get(seed, {}).get("loss", ""),
                    "trajectory_prompt_loss": variant.get(seed, {}).get("loss", ""),
                    "rl_only_loss": control.get(seed, {}).get("loss", ""),
                }
            )


def write_report(path: Path, summary: dict, prompt: str) -> None:
    conditions = summary["conditions"]
    comparisons = summary["comparisons"]
    prompt_pair = comparisons["trajectory_vs_default"]
    lines = [
        "# Logic trajectory-prompt ablation",
        "",
        "## Scope",
        "",
        "A trajectory-faithful prompt tests sensitivity to prompt/evaluator wording. Since both variants are still scored by the same canonical evaluator, this ablation assesses conditioning, not evaluator validity.",
        "",
        f"- Campaign ID: `{CAMPAIGN_ID}`",
        f"- Completion status: **{summary['completion_status']}**",
        "- Frozen paper-default condition: final-state BCE wording, request-0 HOF withheld.",
        "- Diagnostic condition: weighted trajectory-L1 wording, request-0 HOF withheld.",
        "- Both conditions: Logic, seeds 0--19, 100 epochs, batch size 1023, cap 102400, CVODE, DeepSeek V4 Flash, full communication, ten proposals every 20 epochs, 10 parallel seed processes, four CVODE workers per seed, and a 3600-second Harness timeout.",
        "",
        "## Launch",
        "",
        "```bash",
        "comparisons/llm_crn_generation/run_logic_trajectory_prompt_campaign.sh",
        "```",
        "",
        "Queued tmux session: `logic-trajectory-prompt`.",
        "",
        "The launcher passes `--task-prompt-variant logic-trajectory`, `--communication-mode full`, and `--withhold-initial-hof` with method `genai_net_llm_flash_logic_trajectory_prompt` and suffix `cvode_llm_flash_logic_trajectory_prompt`.",
        "",
        "## Artifacts",
        "",
        f"- Default campaign: `{CAMPAIGNS / DEFAULT_CAMPAIGN}`",
        f"- Trajectory campaign: `{CAMPAIGNS / VARIANT_CAMPAIGN}`",
        f"- Default comparison artifacts: `{RAW / DEFAULT_METHOD}`",
        f"- Trajectory comparison artifacts: `{RAW / VARIANT_METHOD}`",
        f"- RL-only control: `{RAW / CONTROL_METHOD}`",
        f"- Per-seed CSV: `{summary['csv_path']}`",
        f"- Generated JSON: `{summary['json_path']}`",
        "",
        "## Endpoint results",
        "",
        "| Condition | Complete | Median endpoint loss | Accepted payloads | Beat-HOF requests | LLM rank one |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("default_prompt", "Default BCE prompt"),
        ("trajectory_prompt", "Trajectory-faithful prompt"),
        ("rl_only", "RL only"),
    ):
        row = conditions[key]
        lines.append(
            f"| {label} | {row['completed']}/20 | {format_number(row['median_endpoint_loss'])} | "
            f"{row.get('accepted_payloads', 'n/a')} | {row.get('requests_beating_issue_hof', 'n/a')} | "
            f"{row.get('llm_rank_one_runs', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Win/tie/loss counts favor the first-named condition when its loss is lower. The paired ratio is `reference loss / first-named loss`, so values above one favor the first-named condition.",
            "",
            "| Comparison | n | Win/tie/loss | Median paired ratio | Two-sided Wilcoxon p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("trajectory_vs_default", "Trajectory vs default"),
        ("default_vs_rl", "Default vs RL only"),
        ("trajectory_vs_rl", "Trajectory vs RL only"),
    ):
        row = comparisons[key]
        wtl = "/".join(str(value) for value in row["left_wins_ties_losses"])
        lines.append(
            f"| {label} | {row['n']} | {wtl} | "
            f"{format_number(row['median_paired_ratio_right_over_left'])} | "
            f"{format_number(row['two_sided_wilcoxon_p'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "This is a prompt-conditioning sensitivity diagnostic, not a replacement for the evaluator-controlled main comparison.",
            "",
            "## Frozen trajectory-faithful prompt",
            "",
            f"SHA-256: `{summary['trajectory_prompt_sha256']}`",
            "",
            "```text",
            prompt.rstrip(),
            "```",
            "",
            f"The frozen paper-default prompt hash remains `{summary['default_prompt_sha256']}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "comparisons/llm_crn_generation/LOGIC_TRAJECTORY_PROMPT_ABLATION.md",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "paper/iclr2027_genai_net_llm/generated/logic_trajectory_prompt_ablation.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "paper/iclr2027_genai_net_llm/generated/logic_trajectory_prompt_ablation.csv",
    )
    args = parser.parse_args()

    default = collect_hybrid(DEFAULT_METHOD, DEFAULT_SUFFIX, DEFAULT_CAMPAIGN)
    variant = collect_hybrid(VARIANT_METHOD, VARIANT_SUFFIX, VARIANT_CAMPAIGN)
    control = collect_control()
    comparisons = {
        "trajectory_vs_default": paired_summary(
            variant, default, left_name="trajectory_prompt", right_name="default_prompt"
        ),
        "default_vs_rl": paired_summary(
            default, control, left_name="default_prompt", right_name="rl_only"
        ),
        "trajectory_vs_rl": paired_summary(
            variant, control, left_name="trajectory_prompt", right_name="rl_only"
        ),
    }
    prompt = get_mmc2_task_prompt_variant(
        "logic", variant=PROMPT_VARIANT, solver="CVODE"
    )
    default_prompt = get_reported_mmc2_task_prompt_2026("logic", solver="CVODE")
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "completion_status": campaign_status(),
        "trajectory_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "default_prompt_sha256": hashlib.sha256(default_prompt.encode("utf-8")).hexdigest(),
        "conditions": {
            "default_prompt": condition_summary(default),
            "trajectory_prompt": condition_summary(variant),
            "rl_only": condition_summary(control),
        },
        "comparisons": comparisons,
        "interpretation": interpretation(comparisons["trajectory_vs_default"], default, variant),
        "report_path": str(args.report.resolve()),
        "json_path": str(args.json_output.resolve()),
        "csv_path": str(args.csv_output.resolve()),
    }
    write_csv(args.csv_output.resolve(), default, variant, control)
    args.json_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.json_output.resolve().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.report.resolve(), summary, prompt)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
