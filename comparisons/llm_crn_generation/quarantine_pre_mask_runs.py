#!/usr/bin/env python3
"""Reversibly quarantine LLM runs created before template-ID masking."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN_ROOT = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/paper-campaigns"
)
DEFAULT_RAW_ROOT = ROOT / "comparisons/rpa_search/data/raw"
DEFAULT_QUARANTINE_ROOT = Path(
    "/local0/home/rossin/ai-workspaces/deepseek-test/crn-runs/"
    "quarantine/2026-09-01-pre-template-mask"
)
DEFAULT_INVENTORY = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/pre_mask_quarantine_inventory.json"
)

TASKS = (
    "oscillator_frequency",
    "dose_ultrasensitive",
    "oscillator_mean",
    "dose_biphasic",
    "stochastic_rpa",
    "classifier",
    "dose_hill",
    "logic",
    "rpa",
)
NO_TEMPLATE_TASKS = frozenset({"classifier", "oscillator_mean"})
REASON = (
    "Pre-fix LLM output contract exposed fixed-template reaction IDs. "
    "The run does not implement the finalized proposal space."
)


@dataclass
class Move:
    source: str
    destination: str
    kind: str
    reason: str
    task: str | None = None
    files: int = 0
    bytes: int = 0
    applied: bool = False


def task_from_name(name: str) -> str | None:
    for task in TASKS:
        if name.startswith(f"{task}_") or f"-{task}-seed" in name:
            return task
    return None


def tree_stats(path: Path) -> tuple[int, int]:
    if path.is_file() or path.is_symlink():
        return 1, path.stat().st_size
    files = size = 0
    for root, _directories, names in os.walk(path):
        for name in names:
            candidate = Path(root) / name
            try:
                size += candidate.stat().st_size
                files += 1
            except FileNotFoundError:
                continue
    return files, size


def _move(
    source: Path,
    destination: Path,
    *,
    kind: str,
    reason: str = REASON,
    task: str | None = None,
) -> Move:
    files, size = tree_stats(source)
    return Move(
        source=str(source),
        destination=str(destination),
        kind=kind,
        reason=reason,
        task=task,
        files=files,
        bytes=size,
    )


def _campaign_trials(campaign: Path) -> list[Path]:
    runs = campaign / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []


def campaign_moves(
    campaign_root: Path, quarantine_root: Path
) -> tuple[list[Move], list[dict], list[Path]]:
    moves: list[Move] = []
    retained: list[dict] = []
    split_campaigns: list[Path] = []
    destination_root = quarantine_root / "workspace-campaigns"

    for campaign in sorted(campaign_root.iterdir()):
        if campaign.name.startswith("rl-only-"):
            retained.append(
                {"path": str(campaign), "reason": "RL-only control; native mask was correct"}
            )
            continue
        if not campaign.is_dir():
            moves.append(
                _move(
                    campaign,
                    destination_root / "_metadata" / campaign.name,
                    kind="workspace-metadata",
                )
            )
            continue

        trials = _campaign_trials(campaign)
        safe_trials = [trial for trial in trials if task_from_name(trial.name) in NO_TEMPLATE_TASKS]
        invalid_trials = [trial for trial in trials if trial not in safe_trials]
        if not safe_trials:
            moves.append(
                _move(
                    campaign,
                    destination_root / campaign.name,
                    kind="workspace-campaign",
                )
            )
            continue

        split_campaigns.append(campaign)
        retained.extend(
            {
                "path": str(trial),
                "task": task_from_name(trial.name),
                "reason": "Task has no fixed template reaction IDs",
            }
            for trial in safe_trials
        )
        for trial in invalid_trials:
            task = task_from_name(trial.name)
            moves.append(
                _move(
                    trial,
                    destination_root / campaign.name / "runs" / trial.name,
                    kind="workspace-run",
                    task=task,
                )
            )
        logs = campaign / "logs"
        if logs.is_dir():
            for log in sorted(logs.iterdir()):
                task = task_from_name(log.name)
                if task not in NO_TEMPLATE_TASKS:
                    moves.append(
                        _move(
                            log,
                            destination_root / campaign.name / "logs" / log.name,
                            kind="workspace-log",
                            task=task,
                        )
                    )
    return moves, retained, split_campaigns


def raw_moves(
    raw_root: Path, quarantine_root: Path
) -> tuple[list[Move], list[dict], list[Path]]:
    moves: list[Move] = []
    retained: list[dict] = []
    split_methods: list[Path] = []
    destination_root = quarantine_root / "raw-results"

    for method in sorted(raw_root.glob("genai_net_llm*")):
        runs = sorted(path for path in method.iterdir() if path.is_dir())
        safe_runs = [run for run in runs if task_from_name(run.name) in NO_TEMPLATE_TASKS]
        invalid_runs = [run for run in runs if run not in safe_runs]
        if not safe_runs:
            moves.append(
                _move(method, destination_root / method.name, kind="raw-method")
            )
            continue
        split_methods.append(method)
        retained.extend(
            {
                "path": str(run),
                "task": task_from_name(run.name),
                "reason": "Task has no fixed template reaction IDs",
            }
            for run in safe_runs
        )
        for run in invalid_runs:
            moves.append(
                _move(
                    run,
                    destination_root / method.name / run.name,
                    kind="raw-run",
                    task=task_from_name(run.name),
                )
            )

    diagnostics = raw_root / "diagnostic_archives"
    if diagnostics.exists():
        moves.append(
            _move(
                diagnostics,
                destination_root / "diagnostic_archives",
                kind="diagnostic-archive",
            )
        )
    locks = raw_root / ".campaign_locks"
    if locks.exists():
        moves.append(
            _move(
                locks,
                quarantine_root / "stale-operational-state/.campaign_locks",
                kind="stale-locks",
                reason="No campaigns are active; retained locks are stale operational state.",
            )
        )
    return moves, retained, split_methods


def derived_moves(root: Path, quarantine_root: Path) -> list[Move]:
    moves: list[Move] = []
    destination_root = quarantine_root / "derived-results"
    paper = root / "paper/iclr2027_genai_net_llm"

    figures = paper / "figures"
    if figures.is_dir():
        for artifact in sorted(figures.iterdir()):
            if artifact.name == "README.md" or artifact.stem == "architecture":
                continue
            moves.append(
                _move(
                    artifact,
                    destination_root / "paper/figures" / artifact.name,
                    kind="derived-paper-figure",
                    reason="Generated from one or more pre-fix LLM campaigns.",
                )
            )

    generated = paper / "generated"
    if generated.is_dir():
        keep = {
            "paper_experiment_audit.json",
            "pre_mask_quarantine_inventory.json",
            "prompts_appendix.tex",
        }
        for artifact in sorted(generated.iterdir()):
            if artifact.name in keep:
                continue
            moves.append(
                _move(
                    artifact,
                    destination_root / "paper/generated" / artifact.name,
                    kind="derived-paper-data",
                    reason="Generated from one or more pre-fix LLM campaigns.",
                )
            )

    build = paper / "build"
    if build.exists():
        moves.append(
            _move(
                build,
                destination_root / "paper/build",
                kind="stale-paper-build",
                reason="Compiled manuscript contains pre-fix LLM results.",
            )
        )

    cutoff = datetime(2026, 8, 20, tzinfo=timezone.utc).timestamp()
    comparison_figures = root / "comparisons/rpa_search/figures"
    if comparison_figures.is_dir():
        for artifact in sorted(comparison_figures.iterdir()):
            if artifact.is_file() and artifact.stat().st_mtime >= cutoff:
                moves.append(
                    _move(
                        artifact,
                        destination_root / "comparison-figures" / artifact.name,
                        kind="derived-comparison-figure",
                        reason="Post-20-August analysis derived from pre-fix LLM campaigns.",
                    )
                )
    for suffix in ("pdf", "png", "svg"):
        artifact = root / f"comparisons_with_GA.{suffix}"
        if artifact.exists():
            moves.append(
                _move(
                    artifact,
                    destination_root / "comparison-figures" / artifact.name,
                    kind="derived-comparison-figure",
                    reason="Generated comparison includes pre-fix LLM campaigns.",
                )
            )

    experiment_root = root / "comparisons/llm_crn_generation"
    stale_reports = (
        "BREADTH_ABLATION_MATRIX.md",
        "FLASH_LONG300_RUN_REPORT.aux",
        "FLASH_LONG300_RUN_REPORT.log",
        "FLASH_LONG300_RUN_REPORT.md",
        "FLASH_LONG300_RUN_REPORT.out",
        "FLASH_LONG300_RUN_REPORT.pdf",
        "FLASH_LONG300_RUN_REPORT.tex",
        "HOF_RECONNECTION_REPORT.md",
        "LOGIC_TRAJECTORY_PROMPT_ABLATION.md",
        "PRELIMINARY_DEFAULT_LONG300_CHECKPOINT.json",
        "PRELIMINARY_DEFAULT_LONG300_CHECKPOINT.md",
        "REASONING_ABLATION_REPORT.md",
        "WEEKEND_RUNS.md",
        "paper_campaign_plan.json",
    )
    for name in stale_reports:
        artifact = experiment_root / name
        if artifact.exists():
            moves.append(
                _move(
                    artifact,
                    destination_root / "reports-and-plans" / artifact.name,
                    kind="stale-report-or-plan",
                    reason="Report or plan refers to one or more pre-fix LLM campaigns.",
                )
            )
    return moves


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _snapshot_and_filter_campaign(
    campaign: Path, quarantine_root: Path, *, apply: bool
) -> None:
    metadata_root = quarantine_root / "workspace-campaigns" / campaign.name / "_original_metadata"
    for name in ("campaign_manifest.json", "status.json"):
        source = campaign / name
        if source.is_file() and apply:
            metadata_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, metadata_root / name)


def _filter_split_campaign(campaign: Path, quarantine_root: Path) -> None:
    remaining_tasks = sorted(
        {
            task
            for trial in _campaign_trials(campaign)
            if (task := task_from_name(trial.name)) is not None
        }
    )
    manifest_path = campaign / "campaign_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest["tasks"] = remaining_tasks
        manifest["quarantine_split"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "quarantine_root": str(quarantine_root),
            "retained_tasks": remaining_tasks,
        }
        _write_json(manifest_path, manifest)
    status_path = campaign / "status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        for field in ("active", "completed", "failed", "pending"):
            status[field] = [
                record for record in status.get(field, []) if record.get("task") in remaining_tasks
            ]
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        status["quarantine_root"] = str(quarantine_root)
        _write_json(status_path, status)
    _write_json(
        campaign / "QUARANTINE_SPLIT.json",
        {
            "quarantine_root": str(quarantine_root),
            "retained_tasks": remaining_tasks,
            "reason": REASON,
        },
    )


def _validate_moves(moves: Iterable[Move]) -> None:
    sources: set[str] = set()
    destinations: set[str] = set()
    for move in moves:
        if move.source in sources:
            raise RuntimeError(f"Duplicate quarantine source: {move.source}")
        if move.destination in destinations:
            raise RuntimeError(f"Duplicate quarantine destination: {move.destination}")
        if not Path(move.source).exists():
            raise RuntimeError(f"Quarantine source disappeared: {move.source}")
        if Path(move.destination).exists():
            raise RuntimeError(f"Quarantine destination already exists: {move.destination}")
        sources.add(move.source)
        destinations.add(move.destination)


def build_inventory(
    *,
    campaign_root: Path,
    raw_root: Path,
    quarantine_root: Path,
    include_derived: bool,
) -> tuple[dict, list[Path], list[Path]]:
    campaign_plan, retained_campaigns, split_campaigns = campaign_moves(
        campaign_root, quarantine_root
    )
    raw_plan, retained_raw, split_methods = raw_moves(raw_root, quarantine_root)
    moves = campaign_plan + raw_plan
    if include_derived:
        moves.extend(derived_moves(ROOT, quarantine_root))
    _validate_moves(moves)
    inventory = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "reason": REASON,
        "classification": {
            "retained_llm_tasks": sorted(NO_TEMPLATE_TASKS),
            "retained_controls": "All RL-only campaigns and raw methods",
            "rule": (
                "Pre-fix LLM tasks with one or more fixed template reaction IDs are "
                "quarantined even when no forbidden ID was selected."
            ),
        },
        "roots": {
            "campaigns": str(campaign_root),
            "raw_results": str(raw_root),
            "quarantine": str(quarantine_root),
        },
        "summary": {
            "moves": len(moves),
            "files": sum(move.files for move in moves),
            "bytes": sum(move.bytes for move in moves),
            "gibibytes": sum(move.bytes for move in moves) / 1073741824,
            "retained_campaign_items": len(retained_campaigns),
            "retained_raw_runs": len(retained_raw),
        },
        "retained": {
            "campaigns": retained_campaigns,
            "raw_runs": retained_raw,
        },
        "moves": [asdict(move) for move in moves],
    }
    return inventory, split_campaigns, split_methods


def apply_inventory(
    inventory: dict,
    *,
    quarantine_root: Path,
    split_campaigns: list[Path],
    split_methods: list[Path],
) -> None:
    moves = [Move(**record) for record in inventory["moves"]]
    _validate_moves(moves)
    quarantine_root.mkdir(parents=True, exist_ok=False)
    progress = quarantine_root / "manifest.in_progress.json"
    inventory["status"] = "in_progress"
    _write_json(progress, inventory)
    for campaign in split_campaigns:
        _snapshot_and_filter_campaign(campaign, quarantine_root, apply=True)
    for index, move in enumerate(moves, start=1):
        source = Path(move.source)
        destination = Path(move.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        move.applied = True
        inventory["moves"][index - 1]["applied"] = True
        if index % 25 == 0:
            _write_json(progress, inventory)
    for campaign in split_campaigns:
        _filter_split_campaign(campaign, quarantine_root)
    for method in split_methods:
        _write_json(
            method / "QUARANTINE_SPLIT.json",
            {"quarantine_root": str(quarantine_root), "reason": REASON},
        )
    inventory["status"] = "complete"
    inventory["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(quarantine_root / "manifest.json", inventory)
    progress.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--quarantine-root", type=Path, default=DEFAULT_QUARANTINE_ROOT)
    parser.add_argument("--inventory-output", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    inventory, split_campaigns, split_methods = build_inventory(
        campaign_root=args.campaign_root.expanduser().resolve(),
        raw_root=args.raw_root.expanduser().resolve(),
        quarantine_root=args.quarantine_root.expanduser().resolve(),
        include_derived=args.include_derived,
    )
    _write_json(args.inventory_output.expanduser().resolve(), inventory)
    if args.apply:
        apply_inventory(
            inventory,
            quarantine_root=args.quarantine_root.expanduser().resolve(),
            split_campaigns=split_campaigns,
            split_methods=split_methods,
        )
        _write_json(args.inventory_output.expanduser().resolve(), inventory)
    print(json.dumps(inventory["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
