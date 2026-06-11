#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SINGLE_RUN = ROOT / "comparisons/rne_oscillator/scripts/run_rl4crn_rne_oscillator.py"
OUTPUT_ROOT = ROOT / "comparisons/rne_oscillator/data/raw"
DEFAULT_METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20"
SUMMARY_FIELDS = [
    "seed",
    "run_id",
    "success",
    "best_loss",
    "stop_reason",
    "elapsed_seconds",
    "success_count_so_far",
    "finished_count_so_far",
]


def _result_path(output_root: Path, method: str, run_id: str) -> Path:
    return output_root / method / run_id / "result.json"


def _progress_success(output_root: Path, method: str, run_id: str, threshold: float) -> bool:
    progress_path = output_root / method / run_id / "progress.csv"
    if not progress_path.exists():
        return False
    with progress_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    try:
        return float(rows[-1]["saved_best_loss"]) < float(threshold)
    except (KeyError, ValueError):
        return False


def _load_result(output_root: Path, method: str, run_id: str, threshold: float) -> dict | None:
    path = _result_path(output_root, method, run_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if _progress_success(output_root, method, run_id, threshold):
        return {
            "run_id": run_id,
            "success": True,
            "best_loss": "",
            "stop_reason": "success",
            "elapsed_seconds": "",
        }
    return None


def _append_summary(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _cmd(args: argparse.Namespace, seed: int, run_id: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(SINGLE_RUN),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
        "--output-root",
        str(Path(args.output_root)),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--n-cpus",
        str(args.n_cpus_per_run),
        "--success-threshold",
        str(args.success_threshold),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--method",
        str(args.method),
        "--policy-depth",
        str(args.policy_depth),
        "--deep-layer-size",
        str(args.deep_layer_size),
        "--risk",
        str(args.risk),
        "--max-risk",
        str(args.max_risk),
        "--risk-schedule",
        str(args.risk_schedule),
        "--risk-update",
        str(args.risk_update),
        "--entropy-weight",
        str(args.entropy_weight),
        "--entropy-schedule",
        str(args.entropy_schedule),
        "--entropy-update-coefficient",
        str(args.entropy_update_coefficient),
        "--minimum-entropy-weight",
        str(args.minimum_entropy_weight),
        "--structure-entropy-weight",
        str(args.structure_entropy_weight),
        "--continuous-entropy-weight",
        str(args.continuous_entropy_weight),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list. Overrides --seed-start/--n-runs.",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Optional text file with one seed per line, or comma-separated seeds. Overrides --seed-start/--n-runs.",
    )
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--n-cpus-per-run", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--success-threshold", type=float, default=20.0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--policy-depth", type=int, default=5)
    parser.add_argument("--deep-layer-size", type=int, default=256)
    parser.add_argument("--risk", type=float, default=0.85)
    parser.add_argument("--max-risk", type=float, default=0.9)
    parser.add_argument("--risk-schedule", type=int, default=20)
    parser.add_argument("--risk-update", type=float, default=0.005)
    parser.add_argument("--entropy-weight", type=float, default=5e-3)
    parser.add_argument("--entropy-schedule", type=int, default=1000)
    parser.add_argument("--entropy-update-coefficient", type=float, default=1.0)
    parser.add_argument("--minimum-entropy-weight", type=float, default=0.0)
    parser.add_argument("--structure-entropy-weight", type=float, default=4.0)
    parser.add_argument("--continuous-entropy-weight", type=float, default=1.0)
    parser.add_argument("--summary-csv", default=str(ROOT / "comparisons/rne_oscillator/data/rne_oscillator_100_runs_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20.csv"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    summary_csv = Path(args.summary_csv)
    if args.seeds_file:
        seed_text = Path(args.seeds_file).read_text(encoding="utf-8")
        seeds = [int(tok) for tok in seed_text.replace(",", "\n").split() if tok.strip()]
    elif args.seeds:
        seeds = [int(tok) for tok in str(args.seeds).replace(",", "\n").split() if tok.strip()]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_runs))
    pending = [(seed, f"seed{seed:03d}") for seed in seeds]
    running: list[tuple[int, str, subprocess.Popen]] = []
    finished_seen: set[str] = set()
    success_count = 0
    finished_count = 0

    while pending or running:
        while pending and len(running) < int(args.max_parallel):
            seed, run_id = pending.pop(0)
            existing = _load_result(output_root, args.method, run_id, args.success_threshold)
            if existing is not None:
                success_count += int(bool(existing.get("success")))
                finished_count += 1
                finished_seen.add(run_id)
                _append_summary(
                    summary_csv,
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "success": bool(existing.get("success")),
                        "best_loss": existing.get("best_loss", ""),
                        "stop_reason": existing.get("stop_reason", "existing"),
                        "elapsed_seconds": existing.get("elapsed_seconds", ""),
                        "success_count_so_far": success_count,
                        "finished_count_so_far": finished_count,
                    },
                )
                print(f"[skip] {run_id} already complete/successful", flush=True)
                continue

            cmd = _cmd(args, seed, run_id)
            print(f"[start] {run_id}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT)
            running.append((seed, run_id, proc))

        still_running: list[tuple[int, str, subprocess.Popen]] = []
        for seed, run_id, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((seed, run_id, proc))
                continue
            if code != 0:
                raise subprocess.CalledProcessError(code, proc.args)

            result = _load_result(output_root, args.method, run_id, args.success_threshold) or {}
            success_count += int(bool(result.get("success")))
            finished_count += 1
            finished_seen.add(run_id)
            _append_summary(
                summary_csv,
                {
                    "seed": seed,
                    "run_id": run_id,
                    "success": bool(result.get("success")),
                    "best_loss": result.get("best_loss", ""),
                    "stop_reason": result.get("stop_reason", ""),
                    "elapsed_seconds": result.get("elapsed_seconds", ""),
                    "success_count_so_far": success_count,
                    "finished_count_so_far": finished_count,
                },
            )
            print(
                f"[done] {run_id} success={bool(result.get('success'))} "
                f"success_count={success_count}/{finished_count}",
                flush=True,
            )
        running = still_running
        if running:
            time.sleep(5)

    print(f"[complete] successes={success_count}/{len(finished_seen)} summary={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
