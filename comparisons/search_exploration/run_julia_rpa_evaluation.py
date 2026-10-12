"""One full RPA Julia-stream evaluation; reuse completed controls, never rerun them."""

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from RL4CRN.search_exploration.engine import atomic_json
from comparisons.search_exploration.run_streams_evaluation import read_status


BASE = Path("/local1/rossin/search-exploration")
REFERENCES = {
    "Evolution-fit hybrid": BASE / "rpa-streams-v2-100epoch-20260914/run",
    "Standalone GA": BASE / "rpa-remaining-controls-20260915/standalone-ga/run",
}


def prepare(reference, completed):
    if reference["schema"] != "search-streams-v2" or reference["task"] != "rpa":
        raise ValueError("Expected corrected RPA stream reference")
    if completed["status"] != "complete":
        raise ValueError("Reference must already be complete")
    config = dict(reference["config"])
    if (config["seed"], config["epochs"], config["batch_size"]) != (0, 100, 1023):
        raise ValueError("Unexpected reference seed/horizon/batch")
    config.update(evaluation_budget=completed["counters"]["expensive_evaluations"], fit_evaluations=24)
    return config


def comparison(root):
    paths = {"Julia-fit hybrid": root / "run", **REFERENCES}
    rows = []
    for name, path in paths.items():
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "completed.json").read_text())
        with sqlite3.connect(f"file:{path / 'exploration.sqlite'}?mode=ro", uri=True) as db:
            count = db.execute("SELECT count(DISTINCT json_extract(payload,'$.topology_hash')) "
                "FROM designs WHERE json_extract(payload,'$.qualified')=1").fetchone()[0]
        metrics = status["metrics"]
        rows.append({"method": name, "path": str(path), "seed": manifest["config"]["seed"],
            "threshold": manifest["config"]["quality_threshold"], "loss": metrics["Archive/best_loss"],
            "qualified_discovered": count, "qualified_retained": metrics["Archive/qualified_topologies"],
            "evaluations": status["counters"]["expensive_evaluations"],
            "joint_evaluations": status["counters"].get("julia/joint_evaluations", 0),
            "minutes": status["elapsed_seconds"] / 60})
    if len({(r["seed"], r["threshold"]) for r in rows}) != 1:
        raise ValueError("Reference seed or qualification threshold mismatch")
    atomic_json(root / "comparison.json", rows)
    lines = ["# RPA Julia Fitting: Full Evaluation", "",
        "Single-seed engineering comparison. Completed controls are read-only; no controls were rerun.", "",
        "| Method | Best loss | Qualified discovered | Qualified retained | Canonical + joint attempts | Joint attempts | Minutes |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['loss']:.8g} | {r['qualified_discovered']} | "
            f"{r['qualified_retained']} | {r['evaluations']} | {r['joint_evaluations']} | {r['minutes']:.1f} |")
    lines += ["", "The two hybrids use 100 search epochs, batch 1023 and seed 0. GA stops at its "
        "214,793-evaluation budget, not at 100 generations. The Julia arm uses the same upper attempt "
        "budget but is not forced to spend unused credit. A joint loss/gradient evaluation is more work "
        "than a state-only canonical evaluation: counts are accounting units, not equal compute. "
        "Worker allocation, optimizer/admission policy and asynchronous completion schedules differ. "
        "The GA live archive is capped at 30 versus 2,000 in the hybrids; use discovered counts for "
        "coverage comparisons, not retained counts alone. Runtime is descriptive, not a controlled "
        "hardware-throughput comparison. No significance claim is supported by one seed.", "",
        "Pruning is separate postprocessing and is excluded from all search endpoints above.", "",
        "[Julia progress plots and diagnostics](run/REPORT.md)"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def recover_reports(root):
    from comparisons.search_exploration.report_julia_stream import render
    root = Path(root).resolve()
    status = read_status(root / "run")
    if status.get("status") != "complete" or not (root / "run/completed.json").exists():
        raise ValueError("Report-only recovery requires a completed experiment")
    previous = json.loads((root / "progress.json").read_text())
    recovery = {"started": time.time(), "previous_progress": previous,
        "operation": "saved-artifact report recovery; no search/evaluator calls"}
    atomic_json(root / "report_recovery.json", recovery)
    render(root / "run")
    comparison(root)
    atomic_json(root / "report_recovery.json", {**recovery, "completed": time.time()})
    atomic_json(root / "progress.json", {"phase": "complete", "updated": time.time(),
        "run": status, "report": str(root / "REPORT.md"), "report_recovery": str(root / "report_recovery.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--report-only", action="store_true", help="Repair reports from completed data; no simulations")
    parser.add_argument("--comet-project")
    args = parser.parse_args()
    if args.report_only:
        recover_reports(args.output)
        return
    if args.project is None:
        parser.error("--project is required to launch a search")
    reference_path = REFERENCES["Evolution-fit hybrid"]
    manifest = json.loads((reference_path / "manifest.json").read_text())
    config = prepare(manifest, json.loads((reference_path / "completed.json").read_text()))
    for path in REFERENCES.values():
        if json.loads((path / "completed.json").read_text())["status"] != "complete":
            raise ValueError("Do not replace or rerun unfinished references")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "exploration.json", config)
    atomic_json(root / "task.json", manifest["task_config"])
    command = [sys.executable, "-m", "comparisons.search_exploration.run_streams",
        "--task", "rpa", "--task-config", str(root / "task.json"),
        "--exploration-config", str(root / "exploration.json"), "--output", str(root / "run"),
        "--embedding-size", str(manifest["policy"]["deep_layer_size"]),
        "--search-workers", "32", "--optimization-workers", "8", "--queue-capacity", "32",
        "--optimize-per-epoch", "32", "--julia-project", str(args.project.resolve()), "--post-prune"]
    if args.comet_project:
        command += ["--comet-project", args.comet_project]
    atomic_json(root / "plan.json", {"command": command, "references": {k: str(v) for k, v in REFERENCES.items()},
        "config_changes_from_hybrid": {k: [manifest["config"][k], v] for k, v in config.items() if manifest["config"][k] != v},
        "new_llm_calls": 0, "comet_project": args.comet_project, "created": time.time(),
        "comparison": "same seed and hybrid search horizon; unequal actual work/cost explicitly reported"})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    process = None
    try:
        with (root / "run.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                atomic_json(root / "progress.json", {"phase": "running", "updated": time.time(),
                    "pid": process.pid, "run": read_status(root / "run")})
                time.sleep(15)
        if process.returncode != 0:
            raise RuntimeError(f"Search exited {process.returncode}; see run.log; no automatic retry")
        atomic_json(root / "progress.json", {"phase": "reporting", "run": read_status(root / "run")})
        subprocess.run([sys.executable, "-m", "comparisons.search_exploration.report_julia_stream",
            str(root / "run")], env=env, check=True)
        comparison(root)
        atomic_json(root / "progress.json", {"phase": "complete", "updated": time.time(),
            "run": read_status(root / "run"), "report": str(root / "REPORT.md")})
    except BaseException as error:
        atomic_json(root / "controller_failed.json", {"type": type(error).__name__, "message": str(error)})
        atomic_json(root / "progress.json", {"phase": "failed", "updated": time.time(), "error": str(error),
            "run": read_status(root / "run")})
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


if __name__ == "__main__":
    main()
