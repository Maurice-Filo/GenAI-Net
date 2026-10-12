"""Run an isolated no-RL control of a completed independent-streams experiment."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from RL4CRN.search_exploration.engine import atomic_json
from comparisons.search_exploration.run_streams_evaluation import read_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation", choices=("no-rl", "no-mutation"), default="no-rl")
    args = parser.parse_args()
    reference = args.reference.resolve()
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest["schema"] != "search-streams-v2" or read_status(reference)["status"] != "complete":
        raise ValueError("Requires the completed corrected-v2 reference")
    config = dict(manifest["config"])
    field = "rl_fraction" if args.ablation == "no-rl" else "mutation_fraction"
    previous_share = config[field]
    if previous_share <= 0:
        raise ValueError("Reference must use the ablated source")
    config[field] = 0.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "task.json", manifest["task_config"])
    atomic_json(output / "exploration.json", config)
    command = [sys.executable, "-m", "comparisons.search_exploration.run_streams",
        "--task", manifest["task"], "--task-config", str(output / "task.json"),
        "--exploration-config", str(output / "exploration.json"), "--output", str(output / "run"),
        "--embedding-size", str(manifest["policy"]["deep_layer_size"]),
        "--comet-project", "genai-net-exploration"]
    for field in ("search_workers", "optimization_workers", "queue_capacity", "optimize_per_epoch"):
        command += ["--" + field.replace("_", "-"), str(manifest[field])]
    atomic_json(output / "plan.json", {"reference": str(reference), "command": command,
        "intervention": ("Replace RL slots with random proposals; skip all policy/imitation updates"
            if args.ablation == "no-rl" else "Replace archive-mutation slots with random; retain RL and rate fitting"),
        "config_difference": {field: [previous_share, 0.]},
        "budgets": "Same proposal schedule and evaluation cap; report actual consumption separately",
        "failure_policy": "No retries or automatic replacement", "created": time.time()})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    root = Path(__file__).resolve().parents[2]
    process = None
    try:
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                atomic_json(output / "progress.json", {"updated": time.time(), "pid": process.pid,
                    "control": read_status(output / "run"), "reference": str(reference)})
                time.sleep(30)
        if process.returncode or read_status(output / "run")["status"] != "complete":
            raise RuntimeError(f"Control failed: exit={process.returncode}")
        for module, tail in [
            ("analyze_streams", [str(reference), str(output / "run"), "--output", str(output / "analysis")]),
            ("analyze_source_interaction", [str(output / "run"), "--output", str(output / "analysis")])]:
            subprocess.run([sys.executable, "-m", "comparisons.search_exploration." + module, *tail],
                cwd=root, env=env, check=True)
        atomic_json(output / "completed.json", {"status": "complete", "updated": time.time(),
            "analysis": str(output / "analysis/REPORT.md")})
        atomic_json(output / "progress.json", {"phase": "finished", "control": read_status(output / "run")})
    except BaseException as error:
        atomic_json(output / "controller_failed.json", {"type": type(error).__name__, "message": str(error)})
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


if __name__ == "__main__":
    main()
