"""Launch reciprocal source control and standalone GA, then compare four arms."""

import argparse
from contextlib import ExitStack
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
    parser.add_argument("--no-rl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, no_rl = args.reference.resolve(), args.no_rl.resolve()
    manifest = json.loads((reference / "manifest.json").read_text())
    control = json.loads((no_rl / "manifest.json").read_text())
    if any(read_status(p).get("status") != "complete" for p in (reference, no_rl)):
        raise ValueError("Existing hybrid and no-RL runs must be completed; do not rerun them")
    if manifest["schema"] != "search-streams-v2" or manifest["task_config"] != control["task_config"]:
        raise ValueError("Expected corrected-v2 reference and identical task definitions")
    expected = dict(manifest["config"], rl_fraction=0.)
    if expected != control["config"]:
        raise ValueError("No-RL control has unexpected configuration changes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(manifest["config"], mutation_fraction=0.)
    atomic_json(output / "no_mutation_config.json", config)
    atomic_json(output / "task.json", manifest["task_config"])
    paths = {"no-mutation": output / "no-mutation/run", "standalone-ga": output / "standalone-ga/run"}
    commands = {
        "no-mutation": [sys.executable, "-m", "comparisons.search_exploration.run_streams",
            "--task", manifest["task"], "--task-config", str(output / "task.json"),
            "--exploration-config", str(output / "no_mutation_config.json"), "--output", str(paths["no-mutation"]),
            "--embedding-size", str(manifest["policy"]["deep_layer_size"]), "--comet-project", "genai-net-exploration"],
        "standalone-ga": [sys.executable, "-m", "comparisons.search_exploration.run_standalone_ga",
            "--reference", str(reference), "--output", str(paths["standalone-ga"]), "--workers", "16"]}
    for field in ("search_workers", "optimization_workers", "queue_capacity", "optimize_per_epoch"):
        commands["no-mutation"] += ["--" + field.replace("_", "-"), str(manifest[field])]
    atomic_json(output / "plan.json", {"commands": commands, "reference": str(reference), "no_rl": str(no_rl),
        "config_difference": {"mutation_fraction": [manifest["config"]["mutation_fraction"], 0.]},
        "concurrent_evaluator_workers": manifest["search_workers"] + manifest["optimization_workers"] + 16,
        "GA_budget": json.loads((reference / "completed.json").read_text())["counters"]["expensive_evaluations"],
        "seed": config["seed"], "created": time.time(), "new_model_calls": 0,
        "failure_policy": "No retries; preserve failures; independent successful jobs continue"})
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    processes = {}
    try:
        with ExitStack() as stack:
            for name, command in commands.items():
                log = stack.enter_context((output / (name + ".log")).open("w"))
                processes[name] = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            while any(p.poll() is None for p in processes.values()):
                atomic_json(output / "progress.json", {"updated": time.time(), "phase": "running",
                    "jobs": {name: {"pid": p.pid, "exit_code": p.poll(), **read_status(paths[name])}
                        for name, p in processes.items()}})
                time.sleep(30)
        failures = {name: read_status(paths[name]) for name, p in processes.items()
            if p.returncode or read_status(paths[name]).get("status") != "complete"}
        if failures:
            atomic_json(output / "failed_jobs.json", failures)
            raise RuntimeError("Some controls failed; retained their outputs and did not rerun")
        subprocess.run([sys.executable, "-m", "comparisons.search_exploration.analyze_streams",
            str(reference), str(no_rl), str(paths["no-mutation"]), str(paths["standalone-ga"]),
            "--output", str(output / "analysis"), "--discovered"], cwd=root, env=env, check=True)
        result = {"phase": "complete", "updated": time.time(),
            "jobs": {name: read_status(path) for name, path in paths.items()},
            "analysis": str(output / "analysis/REPORT.md")}
        atomic_json(output / "progress.json", result)
        atomic_json(output / "completed.json", result)
    except BaseException as error:
        atomic_json(output / "controller_failed.json", {"type": type(error).__name__, "message": str(error)})
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        raise


if __name__ == "__main__":
    main()
