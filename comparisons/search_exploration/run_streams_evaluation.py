"""Complete a detached asynchronous trial, then refresh its saved-results comparison."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from RL4CRN.search_exploration.engine import atomic_json


def read_status(path):
    file = path / "status.json"
    return json.loads(file.read_text()) if file.exists() else {"status": "initializing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "comparisons.search_exploration.run_streams", "--task", "rpa",
        "--task-config", str(args.reference / "../task.json"),
        "--exploration-config", str(args.reference / "../exploration.json"),
        "--output", str(output / "run"), "--embedding-size", "1024",
        "--search-workers", "32", "--optimization-workers", "16", "--queue-capacity", "512",
        "--optimize-per-epoch", "256", "--comet-project", "genai-net-exploration"]
    atomic_json(output / "plan.json", {"command": command, "reference": str(args.reference),
        "interpretation": "One seed; distinct algorithms; shared cap, not identical consumption",
        "created": time.time(), "failure_policy": "No retries or automatic experiment replacement"})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    env.pop("COMET_API_KEY", None)
    process = None
    try:
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            while process.poll() is None:
                atomic_json(output / "progress.json", {"updated": time.time(), "pid": process.pid,
                    "asynchronous": read_status(output / "run"), "reference": read_status(args.reference)})
                time.sleep(30)
        status = read_status(output / "run")
        if process.returncode or status.get("status") != "complete":
            raise RuntimeError(f"Asynchronous run failed: exit={process.returncode}, status={status}")
        analyze = [sys.executable, "-m", "comparisons.search_exploration.analyze_streams",
            str(output / "run"), str(args.reference), "--output", str(output / "analysis")]
        subprocess.run(analyze, cwd=root, env=env, check=True)
        # Do not stop or restart the independently owned synchronous reference.
        while read_status(args.reference).get("status") not in ("complete", "failed"):
            atomic_json(output / "progress.json", {"updated": time.time(), "asynchronous": status,
                "reference": read_status(args.reference), "phase": "waiting_for_reference"})
            time.sleep(60)
        subprocess.run(analyze, cwd=root, env=env, check=True)
        atomic_json(output / "completed.json", {"updated": time.time(), "asynchronous": status["status"],
            "reference": read_status(args.reference)["status"], "analysis": str(output / "analysis/REPORT.md")})
        atomic_json(output / "progress.json", {"updated": time.time(), "phase": "finished",
            "asynchronous": status, "reference": read_status(args.reference)})
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
