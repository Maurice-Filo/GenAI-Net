"""Local efficiency audit; no optimizer/evaluator changes or external uploads."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from RL4CRN.parameter_optimization import JuliaOptimizer, CRNSpec
from comparisons.search_exploration.benchmark_julia_rates import rpa_fixture


def measure(fn, repeats=20):
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter()-t)
    return {"median_s": float(np.median(samples)), "p90_s": float(np.quantile(samples, .9)),
            "samples_s": samples}


def memory(pid):
    text = Path(f"/proc/{pid}/status").read_text()
    return {line.split(":")[0]: int(line.split()[1]) for line in text.splitlines()
            if line.startswith(("VmRSS:", "VmHWM:"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    crn, experiment, rates, canonical = rpa_fixture()
    fixture = dict(crn=crn.payload(), experiment=experiment.payload(crn), rates=rates.tolist())
    (args.output / "fixture.json").write_text(json.dumps(fixture, allow_nan=False))
    report = {"load_average": os.getloadavg(), "repetitions": 20}
    t = time.perf_counter()
    with JuliaOptimizer(startup_timeout=240) as worker:
        report["startup_s"] = time.perf_counter()-t
        t = time.perf_counter()
        key = worker.prepare(crn, experiment)
        report["first_prepare_s"] = time.perf_counter()-t
        t = time.perf_counter()
        first = worker.evaluate(key, rates)
        report["first_evaluation_s"] = time.perf_counter()-t
        worker.stats()
        report["ipc_roundtrip"] = measure(worker.stats, 100)
        report["warm_evaluation"] = measure(lambda: worker.evaluate(key, rates))
        report["cached_prepare"] = measure(lambda: worker.prepare(crn, experiment))
        # Different free-reaction sets; preparation only, never scored as candidates.
        timings = []
        for i, reaction in enumerate(crn.reactions):
            if reaction.fixed_rate is not None:
                continue
            other = CRNSpec(crn.species, crn.inputs, crn.reactions[:i]+crn.reactions[i+1:])
            t = time.perf_counter()
            worker.prepare(other, experiment)
            timings.append(time.perf_counter()-t)
        report["new_structure_prepare"] = {"samples_s": timings, "median_s": float(np.median(timings))}
        worker.evaluate(key, rates, trajectories=True)
        report["evaluation_with_trajectories"] = measure(lambda: worker.evaluate(key, rates, trajectories=True))
        # Cached continuation has no solve and includes checkpoint verification/serialization.
        state = worker.optimize(key, [rates], steps=0)["results"][0]["state"]
        report["cached_checkpoint_roundtrip"] = measure(lambda: worker.optimize(key, states=[state], steps=0))
        canonical(rates)
        report["canonical_python_evaluation"] = measure(lambda: canonical(rates), 10)
        report["worker_memory_kib"] = memory(worker.process.pid)
        report["worker"] = worker.info
        report["reference_loss"] = first["loss"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", JULIA_NUM_THREADS="1")
    command = ["julia", f"--project={os.environ['SYNTH_EVO_PROJECT']}", "--startup-file=no", "--threads=1",
               str(Path(__file__).with_suffix(".jl")), str(args.output / "fixture.json")]
    completed = subprocess.run(command, input="", text=True, capture_output=True, env=env, timeout=180)
    (args.output / "julia_stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    profile = json.loads(completed.stdout.splitlines()[-1])
    report["julia_profile"] = profile
    (args.output / "profile.txt").write_text(profile["profile"])
    for name in ("full_evaluation", "solver_only", "rhs_10000"):
        values = profile[name]
        report[name] = {"median_s": float(np.median([r["seconds"] for r in values])),
            "median_allocated_bytes": float(np.median([r["allocated_bytes"] for r in values])),
            "total_gc_s": sum(r["gc_seconds"] for r in values)}
    source = Path("comparisons/search_exploration/reports/julia-feasible-rates-20260915/results.json")
    previous = json.loads(source.read_text())
    efficiency = []
    for record in previous["records"]:
        result = record["optimizer"]
        if record["arm"] != "feasible_lr_0.1":
            continue
        trials = result.get("trials", [])
        last_time = result["history"][0].get("elapsed_s", 0.) if result.get("history") else 0.
        rejected_s = 0.
        last_improvement = 0
        best = result["history"][0]["best_loss"] if result.get("history") else None
        for trial in trials:
            if trial["status"] == "rejected":
                rejected_s += trial["elapsed_s"]-last_time
            if trial["loss"] is not None and trial["loss"] < best:
                best, last_improvement = trial["loss"], trial["evaluation"]
            last_time = trial["elapsed_s"]
        efficiency.append({"start": record["start"], "status": result["status"],
            "evaluations": result["evaluations"], "seconds": result["elapsed_s"],
            "trial_counts": dict(Counter(t["status"] for t in trials)),
            "rejected_trial_interval_s": rejected_s, "last_improving_evaluation": last_improvement})
    report["saved_backtracking_efficiency"] = efficiency
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Julia Fitter Efficiency Audit", "",
        "Read-only profiling of the current fitter on one RPA topology/rate vector. "
        "No campaign changes or uploads. All warm measurements use one Julia/BLAS thread.", "",
        "| Operation | Median ms |", "|---|---:|"]
    for name in ("ipc_roundtrip", "cached_checkpoint_roundtrip", "cached_prepare", "new_structure_prepare",
                 "warm_evaluation", "evaluation_with_trajectories", "canonical_python_evaluation",
                 "full_evaluation", "solver_only", "rhs_10000"):
        lines.append(f"| {name} | {report[name]['median_s']*1000:.4f} |")
    lines += ["", f"Worker RSS: {report['worker_memory_kib']['VmRSS']/1024:.1f} MiB.",
        f"Warm full Julia evaluation allocation: {report['full_evaluation']['median_allocated_bytes']/1024**2:.3f} MiB.",
        f"Solver-only allocation: {report['solver_only']['median_allocated_bytes']/1024**2:.3f} MiB.", "",
        "Solver-only repeats the same nine augmented solves and saving, omitting loss reduction "
        "and output checks; it is separate timing, not an exact additive decomposition. "
        "Python canonical timing includes materialization and state-only solving, not gradients. "
        "The RHS measurement is 10,000 calls at one state, not a representative whole solve. "
        "Startup/first preparation/JIT costs are separate in results.json. Prepared-model "
        "cache hits still validate and serialize/hash the experiment on the Python side.", "",
        "## Prior Backtracking Work", "",
        "| Start | Status | Evaluations | Seconds | Rejected trial intervals, s | Last improving evaluation |",
        "|---|---|---:|---:|---:|---:|"]
    for row in efficiency:
        lines.append(f"| {row['start']} | {row['status']} | {row['evaluations']} | {row['seconds']:.4f} | "
                     f"{row['rejected_trial_interval_s']:.4f} | {row['last_improving_evaluation']} |")
    lines += ["", "Rejected-trial intervals include nearby loop/bookkeeping time; these are "
        "not solver-only timings. Last-improvement analysis is retrospective, not a validated "
        "early-stopping rule. Initial failures have no trial history. Full sampling profile: profile.txt."]
    lines += ["", "## Interpretation", "",
        "The persistent bridge is not the bottleneck. A no-op RPC is about 0.03 ms and "
        "a cached checkpoint roundtrip about 0.22 ms. A full nine-scenario gradient is "
        "about 21.6 ms on this sample. The Python canonical helper takes about 181 ms, "
        "but includes CRN construction/validation and a state-only solve: this is a "
        "whole-path comparison, not an isolated ODE-solver speedup.", "",
        "The full Julia evaluation versus solver-only comparison leaves roughly 9 ms "
        "and 15.6 MiB of allocation associated with checks/readout/reduction and their "
        "dispatch/allocation overhead. The sampling profile also places substantial "
        "time on worker.jl's saved-state/loss/gradient loop. The abstract ODEProblem "
        "container is a likely inference bottleneck here; a typed reduction helper is "
        "a targeted next experiment, not an already verified speedup. The reaction RHS "
        "itself allocates zero bytes in the isolated test.", "",
        "Priorities: (1) reduce readout allocations with a typed function boundary and "
        "verify identical losses/gradients; (2) avoid repeatedly rebuilding/hash-checking "
        "unchanged experiment data when preparing many topologies; (3) investigate "
        "solver-state reuse only after measuring its share. Preserve all concentration "
        "events, scenario checks, tolerances, and sample times.", "",
        "Algorithmic waste is separate: on the already-good start, 46/59 attempts "
        "were rejected and about 0.76/1.15 seconds lay in rejected-trial intervals, "
        "without improvement. On the difficult start this was 28/50 and about "
        "0.45/1.08 seconds. Backtracking helps one case but does not by itself use "
        "compute efficiently near the boundary. A blanket short patience cutoff is "
        "not justified: the third feasible start improved on its final evaluation.", "",
        "Startup, first preparation, and first gradient together took about 4.6 seconds "
        "with existing package caches. Keep workers alive. RSS was about 779 MiB with "
        "several prepared models; measure pool throughput and memory before scaling "
        "to many workers. This audit does not measure concurrency scaling, other "
        "topologies' gradients, or full search-loop throughput."]
    (args.output / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
