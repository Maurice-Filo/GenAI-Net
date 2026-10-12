"""Isolated CPU NN-size benchmark; no search, evaluator, checkpoint or model calls."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def describe_times(records):
    return {key: {"median": statistics.median(r[key] for r in records),
                  "min": min(r[key] for r in records), "max": max(r[key] for r in records)}
            for key in ("sampling_seconds", "update_seconds", "total_seconds")}


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def run_case(task_name, embedding, samples, elites, repeats, warmup, seed):
    import numpy as np
    import torch
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
    from RL4CRN.search_exploration.config import ExplorationConfig
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.search_exploration.policy import StructuralLearner
    from RL4CRN.search_exploration.session import build_session

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = ExplorationConfig(seed=seed, sil_batch_size=elites)
    evaluator, task, session_cfg = build(task_name, json.loads(CONFIGS[task_name].read_text()), cfg.rate_min, cfg.rate_max)
    session_cfg.policy.deep_layer_size = embedding
    session_cfg.train.n_cpus, session_cfg.train.batch_size, session_cfg.train.seed = 1, samples, seed
    session = build_session(session_cfg, task, device="cpu")
    learner = StructuralLearner(session, evaluator, cfg)
    allowed = sorted(r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids)
    rng = np.random.default_rng(seed + 1000)
    bank = [{"reaction_ids": sorted(rng.choice(allowed, evaluator.max_added_reactions, replace=False).tolist())}
            for _ in range(elites)]
    bank_hash = hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest()
    context = [.1, .1, .2, .1]
    records = []
    start_load = os.getloadavg()
    for repeat in range(warmup + repeats):
        start = time.perf_counter()
        ids = learner.sample(samples, context)
        sampled = time.perf_counter()
        metrics = learner.update([{"reaction_ids": s, "policy_version": learner.version} for s in ids],
                                 [1.]*samples, context, bank)
        finished = time.perf_counter()
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError("Nonfinite policy metrics")
        if repeat >= warmup:
            records.append({"sampling_seconds": sampled-start, "update_seconds": finished-sampled,
                            "total_seconds": finished-start})
    parameters = list(learner.parameters())
    optimizer_tensors = [v for state in learner.optimizer.state.values() for v in state.values() if torch.is_tensor(v)]
    memory = {"parameter_bytes": tensor_bytes(parameters),
              "gradient_bytes": tensor_bytes(p.grad for p in parameters),
              "optimizer_bytes": tensor_bytes(optimizer_tensors),
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    memory["training_tensor_bytes"] = sum(memory[k] for k in ("parameter_bytes", "gradient_bytes", "optimizer_bytes"))
    return {"task": task_name, "embedding": embedding, "samples": samples, "elites": elites,
            "seed": seed, "warmup": warmup, "repeats": repeats, "cpu_threads": torch.get_num_threads(),
            "hidden_width": session_cfg.policy.width, "hidden_depth": session_cfg.policy.depth,
            "reaction_steps": evaluator.max_added_reactions, "library_size": learner.M,
            "parameters": sum(p.numel() for p in parameters), "torch": torch.__version__,
            "dtype": str(parameters[0].dtype), "imitation_bank_hash": bank_hash,
            "timings": describe_times(records), "measurements": records, "memory": memory,
            "host_load_before": start_load, "host_load_after": os.getloadavg()}


def comparison_rows(cases):
    identities = [(c["task"], c["samples"], c["embedding"]) for c in cases]
    if not cases or len(set(identities)) != len(identities):
        raise ValueError("Empty or duplicate benchmark cases")
    rows = []
    for task, samples in sorted({(c["task"], c["samples"]) for c in cases}):
        group = {c["embedding"]: c for c in cases if c["task"] == task and c["samples"] == samples}
        if len(group) != 2:
            raise ValueError("Each task/batch requires two distinct NN sizes")
        small, large = [group[k] for k in sorted(group)]
        for key in ("elites", "hidden_width", "hidden_depth", "reaction_steps", "imitation_bank_hash", "seed", "dtype",
                    "cpu_threads", "warmup", "repeats", "torch", "library_size"):
            if small[key] != large[key]:
                raise ValueError(f"Unmatched benchmark setting: {key}")
        a, b = small["timings"]["total_seconds"]["median"], large["timings"]["total_seconds"]["median"]
        rows.append({"task": task, "samples": samples, "small_embedding": small["embedding"],
                     "large_embedding": large["embedding"], "small_seconds": a, "large_seconds": b,
                     "speedup_large_over_small": b/a, "time_reduction_percent": 100*(1-a/b),
                     "small_parameters": small["parameters"], "large_parameters": large["parameters"],
                     "small_training_mib": small["memory"]["training_tensor_bytes"]/2**20,
                     "large_training_mib": large["memory"]["training_tensor_bytes"]/2**20})
    return rows


def write_outputs(output, manifest, cases):
    from RL4CRN.search_exploration.engine import atomic_json
    rows = comparison_rows(cases)
    atomic_json(output / "summary.json", {"manifest": manifest, "cases": cases, "comparisons": rows})
    atomic_json(output / "rendering.json", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Rendering can be repeated from saved measurements; execution hashes remain in manifest.json."})
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# NN size: computational performance", "",
             "CPU-only, FP32, one thread per process. Each task/size/batch ran in a fresh process.",
             f"Only the intermediate dimension changed: {manifest['embeddings']}. Hidden width and depth are unchanged.",
             f"After {manifest['warmup']} warm-ups, {manifest['repeats']} measured updates were collected per case.",
             f"Each update uses {manifest['elites']} fixed imitation examples; their hashes match across sizes.", "",
             "| Task | RL proposals | Small NN seconds | Large NN seconds | Speedup |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['task']} | {row['samples']} | {row['small_seconds']:.3f} | {row['large_seconds']:.3f} | {row['speedup_large_over_small']:.2f}x |")
    lines += ["", "Timings are medians for sampling plus a policy/Adam update, not end-to-end search timings.",
              "Rewards are synthetic constants. No CRN losses or search-quality outcomes were measured.",
              "Parameter/gradient/Adam tensor bytes exclude activations, allocator overhead and framework memory.",
              "Peak process RSS includes those effects and library/session initialization; it is reported separately in JSON.",
              "Timing error bars span the minimum and maximum measured times, not confidence intervals.",
              "Measurements within a process are not independent experimental seeds; no significance claim is made.",
              "No solver, LLM, Comet upload, checkpoint, or evaluation database was used.", "",
              "![CPU timings](timing.png)", "", "![Training tensor storage](memory.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    tasks = sorted({c["task"] for c in cases})
    dimensions = sorted(manifest["embeddings"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 4), squeeze=False, constrained_layout=True)
    for ax, task in zip(axes[0], tasks):
        batches = sorted({c["samples"] for c in cases if c["task"] == task})
        for index, (dim, color) in enumerate(zip(dimensions, ("#2878b5", "#666666"))):
            selected = [next(c for c in cases if (c["task"], c["embedding"], c["samples"]) == (task, dim, n)) for n in batches]
            values = [c["timings"]["total_seconds"] for c in selected]
            heights = np.array([v["median"] for v in values])
            errors = [heights - np.array([v["min"] for v in values]), np.array([v["max"] for v in values])-heights]
            ax.bar(np.arange(len(batches))+(index-.5)*.36, heights, width=.36, color=color,
                   label=f"{dim:,}", yerr=errors, capsize=3)
        ax.set(xticks=np.arange(len(batches)), xticklabels=batches, xlabel="RL proposals per update",
               ylabel="Median sampling + update (seconds)", title=task.upper())
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Intermediate dimension", frameon=False)
    fig.savefig(output / "timing.png", dpi=180)
    fig.savefig(output / "timing.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(tasks))
    for index, (dim, color) in enumerate(zip(dimensions, ("#2878b5", "#666666"))):
        selected = [next(c for c in cases if c["task"] == t and c["embedding"] == dim) for t in tasks]
        ax.bar(x+(index-.5)*.36, [c["memory"]["training_tensor_bytes"]/2**20 for c in selected],
               width=.36, color=color, label=f"{dim:,}")
    ax.set(xticks=x, xticklabels=[t.upper() for t in tasks], ylabel="Parameters + gradients + Adam (MiB)")
    ax.legend(title="Intermediate dimension", frameon=False, loc="lower center",
              bbox_to_anchor=(.5, 1.01), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "memory.png", dpi=180)
    fig.savefig(output / "memory.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--embeddings", type=int, nargs=2, default=[1024, 10240])
    parser.add_argument("--tasks", nargs="+", choices=["logic", "rpa"], default=["logic", "rpa"])
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 128])
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if any(n < 1 for n in [*args.embeddings, *args.samples, args.elites, args.repeats, args.warmup]):
        parser.error("All size/count arguments must be positive")
    if args.worker:
        result = run_case(args.tasks[0], args.embeddings[0], args.samples[0], args.elites, args.repeats, args.warmup, args.seed)
        print(json.dumps(result), flush=True)
        return
    if args.output is None or len(set(args.embeddings)) != 2:
        parser.error("An output directory and two distinct dimensions are required")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.samples)) != len(args.samples):
        parser.error("Tasks and batch sizes must not contain duplicates")
    args.output = args.output.resolve()
    ancestor = args.output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < 64*2**20:
        raise OSError("Less than 64 MiB free; refusing to start even a small benchmark")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {k: v for k, v in vars(args).items() if k not in ("output", "worker")}
    sources = [Path(__file__).resolve(), *sorted((ROOT/"RL4CRN/search_exploration").glob("*.py"))]
    manifest["source_hashes"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest["command"] = [sys.executable, "-m", "comparisons.search_exploration.benchmark_nn_sizes", *sys.argv[1:]]
    from RL4CRN.search_exploration.engine import atomic_json
    atomic_json(args.output / "manifest.json", manifest)
    cases = []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/local0/tmp/mpl"}
    try:
        for task in args.tasks:
            for batch_index, count in enumerate(args.samples):
                # Reverse ordering on alternating batches to reduce a fixed size/order confound.
                dimensions = args.embeddings if batch_index % 2 == 0 else list(reversed(args.embeddings))
                for dim in dimensions:
                    command = [sys.executable, "-m", "comparisons.search_exploration.benchmark_nn_sizes", "--worker",
                               "--tasks", task, "--embeddings", str(dim), str(dim), "--samples", str(count),
                               "--elites", str(args.elites), "--repeats", str(args.repeats),
                               "--warmup", str(args.warmup), "--seed", str(args.seed)]
                    process = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=600)
                    if process.returncode:
                        raise RuntimeError(f"Worker failed ({task}, {dim}, {count}): {process.stderr[-4000:]}")
                    result = json.loads(process.stdout)
                    cases.append(result)
                    atomic_json(args.output / f"{task}-{dim}-{count}.json", result)
                    print(json.dumps({"task": task, "embedding": dim, "samples": count,
                                      "median_seconds": result["timings"]["total_seconds"]["median"]}), flush=True)
        write_outputs(args.output, manifest, cases)
        atomic_json(args.output / "completed.json", {"status": "complete", "cases": len(cases)})
    except BaseException as error:
        atomic_json(args.output / "failed.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
