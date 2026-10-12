"""CPU-only policy/cache microbenchmarks; no simulations, model calls or uploads."""

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
from types import MethodType

import numpy as np
import torch

from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.evaluation import build, materialize
from RL4CRN.search_exploration.policy import StructuralLearner
from RL4CRN.search_exploration.session import build_session
from RL4CRN.search_exploration.store import Store


def reference_trajectories(self, count, context, sequences=None):
    envs = [materialize(self.evaluator, {"reaction_ids": [], "rates": []}) for _ in range(count)]
    selected = [[] for _ in envs]
    logps, entropies = [], []
    for step in range(self.evaluator.max_added_reactions):
        distribution = self.distribution(envs, context)
        actions = distribution.sample() if sequences is None else torch.tensor([s[step] for s in sequences])
        logps.append(distribution.log_prob(actions))
        entropies.append(distribution.entropy())
        for i, rid in enumerate(actions.tolist()):
            selected[i].append(rid)
            raw = {"reaction index": rid, "parameters": [self.cfg.initial_rate],
                   "continuous parameters": [self.cfg.initial_rate], "discrete parameters": None}
            envs[i].step(self.evaluator.actuator.actuate(raw), self.evaluator.stepper, raw_action=raw)
    return selected, torch.stack(logps).sum(0), torch.stack(entropies).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("Positive counts required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = ExplorationConfig()
    evaluator, task, sc = build("logic", json.loads(CONFIGS["logic"].read_text()), cfg.rate_min, cfg.rate_max)
    sc.train.n_cpus, sc.train.batch_size = 1, args.samples
    session = build_session(sc, task, device="cpu")
    learner = StructuralLearner(session, evaluator, cfg)
    fast = learner.trajectories
    context = [0., 0., 0., 0.]
    timings = {}
    # Each arm gets its own warm-up. This measures throughput, not a search outcome.
    for label, implementation in (("reference", MethodType(reference_trajectories, learner)), ("tensor", fast)):
        learner.trajectories = implementation
        records = []
        for repeat in range(args.repeats + 1):
            start = time.perf_counter()
            ids = learner.sample(args.samples, context)
            middle = time.perf_counter()
            learner.update([{"reaction_ids": s, "policy_version": learner.version} for s in ids],
                           [1.]*args.samples, context, [{"reaction_ids": s} for s in ids])
            end = time.perf_counter()
            if repeat:
                records.append({"sampling_seconds": middle-start, "update_seconds": end-middle,
                                "total_seconds": end-start})
        timings[label] = {"samples": records, "median": {
            key: statistics.median(r[key] for r in records) for key in records[0]}}
    with tempfile.TemporaryDirectory(prefix="exploration-cache-bench-", dir="/local0/tmp") as temp:
        store = Store(Path(temp) / "cache.sqlite", 1, "benchmark")
        candidate = {"reaction_ids": [1], "rates": [1.]}
        payload = {"loss": .1, "task_info": {"outputs": np.random.default_rng(0).random((16, 1, 1000)).tolist()}}
        store.evaluate(candidate, lambda c: payload)
        cache = {}
        for full in (True, False):
            start = time.perf_counter()
            for _ in range(50):
                store.cached(candidate, include_diagnostics=full)
            cache["full_seconds" if full else "summary_seconds"] = (time.perf_counter() - start) / 50
        store.close()
    result = {"task": "logic", "cpu_threads": 1, "torch": torch.__version__,
              "samples": args.samples, "repeats": args.repeats,
              "embedding": sc.policy.deep_layer_size,
              "parameters": sum(p.numel() for p in learner.parameters()),
              "timings": timings, "cache_per_hit": cache,
              "caveat": "Microbenchmark only; excludes CVODE, disk checkpointing and external model latency."}
    atomic_json(args.output / "benchmark.json", result)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    sampling = [timings[k]["median"]["sampling_seconds"] for k in ("reference", "tensor")]
    update = [timings[k]["median"]["update_seconds"] for k in ("reference", "tensor")]
    ax.bar(["Environment replay", "Tensor replay"], sampling, label="Sampling", color="#2878b5")
    ax.bar(["Environment replay", "Tensor replay"], update, bottom=sampling, label="Update", color="#14866d")
    ax.set_ylabel("Median seconds (one CPU thread)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "policy_timing.png", dpi=180)
    fig.savefig(args.output / "policy_timing.pdf")
    plt.close(fig)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
