"""Search cohorts do not wait for independent, bounded rate-refinement jobs."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import numpy as np

from RL4CRN.search_exploration.archive import descriptor
from RL4CRN.search_exploration.engine import ExplorationTrainer, atomic_json
from RL4CRN.search_exploration.evaluation import materialize
from RL4CRN.search_exploration.store import fingerprint
from .database import StreamResultsDatabase
from .archive import StreamDiscoveryArchive


class StreamTrainer(ExplorationTrainer):
    def __init__(self, *args, search_pool, optimization_pool, optimization_workers,
                 queue_capacity, optimize_per_epoch, **kwargs):
        super().__init__(*args, **kwargs)
        self.discovery = StreamDiscoveryArchive(self.config)
        if queue_capacity < optimization_workers or optimize_per_epoch < 1:
            raise ValueError("Invalid optimization queue limits")
        self.search_pool, self.optimization_pool = search_pool, optimization_pool
        self.optimization_workers = optimization_workers
        self.optimizer_executor = ThreadPoolExecutor(max_workers=optimization_workers, thread_name_prefix="rate-stream")
        self.queue_capacity, self.optimize_per_epoch = queue_capacity, optimize_per_epoch
        self.optimization_jobs = {}
        self.scheduled_keys = set()
        self.optimizer_rng = np.random.default_rng(self.config.seed + 91021)
        self.optimization_history = []
        self.started = time.monotonic()
        self.optimization_merged = 0
        self.optimization_improved = 0
        self.cumulative_failures = 0

    def configure_results_database(self, path, *, every=5, run_id=None, metadata=None, plot_every=20):
        if every < 1 or plot_every < 1:
            raise ValueError("Database cadences must be positive")
        if self._results_db:
            self._results_db.close()
        self._results_db = StreamResultsDatabase(path, run_id=run_id,
            run_metadata={**dict(metadata or {}), "method": getattr(self, "method_name", "search-streams-v2"),
                "task": self.s.task.kind, "provenance_schema": "search-streams-v1",
                "loss_semantics": "canonical physical loss"})
        self._results_db_every, self._results_plot_every = every, plot_every
        self._last_hof_snapshot_key = None
        return self._results_db

    def load(self, path, strict=True):
        raise ValueError("Asynchronous restart requires explicit pending-job reconciliation; no automatic resume")

    def _runtime_state(self):
        result = super()._runtime_state()
        result["stream_state"] = {"schema": "search-streams-v2",
            "scheduled_keys": sorted(self.scheduled_keys),
            "jobs": [{k: v for k, v in j.items() if k != "future"} for j in self.optimization_jobs.values()],
            "optimizer_rng": self.optimizer_rng.bit_generator.state,
            "merged": self.optimization_merged, "improved": self.optimization_improved}
        return result

    def _snapshot_evaluation_counts(self, epoch):
        counters = self.store.counters()
        return {"rl": counters.get("evaluations/search", 0) + counters.get("evaluations/optimization", 0),
                "llm_final": counters.get("evaluations/llm", 0), "llm_tool": 0,
                "total": counters.get("expensive_evaluations", 0)}

    def _record(self, record, source, proposal_id):
        record = dict(record)
        packed = record.pop("packed_diagnostics", None)
        record.update(source=source, proposal_id=proposal_id)
        record["qualified"] = bool(record["simulation_complete"] and record["loss"] < self.config.quality_threshold
            and (not self.config.require_all_rows or record["all_rows_correct"] is True))
        record["region"] = descriptor(record["reaction_ids"], self.library,
            self.evaluator.crn_template.output_labels, self.config.descriptor, record["topology_hash"])
        with self.store.transaction():
            gain, novelty = self.discovery.add(record, self.store)
        incumbent = self.discovery.entries.get(record["topology_hash"])
        admitted = incumbent is not None and incumbent["candidate_hash"] == record["candidate_hash"]
        # Every candidate has a scalar ledger entry; only archive admissions need rich records.
        if admitted:
            self.store.retain_diagnostics(record, packed)
            diagnostics = self.store.cached(record).get("task_info", {})
            env = materialize(self.evaluator, record)
            env.state.last_task_info = {**diagnostics, "reward": record["loss"],
                "exploration_source": source, "exploration_origin": record["origin"],
                "parameter_operator": record.get("parameter_operator"), "parent_hash": record.get("parent_hash"),
                "stream": record.get("stream"), "submitted_epoch": record.get("submitted_epoch")}
            self.s.mult_env.hall_of_fame.add(env)
            self._results_db.record_evaluation(env.state, source=source, epoch=self.state.epoch,
                loss=record["loss"], valid=record["simulation_complete"], task_info=diagnostics,
                metadata={k: record.get(k) for k in ("origin", "source", "parent_hash", "proposal_id",
                    "policy_version", "parameter_operator", "stream", "submitted_epoch", "region")})
        self.store.event("archive_insertion", {k: record.get(k) for k in (
            "source", "origin", "parent_hash", "proposal_id", "policy_version", "parameter_operator",
            "stream", "submitted_epoch", "candidate_hash", "loss") } | {
                "gain": gain, "novelty": novelty, "admitted": admitted, "epoch": self.state.epoch})
        return gain, novelty

    def _merge_ready(self, drain=False):
        merged = 0
        # The ready set is frozen at the boundary, then ordered by submission ID.
        ready = [k for k, j in self.optimization_jobs.items() if drain or j["future"].done()]
        for key in ready:
            job = self.optimization_jobs[key]
            outcome = job["future"].result()
            self.cumulative_failures += len(outcome["failures"])
            if outcome["termination"] == "error":
                raise RuntimeError(f"Rate stream error: {outcome['failures']}")
            best = outcome["best"]
            if best is not None:
                best = {**best, "packed_diagnostics": outcome.get("packed_diagnostics"),
                    "origin": job["candidate"]["origin"], "parent_hash": job["candidate"]["candidate_hash"],
                    "stream": "optimization", "submitted_epoch": job["epoch"],
                    "parameter_operator": self.config.optimizer}
                self._record(best, job["candidate"]["source"], "fit-" + job["candidate"]["proposal_id"])
                self.optimization_improved += int(best["loss"] < job["candidate"]["loss"])
            result = {k: v for k, v in outcome.items() if k not in ("best", "packed_diagnostics")}
            result.update(job_key=key, submitted_epoch=job["epoch"], merged_epoch=self.state.epoch,
                initial_loss=job["candidate"]["loss"], best_loss=best["loss"] if best else None,
                candidate_hash=best["candidate_hash"] if best else None)
            self.store.event("optimization_result", result)
            self.optimization_history.append(result)
            self.optimization_merged += 1
            merged += 1
            del self.optimization_jobs[key]
        return merged

    def _schedule(self, records):
        remaining = max(0, self.queue_capacity - len(self.optimization_jobs))
        limit = min(self.optimize_per_epoch, remaining)
        if not limit:
            self.store.count("optimization/queue_full_epochs", 1)
            return 0
        choices = {}
        for record in records:
            key = self.store.evaluation_key(record)
            if key not in self.scheduled_keys:
                choices.setdefault(key, record)
        # Structurally valid candidates with incomplete initial trajectories may
        # be rescued by different rates; they remain eligible for coverage/random slots.
        quality = sorted((k for k in choices if choices[k]["simulation_complete"]),
                         key=lambda k: (choices[k]["loss"], k))
        chosen = quality[:limit // 2]
        remaining_keys = [k for k in choices if k not in chosen]
        # Reserve half the queue admissions for coverage and random exploration.
        def region(key):
            r = choices[key]
            return descriptor(r["reaction_ids"], self.library, self.evaluator.crn_template.output_labels,
                              self.config.descriptor, r["topology_hash"])
        visits = self.store.counts(["region:" + region(k) for k in remaining_keys])
        novelty = sorted(remaining_keys, key=lambda k: (visits["region:" + region(k)], k))
        chosen += novelty[:limit // 4]
        remaining_keys = [k for k in choices if k not in chosen]
        if remaining_keys:
            chosen += self.optimizer_rng.choice(remaining_keys,
                size=min(limit - len(chosen), len(remaining_keys)), replace=False).tolist()
        for key in chosen:
            candidate = {k: v for k, v in choices[key].items() if k not in ("task_info", "packed_diagnostics")}
            seed = int(fingerprint({"seed": self.config.seed,
                "pairs": sorted(zip(candidate["reaction_ids"], candidate["rates"]))})[:8], 16)
            job = {"candidate": candidate, "category": "optimization", "seed": seed, "config": self.config.to_dict()}
            # Journal dispatch before the job can reserve an evaluation.
            self.store.event("optimization_dispatch", {"job_key": key, "epoch": self.state.epoch, "job": job})
            self.scheduled_keys.add(key)
            self.optimization_jobs[key] = {"candidate": candidate, "epoch": self.state.epoch,
                "future": self.optimizer_executor.submit(self.optimization_pool.run, job)}
        self.store.count("optimization/submitted", len(chosen))
        return len(chosen)

    def metrics(self, policy=None, timings=None, phase="search"):
        ranked = self.discovery.ranked()
        return {"Archive/best_loss": ranked[0]["loss"] if ranked else None,
            "Archive/topologies": len(ranked), "Archive/qualified_topologies": sum(r["qualified"] for r in ranked),
            "Archive/qualified_regions": sum(r["qualified"] for r in self.discovery.region_best.values()),
            "Fit/failed_jobs_cumulative": self.cumulative_failures,
            "Optimization/pending": len(self.optimization_jobs), "Optimization/merged": self.optimization_merged,
            "Optimization/improved": self.optimization_improved,
            "Timing/elapsed_seconds": time.monotonic() - self.started,
            **{"Compute/" + k: v for k, v in self.store.counters().items()}, **(policy or {}), **(timings or {})}

    def step_epoch(self):
        start = time.monotonic()
        self._merge_ready()
        if self._checkpoint_id is None:
            self.save(self.output / "checkpoint.pt")
        self._maybe_run_llm_graph(self.state.epoch)
        self.pending = self._new_proposals()
        self.save_pending()
        sample_end = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self.search_pool.slots), thread_name_prefix="search-stream") as executor:
            results = list(executor.map(self.search_pool.run,
                ({"category": "search", "candidate": p} for p in self.pending["proposals"])))
        eval_end = time.monotonic()
        records, rewards = [], []
        contributions = {s: 0. for s in ("rl", "ga", "random", "llm")}
        for proposal, result in zip(self.pending["proposals"], results):
            if result["failures"]:
                raise RuntimeError(str(result["failures"]))
            record = {**result["best"], "stream": "search", "parameter_operator": None,
                      "packed_diagnostics": result.get("packed_diagnostics")}
            gain, novelty = self._record(record, proposal["source"], proposal["proposal_id"])
            rewards.append(gain + self.config.novelty_weight * novelty)
            contributions[proposal["source"]] += gain
            records.append({k: v for k, v in record.items() if k != "packed_diagnostics"})
        record_end = time.monotonic()
        indices = [i for i, p in enumerate(self.pending["proposals"]) if p["source"] == "rl"]
        # Only immediate evaluations supply on-policy advantages. Delayed fits supply elites.
        if self.config.rl_fraction > 0:
            policy = self.learner.update([self.pending["proposals"][i] for i in indices],
                [rewards[i] for i in indices], self.pending["context"], self.discovery.imitation(self.rng))
        else:
            # A no-RL control must not train an unused policy through elite imitation.
            policy = {"Policy/enabled": 0, "Policy/version": self.learner.version,
                "Policy/rl_samples": 0, "Policy/imitation_samples": 0}
        policy_end = time.monotonic()
        self._merge_ready()
        self._schedule(records)
        timings = {"Timing/sample_checkpoint_seconds": sample_end - start,
            "Timing/search_evaluation_seconds": eval_end - sample_end,
            "Timing/record_seconds": record_end - eval_end, "Timing/policy_seconds": policy_end - record_end}
        self.last_metrics = self.metrics(policy, timings)
        self.last_metrics.update({"Source/" + s + "/gain": v for s, v in contributions.items()})
        self.last_metrics["Discovery/mean_reward"] = float(np.mean(rewards))
        self.state.history.append({"epoch": self.state.epoch,
            "best": min(r["loss"] for r in records), "median": float(np.median([r["loss"] for r in records]))})
        self._maybe_persist_hof(self.state.epoch)
        self.flush_results_database()
        self.pending = None
        epoch = self.state.epoch
        self.state.epoch += 1
        self.save(self.output / "checkpoint.pt")
        self.telemetry.metrics(epoch, self.last_metrics)
        atomic_json(self.output / "status.json", {"status": "running", "epoch": self.state.epoch,
            "metrics": self.last_metrics, "updated_at": time.time()})
        if self.state.epoch % self.config.plot_every == 0:
            self.telemetry.plot()

    def finish(self):
        atomic_json(self.output / "status.json", {"status": "draining", "epoch": self.state.epoch,
            "metrics": self.metrics(), "updated_at": time.time()})
        self._merge_ready(drain=True)
        self.last_metrics = self.metrics()
        self.persist_hof_boundary(epoch=self.state.epoch, boundary="terminal_boundary",
            rl_evaluations=self._snapshot_evaluation_counts(self.state.epoch)["rl"],
            llm_tool_evaluations=0, llm_final_evaluations=0,
            total_evaluations=self.store.count("expensive_evaluations"))
        self.flush_results_database()
        self.save(self.output / "checkpoint.pt")
        self.telemetry.metrics(self.state.epoch, self.last_metrics)
        self.telemetry.plot()
        atomic_json(self.output / "optimization_results.json", self.optimization_history)

    def close_streams(self):
        self.optimizer_executor.shutdown(wait=True, cancel_futures=True)
