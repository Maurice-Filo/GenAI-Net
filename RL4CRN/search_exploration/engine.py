"""Opt-in trainer with canonical-loss compatibility and independent discovery reward."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from RL4CRN.utils.input_interface import Trainer
from RL4CRN.utils.hall_of_fame import HallOfFame
from RL4CRN.utils.results_database import serialize_crn
from .archive import DiscoveryArchive, descriptor
from .evaluation import materialize
from .policy import StructuralLearner
from .refinement import fit
from .store import canonical, fingerprint
from .telemetry import Telemetry
from .database import ExplorationResultsDatabase


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    sync_directory(path.parent)


class ExplorationTrainer(Trainer):
    def __init__(self, session, evaluator, service, config, output, logger=None):
        super().__init__(session)
        self.evaluator, self.service, self.config = evaluator, service, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.store = service.store
        if any(len(r.params) != 1 for r in evaluator.library.reactions if r.ID in service.allowed):
            raise ValueError("Topology exploration v1 requires one scalar rate per added reaction")
        self.discovery = DiscoveryArchive(config)
        self.learner = StructuralLearner(session, evaluator, config)
        self.s.agent = self.learner
        self.s.mult_env.hall_of_fame = HallOfFame(config.compatibility_hof_size)
        self.rng = np.random.default_rng(config.seed)
        self.library = {r.ID: r for r in evaluator.library.reactions}
        self.allowed = sorted(service.allowed)
        self.pending = None
        self.last_metrics = {}
        self.external_gain = 0.
        self._restored_llm = None
        self._checkpoint_id = None
        self.telemetry = Telemetry(self.output, logger)
        self.configure_results_database(self.output / "results.sqlite", every=1, plot_every=config.plot_every,
            run_id=self.output.name, metadata={"method": "exploration-v1", "config": config.to_dict(),
            "descriptor_caveat": "Pilot structural roles, not mechanistic equivalence", "task": session.task.kind})

    def context(self):
        best = self.discovery.ranked(1)
        return [len(self.discovery.entries) / self.config.archive_capacity,
                min(1., len(self.discovery.region_best) / self.config.archive_capacity),
                math.exp(-best[0]["loss"] / self.config.quality_threshold) if best else 0.,
                self.store.count("expensive_evaluations") / self.config.evaluation_budget]

    def configure_results_database(self, path, *, every=5, run_id=None, metadata=None, plot_every=20):
        if every < 1 or plot_every < 1:
            raise ValueError("Database cadences must be positive")
        if self._results_db:
            self._results_db.close()
        self._results_db = ExplorationResultsDatabase(path, run_id=run_id,
            run_metadata={"task": self.s.task.kind, **dict(metadata or {}),
                          "provenance_schema": "exploration-v1", "loss_semantics": "canonical physical loss"})
        self._results_db_every, self._results_plot_every = every, plot_every
        self._last_hof_snapshot_key = None
        return self._results_db

    def _maybe_persist_hof(self, epoch, *, force=False, boundary="epoch", evaluation_counts=None):
        key = (int(epoch), boundary)
        if self._last_hof_snapshot_key == key or (not force and epoch % self._results_db_every):
            return
        counts = evaluation_counts or self._snapshot_evaluation_counts(epoch)
        self._results_db.record_hof_snapshot(self.s.mult_env.hall_of_fame, epoch=epoch, boundary=boundary,
            cumulative_rl_evaluations=counts.get("rl", 0),
            cumulative_llm_final_evaluations=counts.get("llm_final", 0),
            cumulative_llm_tool_evaluations=counts.get("llm_tool", 0),
            cumulative_total_evaluations=counts.get("total", 0),
            save_plots=epoch % self._results_plot_every == 0,
            llm_provenance=self._llm_provenance_by_topology)
        self._last_hof_snapshot_key = key

    def _new_proposals(self):
        cfg = self.config
        sources = self.rng.choice(["rl", "ga", "random"], cfg.batch_size,
            p=[cfg.rl_fraction, cfg.mutation_fraction, 1 - cfg.rl_fraction - cfg.mutation_fraction])
        context = self.context()
        sampled = iter(self.learner.sample(int(np.sum(sources == "rl")), context))
        sample_parent = self.discovery.parent_sampler(self.store)
        proposals = []
        for i, source in enumerate(sources):
            parent = None
            initial_rates = [cfg.initial_rate] * self.evaluator.max_added_reactions
            if source == "rl":
                ids = next(sampled)
            elif source == "ga" and (parent := sample_parent(self.rng)) is not None:
                ids = list(parent["reaction_ids"])
                initial_rates = list(parent["rates"])
                pos = int(self.rng.integers(len(ids)))
                ids[pos] = int(self.rng.choice([r for r in self.allowed if r not in ids]))
                initial_rates[pos] = cfg.initial_rate
            else:
                source = "random"
                ids = self.rng.choice(self.allowed, self.evaluator.max_added_reactions, replace=False).tolist()
            proposal = {"reaction_ids": ids, "rates": initial_rates,
                "source": str(source), "origin": parent["origin"] if parent else str(source),
                "parent_hash": parent["candidate_hash"] if parent else None,
                "initialization": "inherited-parent-rates" if parent else "constant",
                "proposal_id": f"epoch{self.state.epoch}-proposal{i}", "policy_version": self.learner.version}
            self.service.validate(proposal)
            proposals.append(proposal)
        with self.store.transaction():
            self.store.count("proposal_attempts", len(proposals))
            for source in ("rl", "ga", "random"):
                n = sum(p["source"] == source for p in proposals)
                if n:
                    self.store.count("proposals/" + source, n)
        return {"epoch": self.state.epoch, "context": context, "proposals": proposals,
                "policy_version": self.learner.version}

    def _fit_burst(self, proposal):
        # Same topology/starting rates use a reproducible local random stream independent of emitter.
        seed = int(fingerprint({"seed": self.config.seed, "pairs": sorted(zip(proposal["reaction_ids"], proposal["rates"]))})[:8], 16)
        result = fit(proposal, self.service, self.config, seed, include_diagnostics=False)
        outcomes = [result]
        parent = result["best"]
        rng = np.random.default_rng(seed)
        for step in range(self.config.structural_refinement_steps):
            if parent is None:
                break
            ids = list(parent["reaction_ids"])
            pos = int(rng.integers(len(ids)))
            ids[pos] = int(rng.choice([r for r in self.allowed if r not in ids]))
            candidate = {**proposal, "reaction_ids": ids, "rates": list(parent["rates"]),
                         "source": "ga", "parent_hash": parent["candidate_hash"],
                         "proposal_id": f"{proposal['proposal_id']}-edit{step}"}
            candidate["rates"][pos] = self.config.initial_rate
            refined = fit(candidate, self.service, self.config, seed + step + 1, include_diagnostics=False)
            outcomes.append(refined)
            if refined["best"] is not None:
                parent = refined["best"]
        return outcomes

    def _record(self, record, source, proposal_id):
        record = dict(record)
        if "task_info" not in record:
            record.update(self.store.cached(record))
        record.update(source=source, proposal_id=proposal_id)
        record["qualified"] = bool(record["simulation_complete"] and record["loss"] < self.config.quality_threshold
            and (not self.config.require_all_rows or record["all_rows_correct"] is True))
        output_labels = self.evaluator.crn_template.output_labels
        record["region"] = descriptor(record["reaction_ids"], self.library, output_labels,
                                       self.config.descriptor, record["topology_hash"])
        with self.store.transaction():
            gain, novelty = self.discovery.add(record, self.store)
        env = materialize(self.evaluator, record)
        env.state.last_task_info = {**record.get("task_info", {}), "reward": record["loss"],
            "exploration_source": source, "exploration_origin": record["origin"],
            "parameter_operator": record.get("parameter_operator"), "parent_hash": record.get("parent_hash")}
        self.s.mult_env.hall_of_fame.add(env)
        provenance = {k: record.get(k) for k in ("source", "origin", "parent_hash", "proposal_id", "policy_version", "parameter_operator")}
        self._results_db.record_evaluation(env.state, source=source, epoch=self.state.epoch,
            loss=record["loss"], valid=record["simulation_complete"], task_info=record.get("task_info", {}),
            metadata={**provenance, "region": record["region"], "archive_gain": gain, "novelty": novelty})
        self.store.event("archive_insertion", {**provenance, "candidate_hash": record["candidate_hash"],
            "gain": gain, "novelty": novelty, "epoch": self.state.epoch})
        return gain, novelty

    def step_epoch(self):
        if self.pending is None:
            if self._checkpoint_id is None:
                self.save(self.output / "checkpoint.pt")
            # Old Harness scheduling/communication rules remain in use; only the evaluator is adapted.
            if self._llm_loop:
                atomic_torch(self.output / "pending.pt", {**self._runtime_state(),
                    "base_checkpoint_id": self._checkpoint_id, "llm_dispatch_in_progress": True})
            self._maybe_run_llm_graph(self.state.epoch)
            self.pending = self._new_proposals()
            self.save_pending()
        pending = self.pending
        if pending["policy_version"] != self.learner.version:
            raise ValueError("Pending cohort policy mismatch")
        folder = self.output / "cohorts" / str(self.state.epoch)
        folder.mkdir(parents=True, exist_ok=True)

        def compute(proposal):
            path = folder / (proposal["proposal_id"] + ".json")
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["proposal"] != proposal:
                    raise ValueError("Pending proposal changed")
                return saved["outcomes"]
            result = self._fit_burst(proposal)
            compact = [{**r, "best": {k: v for k, v in r["best"].items() if k != "task_info"}
                        if r["best"] is not None else None} for r in result]
            atomic_json(path, {"proposal": proposal, "outcomes": compact})
            return result

        # A fixed-policy cohort gives exact on-policy credit while numerical work runs concurrently.
        with ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="exploration-fit") as executor:
            fitted = list(executor.map(compute, pending["proposals"]))
        rewards, physical = [], []
        contributions = {s: 0. for s in ("rl", "ga", "random", "llm")}
        contributions["llm"] = self.external_gain
        failures = 0
        for proposal, burst in zip(pending["proposals"], fitted):
            reward = 0.
            for index, outcome in enumerate(burst):
                failures += int(bool(outcome["failures"]))
                self.store.event("fit_outcome", {"proposal_id": proposal["proposal_id"],
                    **{k: v for k, v in outcome.items() if k != "best"}})
                if outcome["termination"] == "error":
                    raise RuntimeError(f"Refinement implementation/evaluator error: {outcome['failures']}")
                best = outcome["best"]
                if best is None:
                    continue
                best = {**best, "origin": proposal["origin"], "policy_version": proposal["policy_version"]}
                if index == 0:
                    best["parent_hash"] = proposal["parent_hash"]
                # Rate fitting is recorded as refinement, not a raw policy-generated rate vector.
                best["parameter_operator"] = self.config.optimizer
                gain, novelty = self._record(best, proposal["source"] if index == 0 else "ga", proposal["proposal_id"])
                reward += gain + self.config.novelty_weight * novelty
                contributions[proposal["source"]] += gain
                physical.append(best["loss"])
            rewards.append(reward)
        indices = [i for i, p in enumerate(pending["proposals"]) if p["source"] == "rl"]
        policy_metrics = self.learner.update([pending["proposals"][i] for i in indices],
            [rewards[i] for i in indices], pending["context"], self.discovery.imitation(self.rng))
        ranked = self.discovery.ranked()
        metrics = {"Archive/best_loss": ranked[0]["loss"] if ranked else None,
            "Archive/topologies": len(ranked), "Archive/qualified_topologies": sum(r["qualified"] for r in ranked),
            "Archive/qualified_regions": sum(r["qualified"] for r in self.discovery.region_best.values()),
            "Fit/failed_jobs": failures, "Discovery/mean_reward": float(np.mean(rewards)),
            "Archive/live_rate_max": max((max(r["rates"]) for r in ranked), default=0.),
            "Archive/live_rate_min": min((min(r["rates"]) for r in ranked), default=0.),
            **policy_metrics, **{"Source/" + k + "/gain": v for k, v in contributions.items()},
            **{"Compute/" + k: v for k, v in self.store.counters().items()}}
        best = min(physical, default=float("inf"))
        median = float(np.median(physical)) if physical else float("inf")
        epoch = self.state.epoch
        self.state.history.append({"epoch": epoch, "best": best, "median": median})
        self._maybe_persist_hof(epoch)
        self.flush_results_database()
        self.pending = None
        self.external_gain = 0.
        self.state.epoch += 1
        self.last_metrics = metrics
        self.save(self.output / "checkpoint.pt")
        self.telemetry.metrics(epoch, {k: v for k, v in metrics.items() if v is not None})
        atomic_json(self.output / "status.json", {"status": "running", "epoch": self.state.epoch,
            "metrics": metrics, "updated_at": time.time()})
        if self.state.epoch % self.config.plot_every == 0:
            self.telemetry.plot()
        return best, median, np.asarray(physical)

    def _snapshot_evaluation_counts(self, epoch):
        counts = self.store.counters()
        return {"rl": counts.get("evaluations/fitting", 0),
                "llm_final": counts.get("evaluations/llm", 0),
                "llm_tool": 0, "total": counts.get("expensive_evaluations", 0)}

    def forbidden_topology_summary(self, limit=10):
        with self.store.lock:
            rows = self.store.db.execute("SELECT key,status FROM processing ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return "Processing states are budget-relative, not forbidden chemistry: " + canonical(rows)

    def retry_failed_cohort(self):
        """Explicit retry only: preserve failed evidence and rerun affected fit jobs."""
        if self.pending is None:
            raise ValueError("No interrupted cohort")
        folder = self.output / "cohorts" / str(self.state.epoch)
        changed = 0
        for proposal in self.pending["proposals"]:
            path = folder / (proposal["proposal_id"] + ".json")
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if any(r["termination"] in ("error", "timeout") for r in payload["outcomes"]):
                backup = path.with_suffix(f".failed-{time.time_ns()}.json")
                path.rename(backup)
                self.store.event("explicit_retry", {"proposal_id": proposal["proposal_id"], "evidence": str(backup)})
                changed += 1
        return changed

    def sample(self, n_samples, sample_hof_size, *, u_list=None, u_spec=None, u_values=None,
               dose_range=None, ic=None, weights=None):
        if any(v is not None for v in (u_list, u_spec, u_values, dose_range, ic, weights)):
            # Alternate tasks are diagnostic evaluations, not reusable training-cache entries.
            self.store.event("diagnostic_sampling", {"n_samples": n_samples, "task_override": True,
                "budget": "outside_training", "parameter_initialization": "constant_without_refinement"})
            return super().sample(n_samples, sample_hof_size, u_list=u_list, u_spec=u_spec,
                u_values=u_values, dose_range=dose_range, ic=ic, weights=weights)
        if n_samples <= 0 or sample_hof_size <= 0:
            raise ValueError("Sample counts must be positive")
        hof = HallOfFame(sample_hof_size)
        for i, ids in enumerate(self.learner.sample(n_samples, self.context())):
            proposal = {"reaction_ids": ids, "rates": [self.config.initial_rate]*len(ids),
                "proposal_id": f"sample-{self.state.epoch}-{i}", "source": "rl", "origin": "rl",
                "parent_hash": None, "policy_version": self.learner.version}
            result = self._fit_burst(proposal)[0]
            if result["termination"] == "error":
                raise RuntimeError(str(result["failures"]))
            if result["best"] is not None:
                result["best"].update(self.store.cached(result["best"]))
                env = materialize(self.evaluator, result["best"])
                env.state.last_task_info = {**result["best"]["task_info"], "reward": result["best"]["loss"]}
                hof.add(env)
        self.s.sample_hof = hof
        return hof

    def _sil_feedback_summary(self):
        return canonical({"enabled": True, "structural_only": True, "top_k": self.config.sil_top_k,
            "batch_size": self.config.sil_batch_size, "last_update": self.learner.last_sil_info,
            "discovery_archive_size": len(self.discovery.entries)})

    def _maybe_run_llm_graph(self, epoch):
        settings = self._llm_loop
        if settings and hasattr(settings["graph"], "client"):
            permitted = settings["cross_communication"] and not (
                settings["withhold_initial_hof"] and epoch == settings["start_epoch"])
            client = settings["graph"].client
            client.exploration_context = ({"schema": "exploration-v1", "epoch": epoch,
                "archive_size": len(self.discovery.entries), "descriptor": self.config.descriptor,
                "successful_regions": sum(r["qualified"] for r in self.discovery.region_best.values()),
                "sil_top_k": self.config.sil_top_k, "processing": self.forbidden_topology_summary(),
                "warning": "Processed means budget spent, not proof of exhaustion; descriptors are not mechanisms"}
                if permitted else None)
        return super()._maybe_run_llm_graph(epoch)

    def configure_llm_graph(self, graph, **kwargs):
        super().configure_llm_graph(graph, **kwargs)
        if self._restored_llm is not None:
            restored = self._restored_llm
            self._llm_loop["history"] = restored["history"]
            self._llm_loop["isolated_candidates"] = restored["isolated_candidates"]
            if restored["memory"] is not None:
                graph.memory = restored["memory"]
            self._restored_llm = None

    def _collect_llm_graph_result(self, **kwargs):
        result = super()._collect_llm_graph_result(**kwargs)
        if result is not None and self._llm_loop["add_to_hall_of_fame"]:
            for index, evaluation in enumerate(result.evaluations):
                record = getattr(evaluation, "exploration_record", None)
                if evaluation.valid and record is not None:
                    record = {**record, "origin": "llm", "parent_hash": None, "policy_version": None}
                    gain, _ = self._record(record, "llm", f"llm-{self.state.epoch}-{index}-{record['candidate_hash'][:10]}")
                    self.external_gain += gain
        return result

    def _runtime_state(self):
        # Cache and fitting journal are durable independently; archive/count commits occur at cohort boundaries.
        with self.store.lock:
            visits = dict(self.store.db.execute("SELECT key,n FROM visits WHERE key LIKE 'region:%' OR key LIKE 'topology:%'"))
        return {"schema": "exploration-v1", "identity": self.store.identity, "config": self.config.to_dict(),
            "epoch": self.state.epoch, "history": self.state.history, "version": self.learner.version,
            "baseline": self.learner.baseline, "archive": self.discovery.state_dict(), "visits": visits,
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "pending": self.pending,
            "pending_llm": len(self._llm_loop.get("jobs", [])) if self._llm_loop else 0,
            "last_metrics": self.last_metrics, "external_gain": self.external_gain,
            "llm_provenance": self._llm_provenance_by_topology,
            "llm_state": {"history": self._llm_loop["history"],
                "isolated_candidates": self._llm_loop["isolated_candidates"],
                "memory": getattr(self._llm_loop["graph"], "memory", None)} if self._llm_loop else None}
    def save(self, path):
        checkpoint_id = uuid.uuid4().hex
        payload = {**self._runtime_state(), "checkpoint_id": checkpoint_id,
                   "policy": self.learner.state_dict(), "optimizer": self.learner.optimizer.state_dict()}
        atomic_torch(path, payload)
        self._checkpoint_id = checkpoint_id

    def save_pending(self):
        if self.pending is None or self._checkpoint_id is None:
            raise ValueError("Pending journal requires a durable base checkpoint and cohort")
        atomic_torch(self.output / "pending.pt", {
            **self._runtime_state(), "base_checkpoint_id": self._checkpoint_id})

    def load(self, path, strict=True):
        # Trusted local checkpoint only. Never load user-supplied pickle files.
        payload = torch.load(path, map_location=self.s.device, weights_only=False)
        if payload.get("schema") != "exploration-v1" or payload["identity"] != self.store.identity or payload["config"] != self.config.to_dict():
            raise ValueError("Checkpoint schema/evaluator/config mismatch; legacy checkpoints are not migrated")
        self._checkpoint_id = payload.get("checkpoint_id")
        journal = Path(path).parent / "pending.pt"
        if journal.exists():
            pending = torch.load(journal, map_location=self.s.device, weights_only=False)
            if pending["base_checkpoint_id"] == self._checkpoint_id:
                if (pending["identity"] != payload["identity"] or pending["config"] != payload["config"]
                        or pending["epoch"] != payload["epoch"] or pending["version"] != payload["version"]):
                    raise ValueError("Pending journal does not match base checkpoint")
                payload.update(pending)
            elif pending["epoch"] > payload["epoch"]:
                raise ValueError("Pending journal refers to a missing newer checkpoint")
            # An older journal is harmless after the next full checkpoint commits.
        if payload["pending_llm"] or payload.get("llm_dispatch_in_progress"):
            raise ValueError("Interrupted LLM request requires explicit reconciliation; no automatic repeat model call")
        self.learner.load_state_dict(payload["policy"], strict=strict)
        self.learner._sample_cache = None
        self.learner.optimizer.load_state_dict(payload["optimizer"])
        self.learner.version, self.learner.baseline = payload["version"], payload["baseline"]
        self.discovery.load_state_dict(payload["archive"])
        self.rng.bit_generator.state = payload["rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        self.state.epoch, self.state.history = payload["epoch"], payload["history"]
        self.pending, self.last_metrics = payload["pending"], payload["last_metrics"]
        self.external_gain = payload.get("external_gain", 0.)
        self._restored_llm = payload.get("llm_state")
        self._llm_provenance_by_topology = payload.get("llm_provenance", {})
        with self.store.transaction():
            self.store.db.execute("DELETE FROM visits WHERE key LIKE 'region:%' OR key LIKE 'topology:%'")
            self.store.db.executemany("INSERT INTO visits VALUES (?,?)", payload["visits"].items())
        for record in self.discovery.ranked(self.config.compatibility_hof_size):
            env = materialize(self.evaluator, record)
            env.state.last_task_info = {**self.store.cached(record).get("task_info", {}),
                "reward": record["loss"], "exploration_source": record["source"],
                "exploration_origin": record["origin"], "parameter_operator": record.get("parameter_operator")}
            self.s.mult_env.hall_of_fame.add(env)
