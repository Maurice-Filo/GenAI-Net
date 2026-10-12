"""Opt-in Julia stream. Existing evolution/Powell streams retain their defaults."""

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import queue
import threading
import time
import zlib

from RL4CRN.parameter_optimization import JuliaOptimizer
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.store import BudgetExhausted, canonical, fingerprint
from .engine import StreamTrainer
from .julia_fitting import TopologyFitter
from .workers import StreamPool


LOSS_AGREEMENT = {"warning_rtol": 2e-4, "warning_atol": 2e-6,
                  "abort_rtol": .01, "abort_atol": 1e-5,
                  "authoritative_loss": "canonical"}


def loss_agreement(julia_loss, canonical_loss, *, mismatch_policy="abort"):
    if mismatch_policy not in ("abort", "warn"):
        raise ValueError("Unknown loss disagreement policy")
    values = (float(julia_loss), float(canonical_loss))
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid Julia/canonical loss; expected finite nonnegative values")
    difference = abs(values[0] - values[1])
    scale = max(values)
    warning_bound = max(LOSS_AGREEMENT["warning_atol"], LOSS_AGREEMENT["warning_rtol"] * scale)
    abort_bound = max(LOSS_AGREEMENT["abort_atol"], LOSS_AGREEMENT["abort_rtol"] * scale)
    large = difference > abort_bound
    status = "error" if large and mismatch_policy == "abort" else "warning" if difference > warning_bound else "agree"
    return {"status": status, "julia_loss": values[0], "canonical_loss": values[1],
        "absolute_difference": difference, "relative_difference": difference / scale if scale else 0.,
        "warning_bound": warning_bound, "abort_bound": abort_bound, "large_disagreement": large,
        "policy": {**LOSS_AGREEMENT, "mismatch_action": mismatch_policy}}


@dataclass(frozen=True)
class JuliaRecipe:
    admissions: int = 32
    evaluations: int = 24
    extensions: int = 8
    extension_evaluations: int = 24
    compute_share: float = .2
    loss_disagreement_policy: str = "abort"

    def __post_init__(self):
        if min(self.admissions, self.evaluations, self.extension_evaluations) < 1 or self.extensions < 0:
            raise ValueError("Invalid Julia recipe budgets")
        if not 0 < self.compute_share < 1:
            raise ValueError("Compute share must be strictly between zero and one")
        if self.loss_disagreement_policy not in ("abort", "warn"):
            raise ValueError("Unknown loss disagreement policy")


class MeasuredSearchPool(StreamPool):
    def __init__(self, *args, **kwargs):
        self.compute_lock = threading.Lock()
        self.compute_seconds = 0.
        super().__init__(*args, **kwargs)

    def run(self, job):
        result = super().run(job)
        with self.compute_lock:
            self.compute_seconds += result.get("evaluation_seconds", 0.)
        return result


def reserve_joint(store, requested):
    # Precharge outstanding work so simultaneous fitters cannot overbook the
    # optimization share. Unused credit is refunded, not counted as a solve.
    with store.transaction():
        available = min(store.budget - store.count("expensive_evaluations"),
            store.budget - store.search_reserve - store.count("evaluations/optimization") - store.count("evaluations/llm"))
        reserved = min(requested, available)
        if reserved <= 0:
            raise BudgetExhausted("No joint-evaluation credit; direct search remains protected")
        store.count("expensive_evaluations", reserved)
        store.count("evaluations/optimization", reserved)
        store.count("julia/reserved", reserved)
        token = store.event("joint_reserved", {"evaluations": reserved})
        return token, reserved


def settle_joint(store, token, reserved, used=None):
    if used is not None and not 0 <= used <= reserved:
        raise ValueError("Joint budget overspent")
    with store.transaction():
        store.count("julia/reserved", -reserved)
        if used is None:
            store.count("julia/unknown_evaluations_upper_bound", reserved)
        else:
            store.count("expensive_evaluations", used - reserved)
            store.count("evaluations/optimization", used - reserved)
            store.count("julia/joint_evaluations", used)
        store.event("joint_settled", {"reservation": token, "used": used,
            "reserved": reserved, "unknown_cost": used is None})


class JuliaFitPool:
    def __init__(self, *, project, workers, library, experiment, store, canonical_pool,
                 output, bounds, timeout, recipe, search_pool):
        self.store, self.canonical_pool = store, canonical_pool
        self.output = Path(output) / "julia_jobs"
        self.output.mkdir(exist_ok=False)
        self.bounds, self.timeout, self.recipe = bounds, timeout, recipe
        self.search_pool = search_pool
        self.lock = threading.Lock()
        self.slots, self.available = [], queue.Queue()
        self.compute_seconds = 0.
        self.job_estimate = 1.
        self.completed_jobs = 0
        self.extended = defaultdict(int)
        try:
            for _ in range(workers):
                worker = JuliaOptimizer(project=project, startup_timeout=240)
                self.slots.append(worker)
                self.available.put(TopologyFitter(worker, library, experiment, bounds))
            atomic_json(self.output / "workers.json", [w.info for w in self.slots])
        except BaseException:
            self.close()
            raise

    def admission_limit(self, pending):
        with self.lock:
            credit = self.recipe.compute_share / (1 - self.recipe.compute_share) * self.search_pool.compute_seconds
            limit = max(0, math.floor((credit - self.compute_seconds) / self.job_estimate) - pending)
            # One cold calibration job before admitting a concurrent wave.
            return min(limit, max(0, 1 - pending)) if not self.completed_jobs else limit

    def _fit(self, fitter, candidate, seed, requested, prior, timeout):
        token, cap = reserve_joint(self.store, requested)
        used = None
        try:
            result = fitter.fit(candidate, seed, cap, prior=prior, timeout=timeout)
            used = result["evaluations"]
            self.store.count("julia/scenario_solves", result["scenario_solves"])
            self.store.count("julia/screen_cache_hits", result["screen_cache_hits"])
            return result
        finally:
            settle_joint(self.store, token, cap, used)

    def run(self, job):
        fitter = self.available.get(timeout=None if self.timeout is None else 120)
        started = time.monotonic()
        candidate, seed = job["candidate"], job["seed"]
        stem = fingerprint({"job": job["job_key"], "epoch": job["epoch"]})
        phases, failures = [], []
        output = {"best": None, "calls": 0, "cache_hits": 0, "failures": failures,
                  "termination": "completed", "parameter_operator": "julia_adagrad"}
        try:
            try:
                result = self._fit(fitter, candidate, seed, self.recipe.evaluations, None, self.timeout)
                phases.append(result)
                improves = result["best_loss"] is not None and result["best_loss"] < candidate["loss"]
                eligible = result["best_rates"] is not None and (improves or job["novel"])
                with self.lock:
                    extend = (eligible and self.extended[job["epoch"]] < self.recipe.extensions
                        and self.admission_limit_unlocked() > 0)
                    if extend:
                        self.extended[job["epoch"]] += 1
                left = None if self.timeout is None else self.timeout - (time.monotonic() - started)
                if extend and (left is None or left > 0):
                    try:
                        continuation = self._fit(fitter, candidate, seed, self.recipe.extension_evaluations, result, left)
                    except BudgetExhausted:
                        output["extension_skipped"] = "global_budget"
                    else:
                        result = continuation
                        phases.append(result)
                # Save optimizer states/trials before making any canonical claim.
                atomic_json(self.output / (stem + ".json"), {"job": job, "phases": phases})
                if result["best_rates"] is not None:
                    fitted = {**candidate, "reaction_ids": result["reaction_ids"], "rates": result["best_rates"]}
                    canonical_result, hit = self.store.evaluate(fitted, self.canonical_pool.evaluate,
                        category="optimization", include_diagnostics=True)
                    output["cache_hits"] += int(hit)
                    output["canonical_loss"] = canonical_result["loss"]
                    output["julia_loss"] = result["best_loss"]
                    if canonical_result["simulation_complete"]:
                        agreement = loss_agreement(result["best_loss"], canonical_result["loss"],
                            mismatch_policy=self.recipe.loss_disagreement_policy)
                        output["loss_agreement"] = agreement
                        if agreement["status"] != "agree":
                            self.store.count("julia/loss_agreement_" + agreement["status"], 1)
                            self.store.event("julia_loss_agreement", {"job_key": job["job_key"],
                                "epoch": job["epoch"], **agreement})
                        if agreement["large_disagreement"]:
                            self.store.count("julia/large_loss_disagreements", 1)
                        if agreement["status"] == "error":
                            raise RuntimeError("Substantial Julia/canonical loss mismatch; inspect saved diagnostics")
                        output["best"] = {**fitted, **{k: v for k, v in canonical_result.items() if k != "task_info"}}
                        output["packed_diagnostics"] = zlib.compress(canonical(canonical_result["task_info"]).encode())
                    else:
                        failures.append({"status": "canonical_incomplete", "loss": canonical_result["loss"]})
                for phase in phases:
                    failures.extend({"status": t["status"], "error": t.get("error")}
                        for t in phase["traces"] if t["status"] not in ("ok", "evaluation_budget_exhausted", "stalled"))
            except BudgetExhausted as error:
                output["termination"] = "budget_exhausted"
                output["budget_note"] = str(error)
            output["calls"] = sum(p["evaluations"] for p in phases)
            output["scenario_solves"] = sum(p["scenario_solves"] for p in phases)
            output["phases"] = len(phases)
            output["job_seconds"] = time.monotonic() - started
            output["evaluation_seconds"] = output["job_seconds"]
            atomic_json(self.output / (stem + ".json"), {"job": job, "phases": phases,
                "outcome": {k: v for k, v in output.items() if k not in ("packed_diagnostics",)}})
            return output
        except BaseException as error:
            atomic_json(self.output / (stem + ".json"), {"job": job, "phases": phases,
                "outcome": {k: v for k, v in output.items() if k != "packed_diagnostics"},
                "fatal_error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            elapsed = time.monotonic() - started
            with self.lock:
                self.compute_seconds += elapsed
                self.completed_jobs += 1
                self.job_estimate = max(.001, elapsed if self.completed_jobs == 1 else .8 * self.job_estimate + .2 * elapsed)
            self.available.put(fitter)

    def admission_limit_unlocked(self):
        credit = self.recipe.compute_share / (1 - self.recipe.compute_share) * self.search_pool.compute_seconds
        return max(0., credit - self.compute_seconds)

    def close(self):
        for worker in self.slots:
            worker.close()
        self.slots = []


class JuliaStreamTrainer(StreamTrainer):
    method_name = "search-streams-julia-v1"

    def _snapshot_evaluation_counts(self, epoch):
        with self.store.lock:
            counts = super()._snapshot_evaluation_counts(epoch)
            reserved = self.store.count("julia/reserved")
            counts["rl"] -= reserved
            counts["total"] -= reserved
        return counts

    def _record(self, record, source, proposal_id):
        if record.get("stream") == "optimization":
            record = {**record, "parameter_operator": "julia_adagrad"}
        return super()._record(record, source, proposal_id)

    def _schedule(self, records):
        pool = self.optimization_pool
        limit = min(pool.recipe.admissions, self.optimize_per_epoch,
            self.queue_capacity - len(self.optimization_jobs), pool.admission_limit(len(self.optimization_jobs)))
        if limit <= 0:
            self.store.count("julia/admission_throttled_epochs", 1)
            return 0
        active = {j["candidate"]["topology_hash"] for j in self.optimization_jobs.values()}
        choices = {}
        for r in records:
            key = r["topology_hash"]
            if key not in active and (key not in choices or r["loss"] < choices[key]["loss"]):
                choices[key] = r
        visits = self.store.counts(["topology:julia:" + k for k in choices])
        quality = sorted((k for k in choices if choices[k]["simulation_complete"]), key=lambda k: (choices[k]["loss"], k))
        chosen = quality[:limit // 2]
        coverage = sorted((k for k in choices if k not in chosen), key=lambda k: (visits["topology:julia:" + k], k))
        chosen += coverage[:limit // 4]
        remaining = [k for k in choices if k not in chosen]
        if remaining:
            chosen += self.optimizer_rng.choice(remaining, size=min(limit - len(chosen), len(remaining)), replace=False).tolist()
        for topology in chosen:
            candidate = {k: v for k, v in choices[topology].items() if k not in ("task_info", "packed_diagnostics")}
            visit = visits["topology:julia:" + topology]
            seed = int(fingerprint({"seed": self.config.seed, "topology": topology, "visit": visit})[:8], 16)
            key = f"{topology}:{visit}"
            job = {"candidate": candidate, "seed": seed, "epoch": self.state.epoch,
                "job_key": key, "novel": visit == 0, "recipe": asdict(pool.recipe)}
            self.store.event("optimization_dispatch", {"job_key": key, "epoch": self.state.epoch, "job": job})
            self.store.count("topology:julia:" + topology, 1)
            self.scheduled_keys.add(key)
            self.optimization_jobs[key] = {"candidate": candidate, "epoch": self.state.epoch,
                "future": self.optimizer_executor.submit(pool.run, job)}
        self.store.count("optimization/submitted", len(chosen))
        return len(chosen)

    def metrics(self, *args, **kwargs):
        with self.store.lock:
            result = super().metrics(*args, **kwargs)
            reserved = self.store.count("julia/reserved")
            result["Compute/budget_committed"] = result.get("Compute/expensive_evaluations", 0)
            for key in ("Compute/expensive_evaluations", "Compute/evaluations/optimization"):
                result[key] = result.get(key, 0) - reserved
        pool = self.optimization_pool
        fit, search = pool.compute_seconds, self.search_pool.compute_seconds
        result.update({"Julia/fit_worker_seconds": fit, "Julia/search_worker_seconds": search,
            "Julia/measured_fit_share": fit / (fit + search) if fit + search else 0.,
            "Julia/target_fit_share": pool.recipe.compute_share, "Julia/job_estimate_seconds": pool.job_estimate})
        return result
