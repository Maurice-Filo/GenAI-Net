"""Persistent process-isolated canonical evaluation with killable call deadlines."""

from copy import deepcopy
import json
import math
import multiprocessing
import queue
import threading
import time

import numpy as np

from .store import fingerprint


def build(task_name, task_config, rate_min, rate_max):
    from comparisons.llm_crn_generation.run_mmc2_harness_smoke import BUILDERS
    from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
    from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
    from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
    config = deepcopy(task_config)
    if str(config[task_name].get("solver", "CVODE")).upper() != "CVODE":
        raise ValueError("First release supports deterministic CVODE tasks only, not SSA")
    crn, components, task, cfg = BUILDERS[task_name](config)
    if str(crn.solver).upper() != "CVODE":
        raise ValueError("Task builder did not select CVODE")
    library = components[0]
    evaluator = LLMCandidateEvaluator(crn_template=crn, library=library,
        max_added_reactions=int(config["search"]["max_added_reactions"]),
        actuator=LibraryActuator(library), stepper=IOCRNStepper(), compute_reward_func=task.compute_reward,
        min_parameter_value=rate_min, max_parameter_value=rate_max, enforce_parameter_bounds=True,
        forbidden_reaction_ids=[library.find_zero_reaction()])
    return evaluator, task, cfg


def materialize(evaluator, candidate):
    from RL4CRN.environments.environment import Environment
    env = Environment(evaluator.crn_template, evaluator.max_added_reactions)
    for rid, rate in zip(candidate["reaction_ids"], candidate["rates"]):
        action = {"reaction index": int(rid), "parameters": [float(rate)],
                  "continuous parameters": [float(rate)], "discrete parameters": None}
        env.step(evaluator.actuator.actuate(action), evaluator.stepper, raw_action=action)
    return env


def evaluate_once(evaluator, candidate):
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.utils.results_database import serialize_crn, _json_dumps
    result = evaluator.evaluate(LLMCandidate(candidate["reaction_ids"], [[v] for v in candidate["rates"]]))
    if not result.valid or result.env is None or not math.isfinite(float(result.loss)):
        raise RuntimeError(f"Canonical evaluation failed: {result.message}")
    info = json.loads(_json_dumps(result.task_info))
    outputs = np.asarray(info.get("outputs", []), dtype=float)
    complete = bool(outputs.size and np.isfinite(outputs).all() and not np.any(outputs == 1e4))
    correct = None
    targets = np.asarray(info.get("setpoint", []), dtype=float)
    if complete and outputs.ndim == 3 and targets.size == outputs.shape[0] * outputs.shape[1] and np.isin(targets, [0., 1.]).all():
        correct = bool(np.all((2 * targets.reshape(outputs.shape[:2]) - 1) * (outputs[:, :, -1] - .5) > 1e-6))
    identity = serialize_crn(result.env.state)
    return {"loss": float(result.loss), "topology_hash": identity["topology_hash"],
            "candidate_hash": identity["candidate_hash"], "simulation_complete": complete,
            "all_rows_correct": correct, "task_info": info}


def _worker(conn, build_args):
    try:
        import torch
        torch.set_num_threads(1)
        evaluator, _, _ = build(*build_args)
        conn.send(("ready", None))
        while True:
            candidate = conn.recv()
            if candidate is None:
                break
            try:
                conn.send(("ok", evaluate_once(evaluator, candidate)))
            except Exception as error:
                conn.send(("error", f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        conn.send(("error", f"Worker initialization: {type(error).__name__}: {error}"))
    finally:
        conn.close()


class EvaluatorPool:
    def __init__(self, build_args, workers, timeout):
        self.args, self.timeout = build_args, timeout
        self.ctx = multiprocessing.get_context("spawn")
        self.available = queue.Queue()
        self.slots = []
        try:
            for _ in range(workers):
                slot = self._start()
                self.slots.append(slot)
                self.available.put(slot)
        except BaseException:
            self.close()
            raise

    def _start(self):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker, args=(child, self.args), daemon=True)
        proc.start()
        child.close()
        if not parent.poll(120):
            self._stop((proc, parent))
            raise TimeoutError("Evaluator initialization timed out")
        status, payload = parent.recv()
        if status != "ready":
            self._stop((proc, parent))
            raise RuntimeError(payload)
        return proc, parent

    @staticmethod
    def _stop(slot):
        proc, conn = slot
        conn.close()
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()

    def evaluate(self, candidate, timeout=None):
        limits = [t for t in (self.timeout, timeout) if t is not None]
        deadline = time.monotonic() + min(limits) if limits else None
        try:
            slot = self.available.get(timeout=None if deadline is None else max(.001, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError("Deadline reached waiting for an evaluator slot")
        proc, conn = slot
        broken = False
        try:
            conn.send(candidate)
            if not conn.poll(None if deadline is None else max(0., deadline - time.monotonic())):
                broken = True
                raise TimeoutError("Canonical evaluation exceeded deadline; worker terminated")
            status, payload = conn.recv()
            if status != "ok":
                raise RuntimeError(payload)
            return payload
        except (EOFError, BrokenPipeError):
            broken = True
            raise RuntimeError("Evaluator worker exited unexpectedly")
        finally:
            if broken:
                self._stop(slot)
                # Replacement initialization is separate from the failed evaluation.
                try:
                    replacement = self._start()
                except BaseException:
                    # Return a dead slot so other calls fail visibly instead of blocking forever.
                    self.available.put(slot)
                    raise
                self.slots[self.slots.index(slot)] = replacement
                slot = replacement
            self.available.put(slot)

    def close(self):
        for slot in self.slots:
            self._stop(slot)
        self.slots = []


class EvaluationService:
    def __init__(self, store, pool, allowed_ids, reaction_count, bounds):
        self.store, self.pool = store, pool
        self.allowed, self.reaction_count, self.bounds = set(allowed_ids), reaction_count, bounds

    def validate(self, candidate):
        ids, rates = candidate["reaction_ids"], candidate["rates"]
        if len(ids) != self.reaction_count or len(rates) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError("Wrong reaction count or duplicate reaction")
        if not set(ids) <= self.allowed:
            raise ValueError("Forbidden reaction")
        if any(type(i) is not int for i in ids):
            raise ValueError("Reaction IDs must be integers")
        if any(not math.isfinite(v) or not self.bounds[0] <= v <= self.bounds[1] for v in rates):
            raise ValueError("Out-of-domain rate")

    def evaluate(self, candidate, *, category="fitting", timeout=None, include_diagnostics=True):
        self.validate(candidate)
        return self.store.evaluate(candidate, lambda c: self.pool.evaluate(c, timeout=timeout),
                                   category=category, include_diagnostics=include_diagnostics)
