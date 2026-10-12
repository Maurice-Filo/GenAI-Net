"""Keep fitting and trajectories in workers; exchange scalar lookup requests."""

import time
import json
import zlib

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.evaluation import EvaluatorPool, build, evaluate_once
from RL4CRN.search_exploration.refinement import fit
from RL4CRN.search_exploration.store import BudgetExhausted, canonical


class RemoteStore:
    def __init__(self, conn):
        self.conn = conn

    def rpc(self, operation, *args):
        self.conn.send(("rpc", (operation, args)))
        kind, result = self.conn.recv()
        if kind == "budget":
            raise BudgetExhausted(result)
        if kind != "reply":
            raise RuntimeError(result)
        return result

    def count(self, *args):
        return self.rpc("count", *args)

    def processing_state(self, key):
        return self.rpc("processing_state", key)

    def processing(self, *args):
        return self.rpc("processing", *args)

    def cached(self, candidate, include_diagnostics=False):
        return self.rpc("cached", candidate)


class LocalEvaluation:
    def __init__(self, evaluator, conn, category):
        self.evaluator = evaluator
        self.store = RemoteStore(conn)
        self.category = category
        self.best_diagnostics = None
        self.best_loss = float("inf")
        self.best_hash = None
        self.evaluation_seconds = 0.

    def evaluate(self, candidate, timeout=None, include_diagnostics=False):
        action, value = self.store.rpc("reserve", candidate, self.category)
        if action == "cached":
            return value, True
        started = time.monotonic()
        try:
            result = evaluate_once(self.evaluator, candidate)
        except Exception as error:
            self.store.rpc("fail", value, f"{type(error).__name__}: {error}")
            raise
        self.evaluation_seconds += time.monotonic() - started
        diagnostics = result.pop("task_info")
        self.store.rpc("complete", value, result)
        if result["simulation_complete"] and result["loss"] < self.best_loss:
            self.best_loss = result["loss"]
            self.best_hash = result["candidate_hash"]
            self.best_diagnostics = diagnostics
        return result, False


def worker(conn, build_args):
    try:
        import torch
        torch.set_num_threads(1)
        evaluator, _, _ = build(*build_args)
        conn.send(("ready", None))
        while True:
            job = conn.recv()
            if job is None:
                break
            started = time.monotonic()
            candidate = job["candidate"]
            service = LocalEvaluation(evaluator, conn, job["category"])
            try:
                if job["category"] == "standalone":
                    # Optional existing evaluator clients own reservation/accounting.
                    result = evaluate_once(evaluator, candidate)
                    diagnostics = result.pop("task_info")
                    conn.send(("ok", {"best": result,
                        "packed_diagnostics": zlib.compress(canonical(diagnostics).encode())}))
                    continue
                elif job["category"] == "search":
                    result, hit = service.evaluate(candidate)
                    output = {"best": {**candidate, **result}, "calls": 1, "cache_hits": int(hit),
                              "failures": [], "termination": "completed"}
                else:
                    output = fit(candidate, service, ExplorationConfig(**job["config"]), job["seed"], include_diagnostics=False)
                best = output["best"]
                if best is not None and best["candidate_hash"] == service.best_hash:
                    output["packed_diagnostics"] = zlib.compress(canonical(service.best_diagnostics).encode())
                output["evaluation_seconds"] = service.evaluation_seconds
                output["job_seconds"] = time.monotonic() - started
                conn.send(("ok", output))
            except Exception as error:
                conn.send(("error", f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        conn.send(("error", f"Worker initialization: {type(error).__name__}: {error}"))
    finally:
        conn.close()


class StreamPool(EvaluatorPool):
    def __init__(self, build_args, workers, timeout, store, fit_seconds):
        self.store, self.fit_seconds = store, fit_seconds
        super().__init__(build_args, workers, timeout)

    def _start(self):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=worker, args=(child, self.args), daemon=True)
        proc.start()
        child.close()
        if not parent.poll(120):
            self._stop((proc, parent))
            raise TimeoutError("Stream evaluator initialization timed out")
        kind, result = parent.recv()
        if kind != "ready":
            self._stop((proc, parent))
            raise RuntimeError(result)
        return proc, parent

    def evaluate(self, candidate, timeout=None):
        result = self.run({"category": "standalone", "candidate": candidate, "timeout": timeout})
        return {**result["best"], "task_info": json.loads(zlib.decompress(result["packed_diagnostics"]))}

    def run(self, job):
        slot = self.available.get(timeout=None if self.timeout is None else 120)
        proc, conn = slot
        active_key = None
        broken = False
        limit = self.timeout if job["category"] in ("search", "standalone") else self.fit_seconds
        if job.get("timeout") is not None:
            limit = job["timeout"] if limit is None else min(limit, job["timeout"])
        deadline = None if limit is None else time.monotonic() + limit + (0 if job["category"] == "standalone" else 10)
        call_deadline = deadline
        try:
            conn.send(job)
            while True:
                deadlines = [d for d in (deadline, call_deadline) if d is not None]
                wait = max(0., min(deadlines) - time.monotonic()) if deadlines else None
                if not conn.poll(wait):
                    raise TimeoutError("Stream job/evaluation deadline exceeded")
                kind, payload = conn.recv()
                if kind == "ok":
                    return payload
                if kind != "rpc":
                    raise RuntimeError(payload)
                operation, args = payload
                try:
                    if operation == "reserve":
                        result = self.store.reserve(*args)
                        if result[0] == "evaluate":
                            active_key = result[1]
                            call_deadline = None if self.timeout is None else time.monotonic() + self.timeout
                            if deadline is not None:
                                call_deadline = deadline if call_deadline is None else min(deadline, call_deadline)
                    elif operation == "complete":
                        if args[0] != active_key:
                            raise ValueError("Evaluation reservation ownership mismatch")
                        result = self.store.complete(*args)
                        active_key = None
                        call_deadline = deadline
                    elif operation == "fail":
                        if args[0] != active_key:
                            raise ValueError("Failure reservation ownership mismatch")
                        result = self.store.fail(*args)
                        active_key = None
                        call_deadline = deadline
                    elif operation == "cached":
                        result = self.store.cached(*args, include_diagnostics=False)
                    elif operation in ("count", "processing_state", "processing"):
                        result = getattr(self.store, operation)(*args)
                    else:
                        raise ValueError("Unsupported worker operation")
                    conn.send(("reply", result))
                except BudgetExhausted as error:
                    conn.send(("budget", str(error)))
        except BaseException as error:
            broken = True
            if active_key is not None:
                self.store.fail(active_key, f"{type(error).__name__}: {error}")
            self._stop(slot)
            # Fail closed: the coordinator ends the run; no hidden worker/job retry.
            raise
        finally:
            # Return a dead slot too: queued jobs must fail on its closed pipe,
            # not wait forever for a worker after a genuine process error.
            self.available.put(slot)
