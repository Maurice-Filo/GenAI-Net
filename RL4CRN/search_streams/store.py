"""Compact exact lookup, durable accounting, and selective rich diagnostics."""

from collections import OrderedDict
from concurrent.futures import Future
import json
import zlib

from RL4CRN.search_exploration.store import Store, BudgetExhausted, canonical


class CompactStore(Store):
    def __init__(self, path, budget, identity, search_reserve, memory_entries=100000, wait_timeout=120):
        super().__init__(path, budget, identity)
        if not 0 <= search_reserve <= budget:
            raise ValueError("Invalid search budget reserve")
        self.search_reserve = search_reserve
        self.memory_entries = memory_entries
        self.memory = OrderedDict()
        self.wait_timeout = wait_timeout

    def _remember(self, key, summary):
        self.memory[key] = dict(summary)
        self.memory.move_to_end(key)
        while len(self.memory) > self.memory_entries:
            self.memory.popitem(last=False)

    def reserve(self, candidate, category):
        if category not in ("search", "optimization", "llm"):
            raise ValueError("Unknown stream")
        key = self.evaluation_key(candidate)
        with self.lock:
            self.count("evaluation_attempts", 1)
            summary = self.memory.get(key)
            if summary is None:
                row = self.db.execute("SELECT payload FROM evaluation_summaries WHERE key=?", (key,)).fetchone()
                if row is not None:
                    summary = json.loads(row[0])
                    self._remember(key, summary)
            if summary is not None:
                self.count("cache_hits", 1)
                return "cached", dict(summary)
            waiting = self.inflight.get(key)
            if waiting is None:
                with self.transaction():
                    if self.count("expensive_evaluations") >= self.budget:
                        raise BudgetExhausted("Global budget exhausted")
                    if category != "search" and self.count("evaluations/optimization") + self.count("evaluations/llm") >= self.budget - self.search_reserve:
                        raise BudgetExhausted("Optimization share exhausted; search budget protected")
                    if category == "search" and self.count("evaluations/search") >= self.search_reserve:
                        raise BudgetExhausted("Search share exhausted")
                    self.count("expensive_evaluations", 1)
                    self.count("evaluations/" + category, 1)
                    self.event("evaluation_reserved", {"key": key, "category": category})
                self.inflight[key] = Future()
                return "evaluate", key
            self.count("coalesced_hits", 1)
        # Do not hold the database lock while another worker completes this key.
        return "cached", dict(waiting.result(timeout=self.wait_timeout))

    def complete(self, key, summary):
        if "task_info" in summary:
            raise ValueError("Full trajectories must not enter the scalar completion path")
        with self.transaction():
            self.db.execute("INSERT INTO evaluation_summaries VALUES (?,?)", (key, canonical(summary)))
            self.event("evaluation_completed", {"key": key, "loss": summary["loss"]})
            self._remember(key, summary)
        with self.lock:
            self.inflight.pop(key).set_result(dict(summary))

    def fail(self, key, message):
        with self.lock:
            self.event("evaluation_failed", {"key": key, "message": message})
            future = self.inflight.pop(key, None)
            if future is not None:
                future.set_exception(RuntimeError(message))

    def retain_diagnostics(self, candidate, packed):
        if packed is None:
            return
        key = self.evaluation_key(candidate)
        with self.transaction():
            if self.db.execute("SELECT 1 FROM evaluation_summaries WHERE key=?", (key,)).fetchone() is None:
                raise ValueError("Cannot attach diagnostics to an unevaluated candidate")
            self.db.execute("INSERT OR IGNORE INTO diagnostics VALUES (?,?)", (key, packed))

    def cached(self, candidate, *, include_diagnostics=True):
        key = self.evaluation_key(candidate)
        with self.lock:
            summary = self.memory.get(key)
            if summary is None:
                row = self.db.execute("SELECT payload FROM evaluation_summaries WHERE key=?", (key,)).fetchone()
                if row is None:
                    raise KeyError("Evaluation cache record missing")
                summary = json.loads(row[0])
                self._remember(key, summary)
            blob = self.db.execute("SELECT payload FROM diagnostics WHERE key=?", (key,)).fetchone() if include_diagnostics else None
        result = dict(summary)
        if include_diagnostics:
            result["task_info"] = json.loads(zlib.decompress(blob[0])) if blob else {}
        return result

    def evaluate(self, candidate, fn, *, category="search", include_diagnostics=True):
        """Preserve the scalar-evaluator API used by diagnostic sampling/Harness.

        The fast streams use reserve/complete RPC directly. This optional adapter
        retains its diagnostics so legacy consumers can materialize cached results.
        """
        category = "optimization" if category == "fitting" else category
        action, value = self.reserve(candidate, category)
        if action == "cached":
            return self.cached(candidate, include_diagnostics=include_diagnostics), True
        try:
            payload = fn(candidate)
            summary = {k: v for k, v in payload.items() if k != "task_info"}
            self.complete(value, summary)
        except BaseException as error:
            self.fail(value, f"{type(error).__name__}: {error}")
            raise
        if "task_info" in payload:
            self.retain_diagnostics(candidate, zlib.compress(canonical(payload["task_info"]).encode()))
        return (payload if include_diagnostics else summary), False
