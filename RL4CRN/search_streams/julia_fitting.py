"""Topology-global multistart budgets, with explicit resumable AdaGrad states."""

from collections import OrderedDict
from copy import deepcopy
import time

import numpy as np


class TopologyFitter:
    def __init__(self, worker, library, experiment, bounds=(.001, 100.), cache_size=256):
        self.worker, self.library, self.experiment = worker, library, experiment
        self.bounds, self.cache_size = bounds, cache_size
        self.screens = OrderedDict()

    def fit(self, candidate, seed, budget=24, *, prior=None, timeout=60., starts=4):
        if budget < 1 or starts < 1 or (timeout is not None and timeout <= 0):
            raise ValueError("Positive fitting limits required")
        began = time.monotonic()
        pairs = sorted(zip(candidate["reaction_ids"], candidate["rates"]))
        ids, rates = map(list, zip(*pairs))
        model = self.worker.prepare(self.library.select(ids), self.experiment)
        rng = np.random.default_rng(seed)
        traces, states = [], []
        used = solves = hits = 0

        def call(*, start=None, state=None, limit, steps):
            nonlocal used, solves
            left = None if timeout is None else timeout - (time.monotonic() - began)
            if left is not None and left <= 0:
                return None
            result = self.worker.optimize(model, None if state is not None else [start],
                states=[state] if state is not None else None, steps=steps,
                method="adagrad", bounds=self.bounds, learning_rate=.1,
                step_policy="feasible", max_backtracks=8, evaluation_limit=int(limit), timeout=left)["results"][0]
            used += result.get("evaluations", 0)
            solves += result.get("scenario_solves", 0)
            traces.append(result)
            return result

        if prior is None:
            vectors = [np.asarray(rates)]
            if starts > 1:
                vectors.append(np.asarray(rates) * np.exp(rng.normal(0, .5, len(rates))))
            for _ in range(starts - len(vectors)):
                vectors.append(np.exp(rng.uniform(*np.log(self.bounds), size=len(rates))))
            seen = set()
            for vector in vectors:
                vector = np.clip(vector, *self.bounds).tolist()
                key = (model, tuple(float(v).hex() for v in vector))
                if key in seen:
                    continue
                seen.add(key)
                if key in self.screens:
                    result = deepcopy(self.screens[key])
                    self.screens.move_to_end(key)
                    hits += 1
                elif used < budget:
                    result = call(start=vector, limit=1, steps=0)
                    if result is None:
                        break
                    # Cache only complete joint evaluations; failed starts may be retried later.
                    if result.get("best_rates") is not None:
                        self.screens[key] = deepcopy(result)
                        while len(self.screens) > self.cache_size:
                            self.screens.popitem(last=False)
                else:
                    break
                if result.get("best_rates") is not None:
                    states.append(result["state"])
        else:
            if prior["model_key"] != model or prior["reaction_ids"] != ids:
                raise ValueError("Topology continuation mismatch")
            states = deepcopy(prior["states"])

        states.sort(key=lambda s: s["best_loss"])
        states = states[:2]
        for index in range(len(states)):
            remaining = budget - used
            if remaining <= 0:
                break
            allocation = max(1, remaining // (len(states) - index))
            result = call(state=states[index], limit=allocation, steps=allocation)
            if result is None:
                break
            states[index] = result["state"]
        best = min(states, key=lambda s: s["best_loss"]) if states else None
        assert used <= budget
        return {"model_key": model, "reaction_ids": ids, "states": states, "traces": traces,
            "best_rates": best["best_rates"] if best else None,
            "best_loss": best["best_loss"] if best else None,
            "evaluations": used, "scenario_solves": solves, "screen_cache_hits": hits,
            "budget": budget, "elapsed_s": time.monotonic() - began}
