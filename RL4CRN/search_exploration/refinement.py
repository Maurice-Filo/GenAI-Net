"""Budgeted rate refinement; failure is not evidence of topology exhaustion."""

from dataclasses import asdict
import math
import time

import numpy as np
from scipy.optimize import minimize

from .store import BudgetExhausted, fingerprint


class FitLimit(RuntimeError):
    pass


def fit(candidate, service, config, seed, *, include_diagnostics=True):
    # Equivalent action orderings must induce the same optimizer random stream.
    ordered = sorted(zip(candidate["reaction_ids"], candidate["rates"]))
    candidate = {**candidate, "reaction_ids": [i for i, _ in ordered], "rates": [k for _, k in ordered]}
    rng = np.random.default_rng(seed)
    ids = candidate["reaction_ids"]
    key = fingerprint({"pairs": sorted(zip(ids, candidate["rates"])), "optimizer": config.optimizer,
                       "budget": config.fit_evaluations, "seconds": config.fit_seconds,
                       "sigma": config.log_rate_sigma, "seed": seed,
                       "bounds": [config.rate_min, config.rate_max]})
    prior = service.store.processing_state(key)
    if prior and prior[0] == "processed-with-budget":
        cached = prior[1]
        service.store.count("fit_cache_hits", 1)
        if cached["best"] is not None:
            restored = service.store.cached(cached["best"], include_diagnostics=include_diagnostics)
            cached["best"].update(restored)
            cached["best"].update({k: candidate[k] for k in ("source", "proposal_id", "origin", "parent_hash", "policy_version") if k in candidate})
        return cached
    service.store.processing(key, "fitting", {"proposal_id": candidate["proposal_id"]})
    started = time.monotonic()
    best, calls, hits, failures = None, 0, 0, []
    termination = "completed"
    lower, upper = math.log(config.rate_min), math.log(config.rate_max)

    def objective(z):
        nonlocal best, calls, hits
        remaining = None if config.fit_seconds is None else config.fit_seconds - (time.monotonic() - started)
        if calls >= config.fit_evaluations or (remaining is not None and remaining <= 0):
            raise FitLimit("fit budget or deadline reached")
        proposed = {**candidate, "rates": np.clip(np.exp(z), config.rate_min, config.rate_max).tolist()}
        calls += 1
        result, cached = service.evaluate(proposed, timeout=remaining, include_diagnostics=False)
        hits += int(cached)
        record = {**proposed, **result}
        if record["simulation_complete"] and (best is None or record["loss"] < best["loss"]):
            best = record
        return record["loss"]

    try:
        initial = np.log(candidate["rates"])
        objective(initial)
        if config.optimizer == "powell":
            result = minimize(objective, initial, method="Powell", bounds=[(lower, upper)] * len(ids),
                              options={"maxfev": max(1, config.fit_evaluations - 1), "maxiter": config.fit_evaluations})
            termination = "optimizer_converged" if result.success else "optimizer_stopped"
        else:
            while calls < config.fit_evaluations:
                if best is None or rng.random() < .2:
                    z = rng.uniform(lower, upper, len(ids))
                else:
                    z = np.log(best["rates"])
                    z[int(rng.integers(len(ids)))] += rng.normal(0, config.log_rate_sigma)
                objective(np.clip(z, lower, upper))
            termination = "evaluation_limit"
    except (FitLimit, BudgetExhausted) as error:
        termination = type(error).__name__
    except TimeoutError as error:
        termination = "timeout"
        failures.append(str(error))
    except Exception as error:
        termination = "error"
        failures.append(f"{type(error).__name__}: {error}")
    result = {"best": best, "calls": calls, "cache_hits": hits, "failures": failures,
              "termination": termination, "elapsed_seconds": time.monotonic() - started,
              "fit_key": key, "optimizer": config.optimizer}
    state = "processed-with-budget" if best is not None and not failures and termination != "BudgetExhausted" else "retry-eligible"
    service.store.processing(key, state, result)
    if best is not None and include_diagnostics:
        best.update(service.store.cached(best))
    return result
