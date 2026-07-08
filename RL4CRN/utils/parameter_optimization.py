"""Optional fixed-topology parameter optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np


@dataclass
class ParameterOptimizationResult:
    """Outcome of optimizing parameters for one fixed CRN topology."""

    attempted: bool
    success: bool
    loss: float
    state: Any
    message: str
    n_evaluations: int = 0


def optimize_crn_parameters_ipopt(
    state: Any,
    reward_fn: Callable[[Any], Any],
    *,
    maxiter: int = 100,
    log_min: float = -18.0,
    log_max: float = 6.0,
) -> ParameterOptimizationResult:
    """Optimize all reaction parameters for a fixed topology using IPOPT.

    Parameters are optimized in log-space so positivity is enforced by
    construction.  The function requires ``cyipopt`` at runtime; if it is not
    installed, the returned result marks the solve as attempted but unsuccessful.
    """

    try:
        from cyipopt import minimize_ipopt
    except Exception as exc:
        loss = _compute_loss(reward_fn, state)
        return ParameterOptimizationResult(
            attempted=True,
            success=False,
            loss=loss,
            state=state,
            message=f"IPOPT unavailable: {type(exc).__name__}: {exc}",
            n_evaluations=0,
        )

    base_state = state.clone()
    x0 = _flatten_log_parameters(base_state, log_min=log_min, log_max=log_max)
    if x0.size == 0:
        loss = _compute_loss(reward_fn, base_state)
        return ParameterOptimizationResult(
            attempted=True,
            success=True,
            loss=loss,
            state=base_state,
            message="No continuous parameters to optimize.",
            n_evaluations=1,
        )

    eval_count = {"n": 0}

    def objective(x: np.ndarray) -> float:
        candidate = base_state.clone()
        _set_log_parameters(candidate, x)
        eval_count["n"] += 1
        return _compute_loss(reward_fn, candidate)

    bounds = [(float(log_min), float(log_max)) for _ in range(x0.size)]
    try:
        result = minimize_ipopt(
            objective,
            x0,
            bounds=bounds,
            options={
                "max_iter": int(maxiter),
                "print_level": 0,
                "sb": "yes",
            },
        )
        optimized_state = base_state.clone()
        _set_log_parameters(optimized_state, result.x)
        optimized_loss = _compute_loss(reward_fn, optimized_state)
        return ParameterOptimizationResult(
            attempted=True,
            success=bool(getattr(result, "success", False)),
            loss=float(optimized_loss),
            state=optimized_state,
            message=str(getattr(result, "message", "")),
            n_evaluations=int(eval_count["n"]),
        )
    except Exception as exc:
        loss = _compute_loss(reward_fn, base_state)
        return ParameterOptimizationResult(
            attempted=True,
            success=False,
            loss=loss,
            state=base_state,
            message=f"IPOPT failed: {type(exc).__name__}: {exc}",
            n_evaluations=int(eval_count["n"]),
        )


def _compute_loss(reward_fn: Callable[[Any], Any], state: Any) -> float:
    result = reward_fn(state)
    raw = result[0] if isinstance(result, (tuple, list)) else result
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "item"):
        raw = raw.item()
    return float(raw)


def _flatten_log_parameters(state: Any, *, log_min: float, log_max: float) -> np.ndarray:
    values: List[float] = []
    for reaction in getattr(state, "reactions", []):
        for value in getattr(reaction, "params", []):
            value = max(float(value), float(np.exp(log_min)))
            values.append(float(np.clip(np.log(value), log_min, log_max)))
    return np.asarray(values, dtype=float)


def _set_log_parameters(state: Any, log_values: Sequence[float]) -> None:
    idx = 0
    for reaction in getattr(state, "reactions", []):
        n_params = len(getattr(reaction, "params", []))
        if n_params == 0:
            continue
        params = [float(np.exp(v)) for v in log_values[idx : idx + n_params]]
        reaction.set_parameters(params)
        idx += n_params
    if hasattr(state, "compile"):
        state.compile()
    if hasattr(state, "reset"):
        state.reset()
