from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List

import numpy as np

from RL4CRN.utils.input_interface import (
    TaskKindBase,
    TaskSpec,
    overrides_get,
    register_task_kind,
)

from apps.habituation.hallmarks import habituation_hallmarks_loss


def _as_float_list(value, *, name: str) -> List[float]:
    if value is None:
        raise ValueError(f"{name} is required.")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return [float(v) for v in arr]


def _as_weight_dict(value) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("weights must be a dictionary keyed by hallmark name.")
    return {str(k): float(v) for k, v in value.items()}


@register_task_kind
class HabituationHallmarksCustomTaskKind(TaskKindBase):
    """Task kind for the custom six-hallmark habituation objective."""

    kind = "habituation_hallmarks_custom"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "T": "float pulse period for the reference train",
                "Ton": "float ON duration of each pulse",
                "n_pulses": "int number of pulses in reference/frequency/intensity trains",
                "u_values": "List[float] grid for candidate input amplitudes",
            },
            "optional": {
                "A": "reference input amplitude; default is the first component of the evaluated input",
                "T_values": "List[float] periods for hallmark 4; default [T, 4*T/3, 5*T/3]",
                "A_values": "List[float] amplitudes for hallmark 5; default [A, 2*A, 3*A]",
                "weights": "dict with keys hallmark1..hallmark6; missing weights default to 1",
                "amplification_factors": "dict with keys hallmark1..hallmark6; finite component losses >= 1 are multiplied by these factors",
                "h1_kwargs": "dict of keyword overrides for hallmark1_loss",
                "h2_kwargs": "dict of keyword overrides for hallmark2_loss",
                "h3_kwargs": "dict of keyword overrides for hallmark3_loss",
                "h4_kwargs": "dict of keyword overrides for hallmark4_loss",
                "h5_kwargs": "dict of keyword overrides for hallmark5_loss",
                "h6_kwargs": "dict of keyword overrides for hallmark6_loss",
                "invalid_loss": "finite penalty for NaN/inf component losses, default 1e4",
            },
            "notes": (
                "The reward function calls apps.habituation.hallmarks.habituation_hallmarks_loss. "
                "That function stores semantic per-hallmark traces and diagnostics in "
                "state.last_task_info['hallmark_info'], which render_habituation reads later."
            ),
        }

    def validate(self, task: TaskSpec) -> None:
        T = float(overrides_get(task, {}, "T", fallback_attr="T"))
        Ton = float(overrides_get(task, {}, "Ton", fallback_attr="Ton"))
        n_pulses = int(overrides_get(task, {}, "n_pulses", fallback_attr="n_pulses"))
        if T <= 0.0:
            raise ValueError("T must be > 0.")
        if Ton <= 0.0 or Ton >= T:
            raise ValueError("Ton must satisfy 0 < Ton < T.")
        if n_pulses < 2:
            raise ValueError("n_pulses must be >= 2.")
        if overrides_get(task, {}, "u_values", fallback_attr="u_values") is None:
            raise ValueError("habituation_hallmarks_custom requires u_values.")

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        values = _as_float_list(u_values, name="u_values")
        if task.n_inputs is None:
            raise ValueError("need n_inputs")
        return [
            np.asarray(u, dtype=np.float32)
            for u in product(values, repeat=int(task.n_inputs))
        ]

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        T = float(overrides_get(task, overrides, "T", fallback_attr="T"))
        Ton = float(overrides_get(task, overrides, "Ton", fallback_attr="Ton"))
        n_pulses = int(overrides_get(task, overrides, "n_pulses", fallback_attr="n_pulses"))
        A_override = overrides_get(task, overrides, "A", fallback_attr="A", default=None)
        T_values = overrides_get(task, overrides, "T_values", fallback_attr="T_values", default=None)
        A_values = overrides_get(task, overrides, "A_values", fallback_attr="A_values", default=None)
        weights = _as_weight_dict(overrides_get(task, overrides, "weights", fallback_attr="weights", default=None))
        amplification_factors = _as_weight_dict(overrides_get(task, overrides, "amplification_factors", fallback_attr="amplification_factors", default=None))
        invalid_loss = float(overrides_get(task, overrides, "invalid_loss", fallback_attr="invalid_loss", default=task.LARGE_NUMBER))

        h1_kwargs = dict(overrides_get(task, overrides, "h1_kwargs", fallback_attr="h1_kwargs", default={}) or {})
        h2_kwargs = dict(overrides_get(task, overrides, "h2_kwargs", fallback_attr="h2_kwargs", default={}) or {})
        h3_kwargs = dict(overrides_get(task, overrides, "h3_kwargs", fallback_attr="h3_kwargs", default={}) or {})
        h4_kwargs = dict(overrides_get(task, overrides, "h4_kwargs", fallback_attr="h4_kwargs", default={}) or {})
        h5_kwargs = dict(overrides_get(task, overrides, "h5_kwargs", fallback_attr="h5_kwargs", default={}) or {})
        h6_kwargs = dict(overrides_get(task, overrides, "h6_kwargs", fallback_attr="h6_kwargs", default={}) or {})

        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)
            if len(x0_list) != 1:
                raise ValueError("habituation_hallmarks_custom currently expects exactly one initial condition.")

            base_input = np.asarray(u_list_local[0], dtype=float).reshape(-1)
            if base_input.size == 0:
                raise ValueError("u_list entries must be nonempty.")
            A = float(A_override) if A_override is not None else float(base_input[0])

            loss, info = habituation_hallmarks_loss(
                state,
                A=A,
                T=T,
                Ton=Ton,
                n_pulses=n_pulses,
                x0=x0_list[0],
                weights=weights,
                T_values=T_values,
                A_values=A_values,
                h1_kwargs=h1_kwargs,
                h2_kwargs=h2_kwargs,
                h3_kwargs=h3_kwargs,
                h4_kwargs=h4_kwargs,
                h5_kwargs=h5_kwargs,
                h6_kwargs=h6_kwargs,
                amplification_factors=amplification_factors,
                invalid_loss=invalid_loss,
                store_info=True,
                return_info=True,
            )
            state.last_task_info["reward type"] = self.kind
            state.last_task_info["task_info"] = info
            return float(loss), state.last_task_info

        return reward_fn
