from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from RL4CRN.utils.input_interface import (
    TaskKindBase,
    TaskSpec,
    overrides_get,
    register_task_kind,
    build_weights,
)

from RL4CRN.rewards.deterministic import dynamic_tracking_error


def ic_builder_from_list(species_labels: Sequence[str], ic_values: Sequence[Sequence[float]]):
    """
    Build a minimal IC object compatible with RL4CRN's TaskKind interface.

    Parameters
    ----------
    species_labels:
        Names of the CRN species (used only for validation / record keeping).
    ic_values:
        List of initial-condition vectors, one per classification example.

    Returns
    -------
    obj
        Object exposing `get_ic(state) -> List[np.ndarray]`.
    """
    ic_arr = np.asarray(ic_values, dtype=np.float32)
    if ic_arr.ndim != 2:
        raise ValueError("ic_values must be 2D: [[...], [...], ...].")
    if ic_arr.shape[1] != len(species_labels):
        raise ValueError(
            f"Each IC must have length len(species_labels)={len(species_labels)}; got {ic_arr.shape[1]}."
        )

    @dataclass
    class _IC:
        names: List[str]
        values: np.ndarray  # (N, n_species)

        def get_ic(self, state: Any) -> List[np.ndarray]:
            return [self.values[i].copy() for i in range(self.values.shape[0])]

    return _IC(names=list(species_labels), values=ic_arr)


def _build_ic_and_r_list_from_ic_r_maps(
    species_labels: Sequence[str],
    ic_r_maps: Sequence[Tuple[Sequence[Sequence[float]], Sequence[float]]],
) -> Tuple[Any, List[np.ndarray]]:
    """
    Expand (ic_values_block, r_value) blocks into a flat IC list and aligned r_list.

    Parameters
    ----------
    species_labels:
        CRN species labels (used for IC validation).
    ic_r_maps:
        List of (ic_values_block, r_value). r_value is repeated for each IC in the block.

    Returns
    -------
    ic_obj, r_list
        ic_obj.get_ic(state) returns a list of ICs aligned with r_list.
    """
    all_ic: List[List[float]] = []
    r_list: List[np.ndarray] = []

    for ic_values_block, r_value in ic_r_maps:
        block = list(ic_values_block)
        if len(block) == 0:
            continue
        all_ic.extend(block)
        r_vec = np.asarray(r_value, dtype=np.float32).reshape(-1)
        r_list.extend([r_vec.copy() for _ in range(len(block))])

    if len(all_ic) == 0:
        raise ValueError("ic_r_maps produced 0 ICs (all blocks empty?).")

    if len(r_list) != len(all_ic):
        raise RuntimeError("Internal error: mismatch between IC count and target count.")

    ic_obj = ic_builder_from_list(species_labels, all_ic)
    return ic_obj, r_list


def _normalize_u_list(u_list_raw: Any, n_inputs: int) -> List[np.ndarray]:
    """
    Normalize user-provided u_list into List[np.ndarray] of shape (n_inputs,).

    Accepts
    -------
    - [1.] when n_inputs==1
    - [ [..], [..], ... ]
    - np.array([...]) or np.array([[...],[...]])
    """
    if u_list_raw is None:
        raise ValueError("classification requires params['u_list'].")

    # Scalar and single-input convenience
    if isinstance(u_list_raw, (float, int)) and n_inputs == 1:
        return [np.asarray([float(u_list_raw)], dtype=np.float32)]

    arr = np.asarray(u_list_raw, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size == 1 and n_inputs == 1:
            return [arr.reshape(1)]
        if arr.size != n_inputs:
            raise ValueError(f"u_list vector has length {arr.size}, expected n_inputs={n_inputs}.")
        return [arr.reshape(n_inputs)]

    if arr.ndim == 2:
        if arr.shape[1] != n_inputs:
            raise ValueError(f"u_list has shape {arr.shape}, expected second dim n_inputs={n_inputs}.")
        return [arr[i].reshape(n_inputs) for i in range(arr.shape[0])]

    raise ValueError("u_list must be scalar, 1D, or 2D array-like.")


@register_task_kind
class ClassificationTaskKind(TaskKindBase):
    """
    Classification task: map initial condition -> desired output class trajectory.

    Required params
    --------------
    ic_r_maps:
        List of (ic_values_block, r_value) where:
          - ic_values_block is List[List[float]] of ICs
          - r_value is List[float] (class vector), repeated for each IC in the block
    u_list:
        Constant input(s) applied during simulation (usually a single value for 1-input CRNs)
    t_f, n_t:
        Time horizon definition.

    Reward
    ------
    dynamic_tracking_error(state, u_list, x0_list, time_horizon, r_list, w, ...)
    """

    kind = "classification"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "ic_r_maps": "List[(ic_values_block, r_value)]",
                "u_list": "List[float] / List[List[float]] / np.ndarray",
                "t_f": "float",
                "n_t": "int",
            },
            "optional": {
                "weights": "weights spec for build_weights (default 'uniform')",
                "norm": "int (default 1)",
                "relative": "bool (default False)",
                "LARGE_NUMBER": "float (default 1e4)",
            },
        }

    # REQUIRED by TaskKindBase in your RL4CRN version (abstract method)
    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        """
        Fallback u_list if user does not provide one.
        For 1-input CRNs, defaults to [1.0]. For multi-input, defaults to all-ones.
        """
        n_inputs = int(task.n_inputs or 1)
        return [np.ones(n_inputs, dtype=np.float32)]

    def validate(self, task: TaskSpec) -> None:
        ic_r_maps = overrides_get(task, {}, "ic_r_maps", fallback_attr="ic_r_maps")
        if ic_r_maps is None or not isinstance(ic_r_maps, list) or len(ic_r_maps) == 0:
            raise ValueError("classification requires params['ic_r_maps'] as a non-empty list.")

        for pair in ic_r_maps:
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise ValueError("Each ic_r_maps element must be (ic_values_block, r_value).")
            ic_values_block, r_value = pair
            if not isinstance(ic_values_block, (list, tuple)) or len(ic_values_block) == 0:
                raise ValueError("Each ic_values_block must be a non-empty list.")
            _ = np.asarray(ic_values_block, dtype=np.float32)
            _ = np.asarray(r_value, dtype=np.float32).reshape(-1)

        t_f = overrides_get(task, {}, "t_f", fallback_attr="t_f")
        n_t = overrides_get(task, {}, "n_t", fallback_attr="n_t")
        if t_f is None or float(t_f) <= 0:
            raise ValueError("classification requires params['t_f'] > 0.")
        if n_t is None or int(n_t) < 2:
            raise ValueError("classification requires params['n_t'] >= 2.")

        # u_list is required by your spec, but RL4CRN's make_task may set task.u_list too.
        u_list = overrides_get(task, {}, "u_list", fallback_attr="u_list")
        if u_list is None:
            raise ValueError("classification requires params['u_list'].")

        if task.species_labels is None:
            raise ValueError("classification requires species_labels (passed to make_task).")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        ic_r_maps = overrides_get(task, overrides, "ic_r_maps", fallback_attr="ic_r_maps")
        u_list_raw = overrides_get(task, overrides, "u_list", fallback_attr="u_list")

        t_f = float(overrides_get(task, overrides, "t_f", fallback_attr="t_f"))
        n_t = int(overrides_get(task, overrides, "n_t", fallback_attr="n_t"))
        time_horizon = np.linspace(0.0, t_f, n_t, dtype=np.float32)

        weights_spec = overrides_get(task, overrides, "weights", fallback_attr="weights", default="uniform")
        norm = int(overrides_get(task, overrides, "norm", fallback_attr="norm", default=1))
        relative = bool(overrides_get(task, overrides, "relative", fallback_attr="relative", default=False))

        ic_obj, r_list = _build_ic_and_r_list_from_ic_r_maps(task.species_labels, ic_r_maps)

        q = int(np.asarray(r_list[0]).size)
        w = build_weights(q=q, n_t=n_t, w_spec=weights_spec)

        def reward_fn(state: Any):
            n_inputs = int(getattr(state, "num_inputs", None) or task.n_inputs or 1)
            u_list_local = _normalize_u_list(u_list_raw, n_inputs=n_inputs)

            x0_list = ic_obj.get_ic(state)
            if len(x0_list) != len(r_list):
                raise ValueError(
                    f"IC/target mismatch: {len(x0_list)} ICs vs {len(r_list)} targets. "
                    "Check ic_r_maps."
                )

            out = dynamic_tracking_error(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                r_list,
                w,
                norm=norm,
                relative=relative,
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

            state.last_task_info["task_kind"] = "classification"
            state.last_task_info["u_list"] = u_list_local
            state.last_task_info["n_examples"] = len(x0_list)
            return out

        return reward_fn
