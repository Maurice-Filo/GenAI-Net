from typing import Any, Callable, Dict, List

import numpy as np

from RL4CRN.utils.input_interface import (
    TaskKindBase,
    TaskSpec,
    overrides_get,
    register_task_kind,
)


RNE_OSCILLATOR_TIME = np.linspace(0.0, 1.25, 11, dtype=np.float32)
RNE_OSCILLATOR_TARGET = np.asarray(
    [5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0, 30.0, 5.0],
    dtype=np.float32,
)


@register_task_kind
class RNEOscillatorTraceTaskKind(TaskKindBase):
    """RNE nominal oscillator trace matching task.

    ReactionNetworkEvolution.jl's default oscillator setting evolves autonomous
    3-species CRNs against a fixed alternating trace. For each candidate network,
    the original objective computes the L1 error between that trace and every
    species trajectory, then uses the best-matching species.
    """

    kind = "rne_oscillator_trace"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "time_horizon": "1D array of target times; defaults to linspace(0, 1.25, 11)",
                "target_trace": "1D array; defaults to [5, 30, ..., 30, 5]",
                "ic": "IC spec; RNE default is ('values', [[1.0, 5.0, 9.0]])",
                "u_list": "explicit input scenarios; default is one empty input vector",
                "normalize": "bool; divide by number of target points (default False)",
                "LARGE_NUMBER": "float divergence threshold (default 1e4)",
            },
            "notes": (
                "The loss is min_j sum_i |x_j(t_i) - y_i|, matching the inverse of "
                "RNE's default oscillator fitness. The best species is recorded in "
                "last_task_info['best_species_label']."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        if int(task.n_inputs) == 0:
            return [np.asarray([], dtype=np.float32)]
        return [np.zeros(int(task.n_inputs), dtype=np.float32)]

    def validate(self, task: TaskSpec) -> None:
        target = overrides_get(task, {}, "target_trace", fallback_attr=None, default=RNE_OSCILLATOR_TARGET)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        time = self.build_time_horizon(task)

        if target.ndim != 1 or time.ndim != 1:
            raise ValueError("rne_oscillator_trace target_trace and time_horizon must be 1D.")
        if len(target) != len(time):
            raise ValueError("rne_oscillator_trace target_trace length must match time_horizon length.")
        if len(time) < 2:
            raise ValueError("rne_oscillator_trace requires at least two time points.")
        if not np.all(np.diff(time) > 0):
            raise ValueError("rne_oscillator_trace time_horizon must be strictly increasing.")

    def build_time_horizon(self, task: TaskSpec) -> np.ndarray:
        if task.time_horizon is not None and np.asarray(task.time_horizon).size > 0:
            return np.asarray(task.time_horizon, dtype=np.float32).reshape(-1)
        return RNE_OSCILLATOR_TIME.copy()

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        target = np.asarray(
            overrides_get(task, overrides, "target_trace", fallback_attr=None, default=RNE_OSCILLATOR_TARGET),
            dtype=np.float32,
        ).reshape(-1)
        normalize = bool(overrides_get(task, overrides, "normalize", fallback_attr=None, default=False))
        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)
        reward_type = self.kind

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)
            t, x_list, _y_list, last_task_info = state.transient_response(
                u_list_local,
                x0_list,
                time_horizon,
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

            if len(t) != len(time_horizon) or len(x_list) == 0:
                loss = float(task.LARGE_PENALTY)
                state.last_task_info["reward"] = loss
                state.last_task_info["reward type"] = reward_type
                return loss, state.last_task_info

            best_loss = float("inf")
            best_species_index = None
            best_initial_condition_index = None
            errors_by_species = []

            for ic_index, x in enumerate(x_list):
                if np.any(~np.isfinite(x)) or np.any(np.abs(x) >= task.LARGE_NUMBER - 1):
                    continue
                if x.shape[1] != len(target):
                    continue

                species_errors = np.sum(np.abs(x - target[None, :]), axis=1)
                errors_by_species.append(species_errors.astype(float).tolist())
                local_index = int(np.argmin(species_errors))
                local_loss = float(species_errors[local_index])
                if local_loss < best_loss:
                    best_loss = local_loss
                    best_species_index = local_index
                    best_initial_condition_index = ic_index

            if not np.isfinite(best_loss):
                best_loss = float(task.LARGE_PENALTY)

            if normalize:
                best_loss /= float(len(target))

            species_labels = getattr(state, "species_labels", [])
            best_species_label = (
                species_labels[best_species_index]
                if best_species_index is not None and best_species_index < len(species_labels)
                else None
            )

            state.last_task_info["reward"] = best_loss
            state.last_task_info["reward type"] = reward_type
            state.last_task_info["target_time"] = time_horizon
            state.last_task_info["target_trace"] = target
            state.last_task_info["errors_by_species"] = errors_by_species
            state.last_task_info["best_species_index"] = best_species_index
            state.last_task_info["best_species_label"] = best_species_label
            state.last_task_info["best_initial_condition_index"] = best_initial_condition_index
            state.last_task_info["initial_conditions"] = x0_list
            return best_loss, state.last_task_info

        return reward_fn
