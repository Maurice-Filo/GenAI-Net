from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
from RL4CRN.utils.target_fn import build_r_list_from_target
import numpy as np

@register_task_kind
class SSATrackingTaskKind(TaskKindBase):
    """SSA tracking with named-expression targets (or constant)."""
    kind = "ssa_tracking"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "target": "float OR callable with named args (recommended)",
                "u_values": "List[float] values used for default grid inputs",
            },
            "optional": {
                "n_inputs": "int (defaults to template_crn.num_inputs)",
                "t_f": "float (default 100.0)",
                "n_t": "int (default 1000)",
                "ic": "IC spec",
                "weights": "weights spec",
                "u_list": "explicit list of inputs (overrides defaults)",
                "u_spec": "('custom'|'grid'|'linspace', ...) escape hatch",
                "n_trajectories": "int (default 256)",
                "max_threads": "int (default 1024)",
                "norm": "int (default 1)",
                "LARGE_NUMBER": "float (default 1e4)",
                "LARGE_PENALTY": "float (default 1e4)",
            },
            "notes": (
                "Default u_list is cartesian product over u_values repeated n_inputs. "
                "Callable target arg names resolve via input_idx_dict/species_idx_dict."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if u_values is None:
            raise ValueError("ssa_tracking default_u_list requires params['u_values'].")
        if task.n_inputs is None:
            raise ValueError("ssa_tracking default_u_list requires task.n_inputs.")
        return [np.asarray(u, dtype=np.float32) for u in product(list(u_values), repeat=int(task.n_inputs))]

    def build_weights(self, task: TaskSpec, overrides: Dict[str, Any]) -> np.ndarray:
        weights_spec = overrides.get("weights_spec", task.weights_spec)
        return build_weights(q=1, n_t=task.n_t, w_spec=weights_spec)

    def validate(self, task: TaskSpec) -> None:
        target = overrides_get(task, {}, "target", fallback_attr="target")
        if target is None:
            raise ValueError("ssa_tracking task requires params['target'].")
        if not (callable(target) or isinstance(target, (int, float))):
            raise ValueError("ssa_tracking target must be a float/int or a callable with named args.")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        from RL4CRN.rewards.stochastic import dynamic_tracking_error_SSA

        target = overrides_get(task, overrides, "target", fallback_attr="target")
        if target is None:
            raise ValueError("ssa_tracking task requires target.")

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)
        w = overrides.get("weights", None) or self.build_weights(task, overrides)

        n_trajectories = int(overrides_get(task, overrides, "n_trajectories", fallback_attr="n_trajectories", default=task.n_trajectories))
        max_threads = int(overrides_get(task, overrides, "max_threads", fallback_attr="max_threads", default=task.max_threads))

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)

            r_list = build_r_list_from_target(
                target=target,
                template_crn=task.template_crn,
                u_list=u_list_local,
                x0_list=x0_list,
                expand_over_ic=True,
                q=1,
            )

            return dynamic_tracking_error_SSA(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                r_list,
                w,
                n_trajectories=n_trajectories,
                max_threads=max_threads,
                norm=task.norm,
                relative=False,
                LARGE_NUMBER=task.LARGE_NUMBER,
                LARGE_PENALTY=task.LARGE_PENALTY,
            )

        return reward_fn