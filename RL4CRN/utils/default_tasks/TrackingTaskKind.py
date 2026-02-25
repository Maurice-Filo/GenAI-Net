from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
from RL4CRN.utils.target_fn import build_r_list_from_target
import numpy as np
from RL4CRN.rewards.deterministic import dynamic_tracking_error

@register_task_kind
class TrackingTaskKind(TaskKindBase):
    """Deterministic tracking with named-expression targets (or constant)."""
    kind = "tracking"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "target": "float OR callable with named args (recommended)",
                "u_values": "List[float] values used for default grid inputs",
            },
            "optional": {
                "n_inputs": "int (defaults to template_crn.num_inputs)",
                "t_f": "float",
                "n_t": "int",
                "ic": "IC spec",
                "weights": "weights spec",
                "u_list": "explicit u_list",
                "u_spec": "('custom'|'grid'|'linspace', ...) escape hatch",
                "norm": "int (default 1)",
                "LARGE_NUMBER": "float (default 1e4)",
            },
            "notes": (
                "Default u_list is cartesian product over u_values repeated n_inputs. "
                "If target is callable, its argument names are resolved via "
                "template_crn.input_idx_dict and template_crn.species_idx_dict."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if u_values is None:
            raise ValueError("tracking default_u_list requires params['u_values'].")
        if task.n_inputs is None:
            raise ValueError("tracking default_u_list requires task.n_inputs.")
        return [
            np.asarray(u, dtype=np.float32)
            for u in product(list(u_values), repeat=int(task.n_inputs))
        ]

    def build_weights(self, task: TaskSpec, overrides: Dict[str, Any]) -> np.ndarray:
        weights_spec = overrides.get("weights_spec", task.weights_spec)
        return build_weights(q=1, n_t=task.n_t, w_spec=weights_spec)

    def validate(self, task: TaskSpec) -> None:
        target = overrides_get(task, {}, "target", fallback_attr="target")
        if target is None:
            raise ValueError("tracking task requires params['target'].")
        if not (callable(target) or isinstance(target, (int, float))):
            raise ValueError("tracking target must be a float/int or a callable with named args.")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        

        target = overrides_get(task, overrides, "target", fallback_attr="target")
        if target is None:
            raise ValueError("tracking task requires target.")

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)
        w = overrides.get("weights", None) or self.build_weights(task, overrides)

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)

            # NEW: r_list built via helper (constant or callable named-expression)
            r_list = build_r_list_from_target(
                target=target,
                template_crn=task.template_crn,
                u_list=u_list_local,
                x0_list=x0_list,
                expand_over_ic=True,  # safest for tracking-style losses
                q=1,
            )

            return dynamic_tracking_error(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                r_list,
                w,
                norm=task.norm,
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

        return reward_fn