from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
from RL4CRN.utils.target_fn import build_r_list_from_target
import numpy as np

@register_task_kind
class DoseResponseTaskKind(TaskKindBase):
    """Dose-response tracking: target is target(u) (named-arg callable) or legacy target_fn(u0)."""
    kind = "dose_response"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                # Prefer the new unified interface:
                "target": "float OR callable with named args (recommended)",
                "dose_range": "Tuple[u_min, u_max, n]",
            },
            "optional": {
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
                "Default u_list is 1D linspace over dose_range with vectors shape (1,). "
                "If target is callable, its arg names are resolved via input_idx_dict/species_idx_dict."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        dose_range = overrides_get(task, {}, "dose_range", fallback_attr="dose_range")
        if dose_range is None:
            raise ValueError("dose_response default_u_list requires params['dose_range'] = (u_min,u_max,n).")
        u_min, u_max, n = dose_range
        return [
            np.asarray([u], dtype=np.float32)
            for u in np.linspace(float(u_min), float(u_max), int(n), dtype=np.float32)
        ]

    def build_weights(self, task: TaskSpec, overrides: Dict[str, Any]) -> np.ndarray:
        weights_spec = overrides.get("weights_spec", task.weights_spec)
        return build_weights(q=1, n_t=task.n_t, w_spec=weights_spec)

    def validate(self, task: TaskSpec) -> None:
        target = overrides_get(task, {}, "target", fallback_attr="target")

        if target is None:
            raise ValueError("dose_response requires params['target'].")

        if target is not None and not (callable(target) or isinstance(target, (int, float))):
            raise ValueError("dose_response target must be a float/int or a callable with named args.")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        from RL4CRN.rewards.deterministic import dynamic_tracking_error

        target = overrides_get(task, overrides, "target", fallback_attr="target")

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)
        w = overrides.get("weights", None) or self.build_weights(task, overrides)

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
