from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
import numpy as np

@register_task_kind
class LogicTaskKind(TaskKindBase):
    """Logic task: target is logic_fn(u) in {0,1}."""
    kind = "logic"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "logic_fn": "Callable[[np.ndarray], bool] (truth table target)",
            },
            "optional": {
                "n_inputs": "int (defaults to template_crn.num_inputs)",
                "t_f": "float (default 100.0)",
                "n_t": "int (default 1000)",
                "ic": "IC spec, e.g. ('constant', 0.01) or 'zero'",
                "weights": "weights spec, e.g. 'transient' | 'uniform' | 'steady_state' | ('custom', array)",
                "u_list": "explicit list of inputs (overrides defaults)",
                "u_spec": "('custom'| 'grid'| 'linspace', ...) escape hatch",
                "norm": "int (default 1)",
                "LARGE_NUMBER": "float (default 1e4)",
            },
            "notes": "If neither u_list nor u_spec is provided, defaults to full truth-table grid over {0,1}^n_inputs.",
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        if task.n_inputs is None:
            raise ValueError("logic default_u_list requires task.n_inputs.")
        return [
            np.asarray(u, dtype=np.float32)
            for u in product([0.0, 1.0], repeat=int(task.n_inputs))
        ]

    def build_weights(self, task: TaskSpec, overrides: Dict[str, Any]) -> np.ndarray:
        weights_spec = overrides.get("weights_spec", task.weights_spec)
        return build_weights(q=1, n_t=task.n_t, w_spec=weights_spec)

    def validate(self, task: TaskSpec) -> None:
        logic_fn = overrides_get(task, {}, "logic_fn", fallback_attr="logic_fn")
        if logic_fn is None:
            raise ValueError("logic task requires logic_fn (task.params['logic_fn'] recommended).")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        from RL4CRN.rewards.deterministic import dynamic_tracking_error

        logic_fn = overrides_get(task, overrides, "logic_fn", fallback_attr="logic_fn")
        if logic_fn is None:
            raise ValueError("logic task requires logic_fn.")

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)
        w = overrides.get("weights", None) or self.build_weights(task, overrides)

        # per-u targets (truth table)
        r_list_u = [np.array([float(bool(logic_fn(u)))], dtype=np.float32) for u in u_list_local]

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)
            # dynamic_tracking_error may accept either per-u or expanded; keep your current behavior:
            out = dynamic_tracking_error(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                r_list_u,
                w,
                norm=task.norm,
                LARGE_NUMBER=task.LARGE_NUMBER,
            )
            state.last_task_info["u_list"] = u_list_local
            state.last_task_info["logic_fn"] = logic_fn
            return out

        return reward_fn