from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
from RL4CRN.utils.target_fn import build_r_list_from_target
import numpy as np
from RL4CRN.rewards.deterministic import oscillation_error

@register_task_kind
class OscillatorMeanTaskKind(TaskKindBase):
    """Oscillation error with mean targets."""
    kind = "oscillator_mean"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "osc_w": "List[float] oscillation error weights (passed to oscillation_error)",
                "u_values": "List[float] values used for default grid inputs",
                "mean_target": "float OR callable with named args (recommended)",
            },
            "optional": {
                "n_inputs": "int (defaults to template_crn.num_inputs)",
                "t_f": "float (default 100.0)",
                "n_t": "int (default 1000)",
                "t0": "float oscillation start time (default 20.0)",
                "ic": "IC spec",
                "u_list": "explicit list of inputs (overrides defaults)",
                "u_spec": "('custom'|'grid'|'linspace', ...) escape hatch",
                "LARGE_NUMBER": "float (default 1e4)",
            },
            "notes": (
                "mean_target is evaluated per input scenario (and may also depend on ICs via named args). "
                "Callable arg names resolve via input_idx_dict/species_idx_dict."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if u_values is None:
            raise ValueError("oscillator_mean default_u_list requires params['u_values'].")
        if task.n_inputs is None:
            raise ValueError("oscillator_mean default_u_list requires task.n_inputs.")
        return [np.asarray(u, dtype=np.float32) for u in product(list(u_values), repeat=int(task.n_inputs))]

    def validate(self, task: TaskSpec) -> None:
        osc_w = overrides_get(task, {}, "osc_w", fallback_attr="osc_w")
        if osc_w is None:
            raise ValueError("oscillator_mean requires params['osc_w'].")

        mean_target = overrides_get(task, {}, "mean_target", fallback_attr=None)
        if mean_target is None:
            raise ValueError("oscillator_mean requires params['mean_target'] (float or callable).")
        if not (callable(mean_target) or isinstance(mean_target, (int, float))):
            raise ValueError("oscillator_mean mean_target must be a float/int or a callable with named args.")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:

        osc_w = overrides_get(task, overrides, "osc_w", fallback_attr="osc_w")
        t0 = overrides_get(task, overrides, "t0", fallback_attr="t0", default=task.t0)
        mean_target = overrides_get(task, overrides, "mean_target", fallback_attr=None)

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)

            # Use helper to get scalar targets; we only need one scalar per u,
            # but if mean_target depends on IC, we choose the first IC by default.
            # (If you want per-IC oscillator targets, we can extend oscillation_error’s API later.)
            if len(x0_list) == 0:
                raise ValueError("oscillator_mean: ic produced empty x0_list.")

            r_list = build_r_list_from_target(
                target=mean_target,
                template_crn=task.template_crn,
                u_list=u_list_local,
                x0_list=[x0_list[0]],     # evaluate using first IC
                expand_over_ic=False,      # one per u
                q=1,
            )
            mean_list = [np.asarray(r[0], dtype=np.float32).reshape(1) for r in r_list]

            return oscillation_error(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                f_list=None,
                mean_list=mean_list,
                w=osc_w,
                t0=float(t0),
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

        return reward_fn

