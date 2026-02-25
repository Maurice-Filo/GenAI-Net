from itertools import product
from typing import Any, Callable, Dict, List, Tuple
from RL4CRN.utils.input_interface import overrides_get, TaskSpec, TaskKindBase, register_task_kind, build_weights
from RL4CRN.utils.target_fn import build_r_list_from_target
import numpy as np
from RL4CRN.rewards.deterministic import oscillation_error

@register_task_kind
class OscillatorFreqTaskKind(TaskKindBase):
    """Oscillation error with frequency targets."""
    kind = "oscillator_freq"

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "osc_w": "List[float] oscillation error weights",
                "u_values": "List[float] values used for default grid inputs",
                "freq_target": "float OR callable with named args (recommended)",
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
                "freq_target is evaluated per input scenario (and may also depend on ICs via named args). "
                "Callable arg names resolve via input_idx_dict/species_idx_dict."
            ),
        }

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        u_values = overrides_get(task, {}, "u_values", fallback_attr="u_values")
        if u_values is None:
            raise ValueError("oscillator_freq default_u_list requires params['u_values'].")
        if task.n_inputs is None:
            raise ValueError("oscillator_freq default_u_list requires task.n_inputs.")
        return [np.asarray(u, dtype=np.float32) for u in product(list(u_values), repeat=int(task.n_inputs))]

    def validate(self, task: TaskSpec) -> None:
        osc_w = overrides_get(task, {}, "osc_w", fallback_attr="osc_w")
        if osc_w is None:
            raise ValueError("oscillator_freq requires params['osc_w'].")

        freq_target = overrides_get(task, {}, "freq_target", fallback_attr=None)
        if freq_target is None:
            raise ValueError("oscillator_freq requires params['freq_target'] (float or callable).")
        if not (callable(freq_target) or isinstance(freq_target, (int, float))):
            raise ValueError("oscillator_freq freq_target must be a float/int or a callable with named args.")

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        from RL4CRN.rewards.deterministic import oscillation_error

        osc_w = overrides_get(task, overrides, "osc_w", fallback_attr="osc_w")
        t0 = overrides_get(task, overrides, "t0", fallback_attr="t0", default=task.t0)
        freq_target = overrides_get(task, overrides, "freq_target", fallback_attr=None)

        time_horizon = self.build_time_horizon(task)
        u_list_local = self.build_u_list(task, overrides)
        ic_obj = self.build_ic(task, overrides)

        def reward_fn(state: Any):
            x0_list = ic_obj.get_ic(state)
            if len(x0_list) == 0:
                raise ValueError("oscillator_freq: ic produced empty x0_list.")

            r_list = build_r_list_from_target(
                target=freq_target,
                template_crn=task.template_crn,
                u_list=u_list_local,
                x0_list=[x0_list[0]],
                expand_over_ic=False,
                q=1,
            )
            f_list = [np.asarray(r[0], dtype=np.float32).reshape(1) for r in r_list]

            return oscillation_error(
                state,
                u_list_local,
                x0_list,
                time_horizon,
                f_list=f_list,
                mean_list=None,
                w=osc_w,
                t0=float(t0),
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

        return reward_fn

