from typing import Any, Callable, Dict, List
import numpy as np

from RL4CRN.utils.input_interface import (
    overrides_get,
    TaskSpec,
    TaskKindBase,
    register_task_kind,
)
from RL4CRN.rewards.deterministic import dynamic_dataset_fit_error_projected


@register_task_kind
class DataFitTaskKind(TaskKindBase):
    """
    Deterministic data fitting against observed trajectories.

    Required params:
        dataset: List[dict], where each dict has:
            - 'u': input vector, shape (p,)
            - 't_obs': observation times, shape (T_obs,)
            - 'y_obs': observed outputs, shape (q, T_obs)
          Optional:
            - 'x0': initial condition, shape (n,)

    Optional params:
        t_f: float
        n_t: int
        ic: IC spec (used only for samples missing x0)
        norm: int (1 for MAE, 2 for MSE; default 2)
        relative: bool (default False)
        output_weights: np.ndarray shape (q,)
        LARGE_NUMBER: float (default 1e4)
    """
    kind = "datafit"

    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        """
        For data fitting, inputs come from the dataset itself.
        We return the unique u vectors found in the dataset so the abstract
        TaskKindBase contract is satisfied.
        """
        dataset = overrides_get(task, {}, "dataset", fallback_attr="dataset")
        if dataset is None or len(dataset) == 0:
            raise ValueError("datafit default_u_list requires a non-empty dataset.")

        u_list = []
        seen = set()

        for sample in dataset:
            if "u" not in sample:
                raise ValueError("Each dataset sample must contain key 'u'.")
            u = np.asarray(sample["u"], dtype=np.float32).reshape(-1)
            key = tuple(u.tolist())
            if key not in seen:
                seen.add(key)
                u_list.append(u)

        return u_list

    @staticmethod
    def help() -> Dict[str, Any]:
        return {
            "required": {
                "dataset": (
                    "List[dict] with keys "
                    "{'u', 't_obs', 'y_obs'} and optional 'x0'"
                ),
            },
            "optional": {
                "t_f": "float",
                "n_t": "int",
                "ic": "IC spec used if a sample does not provide x0",
                "norm": "int: 1 for MAE, 2 for MSE (default 2)",
                "relative": "bool (default False)",
                "output_weights": "np.ndarray with shape (q,)",
                "LARGE_NUMBER": "float (default 1e4)",
            },
            "notes": (
                "The CRN is simulated on task time_horizon, and each predicted "
                "trajectory is compared against observed data using exact "
                "piecewise-linear MAE/MSE over time."
            ),
        }

    def validate(self, task: TaskSpec) -> None:
        dataset = overrides_get(task, {}, "dataset", fallback_attr="dataset")
        if dataset is None or len(dataset) == 0:
            raise ValueError("datafit task requires params['dataset'] to be non-empty.")

        for i, sample in enumerate(dataset):
            if not isinstance(sample, dict):
                raise ValueError(f"dataset[{i}] must be a dict.")

            for key in ("u", "t_obs", "y_obs"):
                if key not in sample:
                    raise ValueError(f"dataset[{i}] missing required key '{key}'.")

            u = np.asarray(sample["u"])
            t_obs = np.asarray(sample["t_obs"])
            y_obs = np.asarray(sample["y_obs"])

            if u.ndim != 1:
                raise ValueError(f"dataset[{i}]['u'] must be a 1D array.")
            if t_obs.ndim != 1:
                raise ValueError(f"dataset[{i}]['t_obs'] must be a 1D array.")
            if y_obs.ndim != 2:
                raise ValueError(f"dataset[{i}]['y_obs'] must have shape (q, T_obs).")
            if y_obs.shape[1] != t_obs.shape[0]:
                raise ValueError(
                    f"dataset[{i}]['y_obs'].shape[1] must equal len(dataset[{i}]['t_obs'])."
                )
            if len(t_obs) < 2:
                raise ValueError(f"dataset[{i}]['t_obs'] must contain at least 2 time points.")
            if not np.all(np.diff(t_obs) > 0):
                raise ValueError(f"dataset[{i}]['t_obs'] must be strictly increasing.")

            if "x0" in sample:
                x0 = np.asarray(sample["x0"])
                if x0.ndim != 1:
                    raise ValueError(f"dataset[{i}]['x0'] must be a 1D array.")
        
        task.u_list = self.default_u_list(task)

    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        dataset = overrides_get(task, overrides, "dataset", fallback_attr="dataset")
        if dataset is None:
            raise ValueError("datafit task requires dataset.")

        time_horizon = self.build_time_horizon(task)
        ic_obj = self.build_ic(task, overrides)

        norm = overrides.get("norm", getattr(task, "norm", 2))
        relative = overrides.get("relative", getattr(task, "relative", False))
        output_weights = overrides.get(
            "output_weights",
            getattr(task, "output_weights", None),
        )

        def reward_fn(state: Any):
            # Build fallback x0 values for samples that do not explicitly provide x0
            dataset_local = []
            missing_x0_indices = [i for i, sample in enumerate(dataset) if "x0" not in sample]

            fallback_x0_list = None
            if len(missing_x0_indices) > 0:
                fallback_x0_list = ic_obj.get_ic(state)

                if len(fallback_x0_list) == 1:
                    fallback_x0_list = fallback_x0_list * len(missing_x0_indices)
                elif len(fallback_x0_list) != len(missing_x0_indices):
                    raise ValueError(
                        "IC spec must provide either one x0 or one x0 per dataset sample missing x0."
                    )

            miss_ptr = 0
            for sample in dataset:
                sample_local = dict(sample)
                if "x0" not in sample_local:
                    sample_local["x0"] = fallback_x0_list[miss_ptr]
                    miss_ptr += 1
                dataset_local.append(sample_local)

            return dynamic_dataset_fit_error_projected(
                crn=state,
                dataset=dataset_local,
                time_horizon=time_horizon,
                norm=norm,
                relative=relative,
                output_weights=output_weights,
                LARGE_NUMBER=task.LARGE_NUMBER,
            )

        return reward_fn