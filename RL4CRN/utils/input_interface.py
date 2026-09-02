# RL4CRN/utils/input_interface.py
"""
User-facing input interface utilities for GenAI-Net / RL4CRN tutorials.

This module provides:
- lightweight configuration objects with sensible defaults
- a configurator to apply presets and overrides
- a session builder that wires together task/template/library/env/interfaces/policy/agent
- a trainer that supports chunked training, early stopping (Ctrl+C), and save/load checkpoints

The goal is to make tutorial notebooks trivial to run, while keeping all advanced knobs
discoverable via config inspection.
"""
from __future__ import annotations

import pprint
import textwrap
import random
import json
import time

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, asdict
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import os

import cloudpickle
import numpy as np
import torch

from RL4CRN.iocrns.reaction_library import ReactionLibrary
from RL4CRN.iocrns.iocrn import IOCRN
from RL4CRN.utils.forbidden_topologies import (
    ForbiddenTopologyArchive,
    reward_with_forbidden_topologies,
    topology_signature_key,
)
from RL4CRN.utils.hall_of_fame import HallOfFame
from RL4CRN.utils.parameter_optimization import optimize_crn_parameters_ipopt
from RL4CRN.utils.results_database import (
    ResultsDatabase,
    classify_hof_provenance,
    serialize_crn,
)
from abc import ABC, abstractmethod


# ----------------------------
# Small general utilities
# ----------------------------

def get_device(prefer: str = "auto") -> str: # CHECKED ___ OK
    """Select a torch device string.

    Args:
        prefer: Device preference. Options:
            - "auto": choose "cuda" if available, else "cpu"
            - "cpu": force CPU
            - "cuda": force CUDA (raises if not available)

    Returns:
        Device string ("cpu" or "cuda").

    Raises:
        RuntimeError: If prefer="cuda" but CUDA is not available.
        ValueError: If prefer is not one of {"auto", "cpu", "cuda"}.
    """
    prefer = prefer.lower().strip()
    if prefer not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown prefer={prefer!r}. Use 'auto', 'cpu', or 'cuda'.")

    if prefer == "cpu":
        return "cpu"

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None: # CHECKED ___ OK
    """Seed common RNG sources for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Task specification + builders
# ----------------------------

VectorLogic = Callable[[np.ndarray], Union[bool, np.bool_]]


@dataclass
class TaskSpec:
    """Fully materialized task description used by environments.

    Attributes:
        template_crn: Compiled IOCRN template.
        library_components: Tuple (library, M, K, masks).
        species_labels: Species labels used by the task.
        kind: Task kind (e.g., "logic", "tracking", "oscillator_mean", ...).

        # Time
        t_f: Final simulation time.
        n_t: Number of time points.
        time_horizon: 1D array of time points (float32).

        # Inputs
        n_inputs: Number of input channels.
        u_values: Values for grid tasks (tracking/oscillator/SSA).
        dose_range: (u_min, u_max, n) for "dose_response".
        u_spec: Optional input generation spec.
        u_list: List of input vectors (each shape (p,), float32).

        # IC
        ic_spec: IC specification used to build the IC object.
        ic: RL4CRN IC object.

        # Weights / targets
        weights_spec: Weight spec used to build the weight matrix (when applicable).
        weights: Weight matrix (when applicable).
        target: Target spec for tracking/SSA tasks.
        logic_fn: Boolean logic function for "logic".
        target_fn: Target function for dose response.

        # Oscillator knobs
        osc_w: Oscillation error weights.
        t0: Oscillation error start time.

        # SSA knobs
        n_trajectories: SSA number of trajectories.
        max_threads: SSA max threads.
        cv_weight: Robust SSA CV weight.
        rpa_weight: Robust SSA RPA weight.
        relative: Whether to use relative error in SSA rewards.

        # Reward constants
        norm: Norm used in tracking losses.
        LARGE_NUMBER: Large penalty scalar used by deterministic rewards.
        LARGE_PENALTY: Large penalty scalar used by SSA rewards (when applicable).

        compute_reward: Reward callable built from this TaskSpec.

        params: Task-kind-specific parameters (forward-compatible extension point).
    """
    template_crn: IOCRN
    library_components: tuple[ReactionLibrary, int, int, dict[str, Any]]
    species_labels: List[str]
    kind: str

    # Time
    t_f: float = 100.0
    n_t: int = 1000
    time_horizon: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))

    # Inputs
    n_inputs: Optional[int] = None
    u_values: Optional[List[float]] = None
    dose_range: Optional[Tuple[float, float, int]] = None
    u_spec: Optional[tuple] = None
    u_list: List[np.ndarray] = field(default_factory=list)

    # IC
    ic_spec: Union[str, tuple] = "zero"
    ic: Any = None

    # Weights / targets
    weights_spec: Union[str, tuple] = "transient"
    weights: Optional[np.ndarray] = None
    target: Optional[Union[str, float]] = None
    logic_fn: Optional[VectorLogic] = None
    target_fn: Optional[Callable[[float], float]] = None

    # Oscillator knobs
    osc_w: Optional[List[float]] = None
    t0: float = 20.0

    # SSA knobs
    n_trajectories: int = 256
    max_threads: int = 1024
    cv_weight: float = 1.0
    rpa_weight: float = 1.0
    relative: bool = False

    # Reward constants
    norm: int = 1
    LARGE_NUMBER: float = 1e4
    LARGE_PENALTY: float = 1e4

    # Built callable
    compute_reward: Optional[Callable[[Any], Union[float, Tuple[float, Dict[str, Any]]]]] = None

    # task specific information (rendering)
    render_mode: Optional[dict] = "transients"

    # Task-kind specific parameters (forward-compatible extension point)
    params: Dict[str, Any] = field(default_factory=dict)



def make_time_grid(t_f: float = 100.0, n_t: int = 1000) -> np.ndarray: # CHECKED ___ OK
    """Create a uniform time grid.

    Args:
        t_f: Final time.
        n_t: Number of time points.

    Returns:
        Time grid as float32 array of shape (n_t,).
    """
    return np.linspace(0.0, t_f, n_t, dtype=np.float32)


def build_u_list(
    kind: str,
    *,
    n_inputs: Optional[int] = None,
    u_values: Optional[List[float]] = None,
    dose_range: Optional[Tuple[float, float, int]] = None,
    u_spec: Optional[tuple] = None,
) -> List[np.ndarray]: # CHECKED ___ OK
    """Construct a list of inputs for a task kind.

    Args:
        kind: Task kind.
        n_inputs: Number of input channels.
        u_values: Values to enumerate for grid tasks.
        dose_range: (u_min, u_max, n) for "dose_response" tasks.
        u_spec: Optional escape hatch specifying exact input generation:
            - ("custom", u_list)
            - ("grid", values)
            - ("linspace", u_min, u_max, n)

    Returns:
        List of input vectors (float32 arrays).
    """
    if u_spec is not None:
        tag, *args = u_spec
        if tag == "custom": # take input list as-is
            return args[0]
        if tag == "grid": # cartesian product over specified values
            values = args[0]
            dim = n_inputs
            if dim is None:
                raise ValueError("u_spec=('grid', ...) needs n_inputs.")
            return [np.array(u, dtype=np.float32) for u in product(values, repeat=dim)]
        if tag == "linspace": # 1D linspace inputs
            u_min, u_max, n = args
            return [np.array([u], dtype=np.float32) for u in np.linspace(u_min, u_max, n)]
        raise ValueError(f"Unknown u_spec: {u_spec}")

    if kind == "logic":
        return [np.array(u, dtype=np.float32) for u in product([0.0, 1.0], repeat=n_inputs)]

    if kind in ("tracking", "oscillator_mean", "oscillator_freq", "ssa_tracking", "ssa_robust"):
        return [np.array(u, dtype=np.float32) for u in product(u_values, repeat=n_inputs)]

    if kind == "dose_response":
        u_min, u_max, n = dose_range
        return [np.array([u], dtype=np.float32) for u in np.linspace(u_min, u_max, n)]
    
    raise ValueError(f"Unknown task kind: {kind}")


def build_ic(species_labels: List[str], ic_spec: Union[str, tuple]) -> Any: # CHECKED ___ OK
    """Build an RL4CRN IC object from a compact spec.

    Args:
        species_labels: Species names for the CRN.
        ic_spec: One of:
            - "zero"
            - ("constant", value)
            - ("values", values_2d)

    Returns:
        RL4CRN IC instance.

    Raises:
        ValueError: If ic_spec is unknown.
    """
    from RL4CRN.utils.ic import IC

    if ic_spec == "zero":
        return IC(names=species_labels, values=[[0.0 for _ in species_labels]])

    if isinstance(ic_spec, tuple):
        tag = ic_spec[0]
        if tag == "constant":
            val = float(ic_spec[1])
            return IC(names=species_labels, values=[[val for _ in species_labels]])
        if tag == "values":
            return IC(names=species_labels, values=ic_spec[1])
        
    if ic_spec == "from_ss":
        # special tag indicating that ICs should be set to the steady-state for each input (used by some tasks)
        return "from_ss"

    raise ValueError(f"Unknown ic_spec: {ic_spec}")


def build_weights(q: int, n_t: int, w_spec: Union[str, tuple]) -> np.ndarray: # CHECKED ___ OK
    """Build a weight matrix for tracking losses.

    Args:
        q: Output dimension (usually 1).
        n_t: Number of time points.
        w_spec: One of:
            - "steady_state": weight only last time point
            - "uniform": all ones
            - "transient": bias early/late times
            - ("custom", array_like)

    Returns:
        Weight matrix of shape (q, n_t) float32.

    Raises:
        ValueError: If w_spec is unknown.
    """
    if w_spec == "steady_state":
        w = np.zeros((q, n_t), dtype=np.float32)
        w[:, -1] = float(n_t)
        return w

    if w_spec == "uniform":
        return np.ones((q, n_t), dtype=np.float32)

    if w_spec == "transient":
        w = np.ones(n_t, dtype=np.float32)
        w[(len(w) // 5) * 4:] *= 2.0
        w[: (len(w) // 5)] *= 0.25
        return w[None, :]

    if isinstance(w_spec, tuple) and w_spec[0] == "custom":
        return np.asarray(w_spec[1], dtype=np.float32)

    raise ValueError(f"Unknown w_spec: {w_spec}")




def make_task(
    template_crn: IOCRN,
    library_components: tuple[ReactionLibrary, int, int, dict[str, Any]],
    kind: str,
    species_labels: List[str],
    *,
    params: Optional[Dict[str, Any]] = None,
) -> TaskSpec:
    """Create a TaskSpec from a params dictionary and build its reward callable.

    This is the ONLY public constructor: users pass task knobs via `params`.
    Default interpretation of missing fields is delegated to the TaskKind handler.

    Common params keys (shared across many tasks):
        - "t_f": float
        - "n_t": int
        - "n_inputs": int (defaults to template_crn.num_inputs)
        - "ic": Union[str, tuple]               # e.g. "zero", ("constant", 0.01)
        - "weights": Union[str, tuple]          # e.g. "transient", ("custom", ...)
        - "u_spec": tuple                       # only for special-tag generation
        - "u_list": List[np.ndarray]            # explicit scenarios

    Task-specific keys are documented by TaskKind.help().

    Args:
        template_crn: Compiled IOCRN template.
        library_components: Tuple (library, M, K, masks).
        kind: Task kind string.
        species_labels: Species labels used by the task.
        params: Task configuration dictionary.

    Returns:
        TaskSpec with runtime fields (time_horizon/u_list/ic/weights/compute_reward) populated.

    Raises:
        ValueError: If required parameters are missing or inconsistent.
    """
    params = {} if params is None else dict(params)

    # --- Populate "base" fields from params with safe defaults ---
    n_inputs = params.get("n_inputs", None)
    if n_inputs is None:
        # default to the template's declared input dimension
        n_inputs = int(getattr(template_crn, "num_inputs", 0))

    t_f = float(params.get("t_f", 100.0))
    n_t = int(params.get("n_t", 1000))

    ic_spec = params.get("ic", "zero")
    weights_spec = params.get("weights", "transient")

    # Reward constants (optional overrides)
    norm = int(params.get("norm", 1))
    LARGE_NUMBER = float(params.get("LARGE_NUMBER", 1e4))
    LARGE_PENALTY = float(params.get("LARGE_PENALTY", 1e4))

    # Store any explicit runtime caches if user provides them
    # (TaskKind.build_* will decide whether to respect them)
    time_horizon = params.get("time_horizon", None)
    u_list = params.get("u_list", None)
    u_spec = params.get("u_spec", None)

    task = TaskSpec(
        template_crn=template_crn,
        library_components=library_components,
        species_labels=species_labels,
        kind=kind,
        t_f=t_f,
        n_t=n_t,
        n_inputs=int(n_inputs),
        ic_spec=ic_spec,
        weights_spec=weights_spec,
        norm=norm,
        LARGE_NUMBER=LARGE_NUMBER,
        LARGE_PENALTY=LARGE_PENALTY,
        params=params,
    )

    # Optional runtime caches if provided
    if time_horizon is not None:
        task.time_horizon = np.asarray(time_horizon, dtype=np.float32)

    if u_spec is not None:
        task.u_spec = u_spec

    if u_list is not None:
        task.u_list = [np.asarray(u, dtype=np.float32).reshape(-1) for u in u_list]

    # --- Delegate validation + construction to TaskKind ---
    tk = get_task_kind(kind)
    tk.validate(task)

    task.time_horizon = tk.build_time_horizon(task)
    task.u_list = tk.build_u_list(task, overrides={})
    task.ic = tk.build_ic(task, overrides={})
    task.weights = tk.build_weights(task, overrides={})
    task.compute_reward = tk.make_reward_fn(task, overrides={})

    return task


def make_reward_fn_with_overrides(
    task: TaskSpec,
    *,
    u_list: Optional[List[np.ndarray]] = None,
    ic_spec: Optional[Union[str, tuple]] = None,
    weights_spec: Optional[Union[str, tuple]] = None,
    **kwargs: Any,
) -> Callable[[Any], Union[float, Tuple[float, Dict[str, Any]]]]:
    """Build a reward function for a TaskSpec, optionally overriding conditions.

    This is the single entry point used by training, sampling, resimulation, and load.

    Args:
        task: Base TaskSpec.
        u_list: Optional replacement list of input vectors.
        ic_spec: Optional IC spec override.
        weights_spec: Optional weights spec override.
        **kwargs: Additional task-kind-specific overrides.

    Returns:
        Reward callable accepting a CRN state and returning loss or (loss, info).

    Raises:
        ValueError: If task.kind is unknown or required fields are missing.
    """
    overrides: Dict[str, Any] = dict(kwargs)
    if u_list is not None:
        overrides["u_list"] = u_list
    if ic_spec is not None:
        overrides["ic_spec"] = ic_spec
    if weights_spec is not None:
        overrides["weights_spec"] = weights_spec

    tk = get_task_kind(task.kind)
    tk.validate(task)
    return tk.make_reward_fn(task, overrides=overrides)


# ----------------------------
# Config objects
# ----------------------------

@dataclass
class SolverCfg:
    """Solver configuration.

    Attributes:
        algorithm: Solver name (e.g., "CVODE" or "LSODA").
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    algorithm: str = "CVODE"
    rtol: float = 1e-10
    atol: float = 1e-10


@dataclass
class TrainCfg:
    """Training configuration.

    Attributes:
        epochs: Total number of epochs (you may run in chunks).
        max_added_reactions: Episode length: number of reaction-addition steps.
        render_every: Print progress every N epochs (0 disables).
        hall_of_fame_size: Hall-of-fame capacity in ParallelEnvironments.
        batch_multiplier: Batch size = batch_multiplier * num_cpus (if batch_size is None).
        seed: Random seed for reproducibility.
        n_cpus: CPU count to use. If None, uses os.cpu_count().
        batch_size: If provided, overrides auto batch sizing.
    """
    epochs: int = 300
    max_added_reactions: int = 5
    render_every: int = 10
    hall_of_fame_size: int = 30
    batch_multiplier: int = 10
    seed: int = 0
    n_cpus: Optional[int] = None
    batch_size: Optional[int] = None
    forbidden_topology_m: int = 0
    forbidden_topology_every: int = 5
    forbidden_topology_loss: float = 1e9
    forbidden_topology_start_epoch: int = 0
    forbidden_threshold: float = float("inf")
    forbidden_optimize_with_ipopt: bool = True
    forbidden_ipopt_maxiter: int = 100
    forbidden_ipopt_log_min: float = -18.0
    forbidden_ipopt_log_max: float = 6.0
    forbidden_async: bool = False
    forbidden_optimization_max_evaluations: int = 50
    forbidden_optimization_timeout_seconds: float = 120.0


@dataclass
class PolicyCfg:
    """Policy network configuration.

    Attributes:
        width: Hidden size for encoder/heads.
        depth: Number of layers for encoder/heads.
        deep_layer_size: Size of deep layer block (policy-dependent).
        continuous_distribution: Dict describing continuous parameter distribution.
        entropy_weights_per_head: Entropy coefficients per head.
        ordering_enabled: If True, uses ordered reaction addition policy.
        constraint_strength: Constraint strength for ordered policy.
        zero_reaction_idx: If set, this reaction index is treated as a "no-op" action (allowing multiple re-samples).
        stop_flag: Internal flag to indicate if calling a "zero_reaction" is a stopping condition (instead of a no-op).
    """
    width: int = 1024
    depth: int = 5
    deep_layer_size: int = 10240
    continuous_distribution: Dict[str, Any] = field(default_factory=lambda: {"type": "lognormal_1D"})
    entropy_weights_per_head: Dict[str, float] = field(
        default_factory=lambda: {
            "structure": 2.0,
            "continuous": 1.0,
            "discrete": 0.0,
            "input_influence": 0.0,
        }
    )
    ordering_enabled: bool = False
    constraint_strength: float = float("inf")

    # stopping condition 
    zero_reaction_idx: Optional[int] = None             # If set, this reaction index is treated as a "no-op" action
    stop_flag: bool = field(init=False, default=False)  # Internal flag to indicate if calling a "zero_reaction" is a stopping condition


@dataclass
class AgentCfg:
    """Agent configuration.

    Attributes:
        learning_rate: Optimizer learning rate.
        entropy_scheduler: Scheduler parameters for entropy regularization.
        risk_scheduler: Scheduler parameters for risk-sensitive objective.
        sil_settings: Self-imitation learning configuration.
    """
    learning_rate: float = 1e-4
    entropy_scheduler: Dict[str, Any] = field(
        default_factory=lambda: {
            "entropy_weight": 1e-3,
            "topk_entropy_weight": 1.0,
            "remainder_entropy_weight": 1.0,
            "entropy_update_coefficient": 1,
            "entropy_schedule": 1000,
            "minimum_entropy_weight": 0.0,
        }
    )
    risk_scheduler: Dict[str, Any] = field(
        default_factory=lambda: {
            "risk": 0.95,
            "risk_update": 0.0,
            "max_risk": 1.0,
            "risk_schedule": 1000,
        }
    )
    sil_settings: Dict[str, Any] = field(default_factory=lambda: {"sil_loss_weight": 1.0})


@dataclass
class RenderCfg:
    """Rendering configuration.

    Attributes:
        n_best: Number of top trajectories to render.
        disregarded_percentage: Percentage of trajectories to disregard based on reward (for stochastic tasks).
        mode: Rendering mode. A legacy string such as "transients" is accepted
            and normalized by Trainer.run to the logger-mode dictionary expected
            by the environment renderer.
    """
    n_best: int = 30
    disregarded_percentage: float = 0.5
    mode: Union[str, Dict[str, Any]] = "transients"

@dataclass
class Config:
    """Top-level configuration container.

    Attributes:
        task: Task configuration.
        solver: Solver configuration.
        train: Training configuration.
        library: Library configuration.
        policy: Policy configuration.
        agent: Agent configuration.
        render: Rendering configuration.
    """
    task: TaskSpec = None  
    solver: SolverCfg = field(default_factory=SolverCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    render: RenderCfg = field(default_factory=lambda: RenderCfg())

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a JSON-serializable dictionary.

        Returns:
            Nested dictionary of config values.
        """
        return asdict(self)

    def describe(self, width: int = 120) -> None:
        """Pretty-print the full configuration.

        Args:
            width: Print width for formatting.
        """
        import pprint
        pprint.pprint(self.to_dict(), width=width, sort_dicts=False)


class Configurator:
    """Helpers to create configs from presets and apply overrides."""

    @staticmethod
    def preset(name: str = "balanced") -> Config:
        """Create a config from a named preset.

        Args:
            name: Preset name. Supported:
                - "fast": small networks, looser tolerances
                - "balanced": sensible defaults
                - "quality": larger networks, more capacity
                - "paper": settings used in the GenAI-Net paper experiments

        Returns:
            Config instance.

        Raises:
            ValueError: If preset name is unknown.
        """
        name = name.lower().strip()
        cfg = Config()

        if name == "balanced":
            return cfg

        if name == "fast":
            cfg.train.epochs = 50
            cfg.policy.width = 256
            cfg.policy.depth = 2
            cfg.policy.deep_layer_size = 512
            cfg.solver.rtol = 1e-8
            cfg.solver.atol = 1e-8
            return cfg

        if name == "quality":
            cfg.policy.width = 1536
            cfg.policy.depth = 6
            cfg.policy.deep_layer_size = 16384
            cfg.train.hall_of_fame_size = 50
            return cfg
        
        if name == "paper":
            cfg.policy.width = 1024
            cfg.policy.depth = 5
            cfg.policy.deep_layer_size = 10240
            cfg.train.epochs = 300
            cfg.train.hall_of_fame_size = 50
            cfg.solver.rtol = 1e-10
            cfg.solver.atol = 1e-10
            return cfg

        raise ValueError(f"Unknown preset: {name!r}")

    @staticmethod
    def with_overrides(cfg: Config, **overrides: Dict[str, Any]) -> Config:
        """Return a deep-copied config with nested overrides applied.

        Args:
            cfg: Base config.
            **overrides: Nested dictionaries keyed by top-level sections
                (task, solver, train, library, policy, agent).

        Returns:
            New Config with overrides applied.
        """
        new_cfg = copy.deepcopy(cfg)
        for section_name, section_overrides in overrides.items():
            if not hasattr(new_cfg, section_name):
                raise ValueError(f"Unknown config section: {section_name!r}")
            section_obj = getattr(new_cfg, section_name)
            if not isinstance(section_overrides, dict):
                raise ValueError(f"Overrides for {section_name!r} must be a dict.")
            for k, v in section_overrides.items():
                if not hasattr(section_obj, k):
                    raise ValueError(f"Unknown key {k!r} in section {section_name!r}")
                setattr(section_obj, k, v)
        return new_cfg


# ----------------------------
# RL4CRN wiring helpers
# ----------------------------


def build_envs(
    template: Any,
    max_added_reactions: int,
    batch_size: int,
    hall_of_fame_size: int,
    n_cpus: int,
    logger: Any = None,
):
    """Create parallel environments.

    Args:
        template: IOCRN template.
        max_added_reactions: Episode length.
        batch_size: Number of environments.
        hall_of_fame_size: Hall-of-fame capacity.
        n_cpus: Number of CPUs for parallel execution.
        logger: Optional logger.

    Returns:
        ParallelEnvironments instance.
    """
    from RL4CRN.environments.environment import Environment
    from RL4CRN.environments.parallel_environments import ParallelEnvironments

    crn0 = template.clone()
    envs = [Environment(crn0, max_added_reactions, logger=logger, logger_schedule=1) for _ in range(batch_size)]
    mult_env = ParallelEnvironments(envs, hall_of_fame_size=hall_of_fame_size, N_CPUs=n_cpus, logger=logger)
    return mult_env


def build_interfaces(library: Any, device: str, allow_input_influence: bool = False):
    """Build standard env<->agent interfaces.

    Args:
        library: Reaction library.
        device: Torch device string.
        allow_input_influence: Whether to allow input influence features.

    Returns:
        Tuple (observer, tensorizer, actuator, stepper).
    """
    from RL4CRN.env2agent_interface.explicit_observer import ExplicitObserver
    from RL4CRN.env2agent_interface.explicit_tensorizer import ExplicitTensorizer
    from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
    from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper

    observer = ExplicitObserver(reaction_library=library, allow_input_observation=allow_input_influence)
    tensorizer = ExplicitTensorizer(device=device)
    actuator = LibraryActuator(reaction_library=library)
    stepper = IOCRNStepper()
    return observer, tensorizer, actuator, stepper


def build_policy(
    M: int,
    K: int,
    p: int,
    masks: Dict[str, Any],
    device: str,
    policy_cfg: PolicyCfg,
    target_set_size: int,
):
    """Build the policy instance.

    Args:
        M: Number of reactions in the library.
        K: Number of total library parameters.
        p: Number of input channels in CRN.
        masks: Parameter/logit masks from the library.
        device: Torch device string.
        policy_cfg: PolicyCfg instance.
        target_set_size: Required for ordered policy.

    Returns:
        Policy instance.
    """
    from RL4CRN.policies.add_reaction_by_ordered_index import AddReactionByOrderedIndex
    from RL4CRN.policies.add_reaction_by_index import AddReactionByIndex

    encoder_attributes = {"hidden_size": policy_cfg.width, "num_layers": policy_cfg.depth}
    head_attrs = {"hidden_size": policy_cfg.width, "num_layers": policy_cfg.depth}
    input_influence_head_attributes = {"hidden_size": policy_cfg.width, "num_layers": policy_cfg.depth}

    if policy_cfg.ordering_enabled:
        policy = AddReactionByOrderedIndex(
            M,
            K,
            p,
            encoder_attributes,
            policy_cfg.deep_layer_size,
            head_attrs,
            head_attrs,
            input_influence_head_attributes,
            target_set_size=target_set_size,
            masks=masks,
            allow_input_influence=False,
            device=device,
            continuous_distribution=policy_cfg.continuous_distribution,
            entropy_weights_per_head=policy_cfg.entropy_weights_per_head,
            combinatorial_bias_enabled=True,
            constraint_strength=policy_cfg.constraint_strength,
        )
    else:
        policy = AddReactionByIndex(
            M,
            K,
            p,
            encoder_attributes,
            policy_cfg.deep_layer_size,
            head_attrs,
            head_attrs,
            input_influence_head_attributes,
            masks=masks,
            allow_input_influence=False,
            device=device,
            continuous_distribution=policy_cfg.continuous_distribution,
            entropy_weights_per_head=policy_cfg.entropy_weights_per_head,
            # stopping_condition
            zero_reaction_idx = policy_cfg.zero_reaction_idx,
            stop_flag = policy_cfg.stop_flag if policy_cfg.zero_reaction_idx is not None else False
        )

    return policy


def build_agent(policy: Any, device: str, agent_cfg: AgentCfg, logger: Any = None):
    """Build the REINFORCE(+SIL) agent.

    Args:
        policy: Policy instance.
        device: Torch device string.
        agent_cfg: AgentCfg instance.
        logger: Optional logger.

    Returns:
        REINFORCEAgent instance.
    """
    from RL4CRN.agents.reinforce_agent import REINFORCEAgent

    agent = REINFORCEAgent(
        policy,
        allow_input_influence=False,
        logger=logger,
        learning_rate=agent_cfg.learning_rate,
        entropy_scheduler=agent_cfg.entropy_scheduler,
        risk_scheduler=agent_cfg.risk_scheduler,
        sil_settings=agent_cfg.sil_settings,
        device=device,
    )
    return agent


# ----------------------------
# Session + Trainer
# ----------------------------

@dataclass
class Session:
    """Container for all objects needed to run training and inspection.

    Attributes:
        cfg: Config used to build this session.
        device: Torch device string.
        n_cpus: Number of CPUs used for parallel rollouts.
        batch_size: Number of parallel environments.
        task: Materialized TaskSpec used to compute rewards.
        crn_template: Compiled IOCRN template.
        species_labels: Species labels for template/library.
        library: Reaction library.
        M: Number of reactions in library.
        K: Number of parameters in library.
        masks: Parameter/logit masks from the library.
        p: Number of CRN input channels.
        mult_env: Parallel environments.
        observer: Env->agent observer.
        tensorizer: Observer tensorizer.
        actuator: Agent->env actuator.
        stepper: Environment stepper.
        policy: Policy instance.
        agent: Agent instance.
        sample_hof: an HallOfFame of CRNs from the hall of fame, populated after calling `sample`.
    """
    cfg: Config
    device: str
    n_cpus: int
    batch_size: int

    task: TaskSpec

    crn_template: Any
    species_labels: List[str]

    library: Any
    M: int
    K: int
    masks: Dict[str, Any]
    p: int

    mult_env: Any
    observer: Any
    tensorizer: Any
    actuator: Any
    stepper: Any

    policy: Any
    agent: Any

    sample_hof: Optional[HallOfFame] = None
    forbidden_topologies: ForbiddenTopologyArchive = field(default_factory=ForbiddenTopologyArchive)
    logger: Any = None

    @staticmethod
    def from_config(cfg: Config, task: TaskSpec, device: Optional[str] = None, logger: Any = None) -> "Session":
        """Build a Session from a Config.

        Args:
            cfg: Configuration object.
            task: Materialized TaskSpec object.
            device: Torch device string. If None, auto-selects.

        Returns:
            Initialized Session with all required RL4CRN objects wired up.
        """
        if device is None:
            device = get_device("auto")

        seed_everything(cfg.train.seed)

        n_cpus = cfg.train.n_cpus or (os.cpu_count() or 1)
        batch_size = cfg.train.batch_size or (cfg.train.batch_multiplier * n_cpus)

        cfg.task = task  # ensure task in cfg is the materialized one
        task = cfg.task

        # Template CRN + species labels
        crn_template, species_labels = cfg.task.template_crn, cfg.task.species_labels

        # Library
        library, M, K, masks = cfg.task.library_components
        p = crn_template.num_inputs

        # Environments
        mult_env = build_envs(
            template=crn_template,
            max_added_reactions=cfg.train.max_added_reactions,
            batch_size=batch_size,
            hall_of_fame_size=cfg.train.hall_of_fame_size,
            n_cpus=n_cpus,
            logger=logger,
        )

        # Interfaces
        observer, tensorizer, actuator, stepper = build_interfaces(library, device=device, allow_input_influence=False)

        # Policy + agent
        policy = build_policy(
            M=M,
            K=K,
            p=p,
            masks=masks,
            device=device,
            policy_cfg=cfg.policy,
            target_set_size=crn_template.num_reactions + cfg.train.max_added_reactions,
        )
        agent = build_agent(policy=policy, device=device, agent_cfg=cfg.agent, logger=logger)

        return Session(
            cfg=cfg,
            device=device,
            n_cpus=n_cpus,
            batch_size=batch_size,
            task=task,
            crn_template=crn_template,
            species_labels=species_labels,
            library=library,
            M=M,
            K=K,
            masks=masks,
            p=p,
            mult_env=mult_env,
            observer=observer,
            tensorizer=tensorizer,
            actuator=actuator,
            stepper=stepper,
            policy=policy,
            agent=agent,
            logger=logger
        )

    def sample(
        self,
        n_samples: int,
        sample_hof_size: int,
        *,
        u_list: Optional[List[np.ndarray]] = None,
        u_spec: Optional[tuple] = None,
        u_values: Optional[List[float]] = None,
        dose_range: Optional[Tuple[float, float, int]] = None,
        ic: Optional[Union[str, tuple]] = None,
        weights: Optional[Union[str, tuple]] = None,
    ) -> HallOfFame:
        """Sample CRNs from the current policy without training (evaluation-only).

        This method creates a temporary batch of environments, performs one rollout
        (episode) per environment using the current policy in eval mode, computes
        rewards, and stores the best sampled environments in a dedicated
        `sample_hof` HallOfFame.

        Sampling does not perform any learning updates (no backpropagation).

        Calling this method again replaces the previously stored `sample_hof`, so
        that different checkpoints can store different sample sets.

        Args:
            n_samples: Number of environments to roll out (number of samples drawn).
            sample_hof_size: Capacity of the sample HallOfFame (best K kept).
            u_list: Optional explicit list of input vectors to evaluate.
            u_spec: Optional input generation spec (same as build_u_list):
                ("custom", u_list), ("grid", values), ("linspace", u_min, u_max, n)
            u_values: Optional enumerated values used by build_u_list for grid tasks.
            dose_range: Optional (u_min, u_max, n) for dose_response input generation.
            ic: Optional IC spec override (same format accepted by build_ic).
            weights: Optional weights spec override (same format accepted by build_weights).

        Returns:
            HallOfFame: The newly created sample HallOfFame containing sampled env snapshots.

        Raises:
            ValueError: If n_samples/sample_hof_size are invalid or input dimension mismatch.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if sample_hof_size <= 0:
            raise ValueError("sample_hof_size must be positive.")

        task = self.task

        # --- Build evaluation u_list ---
        if u_list is None and u_spec is not None:
            u_list_eval = build_u_list(
                task.kind,
                n_inputs=task.n_inputs,
                u_values=u_values,
                dose_range=dose_range,
                u_spec=u_spec,
            )
        else:
            u_list_eval = task.u_list if u_list is None else u_list

        u_list_eval = [np.asarray(u, dtype=np.float32).reshape(-1) for u in u_list_eval]

        # Strong input-dimension guardrail (prevents ragged-propensity errors)
        expected_p = int(self.crn_template.num_inputs)
        for i, u in enumerate(u_list_eval):
            if u.size != expected_p:
                raise ValueError(
                    f"[Session.sample] u_list[{i}] length={u.size} != template_crn.num_inputs={expected_p}"
                )

        # --- Build reward function from TaskSpec + overrides ---
        reward_fn = make_reward_fn_with_overrides(
            task,
            u_list=u_list_eval,
            ic_spec=ic,
            weights_spec=weights,
        )

        # --- Create a fresh sample HoF (replace previous) ---
        self.sample_hof = HallOfFame(max_size=sample_hof_size)

        # --- Temporary environments (do not touch training envs) ---
        from RL4CRN.environments.environment import Environment
        from RL4CRN.environments.parallel_environments import ParallelEnvironments

        crn0 = self.crn_template.clone()
        envs = [
            Environment(crn0, self.cfg.train.max_added_reactions, logger=None, logger_schedule=1)
            for _ in range(n_samples)
        ]
        sample_env = ParallelEnvironments(
            envs,
            hall_of_fame_size=0,     # we will use *our* HoF, so disable internal one
            N_CPUs=self.n_cpus,
            logger=None,
        )

        policy = self.agent.policy
        was_training = policy.training
        policy.eval()

        try:
            sample_env.reset()
            with torch.no_grad():
                for _ in range(self.cfg.train.max_added_reactions):
                    obs = sample_env.observe(self.observer, self.tensorizer)
                    actions, raw_actions = self.agent.act(obs, self.actuator)
                    sample_env.step(actions, self.stepper, raw_actions=raw_actions)

            # Evaluate rewards (fills env.state.last_task_info['reward'])
            _ = sample_env.get_reward(reward_fn)

        finally:
            policy.train(was_training)

        # --- Add all sampled environments into the sample HallOfFame ---
        # ParallelEnvironments typically stores its live envs in `envs`
        self.sample_hof.add_all(sample_env.envs)

        return self.sample_hof


@dataclass
class TrainState:
    """Training state that persists across chunked runs.

    Attributes:
        epoch: Next epoch index to run.
        history: List of dicts with keys {"epoch","best","median"}.
    """
    epoch: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    """Chunkable trainer with stop/resume and checkpointing."""

    def __init__(self, session: Session):
        """Initialize trainer.

        Args:
            session: Built Session containing envs, agent, and task reward function.
        """
        self.s = session
        self.state = TrainState()
        self._loaded_hof: Optional[List[Any]] = None
        self._loaded_cfg: Optional[dict] = None
        self._llm_loop: Optional[Dict[str, Any]] = None
        self._results_db: Optional[ResultsDatabase] = None
        self._results_db_every: int = 1
        self._results_plot_every: int = 20
        self._last_hof_snapshot_epoch: Optional[int] = None
        self._forbidden_executor: Optional[ThreadPoolExecutor] = None
        self._forbidden_job: Optional[Dict[str, Any]] = None
        self._forbidden_scheduled_signatures: set[bytes] = set()
        self._forbidden_optimization_evaluations = 0
        self._llm_provenance_by_topology: Dict[str, Dict[str, Any]] = {}
        self._rl_seen_topologies: set[str] = set()

    def configure_results_database(
        self,
        path: Union[str, os.PathLike],
        *,
        every: int = 5,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        plot_every: int = 20,
    ) -> ResultsDatabase:
        """Persist analyzed CRNs and periodic Hall-of-Fame snapshots to SQLite.

        Database writes are handled by one background thread.  ``every`` only
        controls HoF snapshots; LLM rounds and parameter optimizations are
        recorded whenever they complete.
        """

        if every <= 0:
            raise ValueError("Results database cadence `every` must be positive.")
        if plot_every <= 0:
            raise ValueError("HoF plot cadence `plot_every` must be positive.")
        if self._results_db is not None:
            self._results_db.close()
        run_metadata = dict(metadata or {})
        task = getattr(self.s, "task", None)
        task_kind = getattr(task, "kind", None)
        if task_kind is not None:
            run_metadata.setdefault("task", str(task_kind))
            run_metadata.setdefault("task_kind", str(task_kind))
        self._results_db = ResultsDatabase(
            path,
            run_id=run_id,
            run_metadata=run_metadata,
        )
        self._results_db_every = int(every)
        self._results_plot_every = int(plot_every)
        self._last_hof_snapshot_epoch = None
        return self._results_db

    def flush_results_database(self) -> None:
        """Wait for already queued database writes to become durable."""

        if self._results_db is not None:
            self._results_db.flush()

    def close_results_database(self) -> None:
        """Flush and close the configured results database writer."""

        if self._results_db is not None:
            self._results_db.close()
            self._results_db = None

    def _maybe_persist_hof(self, epoch: int, *, force: bool = False) -> None:
        database = self._results_db
        if database is None or self._last_hof_snapshot_epoch == epoch:
            return
        if not force and epoch % self._results_db_every != 0:
            return
        hof = getattr(self.s.mult_env, "hall_of_fame", None)
        if hof is None:
            return
        database.record_hof_snapshot(
            hof,
            epoch=epoch,
            save_plots=force or epoch % self._results_plot_every == 0,
            llm_provenance=self._llm_provenance_by_topology,
        )
        logger = self.s.logger
        if logger is not None and hasattr(logger, "log_metric"):
            counts: Dict[str, int] = {}
            for env in hof:
                state = getattr(env, "state", env)
                provenance = classify_hof_provenance(
                    serialize_crn(state),
                    dict(getattr(state, "last_task_info", {}) or {}),
                    self._llm_provenance_by_topology,
                )
                label = str(provenance["provenance_class"])
                counts[label] = counts.get(label, 0) + 1
            for label, count in counts.items():
                logger.log_metric(
                    f"Provenance/HOF {label.replace('_', ' ').title()}",
                    count,
                    step=epoch,
                )
        self._last_hof_snapshot_epoch = epoch

    def _log_loss_components(self, step: int) -> None:
        logger = self.s.logger
        if logger is None:
            return

        component_values: Dict[str, List[float]] = {}
        for env in self.s.mult_env.envs:
            comps = env.state.last_task_info.get("component_losses", {})
            if not isinstance(comps, dict):
                continue
            for name, value in comps.items():
                try:
                    component_values.setdefault(str(name), []).append(float(value))
                except (TypeError, ValueError):
                    continue

        for name, values in component_values.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            logger.log_metric(f"Component: {name} Average", float(np.mean(arr)), step=step)
            logger.log_metric(f"Component: {name} Best", float(np.min(arr)), step=step)
            logger.log_metric(f"Component: {name} Median", float(np.median(arr)), step=step)

    def configure_llm_graph(
        self,
        graph: Any,
        *,
        every: int,
        task_description: str,
        num_candidates: int = 10,
        start_epoch: int = 0,
        stop_epoch: Optional[int] = None,
        add_to_hall_of_fame: bool = True,
        cross_communication: bool = True,
        withhold_initial_hof: bool = False,
        jsonl_path: Optional[Union[str, os.PathLike]] = None,
        max_in_flight: int = 1,
    ) -> None:
        """Attach an LLM proposal graph to the RL training loop.

        Harness generation runs on background workers. The RL thread polls at
        epoch boundaries and inserts completed valid CRNs into the Hall of Fame
        used by self-imitation learning (SIL). Slow model calls therefore do not
        block rollouts, policy updates, or later cadence-triggered model calls.

        Args:
            graph: Object exposing ``run_round(...)``; typically
                ``RL4CRN.llm.DeciderWriterCRNGraph``.
            every: Call the graph every this many epochs.  Set to 0 through
                ``clear_llm_graph`` instead of passing non-positive values.
            task_description: Text task context passed to the LLM graph.
            num_candidates: Number of CRNs requested at each LLM generation
                round.  Defaults to 10.
            start_epoch: First epoch at which LLM proposals are allowed.
            stop_epoch: Optional last epoch at which LLM proposals are allowed.
            add_to_hall_of_fame: If True, valid LLM candidates are inserted into
                the training Hall of Fame before SIL replay.
            cross_communication: If False, do not expose RL HOF, SIL, or
                exclusion state to the LLM. Valid candidates can be retained
                outside the RL HOF for terminal pooling.
            withhold_initial_hof: If True, request zero receives an empty HOF
                snapshot while later requests retain full cross-communication.
            jsonl_path: Optional audit log for every evaluated LLM candidate.
            max_in_flight: Maximum simultaneous model requests. Values above
                one require ``graph.fork()`` so each request owns isolated
                mutable client and workspace state.
        """
        if every <= 0:
            raise ValueError("LLM graph cadence `every` must be positive.")
        if num_candidates <= 0:
            raise ValueError("LLM graph `num_candidates` must be positive.")
        if max_in_flight <= 0:
            raise ValueError("LLM graph `max_in_flight` must be positive.")
        if not hasattr(graph, "run_round"):
            raise ValueError("graph must expose a run_round(...) method.")
        if max_in_flight > 1 and not hasattr(graph, "fork"):
            raise ValueError("Concurrent LLM calls require graph.fork().")

        self.clear_llm_graph(wait=False)

        self._llm_loop = {
            "graph": graph,
            "every": int(every),
            "task_description": str(task_description),
            "num_candidates": int(num_candidates),
            "start_epoch": int(start_epoch),
            "stop_epoch": None if stop_epoch is None else int(stop_epoch),
            "add_to_hall_of_fame": bool(add_to_hall_of_fame),
            "cross_communication": bool(cross_communication),
            "withhold_initial_hof": bool(withhold_initial_hof),
            "isolated_candidates": [],
            "jsonl_path": jsonl_path,
            "history": [],
            "max_in_flight": int(max_in_flight),
            "executor": ThreadPoolExecutor(
                max_workers=int(max_in_flight), thread_name_prefix="rl4crn-llm"
            ),
            "jobs": [],
        }

    def forbidden_topology_summary(self, limit: int = 10) -> str:
        """Return LLM-facing text describing archived forbidden topologies."""

        archive = self._forbidden_topologies()
        summary = archive.format_for_prompt(limit=limit)
        if self._forbidden_job is None:
            return summary
        state = self._forbidden_job["state"]
        reaction_ids = (
            sorted(int(value) for value in state.gather_reaction_IDs())
            if hasattr(state, "gather_reaction_IDs")
            else []
        )
        return (
            summary
            + "\nProcessing now (do not duplicate while pending): "
            + f"reaction_ids={reaction_ids}; launched_epoch={self._forbidden_job['launched_epoch']}."
        )

    def _forbidden_topologies(self) -> ForbiddenTopologyArchive:
        archive = getattr(self.s, "forbidden_topologies", None)
        if archive is None:
            archive = ForbiddenTopologyArchive()
            setattr(self.s, "forbidden_topologies", archive)
        return archive

    def _refresh_forbidden_topologies(self, epoch: int) -> int:
        cfg = self.s.cfg.train
        m = int(getattr(cfg, "forbidden_topology_m", 0) or 0)
        if m <= 0:
            return 0
        every = max(1, int(getattr(cfg, "forbidden_topology_every", 1) or 1))
        start = int(getattr(cfg, "forbidden_topology_start_epoch", 0) or 0)
        if epoch < start or (epoch - start) % every != 0:
            return self._harvest_forbidden_job(epoch)
        if bool(getattr(cfg, "forbidden_async", False)):
            added = self._harvest_forbidden_job(epoch)
            self._schedule_forbidden_job(epoch=epoch, maximum=m)
            return added
        archive = self._forbidden_topologies()
        tic = time.perf_counter()
        added = self._archive_forbidden_from_hof(epoch=epoch, m=m)
        elapsed = time.perf_counter() - tic
        total = len(archive)
        if self.s.logger is not None and hasattr(self.s.logger, "log_metric"):
            self.s.logger.log_metric("Forbidden Topologies/Added", added, step=epoch)
            self.s.logger.log_metric("Forbidden Topologies/Total", total, step=epoch)
            self.s.logger.log_metric("Forbidden Topologies/Timing Total Seconds", elapsed, step=epoch)
        if added:
            print(
                f"[epoch {epoch}] archived {added} forbidden topologies | "
                f"total={total} | archive_time={elapsed:.3g}s"
            )
        return added

    def _schedule_forbidden_job(self, *, epoch: int, maximum: int) -> bool:
        if self._forbidden_job is not None:
            return False
        if len(self._forbidden_scheduled_signatures) >= int(maximum):
            return False
        hof = self.s.mult_env.hall_of_fame
        if hof is None:
            return False
        archive = self._forbidden_topologies()
        selected = None
        for rank, env in enumerate(hof):
            signature = topology_signature_key(env.state)
            if signature in self._forbidden_scheduled_signatures or archive.contains_state(env.state):
                continue
            selected = (rank, env.state.clone(), signature)
            break
        if selected is None:
            return False
        rank, state, signature = selected
        if self._forbidden_executor is None:
            self._forbidden_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rl4crn-topology-processing"
            )
        cfg = self.s.cfg.train
        future = self._forbidden_executor.submit(
            optimize_crn_parameters_ipopt,
            state,
            self.s.task.compute_reward,
            maxiter=int(getattr(cfg, "forbidden_ipopt_maxiter", 100)),
            log_min=float(getattr(cfg, "forbidden_ipopt_log_min", -18.0)),
            log_max=float(getattr(cfg, "forbidden_ipopt_log_max", 6.0)),
            max_evaluations=int(getattr(cfg, "forbidden_optimization_max_evaluations", 50)),
            timeout_seconds=float(getattr(cfg, "forbidden_optimization_timeout_seconds", 120.0)),
        )
        self._forbidden_scheduled_signatures.add(signature)
        self._forbidden_job = {
            "future": future,
            "state": state,
            "signature": signature,
            "rank": rank,
            "launched_epoch": epoch,
            "started": time.perf_counter(),
        }
        return True

    def _harvest_forbidden_job(self, epoch: int, *, wait: bool = False) -> int:
        job = self._forbidden_job
        if job is None or (not wait and not job["future"].done()):
            return 0
        result = job["future"].result()
        elapsed = time.perf_counter() - job["started"]
        self._forbidden_optimization_evaluations += int(result.n_evaluations)
        archive = self._forbidden_topologies()
        inserted = archive.add_state(
            result.state,
            loss=float(result.loss),
            epoch=epoch,
            rank=int(job["rank"]),
            source="hall_of_fame_bounded_ipopt",
            optimization_attempted=True,
            optimization_success=bool(result.success),
            optimization_message=result.message,
            exclusion_reason="bounded parameter processing completed; topology hard-excluded",
        )
        if self._results_db is not None:
            self._results_db.record_optimization(
                job["state"],
                result.state,
                epoch=epoch,
                rank=int(job["rank"]),
                original_loss=float(
                    (getattr(job["state"], "last_task_info", {}) or {}).get("reward", float("inf"))
                ),
                optimized_loss=float(result.loss),
                attempted=True,
                success=bool(result.success),
                message=result.message,
                elapsed_seconds=elapsed,
                n_evaluations=int(result.n_evaluations),
                stored=True,
            )
        self._log_forbidden_archive_decision(
            epoch=epoch,
            rank=int(job["rank"]),
            original_loss=float(
                (getattr(job["state"], "last_task_info", {}) or {}).get("reward", float("inf"))
            ),
            archive_loss=float(result.loss),
            threshold=float("inf"),
            stored=True,
            inserted=inserted,
            optimization_attempted=True,
            optimization_success=bool(result.success),
            optimization_message=result.message,
            optimization_seconds=elapsed,
            optimization_evaluations=int(result.n_evaluations),
        )
        self._forbidden_job = None
        return int(inserted)

    def wait_for_forbidden_topologies(self) -> int:
        """Wait for the current bounded job and merge it; no new job is launched."""

        added = self._harvest_forbidden_job(self.state.epoch, wait=True)
        if self._forbidden_executor is not None:
            self._forbidden_executor.shutdown(wait=True, cancel_futures=False)
            self._forbidden_executor = None
        return added

    def forbidden_optimization_evaluations(self) -> int:
        return int(self._forbidden_optimization_evaluations)

    def _archive_forbidden_from_hof(self, *, epoch: int, m: int) -> int:
        hof = self.s.mult_env.hall_of_fame
        if hof is None or m <= 0:
            return 0

        cfg = self.s.cfg.train
        archive = self._forbidden_topologies()
        threshold = float(getattr(cfg, "forbidden_threshold", float("inf")))
        use_ipopt = bool(getattr(cfg, "forbidden_optimize_with_ipopt", True))
        added = 0

        for rank, env in enumerate(hof):
            if rank >= int(m):
                break
            info = getattr(env.state, "last_task_info", {}) or {}
            original_loss = float(info.get("reward", float("inf")))
            opt_attempted = False
            opt_success = False
            opt_message = "optimization disabled"
            opt_elapsed = 0.0
            opt_evaluations = 0
            archive_state = env.state
            archive_loss = original_loss

            if use_ipopt:
                tic = time.perf_counter()
                result = optimize_crn_parameters_ipopt(
                    env.state,
                    self.s.task.compute_reward,
                    maxiter=int(getattr(cfg, "forbidden_ipopt_maxiter", 100)),
                    log_min=float(getattr(cfg, "forbidden_ipopt_log_min", -18.0)),
                    log_max=float(getattr(cfg, "forbidden_ipopt_log_max", 6.0)),
                )
                opt_elapsed = time.perf_counter() - tic
                opt_attempted = result.attempted
                opt_success = result.success
                opt_message = result.message
                opt_evaluations = int(getattr(result, "n_evaluations", 0) or 0)
                if result.success:
                    archive_state = result.state
                    archive_loss = float(result.loss)

            should_store = bool(opt_success or archive_loss <= threshold)
            if should_store:
                inserted = archive.add_state(
                    archive_state,
                    loss=archive_loss,
                    epoch=epoch,
                    rank=rank,
                    source="hall_of_fame_ipopt" if opt_attempted else "hall_of_fame",
                    optimization_attempted=opt_attempted,
                    optimization_success=opt_success,
                    optimization_message=opt_message,
                    exclusion_reason=(
                        "parameter optimization completed; topology fully processed"
                        if opt_success
                        else "topology evaluated and processing threshold satisfied"
                    ),
                )
                added += int(inserted)

            if self._results_db is not None:
                self._results_db.record_optimization(
                    env.state,
                    archive_state,
                    epoch=epoch,
                    rank=rank,
                    original_loss=original_loss,
                    optimized_loss=archive_loss,
                    attempted=opt_attempted,
                    success=opt_success,
                    message=opt_message,
                    elapsed_seconds=opt_elapsed,
                    n_evaluations=opt_evaluations,
                    stored=should_store,
                )
                self._results_db.record_evaluation(
                    archive_state,
                    source="ipopt" if opt_attempted else "hof_analysis",
                    epoch=epoch,
                    loss=archive_loss,
                    valid=should_store,
                    message=opt_message,
                    metadata={
                        "hof_rank": rank,
                        "optimization_attempted": opt_attempted,
                        "optimization_success": opt_success,
                        "stored_in_forbidden_archive": should_store,
                    },
                )

            self._log_forbidden_archive_decision(
                epoch=epoch,
                rank=rank,
                original_loss=original_loss,
                archive_loss=archive_loss,
                threshold=threshold,
                stored=should_store,
                inserted=should_store and inserted if should_store else False,
                optimization_attempted=opt_attempted,
                optimization_success=opt_success,
                optimization_message=opt_message,
                optimization_seconds=opt_elapsed,
                optimization_evaluations=opt_evaluations,
            )
        return added

    def _log_forbidden_archive_decision(
        self,
        *,
        epoch: int,
        rank: int,
        original_loss: float,
        archive_loss: float,
        threshold: float,
        stored: bool,
        inserted: bool,
        optimization_attempted: bool,
        optimization_success: bool,
        optimization_message: str,
        optimization_seconds: float = 0.0,
        optimization_evaluations: int = 0,
    ) -> None:
        logger = self.s.logger
        if logger is None:
            return
        prefix = "Forbidden Topologies"
        if hasattr(logger, "log_metric"):
            logger.log_metric(f"{prefix}/Candidate Original Loss", original_loss, step=epoch)
            logger.log_metric(f"{prefix}/Candidate Archive Loss", archive_loss, step=epoch)
            logger.log_metric(f"{prefix}/Threshold", threshold, step=epoch)
            logger.log_metric(f"{prefix}/Optimization Attempted", int(optimization_attempted), step=epoch)
            logger.log_metric(f"{prefix}/Optimization Success", int(optimization_success), step=epoch)
            logger.log_metric(f"{prefix}/Optimization Seconds", float(optimization_seconds), step=epoch)
            logger.log_metric(f"{prefix}/Optimization Evaluations", int(optimization_evaluations), step=epoch)
            logger.log_metric(f"{prefix}/Stored", int(stored), step=epoch)
            logger.log_metric(f"{prefix}/Inserted", int(inserted), step=epoch)
        record = {
            "epoch": epoch,
            "rank": rank,
            "original_loss": original_loss,
            "archive_loss": archive_loss,
            "threshold": threshold,
            "stored": stored,
            "inserted": inserted,
            "optimization_attempted": optimization_attempted,
            "optimization_success": optimization_success,
            "optimization_message": optimization_message,
            "optimization_seconds": float(optimization_seconds),
            "optimization_evaluations": int(optimization_evaluations),
        }
        text = "Forbidden topology archive decision:\n" + pprint.pformat(record, width=100)
        if hasattr(logger, "log_text"):
            logger.log_text(text)
        if hasattr(logger, "log_asset_data"):
            logger.log_asset_data(
                json.dumps(record, indent=2, sort_keys=True),
                name=f"forbidden_topology_archive_epoch_{epoch}_rank_{rank}.json",
                step=epoch,
            )

    def clear_llm_graph(self, *, wait: bool = False) -> None:
        """Disable LLM calls and shut down their worker, optionally waiting."""

        settings = self._llm_loop
        self._llm_loop = None
        if not settings:
            return
        for job in settings.get("jobs", ()):
            future = job.get("future")
            if future is not None and not future.done():
                future.cancel()
        executor = settings.get("executor")
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)

    def wait_for_llm_graph(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for all pending LLM jobs and merge them on this thread."""

        settings = self._llm_loop
        if not settings:
            return None
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        last_result = None
        while settings.get("jobs"):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            before = len(settings["jobs"])
            result = self._collect_llm_graph_result(
                completed_epoch=self.state.epoch,
                wait=True,
                timeout=remaining,
            )
            if result is not None:
                last_result = result
            if len(settings["jobs"]) == before:
                break
        return last_result

    def _maybe_run_llm_graph(self, epoch: int) -> Optional[Any]:
        settings = self._llm_loop
        if not settings:
            return None

        completed_result = None
        while settings.get("jobs"):
            before = len(settings["jobs"])
            result = self._collect_llm_graph_result(completed_epoch=epoch)
            if result is not None:
                completed_result = result
            if len(settings["jobs"]) == before:
                break
        start_epoch = int(settings["start_epoch"])
        stop_epoch = settings["stop_epoch"]
        every = int(settings["every"])
        if epoch < start_epoch:
            return completed_result
        if stop_epoch is not None and epoch > int(stop_epoch):
            return completed_result
        if (epoch - start_epoch) % every != 0:
            return completed_result

        hof = self.s.mult_env.hall_of_fame
        if bool(settings["add_to_hall_of_fame"]) and hof is None:
            raise ValueError(
                "LLM proposals cannot be inserted because cfg.train.hall_of_fame_size <= 0."
            )
        if bool(settings["add_to_hall_of_fame"]) and not hasattr(hof, "add"):
            raise ValueError("LLM proposals require a writable HallOfFame object with an add(...) method.")
        jobs = settings["jobs"]
        if len(jobs) >= int(settings["max_in_flight"]):
            if self.s.logger is not None and hasattr(self.s.logger, "log_metric"):
                self.s.logger.log_metric("LLM/Pending", len(jobs), step=epoch)
                self.s.logger.log_metric("LLM/Capacity Skipped", 1, step=epoch)
            print(
                f"[epoch {epoch}] LLM cadence reached but {len(jobs)} requests are in flight; "
                "capacity skip logged",
                flush=True,
            )
            return completed_result

        graph = settings["graph"]
        job_graph = graph.fork() if hasattr(graph, "fork") else graph
        cross_communication = bool(settings.get("cross_communication", True))
        withhold_initial_hof = bool(settings.get("withhold_initial_hof", False))
        is_initial_request = epoch == int(settings["start_epoch"])
        hall_snapshot = (
            tuple(env.clone() for env in hof)
            if (
                cross_communication
                and hof is not None
                and not (withhold_initial_hof and is_initial_request)
            )
            else ()
        )
        job = {
            "launched_epoch": epoch,
            "started": time.perf_counter(),
            "hof_size_at_launch": len(hof) if hof is not None else 0,
            "graph": job_graph,
        }
        job["future"] = settings["executor"].submit(
            job_graph.run_round,
            task_description=settings["task_description"],
            forbidden_topologies_text=(
                self.forbidden_topology_summary()
                if cross_communication
                else "Cross-source exclusion state withheld by the no-communication ablation."
            ),
            sil_feedback_text=(
                self._sil_feedback_summary()
                if cross_communication
                else "Cross-source SIL state withheld by the no-communication ablation."
            ),
            num_candidates=int(settings["num_candidates"]),
            hall_of_fame_iter=hall_snapshot,
            add_to_hall_of_fame=None,
            jsonl_path=settings["jsonl_path"],
            logger=self.s.logger,
            step=epoch,
        )
        jobs.append(job)
        if self.s.logger is not None and hasattr(self.s.logger, "log_metric"):
            self.s.logger.log_metric("LLM/Triggered", 1, step=epoch)
            self.s.logger.log_metric("LLM/Background Workers Active", len(jobs), step=epoch)
            self.s.logger.log_metric("LLM/Requested Count", int(settings["num_candidates"]), step=epoch)
        print(
            f"[epoch {epoch}] launched LLM graph in background "
            f"({len(jobs)}/{int(settings['max_in_flight'])} in flight)",
            flush=True,
        )
        return completed_result

    def _sil_feedback_summary(self) -> str:
        """Return the latest completed RL SIL update as compact LLM context."""

        agent = getattr(self.s, "agent", None)
        info = getattr(agent, "last_sil_info", None)
        if not info:
            return "No completed SIL update is available yet."
        return (
            f"enabled={bool(info.get('enabled'))}; step={info.get('step')}; "
            f"hall_of_fame_size={int(info.get('hall_of_fame_size', 0))}; "
            f"sil_loss={info.get('loss')}; loss_weight={info.get('loss_weight')}; "
            f"weighting_scheme={info.get('weighting_scheme', 'unknown')}"
        )

    def _collect_llm_graph_result(
        self,
        *,
        completed_epoch: int,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """Merge a completed background result into the live HoF on the RL thread."""

        settings = self._llm_loop
        if not settings:
            return None
        jobs = settings.get("jobs", [])
        ready = jobs[:1] if wait and jobs else [job for job in jobs if job["future"].done()]
        if not ready:
            return None
        job = ready[0]
        future: Future = job["future"]
        launched_epoch = int(job.get("launched_epoch", completed_epoch))
        elapsed = time.perf_counter() - float(job.get("started", time.perf_counter()))
        hof = self.s.mult_env.hall_of_fame
        before = len(hof) if hof is not None else 0
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            return None
        except Exception as exc:
            jobs.remove(job)
            message = f"LLM graph failed after epoch {launched_epoch}: {exc}"
            failed_validation = dict(
                getattr(
                    getattr(job.get("graph"), "client", None),
                    "last_response_validation",
                    {},
                )
                or {}
            )
            failed_workspace = getattr(job.get("graph"), "client", None)
            failed_workspace = getattr(failed_workspace, "last_workspace", None)
            if failed_workspace is not None:
                failed_validation["provider_call_count"] = int(
                    getattr(failed_workspace, "call_count", 0)
                )
            if self._results_db is not None:
                self._results_db.record_llm_failure(
                    launched_epoch=launched_epoch,
                    completed_epoch=completed_epoch,
                    requested=int(settings["num_candidates"]),
                    elapsed_seconds=elapsed,
                    error=str(exc),
                    response_validation=failed_validation,
                )
            settings["history"].append(
                {
                    "epoch": launched_epoch,
                    "launched_epoch": launched_epoch,
                    "completed_epoch": completed_epoch,
                    "requested": int(settings["num_candidates"]),
                    "valid": 0,
                    "hof_size_before": before,
                    "hof_size_after": before,
                    "elapsed_seconds": elapsed,
                    "returned": failed_validation.get("returned_candidate_count"),
                    "structurally_accepted": failed_validation.get("accepted_candidate_count"),
                    "structurally_rejected": len(
                        failed_validation.get("rejected_candidates", ()) or ()
                    ),
                    "clamped_parameters": failed_validation.get(
                        "clamped_parameter_count", 0
                    ),
                    "error": str(exc),
                }
            )
            if self.s.logger is not None:
                if hasattr(self.s.logger, "log_metric"):
                    self.s.logger.log_metric("LLM/Failed", 1, step=completed_epoch)
                    self.s.logger.log_metric("LLM/Timing Generation Seconds", elapsed, step=completed_epoch)
                    self.s.logger.log_metric(
                        "LLM/Structurally Rejected Members",
                        len(failed_validation.get("rejected_candidates", ()) or ()),
                        step=completed_epoch,
                    )
                    self.s.logger.log_metric(
                        "LLM/Model Requests",
                        int(failed_validation.get("provider_call_count", 0) or 0),
                        step=completed_epoch,
                    )
                if hasattr(self.s.logger, "log_text"):
                    self.s.logger.log_text(message)
            print(f"[epoch {completed_epoch}] {message}", flush=True)
            return None

        jobs.remove(job)
        configured_graph = settings["graph"]
        if job.get("graph") is not configured_graph and hasattr(configured_graph, "memory"):
            configured_graph.memory.update_many(result.evaluations)
        self._register_llm_provenance(
            result,
            launched_epoch=launched_epoch,
            expose_to_rl=bool(settings["add_to_hall_of_fame"]),
        )
        if bool(settings["add_to_hall_of_fame"]) and hof is not None:
            for evaluation in result.evaluations:
                if evaluation.valid and evaluation.env is not None:
                    hof.add(evaluation.env)
        elif not bool(settings.get("cross_communication", True)):
            settings["isolated_candidates"].extend(
                evaluation.env.clone()
                for evaluation in result.evaluations
                if evaluation.valid and evaluation.env is not None
            )
        after = len(hof) if hof is not None else 0
        valid = sum(1 for evaluation in result.evaluations if evaluation.valid)
        tool_evaluations = len(getattr(result, "tool_evaluations", ()) or ())
        response_validation = dict(
            getattr(result, "response_validation", {}) or {}
        )
        if self._results_db is not None:
            self._results_db.record_llm_round(
                result,
                launched_epoch=launched_epoch,
                completed_epoch=completed_epoch,
                elapsed_seconds=elapsed,
                requested=int(settings["num_candidates"]),
            )
        settings["history"].append(
            {
                "epoch": launched_epoch,
                "launched_epoch": launched_epoch,
                "completed_epoch": completed_epoch,
                "requested": int(settings["num_candidates"]),
                "valid": valid,
                "returned": response_validation.get("returned_candidate_count"),
                "structurally_accepted": response_validation.get("accepted_candidate_count"),
                "structurally_rejected": len(
                    response_validation.get("rejected_candidates", ()) or ()
                ),
                "clamped_parameters": response_validation.get("clamped_parameter_count", 0),
                "tool_evaluations": tool_evaluations,
                "total_llm_candidate_evaluations": len(result.evaluations) + tool_evaluations,
                "hof_size_before": before,
                "hof_size_after": after,
                "elapsed_seconds": elapsed,
            }
        )
        if self.s.logger is not None and hasattr(self.s.logger, "log_metric"):
            self.s.logger.log_metric("LLM/Completed", 1, step=completed_epoch)
            self.s.logger.log_metric("LLM/Hall of Fame Size Before", before, step=completed_epoch)
            self.s.logger.log_metric("LLM/Hall of Fame Size After", after, step=completed_epoch)
            self.s.logger.log_metric("LLM/Timing Generation Seconds", elapsed, step=completed_epoch)
            self.s.logger.log_metric("LLM/Timing Trainer Hook Seconds", 0.0, step=completed_epoch)
            self.s.logger.log_metric("LLM/Tool Evaluations", tool_evaluations, step=completed_epoch)
            self.s.logger.log_metric(
                "LLM/Structurally Rejected Members",
                len(response_validation.get("rejected_candidates", ()) or ()),
                step=completed_epoch,
            )
            self.s.logger.log_metric(
                "LLM/Host-Clamped Parameters",
                int(response_validation.get("clamped_parameter_count", 0) or 0),
                step=completed_epoch,
            )
            self.s.logger.log_metric(
                "LLM/Total Candidate Evaluations",
                len(result.evaluations) + tool_evaluations,
                step=completed_epoch,
            )
        best = min(
            (float(ev.loss) for ev in result.evaluations if ev.valid and ev.loss is not None),
            default=float("nan"),
        )
        print(
            f"[epoch {completed_epoch}] "
            f"{'merged' if bool(settings['add_to_hall_of_fame']) else 'retained isolated'} "
            f"background LLM proposal launched at "
            f"epoch {launched_epoch} | requested={int(settings['num_candidates'])} | "
            f"valid={valid} | best={best:.4g} | HoF {before}->{after} | "
            f"llm_time={elapsed:.3g}s",
            flush=True,
        )
        return result

    def _register_llm_provenance(
        self,
        result: Any,
        *,
        launched_epoch: int,
        expose_to_rl: bool,
    ) -> None:
        """Record exact LLM candidates and prior topology ownership for later HOF labels."""

        hof = getattr(self.s.mult_env, "hall_of_fame", None)
        preexisting_topologies = set(self._rl_seen_topologies) | {
            serialize_crn(env.state)["topology_hash"]
            for env in (hof or ())
            if str(
                (getattr(env.state, "last_task_info", {}) or {}).get("source", "RL")
            ).upper() != "LLM"
        }
        validation = dict(getattr(result, "response_validation", {}) or {})
        accepted_indices = list(validation.get("accepted_candidate_indices", ()) or ())
        for result_index, evaluation in enumerate(result.evaluations):
            if not evaluation.valid or evaluation.env is None:
                continue
            crn = serialize_crn(evaluation.env.state)
            writer_index = (
                int(accepted_indices[result_index])
                if result_index < len(accepted_indices)
                else result_index
            )
            proposal_id = f"epoch-{int(launched_epoch)}:writer-member-{writer_index}"
            record = self._llm_provenance_by_topology.get(crn["topology_hash"])
            if record is None:
                record = {
                    "topology_first_emitter": (
                        "RL" if crn["topology_hash"] in preexisting_topologies else "LLM"
                    ),
                    "first_proposal_id": proposal_id,
                    "first_seen_epoch": int(launched_epoch),
                    "candidate_hashes": set(),
                    "exposed_to_rl": bool(expose_to_rl),
                }
                self._llm_provenance_by_topology[crn["topology_hash"]] = record
            record["candidate_hashes"].add(crn["candidate_hash"])
            record["exposed_to_rl"] = bool(record.get("exposed_to_rl") or expose_to_rl)
            info = dict(getattr(evaluation.env.state, "last_task_info", {}) or {})
            info.update(
                {
                    "source": "LLM",
                    "emitter": "LLM",
                    "provenance_class": "direct_llm",
                    "llm_proposal_id": proposal_id,
                    "llm_first_seen_epoch": int(record["first_seen_epoch"]),
                    "topology_first_emitter": record["topology_first_emitter"],
                }
            )
            evaluation.env.state.last_task_info = info
            object.__setattr__(evaluation, "task_info", info)

    def merge_isolated_llm_candidates(self) -> int:
        """Pool retained LLM candidates into the HOF after RL training ends."""

        settings = self._llm_loop
        hof = getattr(self.s.mult_env, "hall_of_fame", None)
        if not settings or hof is None:
            return 0
        candidates = list(settings.get("isolated_candidates", ()))
        for candidate in candidates:
            hof.add(candidate)
        settings["isolated_candidates"] = []
        return len(candidates)

    def _remember_rl_hof_topologies(self) -> None:
        """Remember every RL-emitted HOF topology seen before later LLM proposals."""

        hof = getattr(self.s.mult_env, "hall_of_fame", None)
        for env in (hof or ()):
            info = dict(getattr(env.state, "last_task_info", {}) or {})
            if str(info.get("source", "RL")).upper() == "LLM":
                continue
            self._rl_seen_topologies.add(serialize_crn(env.state)["topology_hash"])

    def llm_graph_history(self) -> List[Dict[str, Any]]:
        """Return summary records for LLM calls made during this trainer session."""
        if not self._llm_loop:
            return []
        return list(self._llm_loop.get("history", []))

    def _normalized_render_mode(self) -> Dict[str, Any]:
        """Return render mode in the dictionary format expected by mult_env.render."""
        mode = self.s.cfg.render.mode
        if isinstance(mode, str):
            return {"style": "logger", "task": mode, "format": "figure"}
        if isinstance(mode, dict):
            return mode
        raise TypeError(
            "cfg.render.mode must be either a task string like 'transients' "
            "or a render-mode dictionary with keys such as style/task/format."
        )

    def resimulate(
        self,
        crns: List[Any],
        *,
        task: Optional[TaskSpec] = None,
        u_list: Optional[List[np.ndarray]] = None,
        u_spec: Optional[tuple] = None,
        u_values: Optional[List[float]] = None,
        dose_range: Optional[Tuple[float, float, int]] = None,
        ic: Optional[Union[str, tuple]] = None,
        weights: Optional[Union[str, tuple]] = None,
        n_cpus: Optional[int] = None,
    ) -> List[Any]:
        """Clone and re-simulate CRNs under a task, optionally overriding conditions.

        This is mainly for re-evaluating existing CRNs (e.g., from the training Hall of Fame)
        under new experimental conditions such as different input scenarios or initial conditions,
        without mutating the original CRN objects.

        The method clones each CRN via `.clone()` before simulation, runs task reward evaluation
        (which triggers transient simulations internally), and returns the cloned CRNs with
        updated `last_task_info`.

        Args:
            crns: List of CRN objects to re-simulate. Each must implement `.clone()`.
            task: Optional TaskSpec to use. If None, defaults to `self.s.task`.
            u_list: Optional explicit list of input vectors for evaluation.
            u_spec: Optional input generation spec (same as `build_u_list`), used if `u_list` is None.
            u_values: Optional enumerated values used by `build_u_list` for grid tasks.
            dose_range: Optional (u_min, u_max, n) for dose_response input generation.
            ic: Optional IC spec override (same format accepted by `build_ic`).
            weights: Optional weights spec override (same format accepted by `build_weights`).
            n_cpus: Optional CPU override for evaluation. If None, uses `self.s.n_cpus`.

        Returns:
            List of cloned CRNs after evaluation. The returned CRNs have fresh `last_task_info`
            corresponding to this re-simulation.

        Raises:
            ValueError: If CRNs do not support `.clone()` or inputs have inconsistent dimensions.
        """
        if not crns:
            return []

        task_local = self.s.task if task is None else task

        # --- Clone first to avoid overwriting old last_task_info ---
        cloned_crns: List[Any] = []
        for i, c in enumerate(crns):
            if not hasattr(c, "clone"):
                raise ValueError(f"CRN at index {i} has no .clone() method.")
            new_c = c.clone()
            new_c.reset()  # reset old task info to avoid confusion
            cloned_crns.append(new_c)
            

        # --- Build evaluation u_list ---
        if u_list is None and u_spec is not None:
            u_list_eval = build_u_list(
                task_local.kind,
                n_inputs=task_local.n_inputs,
                u_values=u_values,
                dose_range=dose_range,
                u_spec=u_spec,
            )
        else:
            u_list_eval = task_local.u_list if u_list is None else u_list

        u_list_eval = [np.asarray(u, dtype=np.float32).reshape(-1) for u in u_list_eval]

        # Guard input dimension mismatch early (prevents ragged propensity / solver failures)
        expected_p = int(task_local.template_crn.num_inputs)
        for j, u in enumerate(u_list_eval):
            if u.size != expected_p:
                raise ValueError(
                    f"[Trainer.resimulate] u_list[{j}] length={u.size} != template_crn.num_inputs={expected_p}"
                )

        # --- Build reward function with overrides ---
        reward_fn = make_reward_fn_with_overrides(
            task_local,
            u_list=u_list_eval,
            ic_spec=ic,
            weights_spec=weights,
        )

        # --- Evaluate rewards in parallel using existing ParallelEnvironments machinery ---
        from RL4CRN.environments.environment import Environment
        from RL4CRN.environments.parallel_environments import ParallelEnvironments

        # NOTE: In RL4CRN, Environment signature is typically (crn0, max_added_reactions, ...)
        # Here we pass cloned CRN as the initial template/state.
        envs = [
            Environment(crn, self.s.cfg.train.max_added_reactions, logger=None, logger_schedule=1)
            for crn in cloned_crns
        ]

        eval_env = ParallelEnvironments(
            envs,
            hall_of_fame_size=0,  # we will not use the internal HoF for this evaluation
            N_CPUs=int(n_cpus) if n_cpus is not None else int(self.s.n_cpus),
            logger=None,
        )

        # Only need reward evaluation (reward_fn runs simulations internally)
        _ = eval_env.get_reward(reward_fn)

        # Return the updated CRN states (the clones)
        return [env.state for env in envs]



    def step_epoch(self) -> Tuple[float, float]:
        """Run a single epoch: rollout, reward eval, and policy update.

        Returns:
            Tuple (best_loss, median_loss) over the batch.
        """
        mult_env = self.s.mult_env
        agent = self.s.agent
        cfg = self.s.cfg

        mult_env.reset()

        for _ in range(cfg.train.max_added_reactions):
            obs = mult_env.observe(self.s.observer, self.s.tensorizer)
            actions, raw_actions = agent.act(obs, self.s.actuator)
            mult_env.step(actions, self.s.stepper, raw_actions=raw_actions)

        # IMPORTANT:
        # Passing a bound method (e.g. self._compute_loss) forces joblib to pickle `self`,
        # which drags in agent/policy and breaks multiprocessing serialization.
        base_reward_fn = self.s.task.compute_reward
        forbidden_signatures = self._forbidden_topologies().signature_set()
        forbidden_loss = float(getattr(cfg.train, "forbidden_topology_loss", 1e9))
        if forbidden_signatures:
            reward_fn = partial(
                reward_with_forbidden_topologies,
                reward_fn=base_reward_fn,
                forbidden_signatures=forbidden_signatures,
                forbidden_loss=forbidden_loss,
            )
        else:
            reward_fn = base_reward_fn
        rewards = mult_env.get_reward(reward_fn)
        self._remember_rl_hof_topologies()
        self._log_loss_components(self.state.epoch)
        self._refresh_forbidden_topologies(self.state.epoch)
        self._maybe_run_llm_graph(self.state.epoch)
        self._maybe_persist_hof(self.state.epoch)

        agent.update(
            rewards,
            step_iteration=self.state.epoch,
            hof=mult_env.hall_of_fame,
            observer=self.s.observer,
            tensorizer=self.s.tensorizer,
            stepper=self.s.stepper,
            use_sil=True,
            sil_weighting_scheme="uniform",
            sil_batch_size=None,
        )

        best = float(np.min(rewards))
        med = float(np.median(rewards))
        self.state.history.append({"epoch": float(self.state.epoch), "best": best, "median": med})
        self.state.epoch += 1
        return best, med, rewards


    def run(self, epochs: int, checkpoint_path: Optional[str] = None) -> None:
        """Run training for a chunk of epochs.

        Args:
            epochs: Number of epochs to run in this chunk.
            checkpoint_path: If provided, saves a checkpoint periodically and on interrupt.
        """
        cfg = self.s.cfg
        self.s.agent.policy.train()

        def _maybe_save(current_epoch: int) -> None:
            if checkpoint_path is None:
                return
            cadence = max(1, cfg.train.render_every) if cfg.train.render_every else 1
            if current_epoch % cadence == 0:
                self.save(checkpoint_path)

        try:
            for _ in range(epochs):
                best, med, rewards = self.step_epoch()
                e = self.state.epoch - 1
                if cfg.train.render_every and (e % cfg.train.render_every == 0):
                    print(f"[epoch {e}] best loss={best:.4g} | median loss={med:.4g}")
                    self.s.mult_env.render(
                        rewards,
                        n_best=self.s.cfg.render.n_best,
                        disregarded_percentage=self.s.cfg.render.disregarded_percentage,
                        mode=self._normalized_render_mode(),
                    )
                _maybe_save(e)

        except KeyboardInterrupt:
            print("\nStopped early (KeyboardInterrupt). You can inspect and resume by calling run(...) again.")
            if checkpoint_path is not None:
                self.save(checkpoint_path)
        finally:
            if self._results_db is not None and self.state.epoch > 0:
                self._maybe_persist_hof(self.state.epoch - 1, force=True)
                self.flush_results_database()

    def best_crn(self) -> Optional[Any]:
        """Return the best CRN currently in the hall of fame.

        Returns:
            Best CRN object if available, else None.
        """
        hof_crns = [env.state for env in self.s.mult_env.hall_of_fame]
        if not hof_crns:
            return None
        return min(hof_crns, key=lambda c: c.last_task_info.get("reward", np.inf))

    def sample(
        self,
        n_samples: int,
        sample_hof_size: int,
        *,
        u_list: Optional[List[np.ndarray]] = None,
        u_spec: Optional[tuple] = None,
        u_values: Optional[List[float]] = None,
        dose_range: Optional[Tuple[float, float, int]] = None,
        ic: Optional[Union[str, tuple]] = None,
        weights: Optional[Union[str, tuple]] = None,
    ) -> HallOfFame:
        """Convenience wrapper around Session.sample to sample from the current policy."""
        return self.s.sample(
            n_samples,
            sample_hof_size,
            u_list=u_list,
            u_spec=u_spec,
            u_values=u_values,
            dose_range=dose_range,
            ic=ic,
            weights=weights,
        )

    def inspect(
        self,
        crn: Any,
        *,
        plot: bool = True,
        plot_type: Optional[str] = None,
        title: str = "CRN",
        **kwargs,
    ) -> Any:
        """Print and optionally plot a given CRN.

        Args:
            crn: The CRN object to inspect.
            plot: If True, call the appropriate plot method on the CRN (if available).
            plot_type: Optional plot suffix (e.g., "transient_response", "logic_response").
                If None, it is inferred from `self.s.cfg.task.kind`.
            title: Header label for the printed inspection.
            **kwargs: Passed through to the selected plotting function.

        Returns:
            The same CRN object (for convenience).
        """
        if crn is None:
            print(f"{title}: None")
            return None

        print(f"{title} loss:", crn.last_task_info.get("reward", None))
        print(crn)

        if not plot:
            return crn

        if plot_type is None:
            kind = getattr(self.s.cfg.task, "kind", None)
            kind_to_plot = {
                "logic": "logic_response",
                "dose_response": "dose_response",
                "oscillator_mean": "frequency_content",
                "oscillator_freq": "frequency_content",
                "tracking": "transient_response",
                "classifiers": "phase_portrait",
                "ssa_tracking": "SSA_transient_response",
                "ssa_robust": "SSA_transient_response",
                "habituation": "transient_response_piecewise",
                "habituation_gap" : "transient_response_piecewise",
                "habituation_hallmarks" : "habituation_hallmarks",
                "habituation_hallmarks_mmc2" : "habituation_hallmarks",
                "classification" : "phase_portrait"
            }
            plot_type = kind_to_plot.get(kind, "transient_response")
            if plot_type == "transient_response" and kind not in (None, "transient_response"):
                print(f"WARNING: Unknown task kind {kind!r}, defaulting to plot_transient_response().")

        method_name = f"plot_{plot_type}"
        plot_fn = getattr(crn, method_name, None)

        if plot_fn is None:
            print(f"WARNING: CRN has no method {method_name}(). Skipping plot.")
            return crn

        plot_fn(**kwargs)
        return crn


    def inspect_best(
        self,
        *,
        plot: bool = True,
        plot_type: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """Inspect the current best CRN in the Hall of Fame."""
        best = self.best_crn()
        if best is None:
            print("Hall of Fame is empty.")
            return None
        return self.inspect(best, plot=plot, plot_type=plot_type, title="Best CRN", **kwargs)


    def inspect_hof(
        self,
        idx: int,
        *,
        plot: bool = True,
        plot_type: Optional[str] = None,
        sort_by_reward: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        """Inspect a Hall-of-Fame CRN by index.

        Args:
            idx: Index into the HoF list. If `sort_by_reward=True`, index is taken
                after sorting by ascending reward.
            plot: If True, plot (if possible).
            plot_type: Optional plot suffix; inferred from task kind if None.
            sort_by_reward: If True, sort HoF by `last_task_info['reward']` ascending.

        Returns:
            Selected CRN if available, else None.
        """
        hof_items = list(self.s.mult_env.hall_of_fame)
        if not hof_items:
            print("Hall of Fame is empty.")
            return None

        crns = [item.state for item in hof_items]

        if sort_by_reward:
            crns = sorted(crns, key=lambda c: c.last_task_info.get("reward", float("inf")))

        if idx < 0 or idx >= len(crns):
            print(f"Index out of range: idx={idx}, HoF size={len(crns)}")
            return None

        return self.inspect(crns[idx], plot=plot, plot_type=plot_type, title=f"HoF[{idx}]", **kwargs)

    def save(self, path: str) -> None:
        """Save a training checkpoint.

        Args:
            path: File path to save.
        """
        payload = {
            "config": self.s.cfg.to_dict(),
            "epoch": self.state.epoch,
            "history": self.state.history,
            "policy_state_dict": self.s.agent.policy.state_dict(),
            "hall_of_fame_crns": [env.state for env in self.s.mult_env.hall_of_fame],
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "sample_hof_envs": list(self.s.sample_hof) if getattr(self.s, "sample_hof", None) is not None else [],
            "sample_hof_max_size": getattr(self.s.sample_hof, "max_size", 0) if getattr(self.s, "sample_hof", None) is not None else 0,
        }
        with open(path, "wb") as f:
            cloudpickle.dump(payload, f)
        print(f"Saved checkpoint: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load a training checkpoint.

        Args:
            path: File path to load.
            strict: Passed through to policy.load_state_dict.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            payload = cloudpickle.load(f)

        self.state.epoch = int(payload.get("epoch", 0))
        self.state.history = payload.get("history", [])

        self.s.agent.policy.load_state_dict(payload["policy_state_dict"], strict=strict)

        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])
        if "python_rng" in payload:
            random.setstate(payload["python_rng"])

        # --- Restore HoF into the live mult_env ---
        hof_crns = payload.get("hall_of_fame_crns", []) or []

        class _HOFItem:
            """Minimal wrapper so code can use `env.state`."""
            def __init__(self, state):
                self.state = state

        self.s.mult_env.hall_of_fame = [_HOFItem(crn) for crn in hof_crns]

        # Keep copies for debugging/inspection
        self._loaded_hof = hof_crns
        self._loaded_cfg = payload.get("config", None)

        # --- Restore sample HoF (if present) ---
        sample_envs = payload.get("sample_hof_envs", []) or []
        sample_hof_max_size = int(payload.get("sample_hof_max_size", len(sample_envs)) or len(sample_envs))

        if sample_envs:
            self.s.sample_hof = HallOfFame(max_size=max(1, sample_hof_max_size))
            self.s.sample_hof.add_all(sample_envs)
        else:
            self.s.sample_hof = HallOfFame(max_size=max(1, sample_hof_max_size)) if sample_hof_max_size > 0 else None

        # IMPORTANT: rebuild the reward callable (avoid relying on pickled closures)
        self.s.task.compute_reward = make_reward_fn_with_overrides(self.s.task)

        print(
            f"Loaded checkpoint: {path} (epoch={self.state.epoch}) | "
            f"restored_hof={len(self.s.mult_env.hall_of_fame)}"
        )


    def loaded_hof(self) -> Optional[List[Any]]:
        """Return hall-of-fame CRNs loaded from a checkpoint.

        Returns:
            List of CRN objects if present, else None.
        """
        return self._loaded_hof

    def get_sampled_crns(self) -> List[Any]:
        """Return CRN states from the current sample HoF (best->worst)."""
        if self.s.sample_hof is None:
            return []
        return [env.state for env in self.s.sample_hof]


def make_session_and_trainer(cfg: Config, task: TaskSpec, device: str = "auto", logger: Any = None) -> Trainer:
    """Convenience function to build a session and trainer.

    Args:
        cfg: Configuration.
        task: Materialized TaskSpec object.
        device: Device preference ("auto", "cpu", or "cuda").

    Returns:
        Trainer object.
    """
    dev = get_device(device)
    session = Session.from_config(cfg, task=task, device=dev, logger=logger)
    trainer = Trainer(session)
    return trainer


#### HELPERS for printing and reward smoke tests ####
def print_task_summary(task, max_preview=3):
    """Compact TaskSpec summary."""
    print("Task:", task.kind)
    print("time_horizon:", task.time_horizon.shape, f"[0..{task.time_horizon[-1]}]")
    print("num scenarios:", len(task.u_list))
    if len(task.u_list) > 0:
        print(f"first {min(max_preview, len(task.u_list))} u:", task.u_list[:max_preview])
    print()


def run_smoke_reward(task, state, label=""):
    """Call task.compute_reward on a given state and print normalized output."""
    out = task.compute_reward(state)
    if isinstance(out, tuple):
        loss, info = out
    else:
        loss, info = out, {}
    print(f"[reward smoke{(' - ' + label) if label else ''}] loss={float(loss):.6g} | info_keys={list(info.keys())[:10]}")
    return out



def load_session_and_trainer(
    checkpoint_path: str,
    *,
    task = None,
    device: str = "auto",
    strict: bool = True,
):
    """Load a checkpoint and reconstruct a working Trainer.

    This convenience function rebuilds the Session/Trainer wiring from scratch
    and then applies checkpoint state (policy weights, training state, HoFs,
    RNG state). It also rebuilds runtime-only callables (e.g., task.compute_reward).

    Notes:
        - If `task` is provided, it is used as the task definition and the checkpoint
          policy weights/state are loaded onto it.
        - If `task` is None, this function expects the checkpoint's `config` to
          contain a serializable TaskSpec under `config['task']`.

    Args:
        checkpoint_path: Path to the checkpoint file created by `Trainer.save`.
        task: Optional TaskSpec to use instead of the checkpoint's saved task.
        device: Device preference ("auto", "cpu", "cuda").
        strict: Whether to strictly enforce key matching in `load_state_dict`.

    Returns:
        Trainer object fully reconstructed and ready to use.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If required keys are missing and `task` is not provided.
        ValueError: If task reconstruction fails.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    with open(checkpoint_path, "rb") as f:
        payload = cloudpickle.load(f)

    cfg_dict = payload.get("config", None)
    if cfg_dict is None:
        raise KeyError("Checkpoint is missing 'config'.")

    # ----------------------------
    # Rebuild Config object
    # ----------------------------
    # We rebuild only the core cfg sections; cfg.task is set later.
    cfg = Config()
    if "solver" in cfg_dict:
        for k, v in cfg_dict["solver"].items():
            if hasattr(cfg.solver, k):
                setattr(cfg.solver, k, v)

    if "train" in cfg_dict:
        for k, v in cfg_dict["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)

    if "policy" in cfg_dict:
        for k, v in cfg_dict["policy"].items():
            if hasattr(cfg.policy, k):
                setattr(cfg.policy, k, v)

    if "agent" in cfg_dict:
        for k, v in cfg_dict["agent"].items():
            if hasattr(cfg.agent, k):
                setattr(cfg.agent, k, v)

    # ----------------------------
    # Rebuild TaskSpec
    # ----------------------------
    if task is None:
        task_dict = cfg_dict.get("task", None)
        if task_dict is None:
            raise KeyError(
                "Checkpoint config has no 'task'. "
                "Pass `task=...` to load_session_and_trainer()."
            )

        # Expect TaskSpec was serialized via asdict (so nested dicts/lists/arrays)
        # We reconstruct TaskSpec by passing known fields (ignore unknown keys).
        ts_kwargs = {}
        for field_name in TaskSpec.__dataclass_fields__.keys():
            if field_name in task_dict:
                ts_kwargs[field_name] = task_dict[field_name]

        task = TaskSpec(**ts_kwargs)

        # Ensure arrays are numpy arrays where needed
        if not isinstance(task.time_horizon, np.ndarray):
            task.time_horizon = np.asarray(task.time_horizon, dtype=np.float32)

        # Ensure u_list entries are numpy arrays
        task.u_list = [np.asarray(u, dtype=np.float32) for u in (task.u_list or [])]

    # Rebuild runtime parts that should not rely on pickling
    if task.time_horizon is None or task.time_horizon.size == 0:
        task.time_horizon = make_time_grid(task.t_f, task.n_t)

    if task.u_list is None or len(task.u_list) == 0:
        tk = get_task_kind(task.kind)
        tk.validate(task)
        task.u_list = tk.build_u_list(task, overrides={})

    # IC
    task.ic = build_ic(task.species_labels, task.ic_spec)

    # Weights for relevant tasks
    if task.kind in ("logic", "tracking", "dose_response", "ssa_tracking", "ssa_robust"):
        task.weights = build_weights(q=1, n_t=task.n_t, w_spec=task.weights_spec)

    # Reward callable
    task.compute_reward = make_reward_fn_with_overrides(task)

    # ----------------------------
    # Rebuild Session + Trainer
    # ----------------------------
    trainer = make_session_and_trainer(cfg, task, device=device)
    session = trainer.s

    # ----------------------------
    # Load policy weights + trainer state
    # ----------------------------
    trainer.s.agent.policy.load_state_dict(payload["policy_state_dict"], strict=strict)

    trainer.state.epoch = int(payload.get("epoch", 0))
    trainer.state.history = payload.get("history", [])

    # ----------------------------
    # Restore train HoF into live mult_env
    # ----------------------------
    hof_crns = payload.get("hall_of_fame_crns", []) or []

    class _HOFItem:
        """Minimal wrapper so code can use `env.state`."""
        def __init__(self, state):
            self.state = state

    trainer.s.mult_env.hall_of_fame = [_HOFItem(crn) for crn in hof_crns]
    trainer._loaded_hof = hof_crns
    trainer._loaded_cfg = cfg_dict

    # ----------------------------
    # Restore sample HoF (your HallOfFame container)
    # ----------------------------
    sample_envs = payload.get("sample_hof_envs", []) or []
    sample_hof_max_size = int(payload.get("sample_hof_max_size", len(sample_envs)) or len(sample_envs))

    if sample_envs:
        session.sample_hof = HallOfFame(max_size=max(1, sample_hof_max_size))
        session.sample_hof.add_all(sample_envs)
    else:
        session.sample_hof = HallOfFame(max_size=max(1, sample_hof_max_size)) if sample_hof_max_size > 0 else None

    # ----------------------------
    # Restore RNG state (optional, but nice)
    # ----------------------------
    if "torch_rng" in payload:
        torch.set_rng_state(payload["torch_rng"])
    if "numpy_rng" in payload:
        np.random.set_state(payload["numpy_rng"])
    if "python_rng" in payload:
        random.setstate(payload["python_rng"])

    return trainer





# ----------------------------
# TaskKind interface + registry
# ----------------------------

class TaskKindBase(ABC):
    """Abstract base class for task-kind implementations.

    Each task kind encapsulates:
      - validation of required parameters
      - default semantics for inputs (u_list)
      - construction of weights / reward function

    Defaults must live here, NOT in build_u_list().
    """

    kind: str  # subclasses must set

    @staticmethod
    def help() -> Dict[str, Any]:
        """Describe the expected `params` dictionary for this task kind.

        Returns:
            Dictionary describing required/optional keys and any notes.
        """
        return {
            "required": {},
            "optional": {},
            "notes": "",
        }

    @classmethod
    def pretty_help(
        cls,
        *,
        width: int = 100,
        bullet: str = "-",
        return_str: bool = False,
    ) -> Optional[str]:
        """Pretty-print the task-kind help specification in a Markdown-like list format.

        This uses `cls.help()` (a static method implemented by each TaskKind).
        The expected shape is:

            {
              "required": {<key>: <description>, ...},
              "optional": {<key>: <description>, ...},
              "notes": <string or list of strings>
            }

        Args:
            width: Maximum line width for wrapping descriptions.
            bullet: Bullet marker to use for list items (default "-").
            return_str: If True, return the formatted string instead of printing.

        Returns:
            If return_str=True, returns the formatted help string. Otherwise None.
        """
        spec: Dict[str, Any] = cls.help() if hasattr(cls, "help") else {}
        required: Dict[str, Any] = spec.get("required", {}) or {}
        optional: Dict[str, Any] = spec.get("optional", {}) or {}
        notes = spec.get("notes", "")

        def _wrap(desc: str, *, first_prefix: str, next_prefix: str) -> str:
            # width applies to total line width, so reduce by prefix length
            first_w = max(20, width - len(first_prefix))
            next_w = max(20, width - len(next_prefix))
            # wrap once with first prefix, then subsequent lines with next prefix
            wrapped = textwrap.fill(
                desc,
                width=first_w + len(first_prefix),
                initial_indent=first_prefix,
                subsequent_indent=next_prefix,
                break_long_words=False,
                break_on_hyphens=False,
            )
            # textwrap uses same width for all lines; we already compensated via indent sizes
            return wrapped

        def _format_section(title: str, items: Dict[str, Any]) -> list[str]:
            lines: list[str] = [f"**{title}**"]
            if not items:
                lines.append(f"{bullet} (none)")
                return lines

            for k, v in items.items():
                desc = str(v).strip()
                if not desc:
                    desc = "(no description)"
                # `- key: desc` with wrapped continuation aligned under desc
                first_prefix = f"{bullet} `{k}`: "
                next_prefix = " " * len(first_prefix)
                lines.append(_wrap(desc, first_prefix=first_prefix, next_prefix=next_prefix))
            return lines

        # normalize notes to list[str]
        notes_list: list[str] = []
        if isinstance(notes, (list, tuple)):
            notes_list = [str(x).strip() for x in notes if x is not None and str(x).strip()]
        else:
            s = str(notes).strip() if notes is not None else ""
            if s:
                notes_list = [s]

        lines: list[str] = []
        kind_name = getattr(cls, "kind", cls.__name__)
        lines.append(f"### TaskKind `{kind_name}`")
        lines.append("")
        lines.extend(_format_section("Required params", required))
        lines.append("")
        lines.extend(_format_section("Optional params", optional))

        if notes_list:
            lines.append("")
            lines.append("**Notes**")
            for n in notes_list:
                first_prefix = f"{bullet} "
                next_prefix = "  "
                lines.append(_wrap(n, first_prefix=first_prefix, next_prefix=next_prefix))

        out = "\n".join(lines)

        if return_str:
            return out
        print(out)


    def validate(self, task: TaskSpec) -> None:
        """Validate that the TaskSpec contains required fields.

        Args:
            task: TaskSpec instance.

        Raises:
            ValueError: If required fields are missing or inconsistent.
        """
        return

    def build_time_horizon(self, task: TaskSpec) -> np.ndarray:
        """Build or reuse the time horizon.

        Args:
            task: TaskSpec instance.

        Returns:
            Time grid array of shape (n_t,) float32.
        """
        if isinstance(task.time_horizon, np.ndarray) and task.time_horizon.size > 0:
            return task.time_horizon
        return make_time_grid(task.t_f, task.n_t)

    @abstractmethod
    def default_u_list(self, task: TaskSpec) -> List[np.ndarray]:
        """Default semantics for generating u_list for this kind.

        Args:
            task: TaskSpec instance.

        Returns:
            List of float32 input vectors, each shape (p,).

        Raises:
            ValueError: If required params are missing for default generation.
        """
        raise NotImplementedError

    def build_u_list(self, task: TaskSpec, overrides: Dict[str, Any]) -> List[np.ndarray]:
        """Build or override the u_list for evaluation.

        Precedence:
            overrides['u_list']
            overrides['u_spec']
            task.u_list (if user provided explicit list)
            task.u_spec (special tags only)
            TaskKind.default_u_list(task)  # kind-specific semantics

        Args:
            task: TaskSpec instance.
            overrides: Override dictionary.

        Returns:
            List of input vectors (float32 arrays), each shape (p,).
        """
        if overrides.get("u_list") is not None:
            u_list = overrides["u_list"]
            return [np.asarray(u, dtype=np.float32).reshape(-1) for u in u_list]

        u_spec = overrides.get("u_spec", None)
        if u_spec is not None:
            return build_u_list(n_inputs=task.n_inputs, u_spec=u_spec)

        if task.u_list:
            return [np.asarray(u, dtype=np.float32).reshape(-1) for u in task.u_list]

        if task.u_spec is not None:
            return build_u_list(self.kind, n_inputs=task.n_inputs, u_spec=task.u_spec)

        return self.default_u_list(task)

    def build_ic(self, task: TaskSpec, overrides: Dict[str, Any]) -> Any:
        """Build the IC object from spec or override.

        Args:
            task: TaskSpec instance.
            overrides: Override dictionary, may contain 'ic_spec'.

        Returns:
            RL4CRN IC object.
        """
        ic_spec = overrides.get("ic_spec", task.ic_spec)
        return build_ic(task.species_labels, ic_spec)

    def build_weights(self, task: TaskSpec, overrides: Dict[str, Any]) -> Optional[np.ndarray]:
        """Build weights if needed by the task kind.

        Args:
            task: TaskSpec instance.
            overrides: Override dictionary, may contain 'weights_spec'.

        Returns:
            Weight matrix or None.
        """
        return None

    @abstractmethod
    def make_reward_fn(self, task: TaskSpec, overrides: Dict[str, Any]) -> Callable[[Any], Any]:
        """Construct reward function for this task kind."""
        raise NotImplementedError


_TASK_KIND_REGISTRY: Dict[str, type[TaskKindBase]] = {}

def register_task_kind(cls: type[TaskKindBase]) -> type[TaskKindBase]:
    """Register a TaskKindBase subclass into the global registry.

    Args:
        cls: TaskKind class.

    Returns:
        The same class for decorator usage.

    Raises:
        ValueError: If class does not define 'kind' or kind duplicates.
    """
    kind = getattr(cls, "kind", None)
    if not kind:
        raise ValueError(f"{cls.__name__} must define class attribute `kind`.")
    if kind in _TASK_KIND_REGISTRY:
        raise ValueError(f"Duplicate task kind registration: {kind!r}")
    _TASK_KIND_REGISTRY[kind] = cls
    return cls

def get_task_kind(kind: str) -> TaskKindBase:
    """Instantiate a task-kind handler by name.

    Args:
        kind: Task kind string.

    Returns:
        Instance of a TaskKindBase subclass.

    Raises:
        ValueError: If kind is unknown.
    """
    if kind not in _TASK_KIND_REGISTRY:
        raise ValueError(
            f"Unknown task kind {kind!r}. Registered kinds: {sorted(_TASK_KIND_REGISTRY.keys())}"
        )
    return _TASK_KIND_REGISTRY[kind]()

def overrides_get(
    task: TaskSpec,
    overrides: Dict[str, Any],
    key: str,
    *,
    fallback_attr: Optional[str] = None,
    default: Any = None,
) -> Any:
    """Resolve a parameter using precedence overrides > task.params > task.<attr>.

    Args:
        task: TaskSpec instance.
        overrides: Override dictionary.
        key: Key to search in overrides/task.params.
        fallback_attr: If provided, also search task.<fallback_attr>.
        default: Default if not found.

    Returns:
        Resolved value or default.
    """
    if overrides is not None and key in overrides and overrides[key] is not None:
        return overrides[key]
    if hasattr(task, "params") and isinstance(task.params, dict) and key in task.params and task.params[key] is not None:
        return task.params[key]
    if fallback_attr is not None and hasattr(task, fallback_attr):
        val = getattr(task, fallback_attr)
        if val is not None:
            return val
    return default
