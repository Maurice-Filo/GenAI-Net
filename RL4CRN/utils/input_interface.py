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

from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import os

import cloudpickle
import numpy as np
import torch

from RL4CRN.iocrns.reaction_library import ReactionLibrary
from RL4CRN.iocrns.iocrn import IOCRN
from RL4CRN.utils.hall_of_fame import HallOfFame
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
        mode: Rendering mode, e.g., "transients", "inputs", "final_state", etc.
    """
    n_best: int = 30
    disregarded_percentage: float = 0.5
    mode: str = "transients"

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
        reward_fn = self.s.task.compute_reward
        rewards = mult_env.get_reward(reward_fn)

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
                    self.s.mult_env.render(rewards, n_best=self.s.cfg.render.n_best, disregarded_percentage=self.s.cfg.render.disregarded_percentage, mode=self.s.cfg.render.mode)
                _maybe_save(e)

        except KeyboardInterrupt:
            print("\nStopped early (KeyboardInterrupt). You can inspect and resume by calling run(...) again.")
            if checkpoint_path is not None:
                self.save(checkpoint_path)

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


def make_session_and_trainer(cfg: Config, task: TaskSpec, device: str = "auto", logger: Any = None) -> Tuple[Session, Trainer]:
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
