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

from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import os

import cloudpickle
import numpy as np
import torch


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
class TaskSpec: # CHECKED ___ OK
    """Fully materialized task description used by environments.

    Attributes:
        name: Task name/kind (e.g., "logic", "tracking", "oscillator", ...).
        time_horizon: 1D array of time points.
        u_list: List of input vectors (each shape (p,), float32).
        ic: RL4CRN IC object.
        compute_reward: Reward callable. Must accept the environment state (CRN)
            and return either:
            - float loss, or
            - (float loss, dict info)
        render_mode: Optional rendering configuration.
    """
    name: str
    time_horizon: np.ndarray
    u_list: List[np.ndarray]
    ic: Any
    compute_reward: Callable[[Any], Union[float, Tuple[float, Dict[str, Any]]]]
    render_mode: Optional[dict] = "transients"


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
    p: Optional[int] = None,
    u_values: Optional[List[float]] = None,
    dose_range: Optional[Tuple[float, float, int]] = None,
    u_spec: Optional[tuple] = None,
) -> List[np.ndarray]: # CHECKED ___ OK
    """Construct a list of inputs for a task kind.

    Args:
        kind: Task kind.
        n_inputs: Number of input channels for "logic" tasks.
        p: Number of input channels for non-logic tasks (CRN template inputs).
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
        if tag == "custom":
            return args[0]
        if tag == "grid":
            values = args[0]
            dim = p if p is not None else n_inputs
            if dim is None:
                raise ValueError("u_spec=('grid', ...) needs p or n_inputs.")
            return [np.array(u, dtype=np.float32) for u in product(values, repeat=dim)]
        if tag == "linspace":
            u_min, u_max, n = args
            return [np.array([u], dtype=np.float32) for u in np.linspace(u_min, u_max, n)]
        raise ValueError(f"Unknown u_spec: {u_spec}")

    if kind == "logic":
        if n_inputs is None:
            raise ValueError("logic task needs n_inputs.")
        return [np.array(u, dtype=np.float32) for u in product([0.0, 1.0], repeat=n_inputs)]

    if kind in ("tracking", "oscillator", "ssa_tracking", "ssa_robust"):
        if p is None:
            raise ValueError(f"{kind} task needs p.")
        values = u_values if u_values is not None else [1.0]
        return [np.array(u, dtype=np.float32) for u in product(values, repeat=p)]

    if kind == "dose_response":
        u_min, u_max, n = dose_range if dose_range is not None else (0.0, 10.0, 10)
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


def _reward_to_tuple(reward_out: Union[float, Tuple[float, Dict[str, Any]]]) -> Tuple[float, Dict[str, Any]]:
    """Normalize reward outputs to (loss, info).

    Args:
        reward_out: Either a float loss or (float loss, info dict).

    Returns:
        Tuple (loss, info).
    """
    if isinstance(reward_out, tuple) and len(reward_out) == 2 and isinstance(reward_out[1], dict):
        return float(reward_out[0]), reward_out[1]
    return float(reward_out), {}


def make_task(
    kind: str,
    species_labels: List[str],
    *,
    # time
    t_f: float = 100.0,
    n_t: int = 1000,
    # inputs
    n_inputs: Optional[int] = None,
    p: Optional[int] = None,
    u_values: Optional[List[float]] = None,
    dose_range: Optional[Tuple[float, float, int]] = None,
    u_spec: Optional[tuple] = None,
    # IC / weights
    ic: Union[str, tuple] = "zero",
    weights: Union[str, tuple] = "transient",
    # targets
    logic_fn: Optional[VectorLogic] = None,
    target: Union[str, float, None] = None,
    target_fn: Optional[Callable[[float], float]] = None,
    # oscillator knobs
    osc_w: Optional[List[float]] = None,
    t0: float = 20.0,
    # SSA knobs
    n_trajectories: int = 256,
    max_threads: int = 1024,
    cv_weight: float = 1.0,
    rpa_weight: float = 1.0,
) -> TaskSpec:
    """Create a TaskSpec for several common tutorial tasks.

    Supported kinds:
        - "logic": truth-table of a boolean function over {0,1}^n
        - "tracking": mean-tracking (e.g. copy input 0 or constant target)
        - "dose_response": 1D dose sweep with a provided target function
        - "oscillator": oscillation_error reward
        - "ssa_tracking": SSA tracking reward
        - "ssa_robust": SSA robust tracking reward

    Args:
        kind: Task kind.
        species_labels: Species names (used for IC construction).
        t_f: Final simulation time.
        n_t: Number of time points.
        n_inputs: Input dimension for "logic".
        p: Input dimension for non-logic tasks.
        u_values: Enumerated values for grid inputs (non-logic).
        dose_range: Dose sweep parameters for "dose_response".
        u_spec: Custom input generation spec.
        ic: Initial condition spec.
        weights: Weight spec for tracking tasks.
        logic_fn: Boolean function for "logic".
        target: Target spec for tracking/SSA tasks.
        target_fn: Target function for dose response.
        osc_w: Weight vector for oscillation_error.
        t0: Oscillation error start time.
        n_trajectories: SSA trajectories.
        max_threads: SSA max threads.
        cv_weight: Robust SSA coefficient-of-variation weight.
        rpa_weight: Robust SSA relative peak amplitude weight.

    Returns:
        TaskSpec instance.

    Raises:
        ValueError: If kind is unknown or required knobs are missing.
    """
    from RL4CRN.rewards.deterministic import dynamic_tracking_error, oscillation_error
    from RL4CRN.rewards.stochastic import dynamic_tracking_error_SSA, robust_tracking_loss_SSA

    time_horizon = make_time_grid(t_f, n_t)
    u_list = build_u_list(
        kind, n_inputs=n_inputs, p=p, u_values=u_values, dose_range=dose_range, u_spec=u_spec
    )
    ic_obj = build_ic(species_labels, ic)

    if kind == "logic":
        if logic_fn is None:
            raise ValueError("logic task needs logic_fn.")
        r_list = [np.array([float(bool(logic_fn(u)))], dtype=np.float32) for u in u_list]
        w = build_weights(q=1, n_t=n_t, w_spec=weights)

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            out = dynamic_tracking_error(
                state, u_list, x0_list, time_horizon, r_list, w, norm=1, LARGE_NUMBER=1e4
            )
            return _reward_to_tuple(out)

        return TaskSpec("logic", time_horizon, u_list, ic_obj, compute_reward)

    if kind == "tracking":
        if target == "copy_input0":
            r_list = [np.array([u[0]], dtype=np.float32) for u in u_list]
        elif isinstance(target, (int, float)):
            r_list = [np.array([float(target)], dtype=np.float32) for _ in u_list]
        else:
            raise ValueError("tracking needs target='copy_input0' or a constant float target.")
        w = build_weights(q=1, n_t=n_t, w_spec=weights)

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            out = dynamic_tracking_error(
                state, u_list, x0_list, time_horizon, r_list, w, norm=1, LARGE_NUMBER=1e4
            )
            return _reward_to_tuple(out)

        return TaskSpec("tracking", time_horizon, u_list, ic_obj, compute_reward)

    if kind == "dose_response":
        if target_fn is None:
            raise ValueError("dose_response needs target_fn(u)->y*.")
        w = build_weights(q=1, n_t=n_t, w_spec=weights)

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            r_list = [np.array([target_fn(float(u[0]))], dtype=np.float32) for u in u_list] * len(x0_list)
            out = dynamic_tracking_error(
                state, u_list, x0_list, time_horizon, r_list, w, norm=1, LARGE_NUMBER=1e4
            )
            return _reward_to_tuple(out)

        return TaskSpec("dose_response", time_horizon, u_list, ic_obj, compute_reward)

    if kind == "oscillator":
        mean_list = [np.array([u[0]], dtype=np.float32) for u in u_list]
        w_local = osc_w if osc_w is not None else [0.4, 0.0, 0.2, 0.4]

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            out = oscillation_error(
                state,
                u_list,
                x0_list,
                time_horizon,
                f_list=None,
                mean_list=mean_list,
                w=w_local,
                t0=t0,
                LARGE_NUMBER=1e4,
            )
            return _reward_to_tuple(out)

        return TaskSpec("oscillator", time_horizon, u_list, ic_obj, compute_reward)

    if kind == "ssa_tracking":
        if target == "copy_input0":
            r_list = [np.array([u[0]], dtype=np.float32) for u in u_list]
        else:
            raise ValueError("ssa_tracking currently supports target='copy_input0'.")
        w = build_weights(q=1, n_t=n_t, w_spec=weights)

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            out = dynamic_tracking_error_SSA(
                state,
                u_list,
                x0_list,
                time_horizon,
                r_list,
                w,
                n_trajectories=n_trajectories,
                max_threads=max_threads,
                norm=1,
                relative=False,
                LARGE_NUMBER=1e4,
                LARGE_PENALTY=1e4,
            )
            return _reward_to_tuple(out)

        return TaskSpec("ssa_tracking", time_horizon, u_list, ic_obj, compute_reward)

    if kind == "ssa_robust":
        if target == "copy_input0":
            r_list = [np.array([u[0]], dtype=np.float32) for u in u_list]
        else:
            raise ValueError("ssa_robust currently supports target='copy_input0'.")
        w = build_weights(q=1, n_t=n_t, w_spec=weights)

        def compute_reward(state: Any) -> Tuple[float, Dict[str, Any]]:
            x0_list = ic_obj.get_ic(state)
            out = robust_tracking_loss_SSA(
                state,
                u_list,
                x0_list,
                time_horizon,
                r_list,
                w,
                n_trajectories=n_trajectories,
                max_threads=max_threads,
                norm=1,
                relative=True,
                LARGE_NUMBER=1e3,
                LARGE_PENALTY=100,
                cv_weight=cv_weight,
                rpa_weight=rpa_weight,
            )
            return _reward_to_tuple(out)

        return TaskSpec("ssa_robust", time_horizon, u_list, ic_obj, compute_reward)

    raise ValueError(f"Unknown kind: {kind}")


# ----------------------------
# Config objects
# ----------------------------

@dataclass
class TaskCfg:
    """Task configuration.

    Attributes:
        kind: Task kind used by make_task().
        n_inputs: Number of binary input channels for logic tasks.
        logic_fn: Vectorized logic function mapping u -> bool (logic tasks).
        t_f: Final simulation time.
        N_t: Number of time points.
        ic_value: Default initial concentration for ("constant", ic_value).
        weights: Weight spec for tracking-style tasks.
        target: Target spec for tracking tasks.
    """
    kind: str = "logic"
    n_inputs: int = 3
    logic_fn: VectorLogic = lambda u: bool(np.all(u))
    t_f: float = 100.0
    N_t: int = 1000
    ic_value: float = 0.01
    weights: Union[str, tuple] = "steady_state"
    target: Union[str, float, None] = None
    target_fn: Optional[Callable[[float], float]] = None
    dose_range: Optional[Tuple[float, float, int]] = None


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
class LibraryCfg:
    """Reaction library configuration.

    Attributes:
        order: Maximum reaction order for the mass-action library.
        include_dilution: Whether to include dilution reactions in the template.
    """
    order: int = 2
    include_dilution: bool = False


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
class Config:
    """Top-level configuration container.

    Attributes:
        task: Task configuration.
        solver: Solver configuration.
        train: Training configuration.
        library: Library configuration.
        policy: Policy configuration.
        agent: Agent configuration.
    """
    task: TaskCfg = field(default_factory=TaskCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    library: LibraryCfg = field(default_factory=LibraryCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)

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

def build_template_crn(
    n_inputs: int,
    include_dilution: bool = False,
    solver: SolverCfg = SolverCfg(),
    n_support_species: int = 0,
    dilution_rate: float = 0.05,
):
    """Build and compile the template IO-CRN.

    Args:
        n_inputs: Number of inputs.
        include_dilution: Whether to include dilution reactions.
        solver: Solver configuration.
        n_support_species: Number of support species to include.
        dilution_rate: Dilution rate for species.

    Returns:
        Tuple (crn_template, species_labels).
    """
    from RL4CRN.iocrns.iocrn import IOCRN
    from RL4CRN.iocrns.reactions import MassAction

    productions = []
    dilutions = []
    species_labels = [f"X_{i+1}" for i in range(n_inputs)]

    for i, s in enumerate(species_labels):
        productions.append(
            MassAction(
                reactant_labels=[],
                product_labels=[s],
                input_channels=[f"u_{i+1}"],
                params=[1.0],
                params_controllability=[True],
            )
        )
        if include_dilution:
            dilutions.append(
                MassAction(
                    reactant_labels=[s],
                    product_labels=[],
                    input_channels=[None],
                    params=[dilution_rate],
                    params_controllability=[True],
                )
            )

    for j in range(n_support_species):
        support_label = f"S_{j+1}"
        species_labels.append(support_label)
        if include_dilution:
            dilutions.append(
                MassAction(
                    reactant_labels=[support_label],
                    product_labels=[],
                    input_channels=[None],
                    params=[dilution_rate],
                    params_controllability=[True],
                )
            )

    species_labels.append("OUT")

    crn_template = IOCRN(
        productions + dilutions,
        output_labels=["OUT"],
        solver=solver.algorithm,
        rtol=solver.rtol,
        atol=solver.atol,
    )
    crn_template.compile()
    return crn_template, species_labels


def build_MAK_library(crn_template: Any, species_labels: List[str], order: int):
    """Construct and attach a mass-action reaction library.

    Args:
        crn_template: Compiled IOCRN template.
        species_labels: Species labels used by the library.
        order: Reaction order.

    Returns:
        Tuple (library, M, K, masks).
    """
    from RL4CRN.iocrns.reaction_library import construct_mass_action_library

    library = construct_mass_action_library(species_labels=species_labels, order=order)
    crn_template.set_library_context(library)

    M = len(library.reactions)
    K = library.get_num_parameters()
    masks = {
        "continuous": library.get_parameter_mask(mode="continuous"),
        "discrete": library.get_parameter_mask(mode="discrete"),
        "logit": library.get_logit_mask(),
    }
    return library, M, K, masks


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

    @staticmethod
    def from_config(cfg: Config, device: Optional[str] = None) -> "Session":
        """Build a Session from a Config.

        Args:
            cfg: Configuration object.
            device: Torch device string. If None, auto-selects.

        Returns:
            Initialized Session with all required RL4CRN objects wired up.
        """
        if device is None:
            device = get_device("auto")

        seed_everything(cfg.train.seed)

        n_cpus = cfg.train.n_cpus or (os.cpu_count() or 1)
        batch_size = cfg.train.batch_size or (cfg.train.batch_multiplier * n_cpus)

        # Template CRN + species labels
        crn_template, species_labels = build_template_crn(
            n_inputs=cfg.task.n_inputs,
            include_dilution=cfg.library.include_dilution,
            solver=cfg.solver,
        )

        # Library
        library, M, K, masks = build_MAK_library(crn_template, species_labels, order=cfg.library.order)
        p = crn_template.num_inputs

        # Task (end-of-file task API)
        task = make_task(
            cfg.task.kind,
            species_labels=species_labels,
            t_f=cfg.task.t_f,
            n_t=cfg.task.N_t,
            n_inputs=cfg.task.n_inputs,
            p=p,
            ic=("constant", cfg.task.ic_value),
            weights=cfg.task.weights,
            logic_fn=cfg.task.logic_fn if cfg.task.kind == "logic" else None,
            target=cfg.task.target,
            target_fn=cfg.task.target_fn,          # <-- ADD THIS
            dose_range=cfg.task.dose_range,        # <-- ADD THIS (or default inside make_task)
        )

        # Environments
        mult_env = build_envs(
            template=crn_template,
            max_added_reactions=cfg.train.max_added_reactions,
            batch_size=batch_size,
            hall_of_fame_size=cfg.train.hall_of_fame_size,
            n_cpus=n_cpus,
            logger=None,
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
        agent = build_agent(policy=policy, device=device, agent_cfg=cfg.agent, logger=None)

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
        )


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
        return best, med


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
                best, med = self.step_epoch()
                e = self.state.epoch - 1
                if cfg.train.render_every and (e % cfg.train.render_every == 0):
                    print(f"[epoch {e}] best loss={best:.4g} | median loss={med:.4g}")
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

    def inspect_best(self, plot: bool = True) -> Optional[Any]:
        """Print and optionally plot the current best CRN.

        Args:
            plot: If True, calls best_crn.plot_transient_response() when available.

        Returns:
            Best CRN if present, else None.
        """
        best = self.best_crn()
        if best is None:
            print("Hall of Fame is empty.")
            return None

        print("Best loss:", best.last_task_info.get("reward", None))
        print(best)
        if plot and hasattr(best, "plot_transient_response"):
            best.plot_transient_response()
        return best

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


def make_session_and_trainer(cfg: Config, device: str = "auto") -> Tuple[Session, Trainer]:
    """Convenience function to build a session and trainer.

    Args:
        cfg: Configuration.
        device: Device preference ("auto", "cpu", or "cuda").

    Returns:
        Tuple (session, trainer).
    """
    dev = get_device(device)
    session = Session.from_config(cfg, device=dev)
    trainer = Trainer(session)
    return session, trainer
