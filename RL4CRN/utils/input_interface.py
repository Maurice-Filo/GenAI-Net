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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import copy
import os
import cloudpickle

import numpy as np
import torch


# ----------------------------
# Small general utilities
# ----------------------------

def get_device(prefer: str = "auto") -> str:
    """Return the torch device string to use.

    Args:
        prefer: Device preference. Options:
            - "auto": choose "cuda" if available, else "cpu"
            - "cpu": force CPU
            - "cuda": force CUDA (will raise if not available)

    Returns:
        Device string, either "cpu" or "cuda".

    Raises:
        RuntimeError: If prefer="cuda" but CUDA is not available.
        ValueError: If prefer is not one of {"auto","cpu","cuda"}.
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

    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    """Seed common RNG sources for reproducibility.

    Args:
        seed: Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Task builder
# ----------------------------

VectorLogic = Callable[[np.ndarray], Union[bool, np.bool_]]


def build_logic_task(
    logic_fn: VectorLogic,
    n_inputs: int,
    input_values: Iterable[float] = (0.0, 1.0),
) -> Tuple[List[np.ndarray], List[np.ndarray], VectorLogic]:
    """Build a truth-table dataset for a boolean logic function on a vector input.

    The convention is:
        - inputs are float vectors `u` of shape (n_inputs,)
        - the target output is a single float in {0.0, 1.0} stored as shape (1,)

    Args:
        logic_fn: Function mapping `u` (shape (n_inputs,)) to a boolean-like value.
        n_inputs: Number of input channels (length of u).
        input_values: Values to enumerate per input dimension (default: {0,1}).

    Returns:
        u_list: List of all input vectors (dtype float32).
        r_list: List of targets, each an array of shape (1,) float32.
        logic_fn: The original logic function (returned for convenience).
    """
    u_list = [np.array(u, dtype=np.float32) for u in product(list(input_values), repeat=n_inputs)]
    r_list = [np.array([float(bool(logic_fn(u)))], dtype=np.float32) for u in u_list]
    return u_list, r_list, logic_fn


# ----------------------------
# Config objects
# ----------------------------

@dataclass
class TaskCfg:
    """Task configuration.

    Attributes:
        n_inputs: Number of binary input channels.
        logic_fn: Vectorized logic function mapping u -> bool.
        input_values: Values per input dimension to enumerate.
        t_f: Final simulation time.
        N_t: Number of time points.
        ic_value: Default initial concentration used for all species in IC builder.
        steady_state_weight: Weight multiplier applied to the last time point in w.
    """
    n_inputs: int = 3
    logic_fn: VectorLogic = lambda u: bool(np.all(u))
    input_values: Tuple[float, float] = (0.0, 1.0)
    t_f: float = 100.0
    N_t: int = 1000
    ic_value: float = 0.01
    steady_state_weight: float = 1.0


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
        epochs: Total number of epochs (in tutorial you may run in chunks).
        max_added_reactions: Episode length: number of reaction-addition steps.
        render_every: Print progress every N epochs (0 disables).
        hall_of_fame_size: Hall-of-fame capacity in ParallelEnvironments.
        batch_multiplier: Batch size = batch_multiplier * num_cpus (if batch_size is "auto").
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
    entropy_weights_per_head: Dict[str, float] = field(default_factory=lambda: {
        "structure": 2.0,
        "continuous": 1.0,
        "discrete": 0.0,
        "input_influence": 0.0,
    })
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
    entropy_scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "entropy_weight": 1e-3,
        "topk_entropy_weight": 1.0,
        "remainder_entropy_weight": 1.0,
        "entropy_update_coefficient": 1,
        "entropy_schedule": 1000,
        "minimum_entropy_weight": 0.0,
    })
    risk_scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "risk": 0.95,
        "risk_update": 0.0,
        "max_risk": 1.0,
        "risk_schedule": 1000,
    })
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

        Presets are intended to give users a single switch for speed/quality.

        Args:
            name: Preset name. Supported:
                - "fast": small networks, looser tolerances
                - "balanced": sensible defaults
                - "quality": larger networks, more capacity

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

        raise ValueError(f"Unknown preset: {name!r}")

    @staticmethod
    def with_overrides(cfg: Config, **overrides: Dict[str, Any]) -> Config:
        """Return a deep-copied config with nested overrides applied.

        Example:
            cfg = Configurator.with_overrides(
                Configurator.preset("balanced"),
                task=dict(n_inputs=3, logic_fn=lambda u: u[0] and u[1]),
                train=dict(epochs=200),
            )

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

    Template structure:
        - Input-driven productions: ∅ -> X_i with channel u_i
        - Optional dilution: X_i -> ∅
        - Output species "OUT" is included as a species label

    Args:
        n_inputs: Number of inputs.
        include_dilution: Whether to include dilution reactions.
        solver: Solver configuration.
        n_support_species: Number of support species to include.
        dilution_rate: Dilution rate for species.
    Returns:
        Tuple (crn_template, species_labels).

    Notes:
        This function imports RL4CRN internals locally to keep the module import light.
    """
    # Local imports to avoid forcing RL4CRN heavy imports at module import time.
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


def build_MAK_library(crn_template, species_labels: List[str], order: int):
    """Construct and attach a mass-action reaction library.

    Args:
        crn_template: Compiled IOCRN template.
        species_labels: Species labels used by the library.
        order: Reaction order.

    Returns:
        Tuple (library, M, K, masks) where:

            - library: reaction library object
            - M: number of reactions in library
            - K: number of total parameters
            - masks: dict of parameter/logit masks
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


def dynamic_tracking_loss(ic, u_list, r_list, time_horizon, w, large_number: float = 1e4):
    """Build the reward function closure used by environments.

    Args:
        ic: RL4CRN IC object.
        u_list: List of input vectors.
        r_list: List of target outputs.
        time_horizon: 1D array of simulation time points.
        w: Weight array for dynamic_tracking_error, typically emphasizing steady-state.
        large_number: Penalty constant for invalid simulations.

    Returns:
        Function compute_reward(state) -> float
    """
    from RL4CRN.rewards.deterministic import dynamic_tracking_error

    def compute_reward(state):
        x0_list = ic.get_ic(state)
        return dynamic_tracking_error(
            state,
            u_list,
            x0_list,
            time_horizon,
            r_list,
            w,
            norm=1,
            LARGE_NUMBER=large_number,
        )

    return compute_reward


def build_envs(
    template,
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


def build_interfaces(library, device: str, allow_input_influence: bool = False):
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


def build_agent(policy, device: str, agent_cfg: AgentCfg, logger: Any = None):
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

    This is created once per configuration and reused for stop/resume training.
    """
    cfg: Config
    device: str
    n_cpus: int
    batch_size: int

    # Task + simulation
    u_list: List[np.ndarray]
    r_list: List[np.ndarray]
    logic_fn: VectorLogic
    time_horizon: np.ndarray
    w: np.ndarray

    # CRN + reward
    crn_template: Any
    species_labels: List[str]
    ic: Any
    compute_reward: Callable[[Any], float]

    # Library
    library: Any
    M: int
    K: int
    masks: Dict[str, Any]
    p: int

    # Environment + interfaces
    mult_env: Any
    observer: Any
    tensorizer: Any
    actuator: Any
    stepper: Any

    # Policy + agent
    policy: Any
    agent: Any

    @staticmethod
    def from_config(cfg: Config, device: Optional[str] = None) -> "Session":
        """Build a Session from a Config.

        Args:
            cfg: Configuration object.
            device: Device string. If None, uses cfg + auto selection.

        Returns:
            Initialized Session with all required RL4CRN objects wired up.
        """
        if device is None:
            device = get_device("auto")

        # Seed early for deterministic-ish behavior
        seed_everything(cfg.train.seed)

        n_cpus = cfg.train.n_cpus or (os.cpu_count() or 1)
        batch_size = cfg.train.batch_size or (cfg.train.batch_multiplier * n_cpus)

        # Task
        u_list, r_list, logic_fn = build_logic_task(
            logic_fn=cfg.task.logic_fn,
            n_inputs=cfg.task.n_inputs,
            input_values=cfg.task.input_values,
        )

        # Horizon + weights (steady-state by default)
        time_horizon = np.linspace(0.0, cfg.task.t_f, cfg.task.N_t, dtype=np.float32)
        w = np.zeros((1, cfg.task.N_t), dtype=np.float32)
        w[:, -1] = cfg.task.steady_state_weight * cfg.task.N_t

        # Template CRN + species labels
        crn_template, species_labels = build_template_crn(
            n_inputs=cfg.task.n_inputs,
            include_dilution=cfg.library.include_dilution,
            solver=cfg.solver,
        )

        # IC
        from RL4CRN.utils.ic import IC
        ic = IC(names=species_labels, values=[[cfg.task.ic_value for _ in species_labels]])

        # Reward function
        compute_reward = dynamic_tracking_loss(
            ic=ic,
            u_list=u_list,
            r_list=r_list,
            time_horizon=time_horizon,
            w=w,
        )

        # Library
        library, M, K, masks = build_MAK_library(crn_template, species_labels, order=cfg.library.order)
        p = crn_template.num_inputs

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
            u_list=u_list,
            r_list=r_list,
            logic_fn=logic_fn,
            time_horizon=time_horizon,
            w=w,
            crn_template=crn_template,
            species_labels=species_labels,
            ic=ic,
            compute_reward=compute_reward,
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
            session: Built Session containing envs, agent, and reward fn.
        """
        self.s = session
        self.state = TrainState()
        self._loaded_hof: Optional[List[Any]] = None

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

        rewards = mult_env.get_reward(self.s.compute_reward)

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

        This method is designed for notebooks:

            - You can interrupt with Ctrl+C
            - Then inspect intermediate results
            - Then call run(...) again to resume

        Args:
            epochs: Number of epochs to run in this chunk.
            checkpoint_path: If provided, saves a checkpoint periodically and on interrupt.
        """
        cfg = self.s.cfg
        self.s.agent.policy.train()

        def _maybe_save(current_epoch: int) -> None:
            if checkpoint_path is None:
                return
            # Save at the same cadence as progress prints (or every epoch if render_every=0)
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
            plot: If True, calls best_crn.plot_transient_response().

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

        The checkpoint includes:
            - epoch counter + history
            - policy weights
            - a snapshot of hall-of-fame CRNs
            - RNG states (NumPy + torch)

        Args:
            path: File path to save (passed to torch.save).
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

        Notes:
            - This restores policy weights and training counters.
            - Hall-of-fame CRNs are stored in Trainer._loaded_hof for inspection.
              (Re-inserting into ParallelEnvironments may depend on RL4CRN internals.)

        Args:
            path: File path to load (passed to torch.load).
            strict: Passed through to policy.load_state_dict.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            payload = cloudpickle.load(f)

        # Restore training counters
        self.state.epoch = int(payload.get("epoch", 0))
        self.state.history = payload.get("history", [])

        # Restore weights
        self.s.agent.policy.load_state_dict(payload["policy_state_dict"], strict=strict)

        # Restore RNGs
        if "torch_rng" in payload:
            torch.set_rng_state(payload["torch_rng"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])

        # HOF (kept for inspection unless you implement reinsertion)
        self._loaded_hof = payload.get("hall_of_fame_crns", None)

        # If you *want* to restore cfg (including functions) into the session:
        # NOTE: This won't automatically rebuild session objects; it just gives you the cfg back.
        self._loaded_cfg = payload.get("config", None)

        print(f"Loaded checkpoint: {path} (epoch={self.state.epoch})")

    def loaded_hof(self) -> Optional[List[Any]]:
        """Return hall-of-fame CRNs loaded from a checkpoint.

        Returns:
            List of CRN objects if present, else None.
        """
        return self._loaded_hof


def make_session_and_trainer(
    cfg: Config,
    device: str = "auto",
) -> Tuple[Session, Trainer]:
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
