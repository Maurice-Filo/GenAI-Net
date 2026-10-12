"""Compose legacy interfaces without allocating an unused rate policy."""

from copy import deepcopy
import math
import os

import torch

from RL4CRN.utils.input_interface import (
    Session, build_agent, build_envs, build_interfaces, build_policy, seed_everything,
)


def _consume_rate_initialization(cfg, masks, M):
    """Keep the legacy CPU initialization RNG stream using bounded scratch storage."""
    from RL4CRN.policies.parameter_generator_from_distribution import ParameterGeneratorFromDistribution
    if masks.get("discrete") is not None:
        raise ValueError("Structural exploration supports continuous scalar rates only")
    if masks.get("continuous") is None:
        return
    distribution = deepcopy(cfg.continuous_distribution)
    distribution["dim"] = len(masks["continuous"][0])
    with torch.device("meta"):
        ghost = ParameterGeneratorFromDistribution(distribution=distribution,
            backbone_attributes={"input_size": cfg.deep_layer_size + M,
                                 "hidden_size": cfg.width, "num_layers": cfg.depth}, device="meta")
    scratch = torch.empty(65536, device="cpu")
    for layer in ghost.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        gain = torch.nn.init.calculate_gain("leaky_relu", math.sqrt(5))
        bounds = [(layer.weight.numel(), math.sqrt(3.) * gain / math.sqrt(layer.in_features))]
        if layer.bias is not None:
            bounds.append((layer.bias.numel(), 1 / math.sqrt(layer.in_features)))
        for count, bound in bounds:
            for offset in range(0, count, scratch.numel()):
                scratch[:min(scratch.numel(), count - offset)].uniform_(-bound, bound)


def build_session(cfg, task, *, device="cpu", logger=None):
    """Return the ordinary Session type with a structure-only legacy policy shell."""
    if cfg.policy.ordering_enabled:
        raise ValueError("Exploration rollout uses unordered structural actions")
    seed_everything(cfg.train.seed)
    cfg.task = task
    n_cpus = cfg.train.n_cpus or (os.cpu_count() or 1)
    batch_size = cfg.train.batch_size or cfg.train.batch_multiplier * n_cpus
    template = task.template_crn
    library, M, K, masks = task.library_components
    mult_env = build_envs(template, cfg.train.max_added_reactions, batch_size,
                         cfg.train.hall_of_fame_size, n_cpus, logger)
    observer, tensorizer, actuator, stepper = build_interfaces(library, device, False)
    policy = build_policy(M, K, template.num_inputs,
                         {**masks, "continuous": None, "discrete": None}, device,
                         cfg.policy, template.num_reactions + cfg.train.max_added_reactions)
    _consume_rate_initialization(cfg.policy, masks, M)
    agent = build_agent(policy, device, cfg.agent, logger)
    return Session(cfg=cfg, device=device, n_cpus=n_cpus, batch_size=batch_size,
                   task=task, crn_template=template, species_labels=task.species_labels,
                   library=library, M=M, K=K, masks=masks, p=template.num_inputs,
                   mult_env=mult_env, observer=observer, tensorizer=tensorizer,
                   actuator=actuator, stepper=stepper, policy=policy, agent=agent, logger=logger)
