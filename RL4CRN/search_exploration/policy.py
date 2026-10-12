"""Reuse the legacy encoder/structure head without continuous-policy gradients."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .evaluation import materialize


class StructuralLearner(nn.Module):
    def __init__(self, session, evaluator, config):
        super().__init__()
        self.session, self.evaluator, self.cfg = session, evaluator, config
        base = session.policy
        if base.allow_input_influence:
            raise ValueError("Unsupported input-influence policy")
        # Parameter heads are not part of this module or its optimizer.
        self.encoder = base.encoder
        self.head = base.reaction_structure_head
        self.context_head = nn.Linear(4, base.deep_layer_size, bias=False).to(session.device)
        nn.init.zeros_(self.context_head.weight)
        self.M = base.M
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.version = 0
        self.baseline = 0.
        self.last_sil_info = {}
        self.act_context = [0., 0., 0., 0.]
        observer = session.observer
        from RL4CRN.env2agent_interface.explicit_observer import ExplicitObserver
        from RL4CRN.env2agent_interface.explicit_tensorizer import ExplicitTensorizer
        if type(observer) is not ExplicitObserver or type(session.tensorizer) is not ExplicitTensorizer:
            raise ValueError("Tensor rollout requires the explicit observer/tensorizer contract")
        if observer.allow_input_observation:
            raise ValueError("Tensor rollout does not support input influence")
        initial = session.tensorizer.tensorize(observer.observe(evaluator.crn_template)).float()
        offsets = [session.library.parameter_lookup_table[i] for i in range(self.M)]
        self.register_buffer("initial_observation", initial, persistent=False)
        self.register_buffer("parameter_offsets", torch.tensor(offsets, device=session.device), persistent=False)
        forbidden = torch.zeros(self.M, dtype=torch.bool, device=session.device)
        forbidden[list(evaluator.forbidden_reaction_ids)] = True
        self.register_buffer("forbidden_mask", forbidden, persistent=False)
        self._sample_cache = None

    @property
    def policy(self):
        return self

    def distribution(self, envs, context):
        state = torch.stack([self.session.tensorizer.tensorize(self.session.observer.observe(env.state))
                             for env in envs]).float()
        return self.distribution_from_state(state, context)

    def distribution_from_state(self, state, context):
        h = self.encoder(state)
        ctx = torch.as_tensor(context, dtype=h.dtype, device=h.device).reshape(1, 4)
        logits = self.head(h + self.context_head(ctx))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite structural logits")
        mask = state[:, :self.M].bool().clone()
        mask |= self.forbidden_mask
        if mask.all(dim=1).any():
            raise ValueError("No admissible structural actions")
        return Categorical(logits=logits.masked_fill(mask, -torch.inf))

    def forward(self, state, mode="full", action=None, structure_temp=None):
        if mode != "full":
            raise ValueError("Structural policy supports full structural actions only")
        dist = self.distribution_from_state(state, self.act_context)
        ids = dist.sample() if action is None else torch.tensor([a["reaction index"] for a in action], device=state.device)
        logps = dist.log_prob(ids)
        if action is not None:
            return logps
        raw = [{"reaction index": rid, "parameters": [self.cfg.initial_rate],
                "continuous parameters": [self.cfg.initial_rate], "discrete parameters": None}
               for rid in ids.detach().cpu().tolist()]
        return raw, logps, dist.entropy()

    def act(self, states, actuator, mode="full"):
        raw, _, _ = self.forward(states, mode=mode)
        return [actuator.actuate(a) for a in raw], raw

    def trajectories(self, count, context, sequences=None):
        if count < 1:
            raise ValueError("Trajectory count must be positive")
        state = self.initial_observation.expand(count, -1).clone()
        rows = torch.arange(count, device=state.device)
        supplied = torch.as_tensor(sequences, dtype=torch.long, device=state.device) if sequences is not None else None
        if supplied is not None and supplied.shape != (count, self.evaluator.max_added_reactions):
            raise ValueError("Incorrect structural sequence shape")
        reuse = (self._sample_cache is not None and sequences is not None
                 and self._sample_cache[0] == self.version
                 and self._sample_cache[1] == tuple(map(tuple, sequences)))
        prefix_states = []
        selected = []
        logps, entropies = [], []
        for step in range(self.evaluator.max_added_reactions):
            if reuse:
                state = self._sample_cache[2][step]
            prefix_states.append(state)
            dist = self.distribution_from_state(state, context)
            actions = dist.sample() if supplied is None else supplied[:, step]
            values = dist.log_prob(actions)
            if not torch.isfinite(values).all():
                raise ValueError("Invalid imitation action under structural mask")
            logps.append(values)
            entropies.append(dist.entropy())
            selected.append(actions)
            if not reuse and step + 1 < self.evaluator.max_added_reactions:
                # Do not modify a prefix retained by autograd or the replay cache.
                state = state.clone()
                state[rows, actions] = 1.
                state[rows, self.M + self.parameter_offsets[actions]] = self.cfg.initial_rate
        selected = torch.stack(selected, dim=1).detach().cpu().tolist()
        if supplied is None:
            self._sample_cache = (self.version, tuple(map(tuple, selected)), prefix_states)
        return selected, torch.stack(logps).sum(0), torch.stack(entropies).mean(0)

    @torch.no_grad()
    def sample(self, count, context):
        self.act_context = list(context)
        if count == 0:
            return []
        return self.trajectories(count, context)[0]

    def update(self, proposals, rewards, context, elites):
        policy_loss = torch.zeros((), device=self.session.device)
        entropy = policy_loss.clone()
        if proposals:
            if any(p["policy_version"] != self.version for p in proposals):
                raise ValueError("Stale structural trajectory; no uncorrected off-policy update")
            _, logps, entropies = self.trajectories(len(proposals), context, [p["reaction_ids"] for p in proposals])
            reward = torch.tensor(rewards, dtype=logps.dtype, device=logps.device)
            advantage = reward - self.baseline
            policy_loss = -(advantage.detach() * logps).mean()
            entropy = entropies.mean()
        imitation = policy_loss.new_zeros(())
        if elites:
            # Reaction ordering is canonical for demonstrations; one record per topology.
            sequences = [sorted(e["reaction_ids"]) for e in elites]
            _, logps, _ = self.trajectories(len(elites), context, sequences)
            imitation = -logps.mean() / self.evaluator.max_added_reactions
        loss = policy_loss + self.cfg.imitation_weight * imitation - self.cfg.entropy_weight * entropy
        metrics = {"Policy/loss": float(policy_loss.detach()), "Policy/entropy": float(entropy.detach()),
                   "Policy/imitation": float(imitation.detach()), "Policy/rl_samples": len(proposals),
                   "Policy/imitation_samples": len(elites), "Policy/version": self.version}
        if loss.requires_grad:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy objective")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg.gradient_clip, error_if_nonfinite=True)
            self.optimizer.step()
            metrics["Policy/gradient_norm"] = float(norm)
            self.version += 1
        if rewards:
            self.baseline = .9 * self.baseline + .1 * float(np.mean(rewards))
        self.last_sil_info = {"loss": float(imitation.detach()), "loss_weight": self.cfg.imitation_weight,
                              "batch_size": len(elites), "structural_only": True}
        self._sample_cache = None
        return metrics
