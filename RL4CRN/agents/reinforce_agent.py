r"""
REINFORCE-style policy gradient agent.

This module provides **`REINFORCEAgent`**, a policy-gradient agent that
collects per-step log-probabilities and entropies during rollout, and performs
an update using a *risk-seeking* REINFORCE objective with an entropy bonus.
Optionally, it can add a self-imitation learning (SIL) loss computed from a
hall-of-fame buffer.

Terminology:
    This code treats the optimization target as a *loss* to be minimized
    (smaller is better). The variable name `rewards` in `update` actually
    represents per-sample final losses.

Risk-seeking objective:
    Let $\ell_i$ be the final loss for sample $i$ in a batch of size
    $N$. Let $\pi_\theta$ be the policy and let
    $\log \pi_\theta(a_{i,t} \mid s_{i,t})$ be the log-probability of the
    action chosen at step $t$.

    A risk parameter $r \in [0, 1]$ defines a top-k subset
    $\mathcal{K}$ of the best samples (lowest losses), where:

    $$k = \left\lfloor N (1 - r) \right\rfloor$$

    The code computes a baseline $b$ as the *worst* (largest) loss among
    these top-k samples (or $\max_i \ell_i$ if $k = 0$), ensuring
    non-negative weights within the selected subset.

    The (un-normalized) policy-gradient loss term used in the implementation is:

    

    $$\mathcal{L}_{\text{PG}}
        = \frac{1}{k} \sum_{i \in \mathcal{K}} (\ell_i - b) \sum_t \log \pi_\theta(a_{i,t}\mid s_{i,t})$$

Entropy regularization:
    An entropy term is subtracted from the objective to encourage exploration.
    The implementation tracks entropies per step and forms a batch-level entropy
    statistic with separate weights for the top-k subset and the remainder.

Self-imitation learning (optional):
    If enabled, an additional term $\mathcal{L}_{\text{SIL}}$ is added to
    the total loss. It replays trajectories from a hall-of-fame buffer and
    reinforces actions whose final loss improves upon the current batch best.

Notes:
    - `act` stores tensors needed for the later update in internal lists.
      Call `update` once per rollout batch to clear this state.
    - This agent assumes the policy supports a forward signature compatible with
      this code (see `act` for details).
"""

import torch
import time
import numpy as np
from RL4CRN.agents.abstract_agent import AbstractAgent
from RL4CRN.environments.serial_environments import SerialEnvironments

class REINFORCEAgent(AbstractAgent):
    """Risk-seeking REINFORCE agent with entropy regularization and optional SIL.

    The agent performs batched rollouts, storing per-step log-probabilities and
    entropies. During `update`, it selects the best samples according to a
    risk parameter, forms a baseline from that subset, and applies a REINFORCE-
    style policy gradient update with an entropy bonus.

    Args:
        policy: Policy network used to sample actions and return log-probabilities
            and entropies. It must support being called as
            `policy(states, mode=...)` and return `(raw_actions, logPs, entropies)`
            for action sampling. For SIL replay, it must also support being called
            as `policy(observations, mode='full', action=raw_actions)` and return
            per-sample log-probabilities.
        allow_input_influence: Whether actions may include input influence. The
            `'parameters'` mode with input influence is not implemented.
        logger: Optional logger providing `log_metric(name, value, step=...)`.
        learning_rate: Learning rate for the Adam optimizer.
        entropy_scheduler: Dictionary controlling entropy regularization. If empty,
            defaults are used. Supported keys:

            - `entropy_weight`: global multiplier for entropy regularization
            - `topk_entropy_weight`: weight for entropy of top-k subset
            - `remainder_entropy_weight`: weight for entropy of the remainder subset
            - `entropy_update_coefficient`: multiplicative update factor
            - `entropy_schedule`: update period (iterations)
            - `minimum_entropy_weight`: lower bound for `entropy_weight`
        risk_scheduler: Dictionary controlling the risk parameter. If empty,
            defaults are used. Supported keys:

            - `risk`: initial risk value $r$ (higher means fewer samples used)
            - `risk_update`: additive increment for `risk`
            - `max_risk`: upper bound for `risk`
            - `risk_schedule`: update period (iterations)
        sil_settings: Dictionary controlling self-imitation learning. If empty,
            defaults are used. Supported keys:

            - `sil_loss_weight`: multiplier for the SIL term
            - `use_adaptive_baseline`: if True, uses an exponential moving baseline
            - `baseline_annealing_rate`: EMA coefficient for adaptive baseline
        device: Torch device. If None, defaults to CPU.

    Attributes:
        logPs_sequence: List of tensors containing per-step log-probabilities.
        entropies_sequence: List of tensors containing per-step entropies.
        entropy_scheduler: Dictionary of entropy scheduling parameters.
        risk_scheduler: Dictionary of risk scheduling parameters.
    """

    def __init__(self, policy, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_scheduler={}, risk_scheduler={}, sil_settings={}, device=None):
        
        super(REINFORCEAgent, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.allow_input_influence = allow_input_influence

        # Neural Networks
        self.policy = policy.to(self.device) 

        # Trajectories
        self.logPs_sequence = []
        self.entropies_sequence = []

        # Torch training
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.logger = logger

        # Entropy scheduler
        if not entropy_scheduler:
            entropy_scheduler = {'entropy_weight': 1.0, 'topk_entropy_weight': 1.0, 'remainder_entropy_weight': 1.0, 'entropy_update_coefficient': 1.0, 'entropy_schedule': 20, 'minimum_entropy_weight': 0.0}
        self.entropy_scheduler = entropy_scheduler
        self.entropy_scheduler['topk_entropy_weight'] = entropy_scheduler.get('topk_entropy_weight', 1.0)
        self.entropy_scheduler['remainder_entropy_weight'] = entropy_scheduler.get('remainder_entropy_weight', 1.0)
        self.entropy_counter = 0

        # Risky policy scheduler
        if not risk_scheduler:
            risk_scheduler = {'risk': 0.9, 'risk_update': 0.0, 'max_risk': 1.00, 'risk_schedule': 20}
        self.risk_scheduler = risk_scheduler
        self.risk_counter = 0

        if not sil_settings:
            sil_settings = {'sil_loss_weight': 1.0}
        self.sil_settings = sil_settings

        self.use_adaptive_baseline = sil_settings.get('use_adaptive_baseline', False)
        self.baseline_annealing_rate = sil_settings.get('baseline_annealing_rate', 0.95)
        self.adaptive_baseline = None
        
    def act(self, states, actuator, mode='full'):
        """Sample actions from the policy for a batch of states.

        This method performs the forward pass through the policy, stores the
        resulting log-probabilities and entropies for the later update, and
        converts raw policy outputs into environment actions via the provided
        actuator.

        Args:
            states: Batch of observed states (typically a tensor of shape
                `(N, state_dim)`).
            actuator: Actuator that converts raw policy actions into environment
                actions (must provide `actuate(policy_action)`).
            mode: Policy mode. Expected values include `'full'`, `'partial'`, and
                `'parameters'` (depending on the policy implementation).

        Returns:
            A tuple `(actions, raw_actions)`:

                - actions: list of environment actions produced by the actuator.
                - raw_actions: list of raw policy actions prior to actuation
                  (used by self-imitation learning).

        Raises:
            NotImplementedError: If `mode == 'parameters'` and `allow_input_influence`
                is True.
        """

        super(REINFORCEAgent, self).act()
        tic_forward = time.time()

        # Check if the observed IOCRN has unknown parameters
        if mode == 'parameters' and self.allow_input_influence:
            raise NotImplementedError("The cases of unknown rate constants and/or allow input influence are not implemented yet.")
        else:
            raw_actions, logPs, entropies = self.policy(states, mode=mode)
            self.logPs_sequence.append(logPs)
            self.entropies_sequence.append(entropies)
        toc_forward = time.time()

        # Log the forward pass time and return the actions
        if self.logger is not None:
            self.logger.log_metric('Timing: Forward', toc_forward - tic_forward, step=None)    

        actions = [actuator.actuate(a) for a in raw_actions]
        return actions, raw_actions
    
    # TODO this might be general enough to be in its separate utility file (maybe)
    def self_imitation_learingin_loss(self, hof, final_loss_for_each_sample, top_k_indices, weighting_scheme='uniform', observer=None, tensorizer=None, stepper=None, sil_batch_size=None):
        r"""Compute self-imitation learning (SIL) loss using hall-of-fame samples.

        The SIL term replays trajectories from the hall-of-fame (HoF) buffer and
        reinforces actions that yield a loss better than the current batch best.

        For each HoF sample $j$, the advantage used in the implementation is:

        

        $$    A_j = \max(0, \ell_{\text{best}} - \ell^{\text{HoF}}_j)$$

        where $\ell_{\text{best}}$ is the best (lowest) loss in the
        current batch among the selected top-k samples. Constrained to positive
        advantages, this encourages the agent to imitate only those HoF samples
        that improve upon its current best performance.

        The SIL objective is:


        $$\mathcal{L}_{\text{SIL}} = -\frac{1}{M} \sum_{j=1}^M w_j A_j \log \pi_\theta(\tau_j)$$

        where $\log \pi_\theta(\tau_j)$ is the sum of log-probabilities
        assigned by the current policy to the replayed trajectory $\tau_j$
        and $w_j$ are optional sample weights.

        Args:
            hof: Hall-of-fame buffer providing `__len__` and `sample(batch_size)`.
                Each sample must be cloneable (used to create replay environments)
                and must provide `get_raw_action(j)` and `get_action(j)` for each
                replay step.
            final_loss_for_each_sample: Tensor-like object containing the final
                per-sample loss for the current batch.
            top_k_indices: Tensor of indices of the selected top-k samples in the
                current batch.
            weighting_scheme: Currently only `'uniform'` is supported.
            observer: Observer used to produce observations from environments.
            tensorizer: Tensorizer used to convert observations to tensors.
            stepper: Stepper used to apply actions in the environment.
            sil_batch_size: Number of HoF samples to replay. Defaults to `len(hof)`.

        Returns:
            Scalar SIL loss (float or tensor). Returns 0.0 if HoF is empty.

        Raises:
            ValueError: If `observer`, `tensorizer`, or `stepper` is not provided.
            NotImplementedError: If `weighting_scheme` is not implemented.
        """

        if observer is None or tensorizer is None or stepper is None:
            raise ValueError("Observer, tensorizer, and stepper must be provided for self-imitation learning loss computation.")

        sil_loss_value = 0.0

        if hof is None or len(hof) == 0:
            return sil_loss_value
        
        if sil_batch_size is None:
            sil_batch_size = len(hof)

        current_batch_best_loss = final_loss_for_each_sample[top_k_indices[0]] 

        samples = hof.sample(sil_batch_size)

        if weighting_scheme == 'uniform':
            len_samples = len(samples)
            weights = torch.ones(len_samples, device=self.device)
        else:
            raise NotImplementedError(f"{weighting_scheme} weighting scheme not implemented.")

        hof_envs = SerialEnvironments([s.clone() for s in samples], hall_of_fame_size=0, logger=None)
        hof_envs.reset()

        if len(samples) > 0:

            max_added_reactions = samples[0].max_added_reactions

            all_logPs = None
            for j in range(max_added_reactions):
                observations = hof_envs.observe(observer, tensorizer)
                raw_actions = [ s.get_raw_action(j) for s in samples ]
                logPs = self.policy(observations, mode='full', action=raw_actions)
                actions = [s.get_action(j) for s in samples]
                if all_logPs is None:
                    all_logPs = logPs
                else:
                    all_logPs += logPs
                hof_envs.step(actions, stepper)
            
            # read the rewards 
            final_losses_hof = torch.tensor([s.state.last_task_info['reward'] for s in samples], device=self.device, dtype=torch.float32)

            # remove the baseline
            advantages = current_batch_best_loss - final_losses_hof  # shape (len_hof,)
            # remove negative advantages
            advantages = torch.clamp(advantages, min=0.0).detach()   # shape (len_hof,) # detach shouldn't be necessary but just to be sure
            # TODO: normalize weights?
            
            sil_loss = -(all_logPs*weights*advantages).mean()

            return sil_loss

        return 0.

    
    def update(self, rewards, step_iteration=None, hof=None, use_sil=False, sil_weighting_scheme='uniform', observer=None, tensorizer=None, stepper=None, sil_batch_size=None):
        """Update the policy using stored rollout statistics and final losses.

        This method consumes the sequences collected by `act` and performs
        a single optimization step.

        Important:
            The argument `rewards` is treated as *final losses* to minimize.
            Smaller values are better.

        Overview of computations:
            - Stack and sum log-probabilities and entropies across steps.
            - Select top-k samples according to the risk parameter.
            - Compute a baseline (worst loss in top-k, or batch max if k=0).
            - Form the policy gradient loss on the selected subset.
            - Subtract an entropy regularization term.
            - Optionally add a self-imitation learning (SIL) term.
            - Backpropagate, clip gradients, and update policy parameters.

        Args:
            rewards: List/array/tensor of length `N` containing final per-sample
                losses for the batch.
            step_iteration: Optional integer step for logging.
            hof: Optional hall-of-fame buffer used for SIL.
            use_sil: If True, adds the SIL loss term.
            sil_weighting_scheme: Weighting scheme for SIL samples. Currently only
                `'uniform'` is supported.
            observer: Observer used for SIL replay.
            tensorizer: Tensorizer used for SIL replay.
            stepper: Stepper used for SIL replay.
            sil_batch_size: Number of HoF samples to replay.

        Returns:
            None.

        Raises:
            RuntimeError: If called before any `act` call (no stored logPs).
        """
        
        super(REINFORCEAgent, self).update(rewards)
        tic_backward = time.time()

        # Retrieve the information from the forward pass
        self.optimizer.zero_grad()
        final_loss_for_each_sample = rewards # list of size N
        sum_logPs = torch.sum(torch.stack(self.logPs_sequence, dim=1), dim=1) # shape (N,), self.logPs_sequence is a list (length=total number of actions) of tensors of shape (N,)
        sum_entropies = torch.sum(torch.stack(self.entropies_sequence, dim=1), dim=1) # shape (N,), self.entropies_sequence is a list (length=total number of actions) of tensors of shape (N,)
        N = self.logPs_sequence[0].shape[0]

        # Tensorize the rewards
        final_loss_for_each_sample = torch.tensor(final_loss_for_each_sample, device=sum_logPs.device, dtype=sum_logPs.dtype).detach() # shape (N,)

        # Risky policy gradient
        k = int(N * (1. - self.risk_scheduler['risk']))
        top_k = torch.topk(final_loss_for_each_sample, k, largest=False).indices # shape (int(N * (1. - self.risk_scheduler['risk'])),)

        # Compute the gradients with baseline (important: baseline = worst loss in top k, so that the weights are non-negative)
        if not self.use_adaptive_baseline:
            if top_k.numel() == 0:
                baseline = final_loss_for_each_sample.max()
            else:
                baseline = final_loss_for_each_sample[top_k[-1]]
        else:
            if self.adaptive_baseline is None:
                self.adaptive_baseline = final_loss_for_each_sample[top_k[-1]] if top_k.numel() > 0 else final_loss_for_each_sample.max().item()
            else:
                current_best = final_loss_for_each_sample[top_k[0]] if top_k.numel() > 0 else final_loss_for_each_sample.min().item()
                self.adaptive_baseline = self.baseline_annealing_rate * self.adaptive_baseline + (1 - self.baseline_annealing_rate) * current_best
            baseline = self.adaptive_baseline

        # baseline = torch.mean(final_loss_for_each_sample[top_k]).detach()  # shape (1,)
        # advantages = final_loss_for_each_sample[top_k] - baseline  # shape (N,)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantage
        # loss_for_gradient =  advantages.detach() * sum_logPs[top_k] # shape (k,)
        loss_for_gradient =  (final_loss_for_each_sample[top_k] - baseline) * sum_logPs[top_k] # shape (k,)

        # Compute the entropy component of the gradient
        # entropy_for_gradient = torch.mean(sum_entropies + sum_entropies.detach() * sum_logPs) # shape (1,)
        entropy_batch = torch. mean(sum_entropies) # shape (1,)
        entropy_topk = torch.mean(sum_entropies[top_k]) # shape (1,)
        entropy_remainder = (N * entropy_batch - k * entropy_topk) / (N - k) if N > k else 0.0 # shape (1,)
 
        entropy_for_gradient = self.entropy_scheduler['topk_entropy_weight'] * (k/N) * entropy_topk + self.entropy_scheduler['remainder_entropy_weight'] * ((N-k)/N) * entropy_remainder # shape (1,)

        loss_for_gradient = loss_for_gradient - self.entropy_scheduler['entropy_weight'] * entropy_for_gradient # shape (k,)
        loss_for_gradient_entropy_mean = torch.mean(loss_for_gradient)

        # Add self-imitation learning loss if specified
        sil_loss = None
        if use_sil:
            sil_loss = self.self_imitation_learingin_loss(hof, final_loss_for_each_sample, top_k, weighting_scheme=sil_weighting_scheme, observer=observer, tensorizer=tensorizer, stepper=stepper, sil_batch_size=sil_batch_size)
            loss_for_gradient_entropy_mean += sil_loss*self.sil_settings['sil_loss_weight']

        loss_for_gradient_entropy_mean.backward()

        # Do gradient clipping if needed and perform the optimization step
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        toc_backward = time.time()

        # Update the entropy weight and the risk value #TODO: risk scheduler never tested
        if self.entropy_scheduler['entropy_weight'] > self.entropy_scheduler["minimum_entropy_weight"]:
            self.entropy_counter += 1
            if self.entropy_counter % self.entropy_scheduler["entropy_schedule"] == 0:
                self.entropy_scheduler['entropy_weight'] *= self.entropy_scheduler["entropy_update_coefficient"]
        if self.risk_scheduler['risk'] < self.risk_scheduler["max_risk"]:
            self.risk_counter += 1
            if self.risk_counter % self.risk_scheduler["risk_schedule"] == 0:
                self.risk_scheduler['risk'] += self.risk_scheduler["risk_update"]

        # Log the training process
        if self.logger is not None:
            # Compute losses relevant for logging
            best_loss = final_loss_for_each_sample[top_k[0]]
            worst_loss_topk = final_loss_for_each_sample[top_k[-1]]
            avg_loss_topk = final_loss_for_each_sample[top_k].float().mean()
            avg_loss = final_loss_for_each_sample.float().mean()
            worst_loss = final_loss_for_each_sample.float().max()
            total_loss_topk = torch.mean(final_loss_for_each_sample[top_k]) - self.entropy_scheduler['entropy_weight'] * entropy_for_gradient

            # Log the losses
            self.logger.log_metric('Loss: Batch Average', avg_loss.item(), step=step_iteration)
            self.logger.log_metric('Loss: Batch Best', best_loss.item(), step=step_iteration)
            self.logger.log_metric('Loss: Batch Worst', worst_loss.item(), step=step_iteration)
            self.logger.log_metric('Loss: Top-' + str(k) + ' Worst', worst_loss_topk.item(), step=step_iteration)
            self.logger.log_metric('Loss: Top-' + str(k) + ' Average', avg_loss_topk.item(), step=step_iteration)
            self.logger.log_metric('Loss: Top-' + str(k) + ' Total', total_loss_topk.item(), step=step_iteration)

            # Log the entropies
            self.logger.log_metric('Entropy: Batch', entropy_batch.item(), step=step_iteration)
            self.logger.log_metric('Entropy: Top-' + str(k), entropy_topk.item(), step=step_iteration)
            self.logger.log_metric('Entropy: Global Weight', self.entropy_scheduler['entropy_weight'], step=step_iteration)
            self.logger.log_metric('Temperature', self.policy.structure_head_temperature["current_temperature"], step=step_iteration)

            # Log the probabilities
            self.logger.log_metric('LogP: Batch', sum_logPs.mean().item(), step=step_iteration)
            self.logger.log_metric('LogP: Top-' + str(k), sum_logPs[top_k].mean().item(), step=step_iteration)

            # Log the risk value
            self.logger.log_metric('Risk', self.risk_scheduler['risk'], step=step_iteration)

            # Log the timing
            self.logger.log_metric('Timing: Backward', toc_backward - tic_backward, step=step_iteration)

            if sil_loss is not None:
                if isinstance(sil_loss, torch.Tensor):
                    self.logger.log_metric('Loss: SIL', sil_loss.item(), step=step_iteration)
                else:
                    self.logger.log_metric('Loss: SIL', sil_loss, step=step_iteration)

        # Clear the lists of logPs and entropies
        self.logPs_sequence.clear()
        self.entropies_sequence.clear()