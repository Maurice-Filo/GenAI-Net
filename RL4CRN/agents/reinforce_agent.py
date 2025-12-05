import torch
import time
import numpy as np
from RL4CRN.agents.abstract_agent import AbstractAgent
from RL4CRN.environments.serial_environments import SerialEnvironments

class REINFORCEAgent(AbstractAgent):
    def __init__(self, policy, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_scheduler={}, risk_scheduler={}, sil_settings={}, device=None):
        """ Initialize the REINFORCE agent with a policy, learning rate, and optional entropy and risk schedulers.
        Args:
        - policy (torch.nn.Module): The policy network to be used by the agent.
        - allow_input_influence (bool): Whether to allow actions to include input influence. Default is False.
        - logger (Logger): An optional logger for logging metrics.
        - learning_rate (float): The learning rate for the optimizer.
        - entropy_scheduler (dict): A dictionary containing parameters for the entropy scheduler:
            - entropy_weight (float): Initial weight for entropy. It is modified during training.
            - topk_entropy_weight (float): Weight for entropy computed over top-k samples.
            - remainder_entropy_weight (float): Weight for entropy computed over the remaining samples after removing the topk.
            - entropy_update_coefficient (float): Multiplicative coefficient to update the entropy weight.
            - entropy_schedule (int): Number of iterations after which to update the entropy weight.
            - minimum_entropy_weight (float): Minimum value for the entropy weight.
        - risk_scheduler (dict): A dictionary containing parameters for the risky policy scheduler:
            - risk (float): Initial risk value.
            - risk_update (float): Amount to increase the risk value.
            - max_risk (float): Maximum value for the risk.
            - risk_schedule (int): Number of iterations after which to update the risk value.
        - device (torch.device): The device to run the agent on (CPU or GPU). If None, defaults to CPU. """
        
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
        
    def act(self, states, actuator, mode='full'):
        """ Select actions based on the current policy and the observed states.
        Args:
        - states (torch.Tensor): The observed states. Shape: (N, state_dim).
        - actuator (AbstractActuator): The actuator to convert policy actions to environment actions.
        - mode (str): The mode of the policy ('full', 'partial', or 'parameters'). Default is 'full'.
        Returns:
        - actions (list): A list of actions to be taken in the environment. 
        - raw_actions (list): A list of raw actions output by the policy before actuation. (needed for SIL)
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
        """ Compute the self-imitation learning loss using samples from the hall of fame (HoF). """

        if observer is None or tensorizer is None or stepper is None:
            raise ValueError("Observer, tensorizer, and stepper must be provided for self-imitation learning loss computation.")

        sil_loss_value = 0.0

        if hof is None or len(hof) == 0:
            return sil_loss_value
        
        if sil_batch_size is None:
            sil_batch_size = len(hof)

        current_batch_best_loss = final_loss_for_each_sample[top_k_indices[0]] # TODO think how to get the actions or store them, also if this is the best way to implement this things...

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
        """ Update the agent's policy based on the rewards received.
        Args:
        - rewards (list): A list of rewards, at the final step, received for each sample in the batch.
        - step_iteration (int): The current iteration step for logging purposes. """
        
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
        if top_k.numel() == 0:
            baseline = final_loss_for_each_sample.max()
        else:
            baseline = final_loss_for_each_sample[top_k[-1]]
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
                self.logger.log_metric('Loss: SIL', sil_loss.item(), step=step_iteration)

        # Clear the lists of logPs and entropies
        self.logPs_sequence.clear()
        self.entropies_sequence.clear()