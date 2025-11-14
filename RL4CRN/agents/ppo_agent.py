import torch
import time
import numpy as np
import torchrl
from copy import deepcopy
from RL4CRN.agents.abstract_agent import AbstractAgent
from tensordict import TensorDict  
from torchrl.objectives.value import GAE  # put this at the top of ppo_agent.py


class PPOAgent(AbstractAgent):
    def __init__(self, policy, state_value_function, ppo_parameters={}, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_scheduler={}, risk_scheduler={}, device=None):
        """ Initialize the PPO agent with a policy, state value function, learning rate, and optional entropy and risk schedulers.
        Args:
        - policy (torch.nn.Module): The policy network to be used by the agent.
        - state_value_function (torch.nn.Module): The state value function network to be used by the agent.
        - ppo_parameters (dict): A dictionary containing parameters for the PPO algorithm:
            - eps (float): Clipping parameter for the PPO objective.
            - value_loss_weight (float): Weight for the value loss in the total loss.
            - gamma (float): Discount factor for future rewards.
            - lam (float): GAE lambda parameter.
            - update_policy_every (int): Number of iterations after which to update the old policy. Default is 5.
        - allow_input_influence (bool): Whether to allow actions to include input influence. Default is False.
        - logger (Logger): An optional logger for logging metrics.
        - learning_rate (float): The learning rate for the optimizer.
        - entropy_scheduler (dict): A dictionary containing parameters for the entropy scheduler:
            - entropy_weight (float): Initial weight for entropy. It is modified during training.
            - entropy_update_coefficient (float): Multiplicative coefficient to update the entropy weight.
            - entropy_schedule (int): Number of iterations after which to update the entropy weight.
            - minimum_entropy_weight (float): Minimum value for the entropy weight.
        - risk_scheduler (dict): A dictionary containing parameters for the risky policy scheduler:
            - risk (float): Initial risk value.
            - risk_update (float): Amount to increase the risk value.
            - max_risk (float): Maximum value for the risk.
            - risk_schedule (int): Number of iterations after which to update the risk value.
        - device (torch.device): The device to run the agent on (CPU or GPU). If None, defaults to CPU. """

        super(PPOAgent, self).__init__()
        
        # Entropy scheduler
        if not entropy_scheduler:
            entropy_scheduler = {'entropy_weight': 1, 'entropy_update_coefficient': 0.9, 'entropy_schedule': 20, 'minimum_entropy_weight': 1}
        self.entropy_scheduler = entropy_scheduler
        self.entropy_counter = 0

        # Risky policy scheduler
        if not risk_scheduler:
            risk_scheduler = {'risk': 0.8, 'risk_update': 0.00, 'max_risk': 1.00, 'risk_schedule': 20}
        self.risk_scheduler = risk_scheduler
        self.risk_counter = 0

        # PPO parameters
        if not ppo_parameters:
            ppo_parameters = {'eps': 0.2, 'value_loss_weight': 0.01, 'gamma': 1, 'lam': 0.95, 'update_policy_every': 5}
        self.ppo_parameters = ppo_parameters
        self.swap_counter = 0


        self.device = device if device is not None else torch.device('cpu')
        self.allow_input_influence = allow_input_influence

        # Neural Networks
        self.policy = policy.to(self.device)
        self.policy_old = deepcopy(policy).to(self.device)
        self.state_value_function = state_value_function.to(self.device)
        
        # Trajectories
        self.states_sequence = []
        self.actions_sequence = []
        self.logPs_sequence = []
        self.entropies_sequence = []

        # Torch training (as convention, actor is 3x faster lr than critic, we can change this later TODO)
        self.actor_opt  = torch.optim.Adam(self.policy.parameters(), lr=learning_rate*3)

        torch.nn.utils.clip_grad_norm_(self.state_value_function.parameters(), max_norm=1.0)

        self.critic_opt = torch.optim.Adam(self.state_value_function.parameters(), lr=learning_rate)


        self.logger = logger

        self.GAE_module = GAE(
            gamma=self.ppo_parameters['gamma'],
            lmbda=self.ppo_parameters['lam'],
            value_network=None,
            differentiable=False,
            device=self.device
        )

        
    def act(self, states, actuator, mode='full'):
        """ Select actions based on the current policy and the observed states.
        Args:
        - states (torch.Tensor): The observed states. Shape: (N, state_dim).
        - actuator (AbstractActuator): The actuator to convert policy actions to environment actions.
        - mode (str): The mode of the policy ('full', 'partial', or 'parameters'). Default is 'full'.
        Returns:
        - actions (list): A list of actions to be taken in the environment. """

        super(PPOAgent, self).act()
        tic_forward = time.time()

        # Check if the observed IOCRN has unknown rate constants
        if mode == 'parameters' and self.allow_input_influence:
            raise NotImplementedError("The cases of unknown parameters and input influence are not implemented yet.")
        else:
            self.states_sequence.append(states)
            actions, logPs, entropies = self.policy(states, mode='full')
            self.actions_sequence.append(actions)
            self.logPs_sequence.append(logPs)
            self.entropies_sequence.append(entropies)
        toc_forward = time.time()

        # Log the forward pass time and return the actions
        if self.logger is not None:
            self.logger.log_metric('forward_time', toc_forward - tic_forward, step=None)   

        actions = [actuator.actuate(a) for a in actions]        
        return actions
    
    def update(self, losses, step_iteration=None):
        """ Update the agent's policy based on the rewards received.
        Args:
            losses (list): A list of losses, at the final step, received for each sample in the batch.
            step_iteration (int): The current iteration step for logging purposes. """
        
        super(PPOAgent, self).update(losses)
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        tic_backward = time.time()

        # Extract dimensions
        T = len(self.logPs_sequence)  # Final time step
        N = self.logPs_sequence[0].shape[0]  # Number of samples in the batch

        # Convert lists to tensors
        logPs_sequence_tensor = torch.stack(self.logPs_sequence, dim=0).to(self.device)  # Shape: (T, N)
        entropies_sequence_tensor = torch.stack(self.entropies_sequence, dim=0).to(self.device)  # Shape: (T, N)
        rewards_sequence_tensor = torch.zeros_like(logPs_sequence_tensor)
        rewards_sequence_tensor[-1,:] = -torch.tensor(losses, device=self.device) 

        # Compute the PPO loss
        ppo_cpi, ppo_entropy, ppo_value = self.compute_PPO_loss(self.states_sequence, self.actions_sequence, rewards_sequence_tensor, entropies_sequence_tensor)

        # Risky policy gradient and backpropagation
        scores = rewards_sequence_tensor.sum(dim=0)  # Shape: (N,) TODO: is mean better than sum?
        # top_k = torch.topk(scores.detach(), int(N * (1. - self.risk_scheduler['risk'])), largest=False).indices # shape (int(N * (1. - self.risk_scheduler['risk'])),)

        k = max(1, int(N * (1. - self.risk_scheduler['risk'])))
        top_k = torch.topk(scores.detach(), k, largest=True).indices

        (torch.mean(-ppo_cpi[top_k]) - ppo_entropy + ppo_value).backward() 

        # add clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        # also clip critic
        torch.nn.utils.clip_grad_norm_(self.state_value_function.parameters(), max_norm=1.0)

        # Step the optimizer
        self.actor_opt.step()
        self.critic_opt.step()
        toc_backward = time.time()

        # Update the old policy to the new policy
        self.swap_counter += 1
        if self.swap_counter >= self.ppo_parameters['update_policy_every']:
            self.swap_counter = 0
            self.swap()

        # Update the entropy weight and the risk value
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
            self.logger.log_metric('batch average of total sequence entropy', entropies_sequence_tensor.sum(dim=0).mean().item(), step=step_iteration)
            self.logger.log_metric('batch average of total sequence logP', logPs_sequence_tensor.sum(dim=0).mean().item(), step=step_iteration)
            self.logger.log_metric('entropy weight', self.entropy_scheduler['entropy_weight'], step=step_iteration)
            self.logger.log_metric('risk', self.risk_scheduler['risk'], step=step_iteration)
            self.logger.log_metric('backward time', toc_backward - tic_backward, step=step_iteration)
            best = scores[top_k[0]]
            worst = scores[top_k[-1]]
            avg = scores[top_k].float().mean()
            batch_avg = scores.float().mean()
            # here we need the - sign
            self.logger.log_metric('full batch average loss', -batch_avg.item(), step=step_iteration)
            self.logger.log_metric('best loss in the batch top k', -best.item(), step=step_iteration)
            self.logger.log_metric('worst loss in the batch top k', -worst.item(), step=step_iteration)
            self.logger.log_metric('average loss in the batch top k', -avg.item(), step=step_iteration)

        # Clear the trjectories
        self.states_sequence.clear()
        self.actions_sequence.clear()
        self.logPs_sequence.clear()
        self.entropies_sequence.clear()

    def compute_PPO_loss(self, states_sequence, actions_sequence, rewards_sequence, entropies_sequence, masks=1., mode='full'):
        """ Compute the PPO loss for the given states, actions, rewards, and entropies.
        Args:
            states_sequence (list): A (T+1)-list of states.
            actions_sequence (list): A T-list of actions.
            rewards_sequence (torch.Tensor): The rewards of the batch of states received at each time step. Shape: (T, N).
            entropies_sequence (torch.Tensor): The entropies of the actions taken at each time step. Shape: (T, N).
            masks (torch.Tensor or None): A mask tensor indicating which states are valid. Shape: (T, N). If None, all states are considered valid.
            mode (str): The mode of the policy ('full' or 'partial'). Default is 'full'.
        Returns:
            cpi, entropy_bonus, v_loss (torch.Tensor): The computed clipped policy improvement, entropy bonus, and value loss. """
        
        # Compute the GAE
        advantages_sequence, state_value_targets_sequence = self.compute_gae(states_sequence, rewards_sequence, masks)

        # --- per batch --- (it feels that the one per time step is better, but they are very similar)
        # std = advantages_sequence.std()
        # adv_norm = (advantages_sequence - advantages_sequence.mean()) / (
        #     std +1e-8
        # )
        # # clip advantages
        # adv_norm = torch.clamp(adv_norm, -5.0, 5.0)

        # --- per time step --- 
        # advantages_sequence: (T, N)
        mean = advantages_sequence.mean(dim=1, keepdim=True)   # (T, 1)
        std  = advantages_sequence.std(dim=1, keepdim=True)    # (T, 1)
        std  = torch.clamp(std, min=0.1)

        adv_norm = (advantages_sequence - mean) / (std + 1e-8)
        adv_norm = torch.clamp(adv_norm, -5.0, 5.0)

        # Compute the CPI
        cpi = self.clipCPI(states_sequence, actions_sequence, adv_norm, masks, mode).mean(dim=0) # shape: (N,)

        # Compute the entropy bonus
        T = entropies_sequence.shape[0]
        N = entropies_sequence.shape[1]
        entropy_bonus = (entropies_sequence * masks).sum() / (T * N) # .mean(dim=0) # shape: (N,) # TODO we want the global mean entropy over the sequence so shape (1,) 

        # Compute the state value loss
        v_loss = self.state_value_loss(states_sequence, state_value_targets_sequence) # shape: (1,)

        # Sum up the losses # TODO 
        total_loss = -(cpi.mean() + self.entropy_scheduler['entropy_weight'] * entropy_bonus) + self.ppo_parameters['value_loss_weight'] * v_loss # shape: (N,)

        # Log the calculated metrics
        if self.logger is not None:
            self.logger.log_metric('average advantage', advantages_sequence.mean().item())
            self.logger.log_metric('advantage std', advantages_sequence.std().item())
            self.logger.log_metric('average state value targets', state_value_targets_sequence.mean().item())
            self.logger.log_metric('cpi', cpi.mean().item())
            self.logger.log_metric('entropy bonus', entropy_bonus.mean().item())
            self.logger.log_metric('value loss', v_loss.item())
            self.logger.log_metric('total loss (without topk)', total_loss.mean().item())
            self.logger.log_metric('adv_norm std', adv_norm.std().item())
            self.logger.log_metric('adv_norm max', adv_norm.max().item())
            self.logger.log_metric('adv_norm min', adv_norm.min().item())

        # we need to return per-sample cpi for the risky policy gradient, separately as we want to optimize the entropy and value loss globally 
        return cpi, self.entropy_scheduler['entropy_weight'] * entropy_bonus, self.ppo_parameters['value_loss_weight'] * v_loss

    # Older implementation of GAE (kept for reference)
    # @torch.no_grad()
    # def compute_gae(self, states_sequence, rewards_sequence, masks):
    #     """ Compute the Generalized Advantage Estimation (GAE) for the given states and rewards.
    #     Args:
    #         - states_sequence: A (T+1)-list of states.
    #         - rewards (torch.Tensor): The rewards of the batch of states received at each time step. Shape: (T, N).
    #         - masks (torch.Tensor or None): A mask tensor indicating which states are valid. Shape: (T, N). If None, all states are considered valid.
    #     Returns:
    #         advantages_sequence (torch.Tensor): The computed advantages for each state. Shape: (T, N).
    #         state_value_targets_sequence (torch.Tensor): The targets for the value loss. Shape: (T, N). """
    #     # Extract PPO parameters relevant for GAE
    #     gamma = self.ppo_parameters['gamma']
    #     lam = self.ppo_parameters['lam']
    #     # Compute the state values for each state in the sequence
    #     values_sequence = self.v(states_sequence)  # (T+1, N) tensor
    #     # Compute the advantages using GAE
    #     T = rewards_sequence.shape[0]
    #     advantages_sequence = torch.zeros_like(rewards_sequence) # (T, N) tensor
    #     advantages_next = torch.zeros_like(rewards_sequence[0]) # (N,) tensor
    #     for t in reversed(range(T)):
    #         if type(masks) == torch.Tensor:
    #             delta_t = rewards_sequence[t] + gamma * values_sequence[t + 1] * masks[t] - values_sequence[t] # (N,) tensor
    #             advantages_sequence[t]= delta_t + gamma * lam * advantages_next * masks[t]
    #             advantages_next= advantages_sequence[t]
    #         else:
    #             delta_t = rewards_sequence[t] + gamma * values_sequence[t + 1] - values_sequence[t] # (N,) tensor
    #             advantages_sequence[t]= delta_t + gamma * lam * advantages_next
    #             advantages_next= advantages_sequence[t]
    #     # Compute the targets for the value loss
    #     state_value_targets_sequence = advantages_sequence + values_sequence[:-1] # (T, N) tensor
    #     return advantages_sequence, state_value_targets_sequence

    @torch.no_grad()
    def compute_gae(self, states_sequence, rewards_sequence, masks=None):
        """Compute GAE using TorchRL's GAE module.

        Args:
            states_sequence: list of length T+1, each element a tensor of shape (N, D)
                            or a single tensor of shape (T+1, N, D).
            rewards_sequence (torch.Tensor): (T, N)
            masks: torch.Tensor of shape (T, N) with 1=valid, 0=terminal,
                or anything else / None => treated as no mask.
        Returns:
            advantages_sequence: (T, N)
            state_value_targets_sequence: (T, N)
        """

        # --- 0. Handle masks like in your original code ---
        if not isinstance(masks, torch.Tensor):
            masks = None

        # --- 1. Turn the states_sequence into a tensor and compute V(s_t) ---

        # states_sequence is originally a (T+1)-list of (N, D) tensors [in principle we should receive a list]
        if isinstance(states_sequence, list):
            # stack along time dimension -> (T+1, N, D)
            states_tensor = torch.stack(states_sequence, dim=0)
        else:
            # already a tensor, assume shape (T+1, N, D) or (T+1, N, ...)
            states_tensor = states_sequence

        states_tensor = states_tensor.to(rewards_sequence.device) # ensure on same device

        # Flatten time and batch so the critic sees (batch, features)
        # states_tensor: (T+1, N, D)
        T_plus_1, N, D = states_tensor.shape
        states_flat = states_tensor.reshape(-1, D)               # ((T+1)*N, D)

        # StateValueFunction expects input of shape (batch, features) and returns (batch,)
        values_flat = self.state_value_function(states_flat)     # ((T+1)*N,)
        values_sequence = values_flat.reshape(T_plus_1, N)       # (T+1, N)
        values_sequence[-1] = 0.0                                # V(s_{T}) = 0 # We have no value for the terminal state (as in Maurice's definitions)

        # --- 2. Prepare things for TorchRL layout (N, T, 1) ---

        T, N_rewards = rewards_sequence.shape
        assert N_rewards == N, "Batch size mismatch between states and rewards"
        assert T_plus_1 == T + 1, "states_sequence should have length T+1"

        # arrange to (N, T, 1)
        value      = values_sequence[:T].permute(1, 0).unsqueeze(-1)      # (N, T, 1)
        next_value = values_sequence[1:].permute(1, 0).unsqueeze(-1)      # (N, T, 1)
        reward     = rewards_sequence.permute(1, 0).unsqueeze(-1)         # (N, T, 1)

        # manage masks (and done, which in our case are always the same, no mask and terminated)
        if masks is None:
            done = torch.zeros_like(reward, dtype=torch.bool)             # (N, T, 1)
        else:
            masks = masks.to(reward.device)
            # your mask: 1 = non-terminal, 0 = terminal
            done = (~masks.bool()).permute(1, 0).unsqueeze(-1)            # (N, T, 1)

        terminated = done.clone()

        # --- 3. Build TensorDict in the format GAE expects --- (just following TorchRL's conventions)

        td = TensorDict({}, batch_size=(N, T))
        td.set("state_value", value)                       # V(s_t)
        td.set(("next", "state_value"), next_value)        # V(s_{t+1})
        td.set(("next", "reward"), reward)
        td.set(("next", "done"), done)
        td.set(("next", "terminated"), terminated)

        # --- 4. Run GAE (self.GAE_module must have value_network=None) ---

        td = self.GAE_module(td) # note that we define the GAE without a value network, so it uses the precomputed values in the td

        # --- 5. Back to (T, N) layout ---

        advantages_sequence = td.get("advantage").squeeze(-1).permute(1, 0)              # (T, N)
        state_value_targets_sequence = td.get("value_target").squeeze(-1).permute(1, 0)  # (T, N)

        return advantages_sequence, state_value_targets_sequence


    def clipCPI(self, states_sequence, actions_sequence, advantages_sequence, masks, mode='full'):
        """ Compute the clipped policy improvement (CPI) for the given states, actions, and advantages.
        Args:
            - states_sequence: A (T+1)-list of states.
            - actions_sequence: A T-list of actions.
            - advantages_sequence: A tensor of shape (T, N) representing the advantages computed using GAE.
            - masks: A tensor of shape (T, N) indicating which states are valid. If None, all states are considered valid.
            - mode (str): The mode of the policy ('full' or 'partial'). Default is 'full'.
        Returns:
            - clip_cpi_sequence * masks (torch.Tensor): The masked clipped policy improvement for the given states, actions, and advantages. Shape: (T, N). """
        
        # Extract PPO parameters relevant for CPI
        eps = self.ppo_parameters['eps']

        # Compute the old and new action probabilities across the time steps
        old_logPs_sequence = self.p_old(states_sequence, actions_sequence, mode).detach() # (T, N) tensor
        new_logPs_sequence = self.p(states_sequence, actions_sequence, mode) # (T, N) tensor

        # Compute the risk ratio and the clipped policy improvement
        risk_ratio_sequence = torch.exp(new_logPs_sequence - old_logPs_sequence) # (T, N) tensor

        clip_cpi_sequence = torch.min(risk_ratio_sequence * advantages_sequence, torch.clip(risk_ratio_sequence, 1 - eps, 1 + eps) * advantages_sequence) # (T, N) tensor

        # Log the risk ratio if a logger is available
        if self.logger is not None:
            self.logger.log_metric('risk_ratio', risk_ratio_sequence.mean().item())

        return clip_cpi_sequence * masks
    
    def state_value_loss(self, states_sequence, state_value_targets_sequence):
        """ Compute the state value loss for the given states and targets.
        Args:
            states_sequence: A (T+1)-list of states.
            state_value_targets_sequence: A (T, N) tensor representing the targets for the value loss.
        Returns:
            torch.Tensor: The computed state value loss. """
        v_pred = self.v(states_sequence)[:-1]
        return torch.nn.functional.mse_loss(v_pred, state_value_targets_sequence)


    #--------------------------- Functions for applying the policy and value networks across sequences ---------------------------#
    def p(self, states_sequence, actions_sequence, mode='full'):
        logPs = [ self.policy(states, mode, actions) for states,actions in zip(states_sequence[:-1], actions_sequence) ] # A T-list of tensors, each of shape (N,)
        logPs = torch.stack(logPs) # Tensor of shape (T, N)
        return logPs

    def p_old(self, states_sequence, actions_sequence, mode='full'):
        logPs = [ self.policy_old(states, mode, actions) for states,actions in zip(states_sequence[:-1], actions_sequence) ] # A T-list of tensors, each of shape (N,)
        logPs = torch.stack(logPs) # Tensor of shape (T, N)
        return logPs
    
    def v(self, states_sequence):
        values_sequence = [ self.state_value_function(states) for states in states_sequence ] # A (T+1)-list of tensors, each of shape (N,)
        values_sequence = torch.stack(values_sequence) # Tensor of shape (T+1, N)
        values_sequence[-1] = torch.zeros_like(values_sequence[-1])
        return values_sequence
    

    #--------------------------- Functions for swapping old and new policy networks ---------------------------#
    def swap(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.freeze(self.policy_old)
        self.unfreeze(self.policy)

    def freeze(self, net):
        for param in net.parameters():
            param.requires_grad = False
    
    def unfreeze(self, net):
        for param in net.parameters():
            param.requires_grad = True













# import torch
# import time
# import numpy as np
# from copy import deepcopy
# from RL4CRN.agents.abstract_agent import AbstractAgent

# class PPOAgent(AbstractAgent):
#     def __init__(self, policy, state_value_function, ppo_parameters={}, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_scheduler={}, risk_scheduler={}, device=None):
        # """ Initialize the PPO agent with a policy, state value function, learning rate, and optional entropy and risk schedulers.
        # Args:
        # - policy (torch.nn.Module): The policy network to be used by the agent.
        # - state_value_function (torch.nn.Module): The state value function network to be used by the agent.
        # - ppo_parameters (dict): A dictionary containing parameters for the PPO algorithm:
        #     - eps (float): Clipping parameter for the PPO objective.
        #     - value_loss_weight (float): Weight for the value loss in the total loss.
        #     - gamma (float): Discount factor for future rewards.
        #     - lam (float): GAE lambda parameter.
        #     - update_policy_every (int): Number of iterations after which to update the old policy. Default is 5.
        # - allow_input_influence (bool): Whether to allow actions to include input influence. Default is False.
        # - logger (Logger): An optional logger for logging metrics.
        # - learning_rate (float): The learning rate for the optimizer.
        # - entropy_scheduler (dict): A dictionary containing parameters for the entropy scheduler:
        #     - entropy_weight (float): Initial weight for entropy. It is modified during training.
        #     - entropy_update_coefficient (float): Multiplicative coefficient to update the entropy weight.
        #     - entropy_schedule (int): Number of iterations after which to update the entropy weight.
        #     - minimum_entropy_weight (float): Minimum value for the entropy weight.
        # - risk_scheduler (dict): A dictionary containing parameters for the risky policy scheduler:
        #     - risk (float): Initial risk value.
        #     - risk_update (float): Amount to increase the risk value.
        #     - max_risk (float): Maximum value for the risk.
        #     - risk_schedule (int): Number of iterations after which to update the risk value.
        # - device (torch.device): The device to run the agent on (CPU or GPU). If None, defaults to CPU. """
        
#         super(PPOAgent, self).__init__()
#         self.device = device if device is not None else torch.device('cpu')
#         self.allow_input_influence = allow_input_influence

#         # Neural Networks
#         self.policy = policy.to(self.device)
#         self.policy_old = deepcopy(policy).to(self.device)
#         self.state_value_function = state_value_function
        
#         # Trajectories
#         self.states_sequence = []
#         self.actions_sequence = []
#         self.logPs_sequence = []
#         self.entropies_sequence = []

#         # Torch training
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
#         self.logger = logger

#         # Entropy scheduler
#         if not entropy_scheduler:
#             entropy_scheduler = {'entropy_weight': 1, 'entropy_update_coefficient': 0.9, 'entropy_schedule': 20, 'minimum_entropy_weight': 1}
#         self.entropy_scheduler = entropy_scheduler
#         self.entropy_counter = 0

#         # Risky policy scheduler
#         if not risk_scheduler:
#             risk_scheduler = {'initial_risk': 0.8, 'risk_update': 0.00, 'max_risk': 1.00, 'risk_schedule': 20}
#         self.risk_scheduler = risk_scheduler
#         self.risk_counter = 0

#         # PPO parameters
#         if not ppo_parameters:
#             ppo_parameters = {'eps': 0.2, 'value_loss_weight': 0.01, 'gamma': 1, 'lam': 0.95, 'update_policy_every': 5}
#         self.ppo_parameters = ppo_parameters
#         self.swap_counter = 0
        
#     def act(self, states, actuator, mode='full'):
        # """ Select actions based on the current policy and the observed states.
        # Args:
        # - states (torch.Tensor): The observed states. Shape: (N, state_dim).
        # - actuator (AbstractActuator): The actuator to convert policy actions to environment actions.
        # - mode (str): The mode of the policy ('full', 'partial', or 'parameters'). Default is 'full'.
        # Returns:
        # - actions (list): A list of actions to be taken in the environment. """

#         super(PPOAgent, self).act()
#         tic_forward = time.time()

#         # Check if the observed IOCRN has unknown parameters
#         if mode == 'parameters' and self.allow_input_influence:
#             raise NotImplementedError("The cases of unknown parameters and/or allow input influence are not implemented yet.")
#         else:
#             actions, logPs, entropies = self.policy(states, mode=mode)
#             self.states_sequence.append(states)
#             self.actions_sequence.append(actions)
#             self.logPs_sequence.append(logPs)
#             self.entropies_sequence.append(entropies)
#         toc_forward = time.time()

#         # Log the forward pass time and return the actions
#         if self.logger is not None:
#             self.logger.log_metric('forward_time', toc_forward - tic_forward, step=None) 

#         actions = [actuator.actuate(a) for a in actions]           
#         return actions
    
#     def update(self, rewards, step_iteration=None):
#         """ Update the agent's policy based on the rewards received.
#         Args:
#         - rewards (list): A list of rewards, at the final step, received for each sample in the batch.
#         - step_iteration (int): The current iteration step for logging purposes. """
        
#         super(PPOAgent, self).update(rewards)
#         self.optimizer.zero_grad()
#         tic_backward = time.time()

#         # Extract dimensions
#         T = len(self.logPs_sequence)  # Final time step
#         N = self.logPs_sequence[0].shape[0]  # Number of samples in the batch

#         # Convert lists to tensors
#         logPs_sequence_tensor = torch.stack(self.logPs_sequence, dim=0).to(self.device)  # Shape: (T, N)
#         entropies_sequence_tensor = torch.stack(self.entropies_sequence, dim=0).to(self.device)  # Shape: (T, N)
#         rewards_sequence_tensor = torch.zeros_like(logPs_sequence_tensor)
#         rewards_sequence_tensor[-1,:] = torch.tensor(rewards, device=self.device) # Shape: (T, N)

#         # Compute the PPO loss
#         ppo_loss_for_each_sample = self.compute_PPO_loss(self.states_sequence, self.actions_sequence, rewards_sequence_tensor, entropies_sequence_tensor)

#         # Risky policy gradient and backpropagation
#         scores = rewards_sequence_tensor.mean(dim=0)  # Shape: (N,)
#         top_k = torch.topk(scores.detach(), int(N * (1. - self.risk_scheduler['risk'])), largest=False).indices # shape (int(N * (1. - self.risk_scheduler['risk'])),)
#         torch.mean(ppo_loss_for_each_sample[top_k]).backward()

#         # Step the optimizer
#         self.optimizer.step()
#         toc_backward = time.time()

#         # Update the old policy to the new policy
#         self.swap_counter += 1
#         if self.swap_counter >= self.ppo_parameters['update_policy_every']:
#             self.swap_counter = 0
#             self.swap()

#         # Update the entropy weight and the risk value
#         if self.entropy_scheduler['entropy_weight'] > self.entropy_scheduler["minimum_entropy_weight"]:
#             self.entropy_counter += 1
#             if self.entropy_counter % self.entropy_scheduler["entropy_schedule"] == 0:
#                 self.entropy_scheduler['entropy_weight'] *= self.entropy_scheduler["entropy_update_coefficient"]
#         if self.risk_scheduler['risk'] < self.risk_scheduler["max_risk"]:
#             self.risk_counter += 1
#             if self.risk_counter % self.risk_scheduler["risk_schedule"] == 0:
#                 self.risk_scheduler['risk'] += self.risk_scheduler["risk_update"]

        # # Log the training process
        # if self.logger is not None:
        #     self.logger.log_metric('batch average of total sequence entropy', entropies_sequence_tensor.sum(dim=0).mean().item(), step=step_iteration)
        #     self.logger.log_metric('batch average of total sequence logP', logPs_sequence_tensor.sum(dim=0).mean().item(), step=step_iteration)
        #     self.logger.log_metric('entropy weight', self.entropy_scheduler['entropy_weight'], step=step_iteration)
        #     self.logger.log_metric('risk', self.risk_scheduler['risk'], step=step_iteration)
        #     self.logger.log_metric('backward time', toc_backward - tic_backward, step=step_iteration)
        #     best = scores[top_k[0]]
        #     worst = scores[top_k[-1]]
        #     avg = scores[top_k].float().mean()
        #     self.logger.log_metric('best loss in the batch top k', best.item(), step=step_iteration)
        #     self.logger.log_metric('worst loss in the batch top k', worst.item(), step=step_iteration)
        #     self.logger.log_metric('average loss in the batch top k', avg.item(), step=step_iteration)

#         # Clear the trjectories
#         self.states_sequence.clear()
#         self.actions_sequence.clear()
#         self.logPs_sequence.clear()
#         self.entropies_sequence.clear()

#     def compute_PPO_loss(self, states_sequence, actions_sequence, rewards_sequence, entropies_sequence, masks=1., mode='full'):
#         """ Compute the PPO loss for the given states, actions, rewards, and entropies.
#         Args:
#             states_sequence (list): A (T+1)-list of states.
#             actions_sequence (list): A T-list of actions.
#             rewards_sequence (torch.Tensor): The rewards of the batch of states received at each time step. Shape: (T, N).
#             entropies_sequence (torch.Tensor): The entropies of the actions taken at each time step. Shape: (T, N).
#             masks (torch.Tensor or None): A mask tensor indicating which states are valid. Shape: (T, N). If None, all states are considered valid.
#             mode (str): The mode of the policy ('full' or 'partial'). Default is 'full'.
#         Returns:
#             total_loss (torch.Tensor): The computed PPO loss. """
        
#         # Compute the GAE
#         advantages_sequence, state_value_targets_sequence = self.compute_gae(states_sequence, rewards_sequence, masks)
        
#         # Compute the CPI
#         cpi = self.clipCPI(states_sequence, actions_sequence, advantages_sequence, masks, mode).mean(dim=0) # shape: (N,)
        
#         # Compute the entropy bonus
#         entropy_bonus = (entropies_sequence * masks).mean(dim=0) # shape: (N,)
        
#         # Compute the state value loss
#         v_loss = self.state_value_loss(states_sequence, state_value_targets_sequence) # shape: (1,)
        
#         # Sum up the losses
#         total_loss = -(cpi + self.entropy_scheduler['entropy_weight'] * entropy_bonus) + self.ppo_parameters['value_loss_weight'] * v_loss # shape: (N,)
        
#         # Log the calculated metrics
        # if self.logger is not None:
        #     self.logger.log_metric('average advantage', advantages_sequence.mean().item())
        #     self.logger.log_metric('average state value targets', state_value_targets_sequence.mean().item())
        #     self.logger.log_metric('cpi', cpi.mean().item())
        #     self.logger.log_metric('entropy bonus', entropy_bonus.mean().item())
        #     self.logger.log_metric('value loss', v_loss.item())
        #     self.logger.log_metric('total loss', total_loss.mean().item())

#         return total_loss



#     #--------------------------- Helper functions ---------------------------#
#     @torch.no_grad()
#     def compute_gae(self, states_sequence, rewards_sequence, masks):
#         """ Compute the Generalized Advantage Estimation (GAE) for the given states and rewards.
#         Args:
#             - states_sequence: A (T+1)-list of states.
#             - rewards (torch.Tensor): The rewards of the batch of states received at each time step. Shape: (T, N).
#             - masks (torch.Tensor or None): A mask tensor indicating which states are valid. Shape: (T, N). If None, all states are considered valid.
#         Returns:
#             advantages_sequence (torch.Tensor): The computed advantages for each state. Shape: (T, N).
#             state_value_targets_sequence (torch.Tensor): The targets for the value loss. Shape: (T, N). """
        
#         # Extract PPO parameters relevant for GAE
#         gamma = self.ppo_parameters['gamma']
#         lam = self.ppo_parameters['lam']

#         # Compute the state values for each state in the sequence
#         values_sequence = self.v(states_sequence)  # (T+1, N) tensor

#         # Compute the advantages using GAE
#         T = rewards_sequence.shape[0]
#         advantages_sequence = torch.zeros_like(rewards_sequence) # (T, N) tensor
#         advantages_next = torch.zeros_like(rewards_sequence[0]) # (N,) tensor
#         for t in reversed(range(T)):
#             if type(masks) == torch.Tensor:
#                 delta_t = rewards_sequence[t] + gamma * values_sequence[t + 1] * masks[t] - values_sequence[t] # (N,) tensor
#                 advantages_sequence[t]= delta_t + gamma * lam * advantages_next * masks[t]
#                 advantages_next= advantages_sequence[t]
#             else:
#                 delta_t = rewards_sequence[t] + gamma * values_sequence[t + 1] - values_sequence[t] # (N,) tensor
#                 advantages_sequence[t]= delta_t + gamma * lam * advantages_next
#                 advantages_next= advantages_sequence[t]

#         # Compute the targets for the value loss
#         state_value_targets_sequence = advantages_sequence + values_sequence[:-1] # (T, N) tensor

#         return advantages_sequence, state_value_targets_sequence
    
#     def clipCPI(self, states_sequence, actions_sequence, advantages_sequence, masks, mode='full'):
#         """ Compute the clipped policy improvement (CPI) for the given states, actions, and advantages.
#         Args:
#             - states_sequence: A (T+1)-list of states.
#             - actions_sequence: A T-list of actions.
#             - advantages_sequence: A tensor of shape (T, N) representing the advantages computed using GAE.
#             - masks: A tensor of shape (T, N) indicating which states are valid. If None, all states are considered valid.
#             - mode (str): The mode of the policy ('full' or 'partial'). Default is 'full'.
#         Returns:
#             - clip_cpi_sequence * masks (torch.Tensor): The masked clipped policy improvement for the given states, actions, and advantages. Shape: (T, N)."""
        
#         # Extract PPO parameters relevant for CPI
#         eps = self.ppo_parameters['eps']

#         # Compute the old and new action probabilities across the time steps
#         old_logPs_sequence = self.p_old(states_sequence, actions_sequence, mode) # (T, N) tensor
#         new_logPs_sequence = self.p(states_sequence, actions_sequence, mode) # (T, N) tensor

#         # Compute the risk ratio and the clipped policy improvement
#         risk_ratio_sequence = torch.exp(new_logPs_sequence) / (torch.exp(old_logPs_sequence) + 1e-6) # (T, N) tensor
#         clip_cpi_sequence = torch.min(risk_ratio_sequence * advantages_sequence, torch.clip(risk_ratio_sequence, 1 - eps, 1 + eps) * advantages_sequence) # (T, N) tensor

#         # Log the risk ratio if a logger is available
#         if self.logger is not None:
#             self.logger.log_metric('risk ratio', risk_ratio_sequence.mean().item())

#         return clip_cpi_sequence * masks
    
#     def state_value_loss(self, states_sequence, state_value_targets_sequence):
#         """ Compute the state value loss for the given states and targets.
#         Args:
#             states_sequence: A (T+1)-list of states.
#             state_value_targets_sequence: A (T, N) tensor representing the targets for the value loss.
#         Returns:
#             torch.Tensor: The computed state value loss. """
#         v_pred = self.v(states_sequence)[:-1]
#         return torch.nn.functional.mse_loss(v_pred, state_value_targets_sequence)

    

#     #--------------------------- Functions for applying the policy and value networks across sequences ---------------------------#
#     def p(self, states_sequence, actions_sequence, mode='full'):
#         logPs = [ self.policy(states, mode, actions) for states,actions in zip(states_sequence[:-1], actions_sequence) ] # A T-list of tensors, each of shape (N,)
#         logPs = torch.stack(logPs) # Tensor of shape (T, N)
#         return logPs

#     def p_old(self, states_sequence, actions_sequence, mode='full'):
#         logPs = [ self.policy_old(states, mode, actions) for states,actions in zip(states_sequence[:-1], actions_sequence) ] # A T-list of tensors, each of shape (N,)
#         logPs = torch.stack(logPs) # Tensor of shape (T, N)
#         return logPs
    
#     def v(self, states_sequence):
#         values_sequence = [ self.state_value_function(states) for states in states_sequence ] # A (T+1)-list of tensors, each of shape (N,)
#         values_sequence = torch.stack(values_sequence) # Tensor of shape (T+1, N)
#         values_sequence[-1] = torch.zeros_like(values_sequence[-1])
#         return values_sequence
    

#     #--------------------------- Functions for swapping old and new policy networks ---------------------------#
#     def swap(self):
#         self.policy_old.load_state_dict(self.policy.state_dict())
#         self.freeze(self.policy_old)
#         self.unfreeze(self.policy)

#     def freeze(self, net):
#         for param in net.parameters():
#             param.requires_grad = False
    
#     def unfreeze(self, net):
#         for param in net.parameters():
#             param.requires_grad = True