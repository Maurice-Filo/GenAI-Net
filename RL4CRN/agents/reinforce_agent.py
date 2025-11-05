import torch
import time
import numpy as np
from RL4CRN.agents.abstract_agent import AbstractAgent

class REINFORCEAgent(AbstractAgent):
    def __init__(self, policy, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_scheduler = {}, risk_scheduler = {}, device=None):
        """ Initialize the REINFORCE agent with a policy, learning rate, and optional entropy and risk schedulers.
        Args:
        - policy (torch.nn.Module): The policy network to be used by the agent.
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
            entropy_scheduler = {'entropy_weight': 1, 'entropy_update_coefficient': 0.9, 'entropy_schedule': 20, 'minimum_entropy_weight': 1}
        self.entropy_scheduler = entropy_scheduler
        self.entropy_counter = 0

        # Risky policy scheduler
        if not risk_scheduler:
            risk_scheduler = {'risk': 0.8, 'risk_update': 0.00, 'max_risk': 1.00, 'risk_schedule': 20}
        self.risk_scheduler = risk_scheduler
        self.risk_counter = 0
        
    def act(self, states, actuator, mode='full'):
        """ Select actions based on the current policy and the observed states.
        Args:
        - states (torch.Tensor): The observed states. Shape: (N, state_dim).
        - actuator (AbstractActuator): The actuator to convert policy actions to environment actions.
        - mode (str): The mode of the policy ('full', 'partial', or 'parameters'). Default is 'full'.
        Returns:
        - actions (list): A list of actions to be taken in the environment. """

        super(REINFORCEAgent, self).act()
        tic_forward = time.time()

        # Check if the observed IOCRN has unknown parameters
        if mode == 'parameters' and self.allow_input_influence:
            raise NotImplementedError("The cases of unknown rate constants and/or allow input influence are not implemented yet.")
        else:
            actions, logPs, entropies = self.policy(states, mode=mode)
            self.logPs_sequence.append(logPs)
            self.entropies_sequence.append(entropies)
        toc_forward = time.time()

        # Log the forward pass time and return the actions
        if self.logger is not None:
            self.logger.log_metric('forward_time', toc_forward - tic_forward, step=None)    

        actions = [actuator.actuate(a) for a in actions]
        return actions
    
    def update(self, rewards, step_iteration=None):
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

        # Compute the loss that is used to compute the gradient
        final_loss_for_each_sample = torch.tensor(final_loss_for_each_sample, device=sum_logPs.device, dtype=sum_logPs.dtype).detach() # shape (N,)
        loss_for_gradient =  final_loss_for_each_sample * sum_logPs # shape (N,)
        entropy_for_gradient = sum_entropies + sum_entropies.detach() * sum_logPs # shape (N,)
        loss_for_gradient = loss_for_gradient - self.entropy_scheduler['entropy_weight'] * entropy_for_gradient # shape (N,)

        # Risky policy gradient
        top_k = torch.topk(final_loss_for_each_sample, int(N * (1. - self.risk_scheduler['risk'])), largest=False).indices # shape (int(N * (1. - self.risk_scheduler['risk'])),)
        # torch.mean(loss_for_gradient[top_k]).backward() # TODO: add another term to promote entropy for the remaining samples not in the top k
        tt = torch.sum(loss_for_gradient[top_k]) / N
        tt.backward()
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
            self.logger.log_metric('batch average of total sequence entropy', sum_entropies.mean().item(), step=step_iteration)
            self.logger.log_metric('batch average of total sequence logP', sum_logPs.mean().item(), step=step_iteration)
            self.logger.log_metric('entropy weight', self.entropy_scheduler['entropy_weight'], step=step_iteration)
            self.logger.log_metric('risk', self.risk_scheduler['risk'], step=step_iteration)
            self.logger.log_metric('backward time', toc_backward - tic_backward, step=step_iteration)
            best = final_loss_for_each_sample[top_k[0]]
            worst = final_loss_for_each_sample[top_k[-1]]
            avg = final_loss_for_each_sample[top_k].float().mean()
            batch_avg = final_loss_for_each_sample.float().mean()
            self.logger.log_metric('full batch average loss', batch_avg.item(), step=step_iteration)
            self.logger.log_metric('best loss in the batch top k', best.item(), step=step_iteration)
            self.logger.log_metric('worst loss in the batch top k', worst.item(), step=step_iteration)
            self.logger.log_metric('average loss in the batch top k', avg.item(), step=step_iteration)

        # Clear the lists of logPs and entropies
        self.logPs_sequence.clear()
        self.entropies_sequence.clear()