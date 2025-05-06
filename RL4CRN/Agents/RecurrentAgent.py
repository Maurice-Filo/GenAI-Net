import torch
import time
import numpy as np
from RL4CRN.Agents.AbstractAgent import AbstractAgent

class RecurrentAgent(AbstractAgent):
    def __init__(self, env, completor, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_weight=0.01, entropy_update_coefficient=0.9, entropy_schedule=20, minimum_entropy_weight=1, risk=0.8, risk_update=0.00, max_risk=1.00, risk_schedule=20):
        """
        Initialize the RecurrentAgent with the environment and the completor model.
        :param env: The environment to interact with.
        :param completor: The model used to generate reactions. The completor must be an instance of the CRNCompletor class.
        :param allow_input_influence: Whether to allow input influence on the generated reaction.
        :param logger: The logger to log the training process. This is a comet.ml logger
        :param learning_rate: The learning rate for the optimizer.
        :param entropy_weight: The initial weight for the entropy term in the loss function.
        :param entropy_update_coefficient: The coefficient for updating the entropy weight.
        :param entropy_schedule: The number of epochs before updating the entropy weight.
        :param minimum_entropy_weight: The minimum value for the entropy weight.
        :param risk: The initial risk value for the agent.
        :param risk_update: The coefficient for updating the risk value.
        :param max_risk: The maximum value for the risk.
        :param risk_schedule: The number of epochs before updating the risk value.
        """
        super(RecurrentAgent, self).__init__()
        self.env = env
        self.completor = completor
        self.allow_input_influence = allow_input_influence
        self.queue = []
        self.last_logP = None
        self.last_entropy = None
        # Torch training
        self.optimizer = torch.optim.Adam(self.completor.parameters(), lr=learning_rate)
        self.logger = logger
        # Entropy parameters
        self.entropy_weight = entropy_weight
        self.entropy_update_coefficient = entropy_update_coefficient
        self.entropy_schedule = entropy_schedule
        self.minimum_entropy_weight = minimum_entropy_weight
        # Risky policy gradient parameters
        self.risk = risk
        self.risk_update = risk_update
        self.max_risk = max_risk
        self.risk_schedule = risk_schedule
        # Counters for entropy and risk updates
        self.entropy_counter = 0
        self.risk_counter = 0

    def act(self):
        super(RecurrentAgent, self).act()
        tic_forward = time.time()
        # Generate a batch of list of reactions to complete the CRNs and store them into a queue of dimension: (batch size, maximum number of reactions). 
        # The action generation is done only if the queue is empty. 
        if len(self.queue) == 0 or len(self.queue[0]) == 0:
            batched_action_set, total_logP, total_entropy = self.completor()
            self.queue = batched_action_set
            self.last_logP = total_logP
            self.last_entropy = total_entropy
            toc_forward = time.time()
            if self.logger is not None:
                self.logger.log_metric('forward_time', toc_forward - tic_forward)
        # At each call of act, we return a batch of reactions, one for each environment.
        action = [lst.pop(0) for lst in self.queue]            
        return action
    
    def update(self, rewards):
        super(RecurrentAgent, self).update(rewards)
        tic_backward = time.time()
        # Retrieve the information from the forward pass
        self.optimizer.zero_grad()
        loss_for_each_sample = torch.tensor(rewards, requires_grad=False).to(self.completor.device)
        entropy_mean = torch.mean(self.last_entropy)
        n_samples = self.last_logP.shape[0]
        # Compute the loss that is used to compute the gradient
        loss_for_gradient =  (loss_for_each_sample * self.last_logP) - self.entropy_weight * entropy_mean - (self.entropy_weight * entropy_mean.detach() * self.last_logP)
        # Risky policy gradient
        top_k = torch.topk(loss_for_each_sample, int(n_samples * (1.-self.risk)), largest=False).indices
        torch.mean(loss_for_gradient[top_k]).backward()
        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(MyCRN_Generator.parameters(), 0.01)
        self.optimizer.step()
        toc_backward = time.time()
        # Update the entropy weight and the risk value
        if self.entropy_weight > self.minimum_entropy_weight:
            self.entropy_counter += 1
            if self.entropy_counter % self.entropy_schedule == 0:
                self.entropy_weight *= self.entropy_update_coefficient
        if self.risk < self.max_risk:
            self.risk_counter += 1
            if self.risk_counter % self.risk_schedule == 0:
                self.risk += self.risk_update
        # Log the training process
        if self.logger is not None:
            self.logger.log_metric('entropy', self.last_entropy.mean().item())
            self.logger.log_metric('logP', self.last_logP.mean().item())
            self.logger.log_metric('entropy_weight', self.entropy_weight)
            self.logger.log_metric('risk', self.risk)
            self.logger.log_metric('backward_time', toc_backward - tic_backward)
            best = loss_for_each_sample[top_k[0]]
            worst = loss_for_each_sample[top_k[-1]]
            avg = loss_for_each_sample[top_k].float().mean()
            self.logger.log_metric('best_loss', best.item())
            self.logger.log_metric('worst_loss', worst.item())
            self.logger.log_metric('avg_loss', avg.item())