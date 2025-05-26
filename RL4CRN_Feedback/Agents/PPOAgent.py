# from RL4CRN_Feedback.Agents.BimolecularMassActionAgent import BimolecularMassActionAgent
# import time
# import torch

# class PPOAgent(BimolecularMassActionAgent):

#     def __init__(self, env, policy, allow_input_influence=False, logger=None, learning_rate=1e-3, entropy_weight=1., entropy_update_coefficient=0.9, entropy_schedule=20, minimum_entropy_weight=1, risk=0.8, risk_update=0.00, max_risk=1.00, risk_schedule=20, device=None, accumulate_gradients_for=3, update_policy_every=5):
#         super(PPOAgent, self).__init__(env, policy, allow_input_influence, logger, learning_rate, entropy_weight, entropy_update_coefficient, entropy_schedule, minimum_entropy_weight, risk, risk_update, max_risk, risk_schedule, device)
#         self.update_policy_every = update_policy_every
#         self.swap_counter = 0
#         self.accumulate_gradients_for = accumulate_gradients_for
#         self.accumulation_counter = 0

#     def act(self, observation):
#         # super(BimolecularMassActionAgent, self).act()
#         tic_forward = time.time()
#         if self.env.state.num_unknown_parameters > 0:
#             raise NotImplementedError("The case of unknown parameters is not implemented yet.")
#         else:
#             actions, logP, entropy = self.policy(observation, mode='full')
#             self.logPs.append(logP)
#             self.entropies.append(entropy)
#         toc_forward = time.time()
#         if self.logger is not None:
#             self.logger.log_metric('forward_time', toc_forward - tic_forward)           
#         return actions, logP, entropy

#     def update(self, cli_objective, scores=None):
#         if scores is None: # use just the raw score to select the top k
#             scores = cli_objective
#         self.accumulation_counter += 1
#          # super(BimolecularMassActionAgent, self).update(rewards)
#         tic_backward = time.time()
#         # Retrieve the information from the forward pass
#         if self.accumulation_counter >= self.accumulate_gradients_for:
#             self.optimizer.zero_grad()

#         sum_logPs = torch.sum(torch.stack(self.logPs, dim=1), dim=1)
#         sum_entropies = torch.sum(torch.stack(self.entropies, dim=1), dim=1)
#         entropy_mean = torch.mean(sum_entropies)

#         n_samples = self.logPs[0].shape[0]
#         # Compute the loss that is used to compute the gradient
#         loss_for_gradient = cli_objective 
#         # Risky policy gradient
#         top_k = torch.topk(scores, int(n_samples * (1.-self.risk)), largest=False).indices
#         torch.mean(loss_for_gradient[top_k]).backward()

#         if self.logger is not None:
#             # log gradients
#             for name, param in self.policy.named_parameters():
#                 if param.grad is not None:
#                     self.logger.log_metric(name, param.grad.data.cpu().numpy().std())

#         # Clip gradients
#         # torch.nn.utils.clip_grad_norm_(MyCRN_Generator.parameters(), 0.01)
#         if self.accumulation_counter <= self.accumulate_gradients_for:
#             self.optimizer.step()
#             self.accumulation_counter = 0
#         toc_backward = time.time()
#         # Update the entropy weight and the risk value
#         if self.entropy_weight > self.minimum_entropy_weight:
#             self.entropy_counter += 1
#             if self.entropy_counter % self.entropy_schedule == 0:
#                 self.entropy_weight *= self.entropy_update_coefficient
#         if self.risk < self.max_risk:
#             self.risk_counter += 1
#             if self.risk_counter % self.risk_schedule == 0:
#                 self.risk += self.risk_update
#         # Log the training process
#         if self.logger is not None:
#             self.logger.log_metric('entropy', sum_entropies.mean().item())
#             self.logger.log_metric('logP', sum_logPs.mean().item())
#             self.logger.log_metric('entropy_weight', self.entropy_weight)
#             self.logger.log_metric('risk', self.risk)
#             self.logger.log_metric('backward_time', toc_backward - tic_backward)
#             best = scores[top_k[0]]
#             worst = scores[top_k[-1]]
#             avg = scores[top_k].float().mean()
#             self.logger.log_metric('best_loss', best.item())
#             self.logger.log_metric('worst_loss', worst.item())
#             self.logger.log_metric('avg_loss', avg.item())
#         # Clear the lists of logPs and entropies
#         self.logPs.clear()
#         self.entropies.clear()
#         self.swap_counter += 1
#         if self.swap_counter >= self.update_policy_every:
#             self.swap_counter = 0
#             # Swap the policy with the target policy
#             self.policy.swap()

#         # update the entropy within the PPO policy
#         self.policy.entropy_weight = self.entropy_weight

    
from RL4CRN_Feedback.Agents.BimolecularMassActionAgent import BimolecularMassActionAgent
import time
import torch

class PPOAgent(BimolecularMassActionAgent):
    """
    PPOAgent using risk-sensitive policy updates with gradient accumulation and entropy regularization.
    """

    def __init__(
        self, env, policy, allow_input_influence=False, logger=None,
        learning_rate=1e-3, entropy_weight=0.01, entropy_update_coefficient=0.9, 
        entropy_schedule=20, minimum_entropy_weight=1e-3, risk=0.8, risk_update=0.00, 
        max_risk=1.00, risk_schedule=20, device=None, 
        accumulate_gradients_for=3, update_policy_every=5
    ):
        super().__init__(
            env, policy, allow_input_influence, logger, learning_rate, 
            entropy_weight, entropy_update_coefficient, entropy_schedule, 
            minimum_entropy_weight, risk, risk_update, max_risk, risk_schedule, device
        )
        self.update_policy_every = update_policy_every
        self.swap_counter = 0
        self.accumulate_gradients_for = accumulate_gradients_for
        self.accumulation_counter = 0

    def act(self, observation):
        """
        Select an action based on current observation and store log-prob and entropy.
        """
        tic_forward = time.time()
        if self.env.state.num_unknown_parameters > 0:
            raise NotImplementedError("The case of unknown parameters is not implemented yet.")
        
        actions, logP, entropy = self.policy(observation, mode='full')
        self.logPs.append(logP)
        self.entropies.append(entropy)

        toc_forward = time.time()
        if self.logger is not None:
            self.logger.log_metric('forward_time', toc_forward - tic_forward)
        return actions, logP, entropy

    def update(self, cli_objective, scores=None):
        """
        Perform a PPO update using the clipped objective and risk-sensitive filtering.

        Args:
            cli_objective (Tensor): Clipped PPO objective for each sample.
            scores (Tensor, optional): Alternative scores for selecting top-k samples (defaults to `cli_objective`).
        """
        if scores is None:
            scores = cli_objective

        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        self.accumulation_counter += 1
        tic_backward = time.time()

        sum_logPs = torch.sum(torch.stack(self.logPs, dim=1), dim=1)
        sum_entropies = torch.sum(torch.stack(self.entropies, dim=1), dim=1)
        entropy_mean = torch.mean(sum_entropies)

        n_samples = self.logPs[0].shape[0]
        top_k = torch.topk(scores.detach(), int(n_samples * (1. - self.risk)), largest=False).indices

        # PPO policy loss (already computed externally)
        loss = torch.mean(cli_objective[top_k])
        loss.backward()

        # if self.logger is not None:
        #     for name, param in self.policy.named_parameters():
        #         if param.grad is not None:
        #             self.logger.log_metric(name, param.grad.data.cpu().numpy().std())

        if self.accumulation_counter >= self.accumulate_gradients_for:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.accumulation_counter = 0

        toc_backward = time.time()

        # Update entropy weight
        if self.entropy_weight > self.minimum_entropy_weight:
            self.entropy_counter += 1
            if self.entropy_counter % self.entropy_schedule == 0:
                self.entropy_weight *= self.entropy_update_coefficient

        # Update risk threshold
        if self.risk < self.max_risk:
            self.risk_counter += 1
            if self.risk_counter % self.risk_schedule == 0:
                self.risk += self.risk_update

        if self.logger is not None:
            self.logger.log_metric('entropy', entropy_mean.item())
            self.logger.log_metric('logP', sum_logPs.mean().item())
            self.logger.log_metric('entropy_weight', self.entropy_weight)
            self.logger.log_metric('risk', self.risk)
            self.logger.log_metric('backward_time', toc_backward - tic_backward)
            best = scores[top_k[0]]
            worst = scores[top_k[-1]]
            avg = scores[top_k].float().mean()
            self.logger.log_metric('best_loss', best.item())
            self.logger.log_metric('worst_loss', worst.item())
            self.logger.log_metric('avg_loss', avg.item())

        self.logPs.clear()
        self.entropies.clear()

        self.swap_counter += 1
        if self.swap_counter >= self.update_policy_every:
            self.swap_counter = 0
            self.policy.swap()

        self.policy.entropy_weight = self.entropy_weight




    
