import torch
from copy import deepcopy
import numpy as np
 
class PPO(torch.nn.Module):
    """
    Proximal Policy Optimization (PPO) agent with clipped objective and value network.

    Args:
        net (torch.nn.Module): Policy network.
        v_net (torch.nn.Module): Value network.
        eps (float): Clipping parameter epsilon.
        device (str): Device string, e.g., 'cpu' or 'cuda'.
        entropy_weight (float): Weight for entropy bonus.
        logger (Optional): Logging utility with a .log_metric(name, value) method.
    """

    def __init__(self, net, v_net, eps=0.2, device='cpu', entropy_weight=0.01, logger = None):
        super().__init__()
        
        # --- networks ---
        self.net = net.to(device)
        self.net_old = deepcopy(net).to(device)
        self.v_net = v_net.to(device)
        
        # --- hyperparameters ---
        self.eps = eps
        self.entropy_weight = entropy_weight

        # --- system parameters ---
        self.freeze(self.net_old)
        self.device = device
        self.logger = logger

        

    def forward(self, x, mode='full'): # run the policy
        """
        Forward pass through the policy network.

        Args:
            x (Tensor): Input state.
            mode (str): CRN input mode.

        Returns:
            Output from policy network.
        """
        return self.net(x, mode=mode)
    
    def advantage(self, states, rewards, mask, gamma=0.99):
        """
        Compute the advantage function A(s, a) = Q(s, a) - V(s).

        Args:
            states (Tensor): State trajectories.
            rewards (Tensor): Collected rewards.
            mask (Tensor): Binary mask to ignore padding.
            gamma (float): Discount factor.

        Returns:
            Tuple[Tensor, Tensor]: (advantage, discounted_rewards)
        """
        with torch.no_grad():
            values = self.v(states)[:-1] # we need to skip the final state 
            # values [number_of_reactions, batch_size]

        discounts = gamma ** torch.arange(rewards.shape[0]).to(self.device)
        discounts = discounts.unsqueeze(-1).repeat(1,rewards.shape[1])
        discounted_rewards = torch.cumsum((rewards * discounts).flip(0), dim=0).flip(0) / discounts

        advantage = discounted_rewards - values # Q(s,a) - V(s)

        # masking
        advantage = advantage * mask
        discounted_rewards = discounted_rewards * mask

        # normalize advantage
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
        return advantage, discounted_rewards

    def compute_gae(self, states, rewards, mask, gamma=0.99, lam=0.95):
        """
        rewards: [T, B] — immediate rewards
        values: [T+1, B] — value estimates (bootstrap with last state)
        mask: [T, B] — 1 for real, 0 for padding
        """
        with torch.no_grad():
            values = self.v(states)

        # print all shapes:
        # print(f"rewards: {rewards.shape}, values: {values.shape}")

        T = rewards.size(0)
        B = rewards.size(1)
        advantage = torch.zeros_like(rewards)
        last_adv = 0

        for t in reversed(range(T)):
            if type(mask) == torch.Tensor:
                delta = rewards[t] + gamma * values[t + 1] * mask[t] - values[t]
                advantage[t] = delta + gamma * lam * last_adv * mask[t]
                last_adv = advantage[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                advantage[t] = delta + gamma * lam * last_adv
                last_adv = advantage[t]

        returns = advantage + values[:-1]  # target for value loss
        return advantage, returns

    def clipCPI(self, states, actions, advantage, mask):
        """
        Compute the clipped surrogate objective.

        Args:
            states (Tensor): Input states.
            actions (Tensor): Taken actions.
            advantage (Tensor): Advantage estimates.
            mask (Tensor): Binary mask.

        Returns:
            Tensor: Clipped surrogate loss.
        """
        risk_ratio = torch.exp(self.p(states, actions)) / (torch.exp((self.p_old(states, actions))) + 1e-6)
        clip_cpi = torch.min( 
            risk_ratio * advantage,
            torch.clip(risk_ratio, 1 - self.eps, 1 + self.eps) * advantage
        )
        return clip_cpi * mask

    def value_loss(self, states, discounted_rewards, mask):
        """
        Compute value network loss (MSE).

        Args:
            states (Tensor): States.
            discounted_rewards (Tensor): Discounted returns.
            mask (Tensor): Binary mask.

        Returns:
            Tensor: MSE loss.
        """
        v_pred = self.v(states)[:-1] # we need to skip the final state
        # v_pred = v_pred * mask
        # discounted_rewards = discounted_rewards * mask
        return torch.nn.functional.mse_loss(v_pred, discounted_rewards)

    def entropy_bonus(self, entropy, mask):
        """
        Compute entropy regularization term.

        Args:
            entropy (Tensor): Entropy per timestep.
            mask (Tensor): Mask to ignore padding.

        Returns:
            Tensor: Weighted entropy bonus.
        """
        entropy = (entropy * mask).sum(dim=0).mean()
        return self.entropy_weight * entropy

    def reward(self, states, actions, rewards, entropy, mask=1., gamma=0.99, value_loss_weight=0.01, lam=0.95):
        """
        Compute total PPO loss: policy + entropy - value.

        Args:
            states (Tensor): States.
            actions (Tensor): Actions.
            rewards (Tensor): Rewards.
            entropy (Tensor): Policy entropy.
            mask (Tensor): Mask.
            gamma (float): Discount factor.
            value_loss_weight (float): Weight for value loss.
            lam (float): Lambda for GAE.
            
        Returns:
            Tensor: Total loss.
        """
        advantage, returns = self.compute_gae(states, rewards, gamma, mask, lam=lam) # self.advantage(states, rewards, gamma, mask)
        cpi = self.clipCPI(states, actions, advantage.detach(), mask).mean(dim=0)
        entropy_bonus = self.entropy_bonus(entropy, mask)
        v_loss = self.value_loss(states, returns, mask)
        total_loss = -(cpi + entropy_bonus) + value_loss_weight * v_loss
        if self.logger is not None:
            self.logger.log_metric('advantage', advantage.mean().item())
            self.logger.log_metric('discounted_rewards', returns.mean().item())
            self.logger.log_metric('cpi', cpi.mean().item())
            self.logger.log_metric('entropy_bonus', entropy_bonus.item())
            self.logger.log_metric('value_loss', v_loss.item())
            self.logger.log_metric('total_loss', total_loss.mean().item())
        return total_loss

    def swap(self):
        """
        Update the old policy network to match the current one.
        """
        self.net_old.load_state_dict(self.net.state_dict())
        self.freeze(self.net_old)
        self.unfreeze(self.net)
    
    def freeze(self, net):
        """
        Freeze network parameters.

        Args:
            net (torch.nn.Module): Network to freeze.
        """
        for param in net.parameters():
            param.requires_grad = False
    
    def unfreeze(self, net):
        """
        Unfreeze network parameters.

        Args:
            net (torch.nn.Module): Network to unfreeze.
        """
        for param in net.parameters():
            param.requires_grad = True

    # def sample_training(self, state):
    #     probs = self.net(torch.tensor(state).to(self.device))
    #     dist = torch.distributions.Categorical(probs)
    #     action = dist.sample()
    #     entropy = dist.entropy()
    #     return action, dist.log_prob(action), entropy
    
    def sample(self, state):
        """
        Sample an action from the policy network.

        Args:
            state (Tensor): Input state.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (sampled_action, log_prob, entropy)
        """
        samples, log_probability, entropy = self.net(state)
        return samples, log_probability, entropy
            
    # def sample_best(self, state):
    #     state = torch.tensor(state).to(self.device)
    #     with torch.no_grad():
    #         try:
    #             # print(self(state))
    #             return torch.argmax(self(state))
    #         except:
    #             print(self(state))
    #             raise Exception()
            
    def p(self, state, action):
        """
        Compute log-probabilities from current policy.

        Args:
            state (Tensor): States.
            action (Tensor): Actions.

        Returns:
            Tensor: Log-probabilities.
        """
        o = [ self.net.verify(s, a) for s,a in zip(state[:-1], action) ]
        o = torch.stack(o)
        return o

    def p_old(self, state, action):
        """
        Compute log-probabilities from old policy.

        Args:
            state (Tensor): States.
            action (Tensor): Actions.

        Returns:
            Tensor: Log-probabilities.
        """
        o = [ self.net_old.verify(s, a) for s,a in zip(state[:-1], action) ]
        o = torch.stack(o)
        return o
    
    def v(self, state):
        """
        Evaluate value function.

        Args:
            state (Tensor): Input states.

        Returns:
            Tensor: Value estimates.
        """
        o = [ self.v_net(s) for s in state ]
        o = torch.stack(o)
        return o