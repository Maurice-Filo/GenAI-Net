import torch
from RL4CRN.utils.ffnn import FFNN
from RL4CRN.distributions.lognormal import MultivariateLogNormal
from RL4CRN.distributions.categorical import MultiVariateCategorical
from torch.distributions import Categorical, LogNormal

class ParameterGeneratorFromDistribution(torch.nn.Module):
    def __init__(self, distribution, backbone, device='cpu'):
        super().__init__()
        self.backbone = backbone.to(device)
        self.device = device
        self.distribution = distribution

        if distribution["type"] == 'lognormal':
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'lognormal' distribution, 'dim' must be specified in the distribution dictionary.")
            d = int(distribution_dim)
            self.distribution_dim = d
            self.decoder = FFNN(input_size=backbone.output_size, 
                                 output_size=d + d + (d**2 - d)//2, 
                                 hidden_size=backbone.hidden_size, 
                                 num_layers=1).to(device=device)  
            def lognormal_forward(self, x, mask=None):
                
                continuous_distribution_parameters= self.decoder(self.backbone(x))
                continuous_distribution_parameters = torch.nn.functional.softplus(continuous_distribution_parameters)
                mu_log_normal, sigma_log_normal = continuous_distribution_parameters[:, 0], continuous_distribution_parameters[:, 1] # shape: (N,)
                mu_normal = torch.log(mu_log_normal**2 / torch.sqrt(mu_log_normal**2 + sigma_log_normal**2)) 
                sigma_normal = torch.log(1 + sigma_log_normal**2 / mu_log_normal**2)
                reaction_rate_distribution = LogNormal(mu_normal, sigma_normal) # batch of N LogNormal distributions
                samples = reaction_rate_distribution.sample() # shape: (N,)
                entropies = reaction_rate_distribution.entropy() # shape: (N,)
                log_probs = reaction_rate_distribution.log_prob(samples) # shape: (N,)
                samples = samples.unsqueeze(-1) # shape: (N, 1)
                return samples, log_probs, entropies
            self.forward = lognormal_forward.__get__(self)








                # params = self.decoder(self.backbone(x))
                # means = torch.nn.functional.softplus(params[:, :self.distribution_dim])
                # L_diags = torch.nn.functional.softplus(params[:, self.distribution_dim:self.distribution_dim*2])
                # L_off_diags = params[:, self.distribution_dim*2:]
                # L = torch.zeros((x.shape[0], self.distribution_dim, self.distribution_dim), device=self.device)
                # tril_indices = torch.tril_indices(row=self.distribution_dim, col=self.distribution_dim, offset=-1)
                # L[:, tril_indices[0], tril_indices[1]] = L_off_diags
                # L[:, torch.arange(self.distribution_dim), torch.arange(self.distribution_dim)] = L_diags
                # covariance_matrices = torch.matmul(L, L.transpose(-1, -2))
                # mu = means
                # Sigma = covariance_matrices

                # Transform distribution parameters from log-space to normal space
                # Σ_ij = ln(1 + S_ij / (m_i m_j)), applied elementwise
                # μ_i = ln(m_i) - 0.5 * Σ_ii
                # m = means                           # (N, D)
                # S = covariance_matrices             # (N, D, D)
                # outer_mm = m.unsqueeze(-1) * m.unsqueeze(-2)           # (N, D, D)
                # Sigma = torch.log1p(S / outer_mm)                      # (N, D, D)
                # mu = torch.log(m) - 0.5 * torch.diagonal(Sigma, dim1=-1, dim2=-2)

                

            #     # Mask the parameters that do not exist for the current reaction
            #     if mask is not None:
            #         mu = mu * mask
            #         Sigma = Sigma * mask.unsqueeze(-1) * mask.unsqueeze(-2)

            #     # Create the multivariate log-normal distribution
            #     dist = MultivariateLogNormal(loc=mu, covariance_matrix=Sigma)
            #     samples = dist.sample()
            #     log_probs = dist.log_prob(samples)
            #     entropies = dist.entropy()
            #     return samples, log_probs, entropies
            # self.forward = lognormal_forward.__get__(self)

        if distribution["type"] == 'categorical': 
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'dim' must be specified in the distribution dictionary.")
            try:
                categories = distribution["categories"]
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'categories' must be specified in the distribution dictionary.")
            d = int(distribution_dim)
            self.distribution_dim = d
            self.num_categories = sum([c.shape[0] for c in categories])
            self.categories = torch.tensor(categories, device=device)
            self.decoder = FFNN(input_size=backbone.output_size, 
                                 output_size=self.num_categories, 
                                 hidden_size=backbone.hidden_size, 
                                 num_layers=1).to(device=device)  
            def categorical_forward(self, x, logit_mask=None, dimension_mask=None):
                logits = self.decoder(self.backbone(x)).view(-1, self.distribution_dim, self.num_categories)
                if logit_mask is not None:
                    logits = logits.masked_fill(~logit_mask.bool(), float('-inf'))
                dist = MultiVariateCategorical(logits=logits, values=categories)
                samples_indices = dist.sample()
                samples = self.categories[samples_indices]
                samples = samples * dimension_mask if dimension_mask is not None else samples
                log_probs = dist.log_prob(samples_indices).sum(dim=-1)
                entropies = dist.entropy().sum(dim=-1)
                return samples, log_probs, entropies
            self.forward = categorical_forward.__get__(self)

