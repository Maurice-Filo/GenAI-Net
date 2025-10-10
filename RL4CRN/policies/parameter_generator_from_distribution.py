import torch
from RL4CRN.utils.ffnn import FFNN
from RL4CRN.distributions.lognormal import MultivariateLogNormal
from RL4CRN.distributions.categorical import MultiVariateCategorical

class ParameterGeneratorFromDistribution(torch.nn.Module):
    """ A neural network module that generates parameters from specified distributions. 
    Supported distributions: multivariate log-normal, multivariate categorical. 
    The module uses a feedforward neural network (FFNN) as a backbone to output the parameters of the distributions.
    1. For multivariate log-normal distribution, the FFNN outputs the means and covariance matrices of the log-normal distribution.
       The covariance matrices are parameterized via their Cholesky decomposition to ensure positive semi-definiteness.
    2. For multivariate categorical distribution, the FFNN outputs the logits of the categorical distribution.
    The module can also apply masks to the distribution parameters to account for the existence of parameters. """

    def __init__(self, distribution, backbone_attributes, device='cpu'):
        """ Initialize the ParameterGeneratorFromDistribution module.
        Args:
        distribution (dict): A dictionary specifying the type and parameters of the distribution.
                                Supported types: 'lognormal', 'categorical'.
        backbone_attributes (dict): A dictionary specifying the attributes of the backbone network.
                                    Should contain 'input_size', 'hidden_size', and 'num_layers'.
        device (str): The device to run the model on ('cpu' or 'cuda'). """
        
        super().__init__()

        # Initialize the attributes
        self.backbone_attributes = backbone_attributes
        self.device = device
        self.distribution = distribution

        # log-normal distribution
        if distribution["type"] == 'lognormal':

            # Check that the dimension of the distribution is provided
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'lognormal' distribution, 'dim' must be specified in the distribution dictionary.")
            D = int(distribution_dim)
            self.distribution_dim = D

            # Define the backbone network that outputs the parameters of the log-normal distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=D + D + (D**2 - D)//2, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  

            # Define the forward method for multivariate log-normal distribution
            def lognormal_forward(self, x, mask=None, samples=None):
                """ Forward method for generating parameters from a multivariate log-normal distribution.
                Args:
                - x (torch.Tensor): Input tensor of shape (N, input_size).
                - mask (torch.Tensor or None): Optional mask tensor of shape (N, D) indicating the existence of parameters.
                                             If provided, the means and covariance matrices will be masked accordingly.
                - samples (torch.Tensor or None): Optional tensor of shape (N, D) containing samples from the distribution.
                If provided, the log-probabilities will be computed for these samples.
                If None, new samples will be drawn from the distribution.
                Returns:
                - samples (torch.Tensor): Samples drawn from the distribution of shape (N, D).
                - log_probs (torch.Tensor): Log-probabilities of the samples of shape (N,).
                - entropies (torch.Tensor): Entropies of the distributions of shape (N,). """
                
                # Run the backbone to get the unprocessed distribution parameters
                params = self.backbone(x) # shape: (N, D + D + (D**2 - D)//2)

                # Get the batch of means of the log-normal distribution
                means = torch.nn.functional.softplus(params[:, :self.distribution_dim]) # shape: (N, D)

                # Get the batch of covariance matrices of the log-normal distribution using the Cholesky decomposition
                L_diags = torch.nn.functional.softplus(params[:, self.distribution_dim:self.distribution_dim*2]) # shape: (N, D)
                L_off_diags = params[:, self.distribution_dim*2:] # shape: (N, (D**2 - D)//2)
                L = torch.zeros((x.shape[0], self.distribution_dim, self.distribution_dim), device=self.device) # shape: (N, D, D)
                tril_indices = torch.tril_indices(row=self.distribution_dim, col=self.distribution_dim, offset=-1)
                L[:, tril_indices[0], tril_indices[1]] = L_off_diags
                L[:, torch.arange(self.distribution_dim), torch.arange(self.distribution_dim)] = L_diags
                covariance_matrices = torch.matmul(L, L.transpose(-1, -2)) # shape: (N, D, D)

                # Transform distribution parameters from log-space to normal space
                # Formulas: Σ_ij = ln(1 + S_ij / (m_i m_j)), μ_i = ln(m_i) - 0.5 * Σ_ii
                m = means                                                               # shape: (N, D)
                S = covariance_matrices                                                 # shape: (N, D, D)
                outer_mm = m.unsqueeze(-1) * m.unsqueeze(-2)                            # shape: (N, D, D)
                Sigma = torch.log1p(S / outer_mm)                                       # shape: (N, D, D)
                mu = torch.log(m) - 0.5 * torch.diagonal(Sigma, dim1=-1, dim2=-2)

                # Mask the parameters that do not exist for the current reaction
                if mask is not None:
                    mu = mu * mask
                    Sigma = Sigma * mask.unsqueeze(-1) * mask.unsqueeze(-2)

                # Create the multivariate log-normal distributions and compute their entropies
                dist = MultivariateLogNormal(loc=mu, covariance_matrix=Sigma) # batch of N MultivariateLogNormal distributions
                entropies = dist.entropy()  # shape: (N,)

                # Sample from the distributions and compute log-probabilities of the samples
                samples = dist.sample() if samples is None else samples  # shape: (N, D)
                log_probs = dist.log_prob(samples)

                # Return the samples, log-probabilities of the samples, and entropies
                return samples, log_probs, entropies
            
            # Call the forward method for log-normal distribution
            self.forward = lognormal_forward.__get__(self)

        if distribution["type"] == 'categorical': 
            # Check that the dimension of the distribution and the categories are provided
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'dim' must be specified in the distribution dictionary.")
            try:
                categories = distribution["categories"]
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'categories' must be specified in the distribution dictionary.")
            D = int(distribution_dim)
            self.distribution_dim = D

            # Get the total number of categories across all dimensions
            self.num_categories = sum([c.shape[0] for c in categories])
            self.categories = torch.tensor(categories, device=device)

            # Define the backbone network that outputs the logits of the categorical distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=self.num_categories, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  
            
            # Define the forward method for multivariate categorical distribution #TODO: not tested yet
            def categorical_forward(self, x, logit_mask=None, dimension_mask=None, samples=None):

                # Run the backbone to get the logits of the categorical distribution
                logits = self.backbone(x).view(-1, self.distribution_dim, self.num_categories) # shape: (N, D, num_categories)

                # Mask the logits that
                if logit_mask is not None:
                    logits = logits.masked_fill(~logit_mask.bool(), float('-inf'))
                dist = MultiVariateCategorical(logits=logits, values=categories)
                samples_indices = dist.sample() if samples is None else samples
                samples = self.categories[samples_indices]
                samples = samples * dimension_mask if dimension_mask is not None else samples
                log_probs = dist.log_prob(samples_indices).sum(dim=-1)
                entropies = dist.entropy().sum(dim=-1)
                return samples, log_probs, entropies
            self.forward = categorical_forward.__get__(self)

