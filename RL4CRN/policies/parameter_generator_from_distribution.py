"""
RL4CRN.policies.parameter_generator_from_distribution

Neural parameter generator used by RL policies to sample reaction parameters from
learned probability distributions.

This module defines `ParameterGeneratorFromDistribution`, a small wrapper around a
feed-forward backbone (``FFNN``) that maps a conditioning embedding (e.g., an encoded IOCRN
state and/or a reaction choice) to the parameters of a distribution, then provides:

- sampling of parameter vectors (continuous and/or discrete),
- log-probability evaluation of provided samples (for policy-gradient objectives),
- entropy computation (for exploration/regularization).

Supported families include several LogNormal variants (1D, independent multivariate, and a
correlated multivariate LogNormal via an exponentiated MultivariateNormal), a correlated
MultivariateNormal, and multivariate categorical distributions. Optional masks allow a single
fixed-size generator to handle reactions with different effective parameter dimensionalities
and to invalidate impossible categorical choices.
"""

import torch
from RL4CRN.utils.ffnn import FFNN
from RL4CRN.distributions.categorical import MultiVariateCategorical
from RL4CRN.distributions.lognormal import MultivariateLogNormal
from torch.distributions import MultivariateNormal, LogNormal
from torch.distributions import TransformedDistribution, transforms

class ParameterGeneratorFromDistribution(torch.nn.Module):
    """
    Neural module that parameterizes and samples *reaction-parameter vectors* from a chosen
    probability distribution, conditioned on a learned embedding.

    This class is used by policies to generate continuous and/or discrete reaction parameters.
    A small feed-forward backbone (``FFNN``) maps an input embedding ``x`` (typically the encoded
    IOCRN state concatenated with a one-hot reaction choice) to the parameters of a distribution.
    The module then:

    1. constructs the corresponding distribution object,
    2. samples parameters (or evaluates the log-probability of provided samples),
    3. returns samples, summed log-probabilities, and summed entropies per batch element.

    Supported distributions (selected via ``distribution["type"]``):

    Continuous:
        - ``"lognormal_1D"``:
            One-dimensional LogNormal with learned mean/std (in *mean/std* space, converted to
            underlying Normal parameters ``mu, sigma``).
        - ``"lognormal_independent"``:
            Factorized (independent) LogNormal across ``D`` dimensions. The backbone outputs
            per-dimension mean/std (in mean/std space), which are converted to ``mu, sigma`` of
            the underlying Normal.
        - ``"lognormal_processed"``:
            Correlated multivariate LogNormal built as an exponentiated MultivariateNormal.
            The backbone outputs a bounded mean vector ``mu`` and a Cholesky factor ``L`` for the
            covariance in log-space, enabling correlations and ensuring PSD covariance.
            Supports *variable active dimensionality per batch element* via an optional mask.
        - ``"multivariate_normal"``:
            Correlated multivariate Normal with learned mean and Cholesky factor ``L``.
            (Mask currently accepted but not yet used to reduce dimensionality.)

    Discrete:
        - ``"categorical"``:
                Multivariate categorical distribution over a fixed set of categories (shared across
                dimensions). Implemented via ``MultiVariateCategorical``. Supports:

            * ``logit_mask`` to invalidate some category combinations,
            * ``dimension_mask`` to zero-out unused discrete dimensions.

    Masking semantics:
        Masks allow parameter vectors of different effective sizes to coexist in a fixed-size tensor.
        Depending on the distribution:

        - For categorical: ``dimension_mask`` zeros out inactive dimensions; ``logit_mask`` can set
            invalid logits to ``-inf``.
        - For processed lognormal: ``mask`` (shape ``(N, D)``) indicates the number of active
            dimensions per batch element; the implementation groups samples by active dimension count
            and builds smaller distributions for efficiency and numerical stability.

    Inputs-outputs:
        The module's ``forward`` signature depends on the chosen distribution type, but the returned
        values are consistent:
        - ``samples``: tensor of sampled parameters (shape ``(N, D)`` or ``(N, 1)`` for 1D),
        - ``log_probs``: tensor of total log-probabilities per batch element (shape ``(N,)``),
        - ``entropies``: tensor of total entropies per batch element (shape ``(N,)``).

    Notes:
        - The constructor dynamically assigns ``self.forward`` to a distribution-specific implementation.
        This keeps call-sites uniform while allowing different argument sets (e.g., ``mask`` vs
        ``logit_mask``/``dimension_mask``).
        - For LogNormal parameterization in 1D/independent modes, the backbone outputs positive
        mean/std via ``softplus`` and converts them to underlying Normal parameters:
            sigma = sqrt(log(1 + std^2 / mean^2)),  mu = log(mean) - 0.5*sigma^2
        which ensures a valid LogNormal with the requested mean/std.
    """

    def __init__(self, distribution, backbone_attributes, device='cpu'):
        """
        Construct a parameter generator for a specified distribution family.

        Args:
            distribution : dict
                Distribution specification. Must include:

                - ``"type"``: one of
                    ``{"lognormal_1D", "lognormal_independent", "lognormal_processed",
                    "multivariate_normal", "categorical"}``.
                Additional required keys depend on the type:

                - ``"dim"`` (int): required for all multivariate types and categorical.
                - ``"categories"`` (torch.Tensor): required for ``"categorical"``; 1D tensor of
                    category values shared across dimensions.

                Optional keys for ``"lognormal_processed"``:

                - ``squash`` (float): scaling before tanh for mean bounding.
                - ``mu_max`` (float): max absolute value for bounded log-space mean.
                - ``sigma_min`` (float): lower bound added to Cholesky diagonal.
                - ``sigma_max`` (float or None): optional upper bound on Cholesky diagonal.
                - ``off_scale`` (float or None): if set, off-diagonals are bounded by
                    ``off_scale * tanh(off_raw)``.

            backbone_attributes : dict
                FFNN configuration with keys:

                - ``"input_size"`` (int): embedding dimension expected by the generator,
                - ``"hidden_size"`` (int): FFNN hidden width,
                - ``"num_layers"`` (int): number of FFNN layers.
                The backbone output size is determined by the distribution type (e.g., ``2*D`` for
                independent lognormal, ``D + D + D(D-1)/2`` for Cholesky-parameterized covariances).

        device : str or torch.device, default="cpu"
            Device where parameters, backbone, and intermediate tensors are allocated.
        """
        
        super().__init__()

        # Initialize the attributes
        self.backbone_attributes = backbone_attributes
        self.device = device
        self.distribution = distribution

        # --------------------- 1D log-normal distribution ---------------------
        if distribution["type"] == 'lognormal_1D':

            # Define the backbone network that outputs the parameters of the log-normal distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=2, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  

            # Define the forward method for multivariate log-normal distribution
            def lognormal1D_forward(self, x, mask=None, samples=None):
                """ Forward method for generating parameters from a 1D log-normal distribution.
                Args:
                - x (torch.Tensor): Input tensor of shape (N, input_size).
                - mask (torch.Tensor or None): Not used for 1D distribution.
                - samples (torch.Tensor or None): Optional tensor of shape (N,) containing samples from the distribution.
                If provided, the log-probabilities will be computed for these samples.
                If None, new samples will be drawn from the distribution.
                Returns:
                - samples (torch.Tensor): Samples drawn from the distribution of shape (N,).
                - log_probs (torch.Tensor): Log-probabilities of the samples of shape (N,).
                - entropies (torch.Tensor): Entropies of the distributions of shape (N,). """
                
                # Run the backbone to get the unprocessed distribution parameters
                params = self.backbone(x) # shape: (N, 2)

                # Get the batch of means and standard deviation of the log-normal distribution
                means = torch.nn.functional.softplus(params[:, 0])      # shape: (N,)
                stds = torch.nn.functional.softplus(params[:, 1])       # shape: (N,)

                # Transform distribution parameters from log-space to normal space
                sigma = torch.sqrt(torch.log1p(stds**2 / (means**2)))   # shape: (N,)
                mu = torch.log(means) - 0.5 * sigma**2                  # shape: (N,)

                # Construct the 1D lognormal distributions and compute their entropies
                dist = LogNormal(loc=mu.squeeze(-1), scale=sigma.squeeze(-1))           # batch of N LogNormal distributions
                entropies = dist.entropy()  # shape: (N,)

                # Sample from the distributions and compute log-probabilities of the samples
                if samples is None:
                    samples = dist.sample().unsqueeze(-1) # shape: (N, 1)
                log_probs = dist.log_prob(samples.squeeze(-1))

                # Return the samples, log-probabilities of the samples, and entropies
                return samples, log_probs, entropies
            
            # Call the forward method for log-normal distribution
            self.forward = lognormal1D_forward.__get__(self)

        # --------------------- independent log-normal distribution ---------------------
        if distribution["type"] == 'lognormal_independent':

            # Check that the dimension of the distribution is provided
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'lognormal_independent' distribution, 'dim' must be specified in the distribution dictionary.")
            D = int(distribution_dim)
            self.distribution_dim = D

            # Define the backbone network that outputs the parameters of the log-normal distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=2*D, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  

            # Define the forward method for multivariate but independent log-normal distribution
            def lognormal_independent_forward(self, x, mask=None, samples=None):
                """ Forward method for generating parameters from an independent log-normal distribution.
                Args:
                    - x (torch.Tensor): Input tensor of shape (N, input_size).
                    - mask (torch.Tensor or None): Not used for 1D distribution.
                    - samples (torch.Tensor or None): Optional tensor of shape (N,) containing samples from the distribution.
                If provided, the log-probabilities will be computed for these samples.
                If None, new samples will be drawn from the distribution.
                Returns:
                    - samples (torch.Tensor): Samples drawn from the distribution of shape (N,).
                    - log_probs (torch.Tensor): Log-probabilities of the samples of shape (N,).
                    - entropies (torch.Tensor): Entropies of the distributions of shape (N,). """
                
                # Run the backbone to get the unprocessed distribution parameters
                params = self.backbone(x) # shape: (N, 2*D)

                # Get the batch of means and standard deviation of the log-normal distribution
                means = torch.nn.functional.softplus(params[:, 0:D])    # shape: (N,)
                stds = torch.nn.functional.softplus(params[:, D:2*D])   # shape: (N,)

                # Transform distribution parameters from log-space to normal space
                sigma = torch.sqrt(torch.log1p(stds**2 / (means**2)))   # shape: (N,)
                mu = torch.log(means) - 0.5 * sigma**2                  # shape: (N,)

                # Construct the independent lognormal distributions and compute their entropies
                dist = LogNormal(loc=mu, scale=sigma)                   # batch of N LogNormal distributions
                entropies = dist.entropy()                              # shape: (N,D)

                # Sample from the indpendent distributions and compute log-probabilities of the samples
                if samples is None:
                    samples = dist.sample()                             # shape: (N, D)

                eps = 1e-6
                safe_samples = torch.clamp(samples.clone(), min=eps)

                log_probs = dist.log_prob(safe_samples.squeeze(-1))          # shape: (N,D)

                # Sum the log-probabilities and entropies across dimensions
                log_probs = log_probs.sum(dim=-1)                       # shape: (N,)
                entropies = entropies.sum(dim=-1)                       # shape: (N,)

                # Return the samples, log-probabilities of the samples, and entropies
                return samples, log_probs, entropies
            
            # Call the forward method for log-normal distribution
            self.forward = lognormal_independent_forward.__get__(self)

        # --------------------- log-normal (processed) distribution ---------------------
        if distribution["type"] == 'lognormal_processed':

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
            def lognormal_processed_forward(self, x, mask=None, samples=None):
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

                # Flag to indicate if samples have been provided
                samples_provided = True if samples is not None else False
                
                # Run the backbone to get the unprocessed distribution parameters
                params = self.backbone(x) # shape: (N, D + D + (D**2 - D)//2)

                # Get the bounding parameters for mu and Sigma, the dimension D, and the batch size N
                squash = self.distribution.get('squash', 1.0)
                mu_max = self.distribution.get('mu_max', 5.0)
                sigma_min = self.distribution.get('sigma_min', 1e-5)
                sigma_max = self.distribution.get('sigma_max', None)
                off_scale= self.distribution.get('off_scale', None)
                D = self.distribution_dim
                N = x.shape[0]

                # Pre-split the parameters into mu, sigma, and off-diagonal entries
                mu_all = params[:, :D]                                 # (N, D)
                sigma_all = params[:, D:2*D]                           # (N, D)  -> diag(L)
                off_all = params[:, 2*D:]                              # (N, D(D-1)/2) -> strictly lower(L)

                # Collect sub-batches with the same active dimensions
                D_batch = mask.sum(dim=1).long() if mask is not None else torch.full((N,), D, dtype=torch.long, device=self.device)
                unique_D, inverse_indices = torch.unique(D_batch, return_inverse=True)

                # Loop over each sub-batch with the same active dimensions
                samples = samples if samples_provided else torch.zeros((N, D), device=self.device, dtype=params.dtype)  # shape: (N, D)
                log_probs = torch.zeros((N,), device=self.device, dtype=params.dtype)                                   # shape: (N,)
                entropies = torch.zeros((N,), device=self.device, dtype=params.dtype)                                   # shape: (N,)

                for i,d in zip(range(len(unique_D)), unique_D):
                    
                    # Get the indices of the sub-batch with the same active dimensions d
                    idx_i = (inverse_indices == i).nonzero(as_tuple=True)[0]                                            # shape: (subbatch size,)
                    N_i = idx_i.shape[0]

                    # Get the masks for the sub-batch
                    if mask is not None:
                        mask_i = mask[idx_i, :]                                                                         # shape: (subbatch size, D)
                    else:
                        mask_i = torch.ones((N_i, D), device=self.device, dtype=params.dtype)                                # shape: (subbatch size, D)
                    
                    # Slice per-sub-batch heads (prefix first d dims)
                    mu_raw = mu_all[idx_i, :d]
                    sigma_raw = sigma_all[idx_i, :d]                                                                    # diagonal logits for L
                    off_raw = off_all[idx_i, :(d*(d-1))//2]                                                             # strictly-lower entries

                    # Bound mu
                    mu = mu_max * torch.tanh(mu_raw * squash)                                                           # shape: (subbatch size, d)

                    # Build Cholesky factor L directly from (diag, strictly-lower)
                    L = torch.zeros(N_i, d, d, device=self.device, dtype=params.dtype)                                  # shape: (subbatch size, d, d)
                    ar = torch.arange(d, device=self.device)

                    # Diagonal: positive via softplus (+ sigma_min); optional cap
                    diag = torch.nn.functional.softplus(sigma_raw) + sigma_min
                    if sigma_max is not None:
                        diag = torch.clamp(diag, max=sigma_max)
                    L[:, ar, ar] = diag

                    # Strictly-lower: bounded
                    tril_i, tril_j = torch.tril_indices(d, d, offset=-1, device=self.device)
                    if off_scale is None:
                        L[:, tril_i, tril_j] = off_raw          
                    else:                                           
                        L[:, tril_i, tril_j] = off_scale * torch.tanh(off_raw)             
                                    
                    # Create the multivariate log-normal distributions and compute their entropies
                    base = MultivariateNormal(loc=mu, scale_tril=L)                                                     # subbatch of Multi-variate Normal distributions
                    dist = TransformedDistribution(base, [transforms.ExpTransform()])                                   # subbatch of Multi-variate Log-Normal distributions
                    entropies[idx_i] = base.entropy() + mu.sum(-1)                                                      # shape: (subbatch size,)

                    # Sample from the distributions (if samples are not provided) and compute log-probabilities of the samples
                    if not samples_provided:
                        samples_i = dist.sample()                                                                       # shape: (subbatch size, d)
                        samples[idx_i, :d] = samples_i                                                                  # store the samples in the full batch
                    else:
                        samples_i = samples[idx_i, :d]                                                                  # shape: (subbatch size, d)

                    log_probs[idx_i] = dist.log_prob(samples_i)                                                         # store log-probs in the full batch
                                                
                # Return the samples, log-probabilities of the samples, and entropies
                return samples, log_probs, entropies
            
            # Call the forward method for log-normal distribution
            self.forward = lognormal_processed_forward.__get__(self)

        # --------------------- multivariate normal distribution ---------------------
        if distribution["type"] == 'multivariate_normal':

            # Check that the dimension of the distribution is provided
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'multivariate_normal' distribution, 'dim' must be specified in the distribution dictionary.")
            D = int(distribution_dim)
            self.distribution_dim = D

            # Define the backbone network that outputs the parameters of the log-normal distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=D + D + (D**2 - D)//2, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  

            # Define the forward method for multivariate log-normal distribution TODO: masks not used yet
            def multivariate_normal_forward(self, x, mask=None, samples=None):
                """ Forward method for generating parameters from a multivariate normal distribution.
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

                # Flag to indicate if samples have been provided
                samples_provided = True if samples is not None else False
                
                # Run the backbone to get the unprocessed distribution parameters
                params = self.backbone(x) # shape: (N, D + D + (D**2 - D)//2)

                # Get the dimension D, and the batch size N
                D = self.distribution_dim
                N = x.shape[0]

                # Pre-split the parameters into mu, sigma, and off-diagonal entries
                mu = params[:, :D]                                      # (N, D)
                sigma = params[:, D:2*D]                                # (N, D)  -> diag(L)
                off = params[:, 2*D:]                                   # (N, D(D-1)/2) -> strictly lower(L)

                # Loop over each sub-batch with the same active dimensions
                samples = samples if samples_provided else torch.zeros((N, D), device=self.device, dtype=params.dtype)  # shape: (N, D)
                log_probs = torch.zeros((N,), device=self.device, dtype=params.dtype)                                   # shape: (N,)
                entropies = torch.zeros((N,), device=self.device, dtype=params.dtype)                                   # shape: (N,)

                # Build Cholesky factor L directly from (diag, strictly-lower)
                L = torch.zeros(N, D, D, device=self.device, dtype=params.dtype)                                        # shape: (N, D, D)
                ar = torch.arange(D, device=self.device)

                # Diagonal: positive via softplus
                diag = torch.nn.functional.softplus(sigma)
                L[:, ar, ar] = diag

                # Strictly-lower
                tril_i, tril_j = torch.tril_indices(D, D, offset=-1, device=self.device)
                L[:, tril_i, tril_j] = off                  
                                    
                # Create the multivariate normal distributions and compute their entropies
                dist = MultivariateNormal(loc=mu, scale_tril=L)                                                         # batch of Multi-variate Normal distributions
                entropies = dist.entropy()                                                                              # shape: (N,)

                # Sample from the distributions (if samples are not provided) and compute log-probabilities of the samples
                if not samples_provided:
                    samples = dist.sample()                                                                             # shape: (N, D)
                log_probs= dist.log_prob(samples)                                                                       # store log-probs in the full batch
                                                
                # Return the samples, log-probabilities of the samples, and entropies
                return samples, log_probs, entropies
            
            # Call the forward method for log-normal distribution
            self.forward = multivariate_normal_forward.__get__(self)

        # --------------------- categorical distribution ---------------------
        if distribution["type"] == 'categorical': 
            # Check that the dimension of the distribution and the categories are provided
            try:
                distribution_dim = distribution["dim"]
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'dim' must be specified in the distribution dictionary.")
            try:
                categories = distribution["categories"] # 1D tensor, TODO: generalize to different categories per dimension
            except KeyError:
                raise ValueError("For 'categorical' distribution, 'categories' must be specified in the distribution dictionary.")
            D = int(distribution_dim)
            self.distribution_dim = D

            # Get the total number of categories across all dimensions
            self.num_categories = categories.shape[0]
            self.categories = categories.to(device=device)

            # Define the backbone network that outputs the logits of the categorical distribution
            self.backbone = FFNN(input_size=backbone_attributes["input_size"], output_size=self.num_categories * self.distribution_dim, hidden_size=backbone_attributes["hidden_size"], num_layers=backbone_attributes["num_layers"]).to(device=device)  
            
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