import torch
from torch.distributions import Distribution, constraints, MultivariateNormal

class MultivariateLogNormal(Distribution):
    # Parameters live in *log-space*: X ~ N(loc, covariance_matrix), Y = exp(X)
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
    }
    support = constraints.independent(constraints.nonnegative, 1)
    has_rsample = True

    def __init__(self, loc, covariance_matrix, validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.covariance_matrix = torch.as_tensor(covariance_matrix)  # log-space Σ
        self.mvn = MultivariateNormal(self.loc, covariance_matrix=self.covariance_matrix)
        super().__init__(
            batch_shape=self.mvn.batch_shape,
            event_shape=self.mvn.event_shape,
            validate_args=validate_args,
        )

    def sample(self, sample_shape=torch.Size()):
        z = self.mvn.sample(sample_shape)   
        return z.exp()
    

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        x = value.log()
        # log p_Y(y) = log p_X(log y) - sum_i log y_i
        return self.mvn.log_prob(x) - x.sum(dim=-1)

    @property
    def mean(self):
        μ = self.loc
        Σ = self.covariance_matrix
        diagΣ = torch.diagonal(Σ, dim1=-2, dim2=-1)
        return torch.exp(μ + 0.5 * diagΣ)

    @property
    def variance(self):
        # Var(Y_i) = (e^{Σ_ii} - 1) e^{2μ_i + Σ_ii}
        μ = self.loc
        Σ = self.covariance_matrix
        diagΣ = torch.diagonal(Σ, dim1=-2, dim2=-1)
        return (torch.exp(diagΣ) - 1.0) * torch.exp(2 * μ + diagΣ)

    @property
    def real_covariance_matrix(self):
        # Cov(Y_i, Y_j) = exp(μ_i+μ_j + 0.5(Σ_ii+Σ_jj)) * (exp(Σ_ij) - 1)
        μ = self.loc
        Σ = self.covariance_matrix
        diagΣ = torch.diagonal(Σ, dim1=-2, dim2=-1)
        pre = torch.exp(μ[..., :, None] + μ[..., None, :] +
                        0.5 * (diagΣ[..., :, None] + diagΣ[..., None, :]))
        return pre * (torch.exp(Σ) - 1.0)

    def entropy(self):
        # H(Y) = H(X) + sum_i μ_i, with X ~ N(μ, Σ)
        return self.mvn.entropy() + self.loc.sum(-1)
