r"""
Distribution utilities for joint categorical variables.

This module provides:

- `_mixed_radix`, a helper to build mixed-radix multipliers for encoding
  multi-dimensional categorical indices into a single flattened index.
- `MultiVariateCategorical`, a joint categorical distribution over
  $M$ discrete variables represented internally as a single
  `torch.distributions.Categorical` over $\prod_i K_i$ outcomes.

The joint outcome corresponding to per-dimension indices
$\mathbf{i} = (i_0, \dots, i_{M-1})$ is encoded into a flat index:


$$z = \sum_{m=0}^{M-1} i_m \cdot r_m,$$

where $r_m$ are the mixed-radix multipliers:

$$    r_m = \prod_{j=m+1}^{M-1} K_j.$$

Decoding reverses this mapping using integer division and modulo operations.

Notes:
    The distribution can operate in two modes:

    1. **Index mode** via `arities=[K1,...,KM]`: samples are integer indices in
       `0..Ki-1` for each dimension.
    2. **Value mode** via `values=[v1,...,vM]`: samples are explicit numeric
       categories, where each `vm` is a strictly increasing 1D tensor of length
       `Km`. Sampling returns the corresponding values rather than indices.
"""

import torch
from torch.distributions import Distribution, Categorical, constraints

def _mixed_radix(arities: torch.Tensor) -> torch.Tensor:
    r"""Compute mixed-radix multipliers for flattening multi-index categories.

    Given arities $(K_0, \dots, K_{M-1})$, returns a tensor
    $(r_0, \dots, r_{M-1})$ such that a multi-index
    $\mathbf{i} = (i_0, \dots, i_{M-1})$ can be flattened as:

    $$z = \sum_{m=0}^{M-1} i_m \cdot r_m,$$

    with

    $$r_m = \prod_{j=m+1}^{M-1} K_j, \quad r_{M-1} = 1.$$

    Args:
        arities: 1D tensor of positive integers of shape `(M,)`.

    Returns:
        1D tensor of shape `(M,)` containing the mixed-radix multipliers.
    """
    r = torch.ones_like(arities)
    for i in range(len(arities) - 2, -1, -1):
        r[i] = r[i + 1] * arities[i + 1]
    return r

class MultiVariateCategorical(Distribution):
    """Joint categorical distribution over multiple discrete variables.

    This distribution represents a joint categorical distribution over
    $M$ discrete variables by flattening the joint support of size
    $\prod_{m=0}^{M-1} K_m$ into a single categorical random variable.

    Two ways to define the per-dimension categories:

        1. **Index mode**: pass `arities=[K1, ..., KM]`. Each variable takes
           values in `{0, ..., K_m-1}` and samples are returned as integer
           indices of shape `(..., M)`.
        2. **Value mode**: pass `values=[v1, ..., vM]`, where each `vm` is a
           strictly increasing 1D tensor of length `K_m`. Samples are returned
           as values from these tensors (shape `(..., M)`).

    Parameters:
        Exactly one of `logits` or `probs` must be provided. They parameterize
        the flattened joint categorical distribution and must have last dimension
        size $\prod_m K_m$.

    Shapes:
        - Batch shape is inherited from `logits`/`probs` (everything except last dim).
        - Event shape is `(M,)`.

    Validation:
        - `probs` must be a simplex along the last dimension.
        - `logits` can be any real vector along the last dimension.
        - `support` is marked as `constraints.dependent` because exact support
          checking is non-trivial when using explicit `values`. Shape validation
          still applies.

    Examples:
        Index mode:

        >>> dist = MultiVariateCategorical(arities=[2, 3], logits=torch.zeros(6))
        >>> x = dist.sample((5,))  # shape (5, 2), entries in [0..1] and [0..2]

        Value mode:

        >>> vals0 = torch.tensor([10, 20])
        >>> vals1 = torch.tensor([0.1, 0.2, 0.3])
        >>> dist = MultiVariateCategorical(values=[vals0, vals1], probs=torch.ones(6)/6)
        >>> x = dist.sample()  # shape (2,), values from vals0 and vals1
    """
    has_rsample = False

    # Tell PyTorch how to validate parameters:
    # - probs must be simplex along the last dim
    # - logits can be any real vector
    arg_constraints = {
        "probs": constraints.simplex,
        "logits": constraints.real_vector,
    }

    def __init__(self, *, arities=None, values=None, logits=None, probs=None, validate_args=None):
        """Create a multivariate joint categorical distribution.

        Args:
            arities: 1D sequence/tensor `(K1,...,KM)` specifying the number of
                categories per dimension. Used in index mode. Mutually exclusive
                with `values`.
            values: Optional list of 1D tensors specifying explicit categories
                per dimension. Each tensor must be non-empty and strictly
                increasing. Mutually exclusive with `arities`.
            logits: Logits for the flattened joint categorical distribution.
                Must have last dimension size `prod(arities)`. Mutually exclusive
                with `probs`.
            probs: Probabilities for the flattened joint categorical distribution.
                Must have last dimension size `prod(arities)` and be a simplex.
                Mutually exclusive with `logits`.
            validate_args: Passed to `torch.distributions.Distribution`
                to enable/disable argument validation.

        Raises:
            AssertionError: If not exactly one of `logits`/`probs` is provided, or
                if `arities`/`values` are missing/invalid, or if the last dimension
                of `logits`/`probs` does not match the joint cardinality.
            ValueError: If `values` tensors are not 1D, empty, or not strictly
                increasing.
        """

        assert (logits is None) ^ (probs is None), "Pass exactly one of logits or probs."

        if values is not None:
            self._values = [torch.as_tensor(v) for v in values]
            for i, v in enumerate(self._values):
                if v.ndim != 1 or v.numel() == 0 or not torch.all(v[1:] > v[:-1]):
                    raise ValueError(f"values[{i}] must be 1D, nonempty, strictly increasing.")
            self.arities = torch.tensor([v.numel() for v in self._values], dtype=torch.long)
        else:
            assert arities is not None, "Provide arities or values."
            self._values = None
            self.arities = torch.as_tensor(arities, dtype=torch.long)
            assert self.arities.ndim == 1 and torch.all(self.arities > 0)

        self.M = int(self.arities.numel())
        self.radix = _mixed_radix(self.arities)
        Ktot = int(self.arities.prod().item())

        if logits is not None:
            assert logits.shape[-1] == Ktot
            self.base = Categorical(logits=logits)
        else:
            assert probs.shape[-1] == Ktot
            self.base = Categorical(probs=probs)

        # Initialize Distribution (sets up validation machinery)
        super().__init__(self.base.batch_shape, torch.Size([self.M]), validate_args=validate_args)
        if self._validate_args:
            self._validate_args  # validates probs/logits using arg_constraints

    # ----- Distribution API -----
    @property
    def event_shape(self):
        """Shape of a single draw from the distribution (always `(M,)`)."""
        return torch.Size([self.M])
    @property
    def batch_shape(self):
        """Batch shape of the distribution (inherited from base categorical)."""
        return self.base.batch_shape
    @property
    def dtype(self):
        """Data type of samples (values dtype in value mode, else `torch.long`)."""
        return (self._values[0].dtype if self._values is not None else torch.long)

    # Expose params with the expected names so arg_constraints can see them
    @property
    def probs(self):
        """Probabilities of the flattened joint distribution (shape `(..., Ktot)`)."""
        return self.base.probs
    @property
    def logits(self):
        """Logits of the flattened joint distribution (shape `(..., Ktot)`)."""
        return self.base.logits

    # Support: exact checking is tricky with varying Ki/explicit values,
    # so mark as dependent to skip per-sample value checks (shapes still validate).
    @property
    def support(self):
        """Support constraint.

        Marked as `constraints.dependent` because exact element-wise support checks
        are non-trivial when each dimension can have explicit value sets.
        """
        return constraints.dependent

    def sample(self, sample_shape=torch.Size()):
        """Draw samples.

        Args:
            sample_shape: Optional leading sample shape.

        Returns:
            Samples of shape `sample_shape + batch_shape + (M,)`.

                - In index mode (`arities` provided): integer indices in `[0, K_m-1]`.
                - In value mode (`values` provided): values from the provided category
                tensors per dimension.
        """
        flat = self.base.sample(sample_shape)      # (...,)
        idx = self._decode(flat)                   # (..., M) indices
        return self._indices_to_values(idx) if self._values is not None else idx

    def log_prob(self, value):
        """Compute log-probability of a batch of samples.

        Args:
            value: Samples of shape `(..., M)`. In index mode these should be
                integer indices; in value mode they should match the explicit
                category values.

        Returns:
            Tensor of log-probabilities with shape `value.shape[:-1]`.

        Raises:
            ValueError: In value mode, if any entry is not contained in the
                corresponding category set.
        """
        idx = self._values_to_indices(value) if self._values is not None else value.long()
        flat = self._encode(idx)
        return self.base.log_prob(flat)

    # ----- utilities -----
    def joint_table(self):
        """Return the joint probability table reshaped to per-dimension axes.

        Returns:
            Tensor of shape `batch_shape + (K1, ..., KM)` representing the full
            joint probability table.
        """
        shape = (*self.batch_shape, *self.arities.tolist())
        return self.base.probs.reshape(shape)

    def _encode(self, idx):  # (..., M) -> (...,)
        """Flatten per-dimension indices into a single categorical index.

        Args:
            idx: Tensor of indices with shape `(..., M)`.

        Returns:
            Tensor of flattened indices with shape `(...)`.
        """
        return (idx.long() * self.radix.to(idx.device)).sum(dim=-1)

    def _decode(self, flat):  # (...,) -> (..., M)
        """Decode flattened indices into per-dimension indices.

        Args:
            flat: Tensor of flattened indices with shape `(...)`.

        Returns:
            Tensor of per-dimension indices with shape `(..., M)`.
        """
        cur = flat.long()
        radix = self.radix.to(cur.device); ar = self.arities.to(cur.device)
        outs = []
        for i in range(self.M):
            outs.append((cur // radix[i]) % ar[i])
        return torch.stack(outs, dim=-1)

    def _values_to_indices(self, x):
        """Convert explicit category values into per-dimension indices.

        Args:
            x: Tensor of shape `(..., M)` containing explicit category values.

        Returns:
            Long tensor of indices with shape `(..., M)`.

        Raises:
            ValueError: If any value does not belong to the category set of a
                dimension.
        """
        idxs = []
        for i, v in enumerate(self._values):
            vi = v.to(x.device); xi = x[..., i]
            j = torch.searchsorted(vi, xi)
            ok = (j >= 0) & (j < vi.numel()) & (vi[j] == xi)
            if not torch.all(ok):
                raise ValueError(f"value not in categories for dim {i}")
            idxs.append(j)
        return torch.stack(idxs, dim=-1).long()

    def _indices_to_values(self, idx):
        """Convert per-dimension indices into explicit category values.

        Args:
            idx: Long tensor of indices with shape `(..., M)`.

        Returns:
            Tensor of explicit category values with shape `(..., M)`.
        """
        outs = []
        for i, v in enumerate(self._values):
            vi = v.to(idx.device)
            outs.append(vi[idx[..., i]])
        return torch.stack(outs, dim=-1)
