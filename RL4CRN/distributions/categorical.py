import torch
from torch.distributions import Distribution, Categorical, constraints

def _mixed_radix(arities: torch.Tensor) -> torch.Tensor:
    r = torch.ones_like(arities)
    for i in range(len(arities) - 2, -1, -1):
        r[i] = r[i + 1] * arities[i + 1]
    return r

class MultiVariateCategorical(Distribution):
    """
    Joint categorical over M variables (dependent).
    - Either pass `arities=[K1,...,KM]` for categories 0..Ki-1
      or `values=[tensor(K1),...,tensor(KM)]` for explicit numeric categories.
    - Params are a flattened joint over prod(Ki): pass exactly one of logits/probs.
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
    def event_shape(self): return torch.Size([self.M])
    @property
    def batch_shape(self): return self.base.batch_shape
    @property
    def dtype(self): return (self._values[0].dtype if self._values is not None else torch.long)

    # Expose params with the expected names so arg_constraints can see them
    @property
    def probs(self): return self.base.probs
    @property
    def logits(self): return self.base.logits

    # Support: exact checking is tricky with varying Ki/explicit values,
    # so mark as dependent to skip per-sample value checks (shapes still validate).
    @property
    def support(self): return constraints.dependent

    def sample(self, sample_shape=torch.Size()):
        flat = self.base.sample(sample_shape)      # (...,)
        idx = self._decode(flat)                   # (..., M) indices
        return self._indices_to_values(idx) if self._values is not None else idx

    def log_prob(self, value):
        idx = self._values_to_indices(value) if self._values is not None else value.long()
        flat = self._encode(idx)
        return self.base.log_prob(flat)

    # ----- utilities -----
    def joint_table(self):
        shape = (*self.batch_shape, *self.arities.tolist())
        return self.base.probs.reshape(shape)

    def _encode(self, idx):  # (..., M) -> (...,)
        return (idx.long() * self.radix.to(idx.device)).sum(dim=-1)

    def _decode(self, flat):  # (...,) -> (..., M)
        cur = flat.long()
        radix = self.radix.to(cur.device); ar = self.arities.to(cur.device)
        outs = []
        for i in range(self.M):
            outs.append((cur // radix[i]) % ar[i])
        return torch.stack(outs, dim=-1)

    def _values_to_indices(self, x):
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
        outs = []
        for i, v in enumerate(self._values):
            vi = v.to(idx.device)
            outs.append(vi[idx[..., i]])
        return torch.stack(outs, dim=-1)
