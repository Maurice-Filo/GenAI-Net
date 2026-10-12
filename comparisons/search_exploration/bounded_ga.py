"""Opt-in rate-bounded wrapper; preserve the original evolutionary operators."""

from dataclasses import replace

from comparisons.llm_seeded_evolution.search import Evolution


class BoundedEvolution(Evolution):
    def __init__(self, *, bounds, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        if not 0 < bounds[0] <= self.rate_range[0] <= self.rate_range[1] <= bounds[1]:
            raise ValueError("Initialization prior must lie within finite positive bounds")
        import math
        if not math.isfinite(bounds[1]):
            raise ValueError("Finite rate bounds required")
        self.clipped_rates = 0

    def propose(self):
        candidate = super().propose()
        rates = tuple(min(self.bounds[1], max(self.bounds[0], value)) for value in candidate.rates)
        self.clipped_rates += sum(a != b for a, b in zip(candidate.rates, rates))
        return replace(candidate, rates=rates)
