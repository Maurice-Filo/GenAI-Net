"""Explicit pilot settings, independent of legacy TrainCfg defaults."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ExplorationConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 32
    evaluation_budget: int = 10000
    fit_evaluations: int = 16
    fit_seconds: float | None = 60.
    evaluation_seconds: float | None = 30.
    workers: int = 4
    optimizer: str = "evolution"
    rate_min: float = .001
    rate_max: float = 100.
    initial_rate: float = 1.
    log_rate_sigma: float = .5
    archive_capacity: int = 2000
    sil_top_k: int = 30
    sil_batch_size: int = 16
    sil_region_balanced: bool = True
    compatibility_hof_size: int = 30
    quality_threshold: float = .05
    require_all_rows: bool = False
    descriptor: str = "reaction_roles_v1"
    novelty_weight: float = .1
    quality_improvement_weight: float = .25
    rl_fraction: float = .5
    mutation_fraction: float = .25
    structural_refinement_steps: int = 0
    learning_rate: float = 1e-4
    imitation_weight: float = .1
    entropy_weight: float = .001
    gradient_clip: float = 1.
    plot_every: int = 5

    def __post_init__(self):
        ints = (self.epochs, self.batch_size, self.evaluation_budget, self.fit_evaluations,
                self.workers, self.archive_capacity, self.sil_top_k, self.sil_batch_size,
                self.compatibility_hof_size, self.plot_every)
        if any(type(x) is not int or x < 1 for x in ints):
            raise ValueError("Counts must be positive integers")
        if not 0 < self.rate_min <= self.initial_rate <= self.rate_max < math.inf:
            raise ValueError("Invalid rate bounds/initialization")
        if not (0 <= self.rl_fraction <= 1 and 0 <= self.mutation_fraction <= 1
                and self.rl_fraction + self.mutation_fraction < 1):
            raise ValueError("A nonzero random source share is required")
        if self.sil_top_k > self.archive_capacity or self.structural_refinement_steps < 0:
            raise ValueError("Invalid archive/refinement limits")
        for value in (self.fit_seconds, self.evaluation_seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Deadlines must be positive finite seconds or None")
        for value in (self.quality_threshold,
                      self.log_rate_sigma, self.learning_rate, self.gradient_clip):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Scales and deadlines must be positive and finite")
        for value in (self.novelty_weight, self.quality_improvement_weight, self.imitation_weight, self.entropy_weight):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Weights must be finite and nonnegative")
        if self.optimizer not in ("evolution", "powell"):
            raise ValueError("Unsupported optimizer")
        if self.descriptor not in ("reaction_roles_v1", "topology"):
            raise ValueError("Unknown descriptor")

    def to_dict(self):
        return asdict(self)
