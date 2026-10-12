"""Explicit mass-action conventions and serializable experiment definitions."""

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Reaction:
    reactants: tuple[int, ...]
    products: tuple[int, ...]
    input_channel: Optional[int] = None
    fixed_rate: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "reactants", tuple(self.reactants))
        object.__setattr__(self, "products", tuple(self.products))


@dataclass(frozen=True)
class CRNSpec:
    species: tuple[str, ...]
    inputs: tuple[str, ...]
    reactions: tuple[Reaction, ...]

    def __post_init__(self):
        for name in ("species", "inputs", "reactions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.species or not self.reactions:
            raise ValueError("Nonempty species and reactions required")
        for labels in (self.species, self.inputs):
            if len(set(labels)) != len(labels) or any(not isinstance(s, str) or not s for s in labels):
                raise ValueError("Species/input labels must be unique nonempty strings")
        for reaction in self.reactions:
            for counts in (reaction.reactants, reaction.products):
                if len(counts) != len(self.species) or any(
                    not isinstance(v, (int, np.integer)) or v < 0 for v in counts
                ):
                    raise ValueError("Stoichiometry must contain nonnegative integer counts per species")
            u = reaction.input_channel
            if u is not None and (not isinstance(u, int) or not 0 <= u < len(self.inputs)):
                raise ValueError("Invalid input channel")
            k = reaction.fixed_rate
            if k is not None and (not np.isfinite(k) or k < 0):
                raise ValueError("Fixed rates must be finite and nonnegative")

    @property
    def parameter_count(self):
        return sum(r.fixed_rate is None for r in self.reactions)

    def payload(self):
        out = asdict(self)
        for reaction in out["reactions"]:
            for side in ("reactants", "products"):
                reaction[side] = [int(v) for v in reaction[side]]
        return out


class ReactionLibrary:
    """Precompute once; select in caller's rate order without cloning IOCRNs.

    The benchmark adapter excludes any added reaction with the same reactant/
    product pair as a starter reaction, even if its input modulation differs.
    The general CRNSpec does not impose that benchmark-specific restriction.
    """

    def __init__(self, species, inputs, starter, reactions):
        self.species, self.inputs, self.starter = tuple(species), tuple(inputs), tuple(starter)
        self.reactions = MappingProxyType(dict(reactions))
        CRNSpec(self.species, self.inputs, self.starter + tuple(self.reactions.values()))
        if any(r.fixed_rate is None for r in self.starter):
            raise ValueError("Starter reactions must have fixed rates")
        if any(r.fixed_rate is not None for r in self.reactions.values()):
            raise ValueError("Library reactions must have trainable rates")
        fixed = {(r.reactants, r.products) for r in self.starter}
        self.forbidden_ids = frozenset(i for i, r in self.reactions.items()
            if (r.reactants, r.products) in fixed or not any(r.reactants + r.products))

    def select(self, reaction_ids):
        ids = tuple(reaction_ids)
        if not ids or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in ids) or len(set(ids)) != len(ids) or any(
            i not in self.reactions or i in self.forbidden_ids for i in ids
        ):
            raise ValueError("Unknown, duplicate, null, or template-overlapping reaction ID")
        return CRNSpec(self.species, self.inputs, self.starter + tuple(self.reactions[i] for i in ids))

    @classmethod
    def from_iocrn(cls, starter, library):
        from RL4CRN.iocrns.reactions import MassAction
        all_reactions = tuple(starter.reactions) + tuple(library.reactions)
        species = tuple(sorted(set(starter.species_labels).union(
            *(set(r.get_involved_species()) for r in all_reactions))))
        inputs = tuple(starter.input_labels)

        def convert(r, fixed):
            if type(r) is not MassAction:
                raise ValueError("Only plain MassAction reactions are supported")
            channels = [u for u in r.input_channels if u is not None]
            if len(channels) > 1:
                raise ValueError("At most one multiplicative input per reaction")
            return Reaction(tuple(r.reactant_labels.count(s) for s in species),
                            tuple(r.product_labels.count(s) for s in species),
                            inputs.index(channels[0]) if channels else None,
                            float(r.rate_constant) if fixed else None)

        return cls(species, inputs, [convert(r, True) for r in starter.reactions],
                   {i: convert(r, False) for i, r in enumerate(library.reactions)})


@dataclass
class Experiment:
    """Independent constant-input scenarios; outputs are linear species maps.

    Shapes: initial_states[S,N], inputs[S,U], output_matrix[O,N],
    targets[S,O,T], weights broadcastable to [S,O,T]. Loss is the arithmetic
    mean of weighted absolute/squared errors, NOT normalized by weight sum.
    Initial conditions refer to times[0]. No hidden scenario Cartesian product.
    """

    times: object
    initial_states: object
    inputs: object
    output_matrix: object
    targets: object
    weights: object = 1.
    loss: str = "l1"
    relative: bool = False
    solver: str = "cvode"
    rtol: float = 1e-8
    atol: float = 1e-8
    max_steps: int = 100000
    max_species: Optional[float] = None

    def payload(self, crn):
        arrays = {name: np.asarray(getattr(self, name), dtype=float) for name in
                  ("times", "initial_states", "inputs", "output_matrix", "targets", "weights")}
        t, x, u, c = (arrays[n] for n in ("times", "initial_states", "inputs", "output_matrix"))
        if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError("Need at least two strictly increasing times")
        if x.ndim != 2 or x.shape[1] != len(crn.species) or not len(x) or np.any(x < 0):
            raise ValueError("Invalid initial_states [scenario,species]")
        if u.shape != (len(x), len(crn.inputs)) or np.any(u < 0):
            raise ValueError("Invalid inputs [scenario,input]")
        if c.ndim != 2 or c.shape[1] != len(crn.species) or not len(c):
            raise ValueError("Invalid output_matrix [output,species]")
        shape = (len(x), len(c), len(t))
        if arrays["targets"].shape != shape:
            raise ValueError(f"targets must have explicit shape {shape}")
        arrays["weights"] = np.broadcast_to(arrays["weights"], shape)
        if np.any(arrays["weights"] < 0) or not np.any(arrays["weights"] > 0):
            raise ValueError("Weights must be nonnegative with positive total")
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("All experiment values must be finite")
        if self.loss not in ("l1", "l2") or self.solver not in ("cvode", "tsit5"):
            raise ValueError("Supported losses: l1/l2; solvers: cvode/tsit5")
        if any(not np.isfinite(v) or v <= 0 for v in (self.rtol, self.atol)):
            raise ValueError("Tolerances must be positive and finite")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.max_species is not None and (not np.isfinite(self.max_species) or
                self.max_species <= 0 or np.any(x >= self.max_species)):
            raise ValueError("max_species must be positive, finite, and above initial states")
        return {**{k: a.tolist() for k, a in arrays.items()}, "loss": self.loss,
                "relative": bool(self.relative), "solver": self.solver, "rtol": self.rtol,
                "atol": self.atol, "max_steps": self.max_steps,
                "max_species": None if self.max_species is None else float(self.max_species)}
