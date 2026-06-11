from __future__ import annotations

import math
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Sequence


@dataclass
class EvaluationResult:
    loss: float
    task_info: Dict[str, Any]
    ode_simulations: int
    scenario_count: int

    @property
    def performance(self) -> float:
        return -self.loss


def build_candidate(
    template_crn: Any,
    library: Any,
    reaction_ids: Sequence[int],
    rate_constants: Sequence[float],
) -> Any:
    """Build a candidate IOCRN by adding library reactions to the RPA template."""
    if len(reaction_ids) != len(rate_constants):
        raise ValueError("reaction_ids and rate_constants must have the same length.")

    crn = template_crn.clone()
    for rid, rate in zip(reaction_ids, rate_constants):
        reaction = library.get_reaction(int(rid))
        if reaction is None:
            raise ValueError(f"Unknown reaction ID: {rid}")
        reaction.set_parameters([float(rate)])
        crn.add_reaction(reaction)
    return crn


def evaluate_crn(crn: Any, task: Any) -> EvaluationResult:
    """Evaluate a candidate using the shared RPA task."""
    crn.reset()
    with redirect_stdout(StringIO()):
        loss, task_info = task.compute_reward(crn)
    loss = float(loss)
    if not math.isfinite(loss):
        loss = float("inf")
    return EvaluationResult(
        loss=loss,
        task_info=task_info,
        ode_simulations=1,
        scenario_count=len(task.u_list),
    )


def candidate_summary(crn: Any) -> Dict[str, Any]:
    """Return a compact JSON-serializable summary of a candidate CRN."""
    return {
        "inputs": list(getattr(crn, "input_labels", [])),
        "species": list(getattr(crn, "species_labels", [])),
        "output_labels": list(getattr(crn, "output_labels", [])),
        "reaction_ids": [int(r.ID) if r.ID is not None else None for r in crn.reactions],
        "rate_constants": [
            float(getattr(r, "rate_constant", r.params[0]))
            for r in crn.reactions
            if getattr(r, "params", None)
        ],
        "reactions": [str(r) for r in crn.reactions],
    }


def sampled_reaction_ids_and_rates(rng: Any, library: Any, n_reactions: int, rate_range: Sequence[float]):
    """Sample reaction IDs and log-uniform-ish rate constants."""
    low, high = float(rate_range[0]), float(rate_range[1])
    if low <= 0 or high <= 0 or high < low:
        raise ValueError("rate_constant_range must be positive and ordered.")

    zero_id = library.find_zero_reaction()
    valid_ids = [r.ID for r in library.reactions if r.ID != zero_id]
    reaction_ids = rng.choice(valid_ids, size=int(n_reactions), replace=False).tolist()
    rates = 10 ** rng.uniform(math.log10(low), math.log10(high), size=int(n_reactions))
    return [int(rid) for rid in reaction_ids], [float(rate) for rate in rates]
