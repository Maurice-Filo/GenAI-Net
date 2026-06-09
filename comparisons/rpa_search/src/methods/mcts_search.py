from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from comparisons.rpa_search.src.common.evaluator import (
    build_candidate,
    candidate_summary,
    evaluate_crn,
)
from comparisons.rpa_search.src.common.io import (
    CANDIDATE_FIELDS,
    PROGRESS_FIELDS,
    append_csv,
    write_json,
)


@dataclass
class Node:
    reaction_ids: Tuple[int, ...]
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def run_mcts_search(config: Dict[str, Any], run_dir, method: str, run_id: str, components) -> Dict[str, Any]:
    """Small dependency-free MCTS-like topology search over reaction IDs.

    Parameters are sampled at terminal evaluation. This is intentionally minimal:
    it gives us the same adapter shape a CircuiTree grammar/reward wrapper would
    use without requiring the external package for smoke tests.
    """
    template_crn, library_components, task, _cfg = components
    library = library_components[0]
    search = config["search"]
    rng = np.random.default_rng(int(search.get("seed", 0)))

    iterations = int(search.get("mcts_iterations", search.get("candidate_budget", 100)))
    depth = int(search.get("max_added_reactions", 5))
    exploration = float(search.get("mcts_exploration", 1.4))
    rate_range = search.get("rate_constant_range", [0.1, 50.0])

    zero_id = library.find_zero_reaction()
    action_ids = [int(r.ID) for r in library.reactions if r.ID != zero_id]

    root = Node(())
    best_loss = float("inf")
    best_crn = None
    total_ode = 0
    tic = time.time()

    for i in range(1, iterations + 1):
        leaf, path = _select(root, action_ids, depth, exploration, rng)
        reaction_ids = _complete_rollout(leaf.reaction_ids, action_ids, depth, rng)
        rates = _sample_rates(rng, len(reaction_ids), rate_range)
        crn = build_candidate(template_crn, library, reaction_ids, rates)
        result = evaluate_crn(crn, task)
        total_ode += result.ode_simulations

        value = -result.loss
        for node in path:
            node.visits += 1
            node.value_sum += value

        if result.loss < best_loss:
            best_loss = result.loss
            best_crn = crn.clone()

        progress_row = {
            "method": method,
            "run_id": run_id,
            "step": i,
            "candidate_evaluations": i,
            "ode_simulations": total_ode,
            "loss": result.loss,
            "best_so_far_loss": best_loss,
            "performance": result.performance,
            "best_so_far_performance": -best_loss,
            "elapsed_seconds": time.time() - tic,
        }
        append_csv(run_dir / "progress.csv", progress_row, PROGRESS_FIELDS)

        candidate_row = {
            "method": method,
            "run_id": run_id,
            "candidate_id": i,
            "candidate_evaluations": i,
            "ode_simulations": total_ode,
            "loss": result.loss,
            "best_so_far_loss": best_loss,
            "reaction_ids": list(reaction_ids),
            "rate_constants": rates,
        }
        append_csv(run_dir / "candidates.csv", candidate_row, CANDIDATE_FIELDS)
        if i == 1 or i == iterations or i % 50 == 0:
            print(f"[{method} {run_id}] iterations={i} ode_sims={total_ode} best_loss={best_loss:.6g}", flush=True)

    if best_crn is not None:
        (run_dir / "best_network.txt").write_text(str(best_crn), encoding="utf-8")
        write_json(run_dir / "best_network.json", candidate_summary(best_crn))

    return {"best_loss": best_loss, "ode_simulations": total_ode}


def _select(root: Node, action_ids: List[int], depth: int, exploration: float, rng: Any):
    node = root
    path = [node]
    while len(node.reaction_ids) < depth:
        remaining = [a for a in action_ids if a not in node.reaction_ids]
        unexpanded = [a for a in remaining if a not in node.children]
        if unexpanded:
            action = int(rng.choice(unexpanded))
            child = Node(node.reaction_ids + (action,))
            node.children[action] = child
            path.append(child)
            return child, path

        action, node = _best_ucb_child(node, exploration)
        path.append(node)
    return node, path


def _best_ucb_child(node: Node, exploration: float):
    total = max(1, node.visits)
    best_action = None
    best_child = None
    best_score = -float("inf")
    for action, child in node.children.items():
        if child.visits == 0:
            score = float("inf")
        else:
            score = child.mean_value + exploration * math.sqrt(math.log(total + 1) / child.visits)
        if score > best_score:
            best_action, best_child, best_score = action, child, score
    return best_action, best_child


def _complete_rollout(prefix: Tuple[int, ...], action_ids: List[int], depth: int, rng: Any) -> Tuple[int, ...]:
    chosen = list(prefix)
    remaining = [a for a in action_ids if a not in chosen]
    while len(chosen) < depth and remaining:
        action = int(rng.choice(remaining))
        chosen.append(action)
        remaining.remove(action)
    return tuple(chosen)


def _sample_rates(rng: Any, n: int, rate_range):
    low, high = float(rate_range[0]), float(rate_range[1])
    rates = 10 ** rng.uniform(math.log10(low), math.log10(high), size=n)
    return [float(rate) for rate in rates]
