from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from circuitree import CircuitGrammar, CircuiTree
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


class ReactionIdGrammar(CircuitGrammar):
    """CircuiTree grammar whose states are ordered reaction-ID selections."""

    def __init__(self, action_ids: Iterable[int], depth: int):
        super().__init__()
        self.action_ids = tuple(int(a) for a in action_ids)
        self.depth = int(depth)

    def get_actions(self, state: str):
        if self.is_terminal(state):
            return []
        selected = set(_parse_state(state))
        return [a for a in self.action_ids if a not in selected]

    def do_action(self, state: str, action: int) -> str:
        selected = list(_parse_state(state))
        selected.append(int(action))
        return _format_state(selected)

    def is_terminal(self, state: str) -> bool:
        return len(_parse_state(state)) >= self.depth

    def get_unique_state(self, state: str) -> str:
        # Keep order: different reaction-addition orders can have different tree paths.
        return state


class RpaCircuitTree(CircuiTree):
    def __init__(self, *args, config, run_dir, method, run_id, components, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.run_dir = run_dir
        self.method = method
        self.run_id = run_id
        self.template_crn, self.library_components, self.task, _cfg = components
        self.library = self.library_components[0]
        self.rng = np.random.default_rng(int(config["search"].get("seed", 0)))
        self.rate_range = config["search"].get("rate_constant_range", [0.1, 50.0])
        self.candidate_evaluations = 0
        self.ode_simulations = 0
        self.scenario_evaluations = 0
        self.best_loss = float("inf")
        self.best_crn = None
        self.tic = time.time()

    def get_reward(self, state) -> float:
        reaction_ids = _parse_state(state)
        rates = _sample_rates(self.rng, len(reaction_ids), self.rate_range)
        crn = build_candidate(self.template_crn, self.library, reaction_ids, rates)
        result = evaluate_crn(crn, self.task)

        self.candidate_evaluations += 1
        self.ode_simulations += result.ode_simulations
        self.scenario_evaluations += result.scenario_count
        if result.loss < self.best_loss:
            self.best_loss = result.loss
            self.best_crn = crn.clone()

        progress_row = {
            "method": self.method,
            "run_id": self.run_id,
            "step": self.candidate_evaluations,
            "candidate_evaluations": self.candidate_evaluations,
            "ode_simulations": self.ode_simulations,
            "scenario_count": result.scenario_count,
            "scenario_evaluations": self.scenario_evaluations,
            "loss": result.loss,
            "best_so_far_loss": self.best_loss,
            "performance": result.performance,
            "best_so_far_performance": -self.best_loss,
            "elapsed_seconds": time.time() - self.tic,
        }
        append_csv(self.run_dir / "progress.csv", progress_row, PROGRESS_FIELDS)

        candidate_row = {
            "method": self.method,
            "run_id": self.run_id,
            "candidate_id": self.candidate_evaluations,
            "candidate_evaluations": self.candidate_evaluations,
            "ode_simulations": self.ode_simulations,
            "scenario_count": result.scenario_count,
            "scenario_evaluations": self.scenario_evaluations,
            "loss": result.loss,
            "best_so_far_loss": self.best_loss,
            "reaction_ids": list(reaction_ids),
            "rate_constants": rates,
        }
        append_csv(self.run_dir / "candidates.csv", candidate_row, CANDIDATE_FIELDS)

        n_steps = int(self.config["search"].get("mcts_iterations", self.config["search"].get("candidate_budget", 100)))
        if self.candidate_evaluations == 1 or self.candidate_evaluations == n_steps or self.candidate_evaluations % 50 == 0:
            print(
                f"[{self.method} {self.run_id}] iterations={self.candidate_evaluations} "
                f"ode_sims={self.ode_simulations} best_loss={self.best_loss:.6g}",
                flush=True,
            )
        return -result.loss


def run_circuitree(config: Dict[str, Any], run_dir, method: str, run_id: str, components) -> Dict[str, Any]:
    workers = int(config["search"].get("circuitree_workers", 1))
    if workers > 1:
        return _run_circuitree_multistart(config, run_dir, method, run_id, workers)

    template_crn, library_components, _task, _cfg = components
    library = library_components[0]
    search = config["search"]

    zero_id = library.find_zero_reaction()
    action_ids = [int(r.ID) for r in library.reactions if r.ID != zero_id]
    grammar = ReactionIdGrammar(action_ids, depth=int(search.get("max_added_reactions", 5)))

    tree = RpaCircuitTree(
        grammar=grammar,
        root="",
        exploration_constant=float(search.get("mcts_exploration", 1.4)),
        seed=int(search.get("seed", 0)),
        config=config,
        run_dir=run_dir,
        method=method,
        run_id=run_id,
        components=components,
        compute_unique=False,
    )
    n_steps = int(search.get("mcts_iterations", search.get("candidate_budget", 100)))
    tree.search_mcts(n_steps=n_steps, progress_bar=False)

    if tree.best_crn is not None:
        (run_dir / "best_network.txt").write_text(str(tree.best_crn), encoding="utf-8")
        write_json(run_dir / "best_network.json", candidate_summary(tree.best_crn))

    return {"best_loss": tree.best_loss, "ode_simulations": tree.ode_simulations}


def _run_circuitree_multistart(config: Dict[str, Any], run_dir, method: str, run_id: str, workers: int) -> Dict[str, Any]:
    n_steps = int(config["search"].get("mcts_iterations", config["search"].get("candidate_budget", 100)))
    base = n_steps // workers
    remainder = n_steps % workers
    shard_args = []
    shard_root = Path(run_dir) / "_shards"
    for shard_id in range(workers):
        shard_steps = base + (1 if shard_id < remainder else 0)
        if shard_steps <= 0:
            continue
        shard_args.append((config, str(shard_root), method, run_id, shard_id, shard_steps))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        shard_dirs = list(executor.map(_run_circuitree_shard, shard_args))

    rows = []
    for shard_dir in shard_dirs:
        shard_id = int(Path(shard_dir).name.replace("shard_", ""))
        with (Path(shard_dir) / "candidates.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                local_step = int(row["candidate_evaluations"])
                row["_global_step"] = (local_step - 1) * workers + shard_id + 1
                rows.append(row)
    rows.sort(key=lambda row: int(row["_global_step"]))

    best_loss = float("inf")
    best_row = None
    tic = time.time()
    for global_step, row in enumerate(rows, start=1):
        loss = float(row["loss"])
        if loss < best_loss:
            best_loss = loss
            best_row = row
        progress_row = {
            "method": method,
            "run_id": run_id,
            "step": global_step,
            "candidate_evaluations": global_step,
            "ode_simulations": global_step,
            "scenario_count": row["scenario_count"],
            "scenario_evaluations": int(row["scenario_count"]) * global_step,
            "loss": loss,
            "best_so_far_loss": best_loss,
            "performance": -loss,
            "best_so_far_performance": -best_loss,
            "elapsed_seconds": time.time() - tic,
        }
        append_csv(Path(run_dir) / "progress.csv", progress_row, PROGRESS_FIELDS)
        candidate_row = {
            "method": method,
            "run_id": run_id,
            "candidate_id": global_step,
            "candidate_evaluations": global_step,
            "ode_simulations": global_step,
            "scenario_count": row["scenario_count"],
            "scenario_evaluations": int(row["scenario_count"]) * global_step,
            "loss": loss,
            "best_so_far_loss": best_loss,
            "reaction_ids": row["reaction_ids"],
            "rate_constants": row["rate_constants"],
        }
        append_csv(Path(run_dir) / "candidates.csv", candidate_row, CANDIDATE_FIELDS)
        if global_step == 1 or global_step == n_steps or global_step % 50 == 0:
            print(f"[{method} {run_id}] iterations={global_step} ode_sims={global_step} best_loss={best_loss:.6g}", flush=True)

    if best_row is not None:
        import json

        components = _build_components_with_seed(config, int(config["search"].get("seed", 0)))
        template_crn, library_components, _task, _cfg = components
        library = library_components[0]
        reaction_ids = json.loads(best_row["reaction_ids"])
        rates = json.loads(best_row["rate_constants"])
        best_crn = build_candidate(template_crn, library, reaction_ids, rates)
        (Path(run_dir) / "best_network.txt").write_text(str(best_crn), encoding="utf-8")
        write_json(Path(run_dir) / "best_network.json", candidate_summary(best_crn))

    return {"best_loss": best_loss, "ode_simulations": len(rows)}


def _run_circuitree_shard(args) -> str:
    config, shard_root, method, run_id, shard_id, n_steps = args
    shard_config = dict(config)
    shard_config["search"] = dict(config["search"])
    shard_config["search"]["seed"] = int(config["search"].get("seed", 0)) + shard_id
    shard_config["search"]["mcts_iterations"] = int(n_steps)
    shard_config["search"]["candidate_budget"] = int(n_steps)
    shard_config["search"]["circuitree_workers"] = 1
    components = _build_components_with_seed(shard_config, int(shard_config["search"].get("seed", 0)))
    shard_dir = Path(shard_root) / f"shard_{shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_circuitree(shard_config, shard_dir, method, f"{run_id}_shard{shard_id:03d}", components)
    return str(shard_dir)


def _build_components_with_seed(config: Dict[str, Any], seed: int):
    from comparisons.rpa_search.src.common.task_factory import build_components

    seeded = dict(config)
    seeded["search"] = dict(config["search"])
    seeded["search"]["seed"] = int(seed)
    return build_components(seeded)


def _parse_state(state: str) -> Tuple[int, ...]:
    if not state:
        return ()
    return tuple(int(part) for part in state.split(",") if part)


def _format_state(reaction_ids) -> str:
    return ",".join(str(int(rid)) for rid in reaction_ids)


def _sample_rates(rng: Any, n: int, rate_range):
    low, high = float(rate_range[0]), float(rate_range[1])
    rates = 10 ** rng.uniform(np.log10(low), np.log10(high), size=n)
    return [float(rate) for rate in rates]
