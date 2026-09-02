from __future__ import annotations

import time
import warnings
from typing import Any, Dict

import numpy as np

from comparisons.rpa_search.src.common.evaluator import candidate_summary
from comparisons.rpa_search.src.common.io import PROGRESS_FIELDS, append_csv, write_json
from RL4CRN.utils.input_interface import make_session_and_trainer


def run_rl4crn(
    config: Dict[str, Any],
    run_dir,
    method: str,
    run_id: str,
    components,
    logger=None,
) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            np.long
        except AttributeError:
            np.long = np.int_

    _template_crn, _library_components, task, cfg = components
    search = config["search"]
    rl = config["rl4crn"]

    cfg.train.seed = int(search.get("seed", 0))
    cfg.train.max_added_reactions = int(search.get("max_added_reactions", 5))
    cfg.train.epochs = int(rl.get("epochs", 20))
    cfg.train.batch_size = int(rl.get("batch_size", 16))
    cfg.train.n_cpus = int(rl.get("n_cpus", 1))
    cfg.train.render_every = int(rl.get("render_every", 0))
    cfg.train.hall_of_fame_size = int(rl.get("hall_of_fame_size", 30))
    cfg.agent.risk_scheduler["risk"] = min(
        float(cfg.agent.risk_scheduler.get("risk", 0.95)),
        max(0.0, 1.0 - 1.0 / max(1, cfg.train.batch_size)),
    )
    cfg.policy.width = int(rl.get("policy_width", cfg.policy.width))
    cfg.policy.depth = int(rl.get("policy_depth", cfg.policy.depth))
    cfg.policy.deep_layer_size = int(rl.get("deep_layer_size", cfg.policy.deep_layer_size))

    trainer = make_session_and_trainer(cfg, task, logger=logger)

    best_loss = float("inf")
    total_ode = 0
    total_scenarios = 0
    scenario_count = len(task.u_list)
    tic = time.time()

    for step in range(1, cfg.train.epochs + 1):
        best, _median, rewards = trainer.step_epoch()
        total_ode += len(rewards)
        total_scenarios += len(rewards) * scenario_count
        best_loss = min(best_loss, float(best))

        progress_row = {
            "method": method,
            "run_id": run_id,
            "step": step,
            "candidate_evaluations": step * len(rewards),
            "ode_simulations": total_ode,
            "scenario_count": scenario_count,
            "scenario_evaluations": total_scenarios,
            "loss": float(best),
            "best_so_far_loss": best_loss,
            "performance": -float(best),
            "best_so_far_performance": -best_loss,
            "elapsed_seconds": time.time() - tic,
        }
        append_csv(run_dir / "progress.csv", progress_row, PROGRESS_FIELDS)
        print(f"[{method} {run_id}] epoch={step} candidates={step * len(rewards)} ode_sims={total_ode} best_loss={best_loss:.6g}", flush=True)

    best_crn = trainer.best_crn()
    if best_crn is not None:
        (run_dir / "best_network.txt").write_text(str(best_crn), encoding="utf-8")
        write_json(run_dir / "best_network.json", candidate_summary(best_crn))

    history_path = run_dir / "trainer_history.csv"
    for row in trainer.state.history:
        append_csv(history_path, row, ["epoch", "best", "median"])

    return {"best_loss": best_loss, "ode_simulations": total_ode}
