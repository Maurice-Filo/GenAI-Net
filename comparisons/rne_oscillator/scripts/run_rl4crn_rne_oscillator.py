#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "comparisons/rne_oscillator/.mplconfig").resolve()))

from comparisons.rpa_search.src.common.evaluator import candidate_summary
from comparisons.rpa_search.src.common.io import append_csv, write_json
from RL4CRN.utils.crn_builders import build_simple_IOCRN
from RL4CRN.utils.default_tasks.RNEOscillatorTraceTaskKind import (
    RNE_OSCILLATOR_TARGET,
    RNE_OSCILLATOR_TIME,
)
from RL4CRN.utils.input_interface import Configurator, make_session_and_trainer, make_task
from RL4CRN.utils.library_builders import build_MAK_library


DEFAULT_METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_p20"
METHOD = os.environ.get("RL4CRN_RNE_METHOD", DEFAULT_METHOD)
DEFAULT_OUTPUT_ROOT = ROOT / "comparisons/rne_oscillator/data/raw"
PROGRESS_FIELDS = [
    "method",
    "run_id",
    "seed",
    "epoch",
    "candidate_evaluations",
    "batch_size",
    "n_cpus",
    "epoch_best_loss",
    "epoch_median_loss",
    "saved_best_loss",
    "success",
    "stop_reason",
    "elapsed_seconds",
]


def _write_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _save_best(run_dir: Path, best_crn: Any, best_loss: float, task: Any) -> None:
    best_loss, best_info = task.compute_reward(best_crn)
    summary = candidate_summary(best_crn)
    summary["loss"] = float(best_loss)
    summary["best_species_label"] = best_info.get("best_species_label")
    summary["best_species_index"] = best_info.get("best_species_index")

    (run_dir / "best_network.txt").write_text(str(best_crn), encoding="utf-8")
    write_json(run_dir / "best_network.json", summary)
    with (run_dir / "best_network.pkl").open("wb") as f:
        cloudpickle.dump(best_crn, f)


def build_components(
    seed: int,
    batch_size: int,
    n_cpus: int,
    epochs: int,
    render_every: int,
    *,
    policy_depth: int,
    deep_layer_size: int,
    risk: float,
    max_risk: float,
    risk_schedule: int,
    risk_update: float,
    entropy_weight: float,
    entropy_schedule: int,
    entropy_update_coefficient: float,
    minimum_entropy_weight: float,
    structure_entropy_weight: float,
    continuous_entropy_weight: float,
):
    cfg = Configurator.preset("paper")
    cfg.solver.algorithm = "CVODE"
    cfg.solver.rtol = 1e-3
    cfg.solver.atol = 1e-6

    species_labels = ["X_1", "X_2", "X_3"]
    crn, species_labels = build_simple_IOCRN(
        species=species_labels,
        production_input_map={},
        output_species=species_labels,
        solver=cfg.solver,
    )
    library_components = build_MAK_library(crn, species_labels, order=2)
    library, _M, _K, _masks = library_components

    task = make_task(
        template_crn=crn,
        library_components=library_components,
        kind="rne_oscillator_trace",
        species_labels=species_labels,
        params={
            "time_horizon": RNE_OSCILLATOR_TIME,
            "target_trace": RNE_OSCILLATOR_TARGET,
            "ic": ("values", [[1.0, 5.0, 9.0]]),
            "u_list": [np.asarray([], dtype=np.float32)],
            "LARGE_NUMBER": 1e4,
            "LARGE_PENALTY": 1e4,
            "normalize": False,
        },
    )

    cfg.train.max_added_reactions = 8
    cfg.train.epochs = int(epochs)
    cfg.train.batch_size = int(batch_size)
    cfg.train.batch_multiplier = 10
    cfg.train.render_every = 50
    cfg.train.hall_of_fame_size = 50
    cfg.train.seed = int(seed)
    cfg.train.n_cpus = int(n_cpus)
    cfg.policy.depth = int(policy_depth)
    cfg.policy.deep_layer_size = int(deep_layer_size)
    cfg.agent.learning_rate = 1e-4
    cfg.agent.risk_scheduler["risk"] = float(risk)
    cfg.agent.risk_scheduler["max_risk"] = float(max_risk)
    cfg.agent.risk_scheduler["risk_schedule"] = int(risk_schedule)
    cfg.agent.risk_scheduler["risk_update"] = float(risk_update)
    cfg.agent.entropy_scheduler["entropy_weight"] = float(entropy_weight)
    cfg.agent.entropy_scheduler["entropy_schedule"] = int(entropy_schedule)
    cfg.agent.entropy_scheduler["entropy_update_coefficient"] = float(entropy_update_coefficient)
    cfg.agent.entropy_scheduler["minimum_entropy_weight"] = float(minimum_entropy_weight)
    cfg.policy.entropy_weights_per_head["structure"] = float(structure_entropy_weight)
    cfg.policy.entropy_weights_per_head["continuous"] = float(continuous_entropy_weight)
    cfg.policy.zero_reaction_idx = library.find_zero_reaction()
    cfg.policy.stop_flag = True
    cfg.render.n_best = 20
    cfg.render.disregarded_percentage = 0.9
    cfg.render.mode = {
        "style": "logger",
        "task": "transients",
        "format": "image",
        "topology": True,
    }
    return task, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--success-threshold", type=float, default=20.0)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--policy-depth", type=int, default=5)
    parser.add_argument("--deep-layer-size", type=int, default=256)
    parser.add_argument("--risk", type=float, default=0.85)
    parser.add_argument("--max-risk", type=float, default=0.9)
    parser.add_argument("--risk-schedule", type=int, default=20)
    parser.add_argument("--risk-update", type=float, default=0.005)
    parser.add_argument("--entropy-weight", type=float, default=5e-3)
    parser.add_argument("--entropy-schedule", type=int, default=1000)
    parser.add_argument("--entropy-update-coefficient", type=float, default=1.0)
    parser.add_argument("--minimum-entropy-weight", type=float, default=0.0)
    parser.add_argument("--structure-entropy-weight", type=float, default=4.0)
    parser.add_argument("--continuous-entropy-weight", type=float, default=1.0)
    args = parser.parse_args()
    method = str(args.method)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            np.long
        except AttributeError:
            np.long = np.int_

    run_id = args.run_id or f"seed{args.seed:03d}"
    run_dir = Path(args.output_root) / method / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".running"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        print(f"[skip] active lock exists for {run_id}: {lock_path}", flush=True)
        return

    tic = time.time()
    best_loss = float("inf")
    success = False
    stop_reason = "max_epochs"
    task = None
    trainer = None
    try:
        task, cfg = build_components(
            seed=args.seed,
            batch_size=args.batch_size,
            n_cpus=args.n_cpus,
            epochs=args.epochs,
            render_every=args.render_every,
            policy_depth=args.policy_depth,
            deep_layer_size=args.deep_layer_size,
            risk=args.risk,
            max_risk=args.max_risk,
            risk_schedule=args.risk_schedule,
            risk_update=args.risk_update,
            entropy_weight=args.entropy_weight,
            entropy_schedule=args.entropy_schedule,
            entropy_update_coefficient=args.entropy_update_coefficient,
            minimum_entropy_weight=args.minimum_entropy_weight,
            structure_entropy_weight=args.structure_entropy_weight,
            continuous_entropy_weight=args.continuous_entropy_weight,
        )
        _write_config(
            run_dir / "config.json",
            {
                "method": method,
                "run_id": run_id,
                "seed": args.seed,
                "success_threshold": args.success_threshold,
                "task": "rne_oscillator_trace",
                "train": cfg.train.__dict__,
                "solver": cfg.solver.__dict__,
                "policy": cfg.policy.__dict__,
                "agent": cfg.agent.__dict__,
                "target_time": RNE_OSCILLATOR_TIME.astype(float).tolist(),
                "target_trace": RNE_OSCILLATOR_TARGET.astype(float).tolist(),
            },
        )

        trainer = make_session_and_trainer(cfg, task, logger=None)
        trainer.s.agent.policy.train()
        checkpoint_path = run_dir / "checkpoint.pkl"

        for epoch in range(1, args.epochs + 1):
            epoch_best, epoch_median, rewards = trainer.step_epoch()
            candidate_evaluations = epoch * len(rewards)

            best_crn = trainer.best_crn()
            if best_crn is not None:
                candidate_loss, _candidate_info = task.compute_reward(best_crn)
                candidate_loss = float(candidate_loss)
                if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                    best_loss = candidate_loss
                    _save_best(run_dir, best_crn, best_loss, task)

            success = best_loss < float(args.success_threshold)
            stop_reason = "success" if success else "max_epochs"
            append_csv(
                run_dir / "progress.csv",
                {
                    "method": method,
                    "run_id": run_id,
                    "seed": args.seed,
                    "epoch": epoch,
                    "candidate_evaluations": candidate_evaluations,
                    "batch_size": len(rewards),
                    "n_cpus": args.n_cpus,
                    "epoch_best_loss": float(epoch_best),
                    "epoch_median_loss": float(epoch_median),
                    "saved_best_loss": best_loss,
                    "success": success,
                    "stop_reason": stop_reason,
                    "elapsed_seconds": time.time() - tic,
                },
                PROGRESS_FIELDS,
            )
            print(
                f"[{method} {run_id}] epoch={epoch} "
                f"saved_best_loss={best_loss:.6g} success={success}",
                flush=True,
            )

            if success:
                break
            if args.checkpoint_every and epoch % int(args.checkpoint_every) == 0:
                trainer.save(str(checkpoint_path))

        if trainer is not None and args.checkpoint_every:
            trainer.save(str(checkpoint_path))

        if not math.isfinite(best_loss) and trainer is not None:
            best_crn = trainer.best_crn()
            if best_crn is not None:
                best_loss, _ = task.compute_reward(best_crn)
                _save_best(run_dir, best_crn, float(best_loss), task)

        _write_config(
            run_dir / "result.json",
            {
                "method": method,
                "run_id": run_id,
                "seed": args.seed,
                "success": bool(success),
                "success_threshold": float(args.success_threshold),
                "best_loss": float(best_loss),
                "stop_reason": stop_reason,
                "elapsed_seconds": time.time() - tic,
            },
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
