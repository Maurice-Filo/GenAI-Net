#!/usr/bin/env python3
"""Run the independent baseline against a frozen proposal stream (no model calls)."""

from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "comparisons/llm_seeded_evolution/.mplconfig"))

from comparisons.llm_seeded_evolution import METHOD
from comparisons.llm_seeded_evolution.replay import GENERATED, check_source_mode, export_replay, sha256, source_from_audits
from comparisons.llm_seeded_evolution.search import Candidate, Evolution, ScoredCandidate, run_search
from comparisons.rpa_search.src.common.io import CANDIDATE_FIELDS, PROGRESS_FIELDS, append_csv, write_json

_EVALUATOR = None


def build_evaluator(config: dict):
    import torch
    from RL4CRN.agent2env_interface.iocrn_stepper import IOCRNStepper
    from RL4CRN.agent2env_interface.library_actuator import LibraryActuator
    from RL4CRN.llm.candidate_evaluator import LLMCandidateEvaluator
    from comparisons.rpa_search.src.common.task_factory import build_components

    torch.set_num_threads(1)
    crn, library_components, task, _ = build_components(config)
    library = library_components[0]
    evaluator = LLMCandidateEvaluator(
        crn_template=crn, max_added_reactions=config["search"]["max_added_reactions"],
        library=library, stepper=IOCRNStepper(), actuator=LibraryActuator(library),
        compute_reward_func=task.compute_reward, min_parameter_value=0.0,
        enforce_parameter_bounds=False, forbidden_reaction_ids=[library.find_zero_reaction()],
    )
    task_name = config["benchmark"]["task"]
    # Classifier scenarios are initial conditions, not its single inert input slot.
    evaluator.scenario_count = (sum(len(points) for points, _ in task.params["ic_r_maps"])
                                if task_name == "classifier" else
                                2 ** int(config["logic"]["n_inputs"]) if task_name == "logic" else 9)
    return evaluator


def init_worker(config: dict) -> None:
    global _EVALUATOR
    _EVALUATOR = build_evaluator(config)


def evaluate(candidate: Candidate) -> ScoredCandidate:
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.utils.results_database import serialize_crn

    result = _EVALUATOR.evaluate(LLMCandidate(
        list(candidate.reaction_ids), [[x] for x in candidate.rates]))
    if not result.valid or result.env is None:
        # Generated candidates are valid by construction. Unexpected exceptions
        # must not silently become failed simulator evaluations or fake progress.
        raise RuntimeError(f"Canonical candidate evaluation failed: {result.message}")
    if not math.isfinite(float(result.loss)):
        raise RuntimeError("Canonical evaluator returned a nonfinite loss.")
    crn = serialize_crn(result.env.state)
    return ScoredCandidate(candidate, float(result.loss), crn["topology_hash"], crn["crn_text"])


def validate_replay(replay: dict, evaluate_batch) -> dict:
    members = [c for e in replay["events"] for c in e["candidates"]]
    candidates = [Candidate(tuple(c["reaction_ids"]), tuple(c["rates"]), "llm",
                            proposal_id=c["proposal_id"]) for c in members]
    if not candidates:
        raise ValueError("Replay has no accepted proposals to validate.")
    results = evaluate_batch(candidates)
    if len(results) != len(candidates):
        raise RuntimeError("Replay validation returned an incomplete batch.")
    max_error = 0.0
    for member, result in zip(members, results):
        error = abs(result.loss - member["reference_loss"])
        max_error = max(max_error, error)
        if result.topology_hash != member["topology_hash"] or not math.isclose(
            result.loss, member["reference_loss"], rel_tol=1e-7, abs_tol=1e-10
        ):
            raise ValueError(f"Recorded proposal no longer reproduces: {member['proposal_id']}; "
                             f"reference={member['reference_loss']}, actual={result.loss}")
    return {"status": "pass", "validation_evaluations": len(results),
            "max_absolute_loss_error": max_error}


def prepare(args) -> tuple[dict, dict]:
    if args.replay:
        replay = json.loads(Path(args.replay).read_text())
    else:
        if args.source_mode != "none" and not args.source_run:
            raise ValueError("Insertion replay requires an explicit source run or frozen replay.")
        source = Path(args.source_run) if args.source_run else source_from_audits(args.task, args.seed)
        replay = export_replay(source, source_mode=args.source_mode)
    if replay["schema_version"] != 1 or replay["task"] != args.task or replay["seed"] != args.seed:
        raise ValueError("Replay schema, task, or seed does not match the requested run.")
    check_source_mode(replay, args.source_mode)
    config = deepcopy(replay["scientific_config"])
    settings_path = Path(args.settings)
    settings = json.loads(settings_path.read_text())
    budget = args.search_budget if args.search_budget is not None else replay["search_budget"]
    if not 1 <= budget <= replay["search_budget"]:
        raise ValueError("Search budget must be a positive prefix of the source run budget.")
    settings.update(search_budget=budget, workers=args.workers)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.workers < 1 or settings["batch_size"] < 1:
        raise ValueError("Worker count and evolutionary batch size must be positive.")
    config["evolution"] = settings
    config["search"]["seed"] = args.seed
    config["benchmark"].update(
        name=f"{args.task}_llm_seeded_evolution_replay_v1", output_root=str(args.output_root),
        figure_dir=str(ROOT / "comparisons/llm_seeded_evolution/figures"),
        plot_methods=[METHOD], plot_title="LLM-seeded evolutionary search",
    )
    return replay, config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("logic", "rpa", "classifier"), default="logic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-mode", choices=("none", "llm-to-rl"), default="none")
    parser.add_argument("--comet-project", help="Optional authenticated Comet logging; no credentials on the command line.")
    parser.add_argument("--campaign-id", default="llm-seeded-evolution")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--source-run", help="Completed No communication run; default: audited Qwen (Logic/RPA) or Flash (Classifier).")
    inputs.add_argument("--replay", help="Previously exported replay.json; no source database access needed.")
    parser.add_argument("--settings", default=str(Path(__file__).parent / "configs/default.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, help="Evolutionary offspring batch, independent of worker count.")
    parser.add_argument("--search-budget", type=int, help="Optional prefix for a pilot; full source budget by default.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "comparisons/rpa_search/data/raw")
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true", help="Reevaluate every saved proposal and check loss/hash parity.")
    args = parser.parse_args(argv)
    if args.source_mode != "none" and not args.run_id:
        parser.error("Insertion replay requires a distinct --run-id; existing none runs are never reused.")
    replay, config = prepare(args)
    settings = config["evolution"]
    budget = settings["search_budget"]
    pilot = budget != replay["search_budget"]
    family = "flash" if args.task == "classifier" else "local"
    run_id = args.run_id or f"{args.task}_search{budget}_seed{args.seed}_{family}_none_replay_v1"
    if args.validate_only:
        run_id += "_validation"
    config["benchmark"]["plot_run_ids"] = [run_id]
    out = args.output_root / METHOD / run_id
    # Fail closed rather than mixing a new search with partial old trajectories.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output is not empty: {out}. Choose a new run ID.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "replay.json", replay)
    write_json(out / "config.json", config)
    write_json(out / "run_manifest.json", {
        "method": METHOD, "run_id": run_id, "task": args.task, "seed": args.seed,
        "pilot": pilot, "new_model_calls": 0, "comparison": "conditional_on_recorded_llm_stream",
        "source_database_sha256": replay["source_database_sha256"],
        "replay_sha256": sha256(out / "replay.json"), "config_sha256": sha256(out / "config.json"),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted([
            *Path(__file__).parent.glob("*.py"), *ROOT.glob("RL4CRN/**/*.py"),
            *ROOT.glob("comparisons/rpa_search/src/common/*.py"),
            ROOT / "comparisons/llm_crn_generation/paper_breadth_tasks.py",
        ])},
    })
    init_worker(config)
    if sorted(_EVALUATOR.forbidden_reaction_ids) != replay["proposal_space"]["forbidden_reaction_ids"]:
        raise ValueError("Reconstructed reaction mask disagrees with the source contract.")
    allowed = [int(r.ID) for r in _EVALUATOR.library.reactions
               if r.ID not in _EVALUATOR.forbidden_reaction_ids]
    for event in replay["events"]:
        for member in event["candidates"]:
            if not all(0.001 <= rate <= 100.0 for rate in member["rates"]):
                raise ValueError("Saved direct proposal violates its frozen rate bounds.")
    pool = (ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,),
                                mp_context=multiprocessing.get_context("spawn"))
            if args.workers > 1 else None)

    def evaluate_batch(candidates):
        return list(pool.map(evaluate, candidates) if pool else map(evaluate, candidates))

    started = time.monotonic()
    experiment = None
    try:
        if args.comet_project:
            from comet_ml import Experiment
            experiment = Experiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_graph=False, log_git_patch=False,
                log_env_details=False, display_summary_level=0)
            if not experiment.get_key() or not getattr(experiment, "alive", False):
                raise RuntimeError("Comet logging requested but experiment is not authenticated and live.")
            experiment.set_name(run_id)
            experiment.add_tags(["fixed-proposal", "mutation-only-evolution", args.task, args.source_mode])
            experiment.log_parameters({**settings, "seed": args.seed, "task": args.task,
                "model_proposal_source": replay["model"], "source_mode": args.source_mode,
                "campaign_id": args.campaign_id, "new_model_calls": 0})
            experiment.log_other("replay_sha256", sha256(out / "replay.json"))
            experiment.log_other("status", "running")
            write_json(out / "comet.json", {"experiment_key": experiment.get_key(),
                "project": args.comet_project, "campaign_id": args.campaign_id})
        if args.validate_only:
            result = validate_replay(replay, evaluate_batch)
            result["elapsed_seconds"] = time.monotonic() - started
            write_json(out / "validation.json", result)
            print(json.dumps({"output": str(out), **result}), flush=True)
            return
        evolution = Evolution(
            seed=args.seed, allowed_ids=allowed, n_reactions=config["search"]["max_added_reactions"],
            archive_size=config["rl4crn"]["hall_of_fame_size"],
            rate_range=tuple(settings["initial_rate_range"]),
            rate_probability=settings["rate_mutation_probability"],
            topology_probability=settings["reaction_mutation_probability"],
            log_rate_sigma=settings["log_rate_sigma"],
        )
        plan = json.loads((GENERATED / "analysis_plan_v2.json").read_text())
        threshold, top_k = plan["quality_thresholds"][args.task], plan["portfolio_top_k"]
        references = {c["proposal_id"]: c for e in replay["events"] for c in e["candidates"]}
        operations: dict[str, int] = {}
        last_logged = [-1]
        scenario_count = _EVALUATOR.scenario_count

        def record(result, search_count, llm_count):
            candidate = result.candidate
            if candidate.source == "llm":
                reference = references[candidate.proposal_id]
                if result.topology_hash != reference["topology_hash"] or not math.isclose(
                    result.loss, reference["reference_loss"], rel_tol=1e-7, abs_tol=1e-10
                ):
                    raise ValueError(f"Proposal parity failed: {candidate.proposal_id}")
            operations[candidate.source] = operations.get(candidate.source, 0) + 1
            total = search_count + llm_count
            append_csv(out / "candidates.csv", {
                "method": METHOD, "run_id": run_id, "candidate_id": total,
                "candidate_evaluations": total, "ode_simulations": total,
                "scenario_count": scenario_count, "scenario_evaluations": total * scenario_count,
                "loss": result.loss, "best_so_far_loss": evolution.archive.ranked()[0].loss,
                "reaction_ids": candidate.reaction_ids, "rate_constants": candidate.rates,
                "topology_hash": result.topology_hash, "source": candidate.source,
                "parent_hash": candidate.parent_hash, "proposal_id": candidate.proposal_id,
                "search_evaluations": search_count, "llm_evaluations": llm_count,
            }, [*CANDIDATE_FIELDS, "topology_hash", "source", "parent_hash", "proposal_id",
                "search_evaluations", "llm_evaluations"])

        def checkpoint(search_count, llm_count):
            if not evolution.archive.entries:
                return
            ranked = evolution.archive.ranked()
            total = search_count + llm_count
            append_csv(out / "progress.csv", {
                "method": METHOD, "run_id": run_id, "step": search_count,
                "candidate_evaluations": total, "ode_simulations": total,
                "scenario_count": scenario_count, "scenario_evaluations": total * scenario_count,
                "loss": ranked[0].loss, "best_so_far_loss": ranked[0].loss,
                "performance": -ranked[0].loss, "best_so_far_performance": -ranked[0].loss,
                "elapsed_seconds": time.monotonic() - started,
                "search_evaluations": search_count, "llm_evaluations": llm_count,
                "qualified_topologies": sum(x.loss < threshold for x in ranked[:top_k]),
            }, [*PROGRESS_FIELDS, "search_evaluations", "llm_evaluations", "qualified_topologies"])
            if search_count // 1023 != last_logged[0] or search_count == budget:
                last_logged[0] = search_count // 1023
                print(f"[{run_id}] search={search_count}/{budget} llm={llm_count} "
                      f"best={ranked[0].loss:.6g} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if experiment is not None:
                experiment.log_metrics({"best_loss": ranked[0].loss,
                    "qualified_topologies": sum(x.loss < threshold for x in ranked[:top_k]),
                    "search_evaluations": search_count, "llm_evaluations": llm_count,
                    "candidate_evaluations": total, "epoch_equivalent": search_count / 1023,
                    "elapsed_seconds": time.monotonic() - started}, step=total)
                experiment.log_metrics({f"operations/{k}": v for k, v in operations.items()}, step=total)

        search_count, llm_count = run_search(
            evolution, events=replay["events"], search_budget=budget, batch_size=settings["batch_size"],
            evaluate_batch=evaluate_batch, record=record, checkpoint=checkpoint,
        )
        ranked = evolution.archive.ranked()
        archive = [asdict(x) for x in ranked[:top_k]]
        write_json(out / "terminal_archive.json", archive)
        write_json(out / "best_network.json", archive[0])
        (out / "best_network.txt").write_text(ranked[0].crn_text)
        summary = {
            "status": "complete", "method": METHOD, "run_id": run_id, "task": args.task,
            "seed": args.seed, "pilot": pilot, "search_evaluations": search_count,
            "llm_evaluations": llm_count, "candidate_evaluations": search_count + llm_count,
            "new_model_calls": 0, "best_loss": ranked[0].loss,
            "qualified_topologies": sum(x.loss < threshold for x in ranked[:top_k]),
            "quality_threshold": threshold, "portfolio_top_k": top_k, "operations": operations,
            "elapsed_seconds": time.monotonic() - started,
            "replay_sha256": sha256(out / "replay.json"), "config_sha256": sha256(out / "config.json"),
            "terminal_archive_sha256": sha256(out / "terminal_archive.json"),
        }
        write_json(out / "completed.json", summary)
        if experiment is not None:
            experiment.log_other("status", "complete")
        print(json.dumps({"output": str(out), **summary}), flush=True)
    except Exception as error:
        write_json(out / "failed.json", {"status": "failed", "error_type": type(error).__name__,
            "message": str(error), "elapsed_seconds": time.monotonic() - started})
        if experiment is not None:
            experiment.log_other("status", "failed")
        raise
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
