"""Canonical/Julia Logic-5 software check; no LLM calls or search results reused."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from comparisons.search_exploration.freeze_legacy import ROOT, digest, verify
from comparisons.search_exploration.logic5_four_arm_campaign import configuration, PROMPT, REFERENCE


def run(output):
    from RL4CRN.parameter_optimization import JuliaOptimizer
    from RL4CRN.search_exploration.evaluation import build, evaluate_once
    from RL4CRN.search_streams.julia_adapter import task_experiment
    from RL4CRN.search_streams.julia_stream import LOSS_AGREEMENT
    verify()
    reference = json.loads((REFERENCE / "manifest.json").read_text())
    config, _ = configuration(reference, 1)
    evaluator, task, _ = build("logic", config, .001, 100.)
    if evaluator.max_added_reactions != 5 or evaluator.forbidden_reaction_ids != frozenset(range(5)):
        raise ValueError("Unexpected Logic proposal contract")
    library, experiment = task_experiment(evaluator, task)
    ids = [146, 246, 326, 401, 101]
    starts = [[3., 3., 3., .45, .1], [2.4, 2.4, 2.4, .36, .08]]
    rows = []
    began = time.monotonic()
    with JuliaOptimizer(project=reference["julia_project"], startup_timeout=600) as worker:
        model = worker.prepare(library.select(ids), experiment)
        for rates in starts:
            direct = evaluate_once(evaluator, {"reaction_ids": ids, "rates": rates})
            result = worker.evaluate(model, rates, timeout=None)
            if not direct["simulation_complete"] or not np.isclose(
                result["loss"], direct["loss"], rtol=LOSS_AGREEMENT["warning_rtol"],
                atol=LOSS_AGREEMENT["warning_atol"]
            ):
                raise ValueError("Canonical/Julia preflight mismatch")
            rows.append({"rates": rates, "canonical_loss": direct["loss"], "julia_loss": result["loss"]})
        fitted = worker.optimize(model, [starts[1]], steps=2, method="adagrad", learning_rate=.1,
            bounds=(.001, 100.), step_policy="feasible", evaluation_limit=4, timeout=None)["results"][0]
        if fitted.get("best_rates") is None:
            raise ValueError("AdaGrad preflight returned no valid rates")
        direct = evaluate_once(evaluator, {"reaction_ids": ids, "rates": fitted["best_rates"]})
        if not direct["simulation_complete"] or not np.isclose(
            direct["loss"], fitted["best_loss"], rtol=LOSS_AGREEMENT["warning_rtol"],
            atol=LOSS_AGREEMENT["warning_atol"]
        ):
            raise ValueError("Fitted canonical/Julia loss mismatch")
    result = {"status": "passed", "task": "logic", "prompt_sha256": digest(PROMPT),
        "reaction_ids": ids, "checks": rows, "fit_status": fitted["status"],
        "fitted_rates": fitted["best_rates"], "fitted_canonical_loss": direct["loss"],
        "fitted_julia_loss": fitted["best_loss"], "fit_evaluations": fitted["evaluations"],
        "elapsed_seconds": time.monotonic() - began, "new_model_calls": 0,
        "note": "Known historical candidate used only for software verification, not injected into any campaign seed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
