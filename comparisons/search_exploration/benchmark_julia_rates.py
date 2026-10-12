"""Small deterministic optimizer verification, not a search-method comparison."""

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from RL4CRN.parameter_optimization import Experiment, JuliaOptimizer, ReactionLibrary


def rpa_fixture():
    from RL4CRN.search_exploration.evaluation import build, evaluate_once
    from RL4CRN.utils.input_interface import build_weights
    config = json.loads(Path("comparisons/rpa_search/configs/rpa_cvode.json").read_text())
    config["rpa"].update(rtol=1e-8, atol=1e-8)
    evaluator, _, _ = build("rpa", config, .001, 100.)
    library = ReactionLibrary.from_iocrn(evaluator.crn_template, evaluator.library)
    ids = [18, 51, 7, 13, 80]
    crn = library.select(ids)
    inputs = np.asarray(list(product([.5, 1., 1.5], repeat=2)))
    times = np.linspace(0, 100, 1000)
    out = np.zeros((1, len(crn.species)))
    out[0, crn.species.index("X_3")] = 1
    targets = np.broadcast_to(inputs[:, crn.inputs.index("u_1"), None, None], (9, 1, len(times)))
    experiment = Experiment(times, np.full((9, 3), .01), inputs, out, targets,
                            weights=build_weights(1, len(times), "transient"), max_species=1e4)
    rates = np.array([5.79576795, 11.87837895, 47.32750789, .7406736, 2.34991266])

    def canonical(parameters):
        result = evaluate_once(evaluator, {"reaction_ids": ids, "rates": list(parameters)})
        return {"loss": result["loss"], "outputs": np.asarray(result["task_info"]["outputs"]),
                "simulation_complete": result["simulation_complete"]}

    return crn, experiment, rates, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--project", default=None)
    parser.add_argument("--comet-project", help="Optional metrics/plots; existing configuration supplies credentials")
    parser.add_argument("--render-only", action="store_true", help="Render existing results.json; no optimization")
    args = parser.parse_args()
    if args.render_only:
        render_report(args.output, json.loads((args.output / "results.json").read_text()))
        return
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    logger = None
    try:
        if args.comet_project:
            from comet_ml import Experiment as CometExperiment
            logger = CometExperiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_git_patch=False, log_graph=False,
                log_git_metadata=False, log_env_details=False, log_env_host=False,
                log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
            logger.set_name(args.output.name)
            logger.add_tag("optimizer-software-verification")
        run(args, logger)
    finally:
        if logger is not None:
            logger.end()


def run(args, logger):
    args.output.mkdir(parents=True, exist_ok=False)
    crn, experiment, rates, canonical = rpa_fixture()
    started = time.perf_counter()
    with JuliaOptimizer(project=args.project, startup_timeout=240) as worker:
        startup_s = time.perf_counter() - started
        started = time.perf_counter()
        model = worker.prepare(crn, experiment)
        prepare_s = time.perf_counter() - started
        started = time.perf_counter()
        initial = worker.evaluate(model, rates, trajectories=True)
        cold_s = time.perf_counter() - started
        started = time.perf_counter()
        warm = worker.evaluate(model, rates)
        warm_s = time.perf_counter() - started
        reference = canonical(rates)
        trials = {}
        starts = [rates.tolist(), (rates * [.7, 1.2, .8, 1.3, 1.1]).tolist()]
        for method in ("adagrad", "adam"):
            trials[method] = worker.optimize(model, starts, steps=args.steps,
                                             method=method, learning_rate=.1, timeout=300.)
            (args.output / f"{method}.json").write_text(json.dumps(trials[method], indent=2, allow_nan=False) + "\n")
            for i, result in enumerate(trials[method]["results"]):
                if result.get("best_rates") is not None:
                    verification = canonical(result["best_rates"])
                    result["canonical_best_loss"] = verification["loss"]
                    result["canonical_best_complete"] = verification["simulation_complete"]
                    result["julia_best_peak_species"] = worker.evaluate(model, result["best_rates"])["peak_species"]
                if logger:
                    prefix = f"{method}/start_{i}"
                    for item in result.get("history", []):
                        logger.log_metrics({prefix + "/loss": item["loss"],
                                            prefix + "/best_loss": item["best_loss"]}, step=item["step"])
                    logger.log_other(prefix + "/status", result["status"])
        report = dict(label="Single-topology software verification; not comparative search evidence",
            worker=worker.info, command=worker.command, startup_s=startup_s,
            prepare_s=prepare_s, cold_evaluation_s=cold_s, warm_evaluation_s=warm_s,
            initial_loss=initial["loss"], canonical_initial_loss=reference["loss"],
            max_output_discrepancy=float(np.max(np.abs(np.asarray(initial["outputs"]) - reference["outputs"]))),
            warm_repeat_loss=warm["loss"], model_key=model, crn=crn.payload(),
            experiment=experiment.payload(crn), starts=starts, trials=trials, stats=worker.stats(),
            comet=None if logger is None else {"project": args.comet_project, "experiment_key": logger.get_key()})
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    render_report(args.output, report, logger)


def render_report(output, report, logger=None):
    trials = report["trials"]
    startup_s, prepare_s, cold_s, warm_s = (report[k] for k in (
        "startup_s", "prepare_s", "cold_evaluation_s", "warm_evaluation_s"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for method, color in (("adagrad", "#007f73"), ("adam", "#a63b59")):
        for i, result in enumerate(trials[method]["results"]):
            history = result.get("history", [])
            ax.plot([r["step"] for r in history], [r["best_loss"] for r in history],
                    color=color, linestyle="-" if i == 0 else "--", label=f"{method}, start {i+1}",
                    marker="o" if method == "adagrad" else "s", markerfacecolor="none",
                    markersize=7 if method == "adagrad" else 4, markevery=[0, -1] if history else None)
    ax.set(xlabel="Rate update", ylabel="Best evaluated weighted L1 loss", yscale="log",
           title="RPA rate fitting: one fixed topology, two starts")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output / "rate_fitting.png", dpi=180)
    if logger:
        logger.log_figure(figure_name="RPA fixed-topology rate fitting", figure=fig)
        logger.log_metrics({"startup_s": startup_s, "prepare_s": prepare_s,
                            "cold_evaluation_s": cold_s, "warm_evaluation_s": warm_s,
                            "max_output_discrepancy": report["max_output_discrepancy"]})
        logger.log_parameters({"learning_rate": .1, "bounds": [.001, 100.],
                               "task": "rpa", "starts": 2, "model_key": report["model_key"], **report["worker"]})
    plt.close(fig)
    lines = ["# Julia Rate Optimizer Verification", "", report["label"], "",
        f"Julia startup: {startup_s:.3f}s; prepare: {prepare_s:.3f}s; "
        f"first evaluation: {cold_s:.3f}s; warm evaluation: {warm_s:.3f}s.", "",
        f"Initial Julia / canonical loss: {report['initial_loss']:.10g} / {report['canonical_initial_loss']:.10g}.",
        f"Maximum trajectory discrepancy: {report['max_output_discrepancy']:.3g}.", "",
        "| Method | Start | Status | Initial loss | Best Julia loss | Canonical best loss | Canonical complete | Seconds |",
        "|---|---:|---|---:|---:|---:|---|---:|"]
    for method, trial in trials.items():
        for i, r in enumerate(trial["results"]):
            def number(v):
                return "n/a" if v is None else f"{v:.6g}"
            first = r.get("history", [{}])[0].get("loss") if r.get("history") else None
            lines.append(f"| {method} | {i+1} | {r['status']} | {number(first)} | "
                         f"{number(r.get('best_loss'))} | {number(r.get('canonical_best_loss'))} | "
                         f"{r.get('canonical_best_complete')} | {number(r.get('elapsed_s'))} |")
    lines += ["", "![Rate-fitting traces](rate_fitting.png)", "",
        "Curves end at the last valid update. `evaluation_failed` means the restart "
        "stopped before completing its requested updates; the table separately verifies "
        "the retained best with canonical CVODE. No failed iterate is accepted.", "",
        "AdaGrad is the default by user choice. Two starts on one selected topology cannot "
        "establish superiority of either optimizer. Search integration, throughput scaling, "
        "SSA, and arbitrary task objectives are not tested here."]
    for method, trial in trials.items():
        for i, r in enumerate(trial["results"]):
            if r.get("error"):
                lines += [f"- {method}, start {i+1}: {r['error']} "
                          f"({r['completed_evaluations']} successful evaluations, "
                          f"{r['evaluations']} attempted)."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:-4]))
    if logger:
        logger.log_asset(str(output / "REPORT.md"))
    if any(r["status"] not in ("ok", "evaluation_failed") or not r.get("canonical_best_complete")
           or abs(r["canonical_best_loss"] - r["best_loss"]) > 2e-6
           for trial in trials.values() for r in trial["results"]):
        raise RuntimeError("Best-candidate canonical verification failed; see preserved results.json")


if __name__ == "__main__":
    main()
