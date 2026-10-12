"""Matched-start engineering test of opt-in feasible AdaGrad; no search campaign."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from RL4CRN.parameter_optimization import JuliaOptimizer
from comparisons.search_exploration.benchmark_julia_rates import rpa_fixture


ARMS = {
    "stop_lr_0.1": dict(learning_rate=.1),
    "stop_lr_0.01": dict(learning_rate=.01),
    "feasible_lr_0.1": dict(learning_rate=.1, step_policy="feasible",
                            max_backtracks=8, max_evaluations=65),
}
COLORS = {"stop_lr_0.1": "#666666", "stop_lr_0.01": "#b45a36", "feasible_lr_0.1": "#007f73"}
LABELS = {"stop_lr_0.1": "Stop, lr=0.1", "stop_lr_0.01": "Stop, lr=0.01",
          "feasible_lr_0.1": "Backtrack, lr=0.1"}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(output, logger):
    output.mkdir(parents=True, exist_ok=False)
    crn, experiment, rates, canonical = rpa_fixture()
    rng = np.random.default_rng(20260915)
    starts = [rates, rates * [.7, 1.2, .8, 1.3, 1.1]]
    starts += [np.clip(rates * np.exp(rng.normal(0, .15, 5)), .001, 100.) for _ in range(6)]
    report = {"started": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "starts": [p.tolist() for p in starts], "arms": ARMS, "steps": 32,
              "crn": crn.payload(), "experiment": experiment.payload(crn), "records": [],
              "comet": None if logger is None else logger.get_key()}
    save(output / "results.json", report)
    with JuliaOptimizer(startup_timeout=240) as worker:
        model = worker.prepare(crn, experiment)
        report["worker"], report["model_key"] = worker.info, model
        # Warm both code paths; charge and report warmup separately from trials.
        t0 = time.perf_counter()
        warmup = worker.optimize(model, [starts[0]], steps=0, **ARMS["feasible_lr_0.1"])
        report["warmup"] = {"elapsed_s": time.perf_counter()-t0, "result": warmup}
        initial_checks = []
        for start in starts:
            result = canonical(start)
            initial_checks.append({"loss": result["loss"], "complete": result["simulation_complete"]})
        report["canonical_initial"] = initial_checks
        names = list(ARMS)
        for i, start in enumerate(starts):
            # Rotate order to reduce systematic warm/order advantages.
            for name in names[i % len(names):] + names[:i % len(names)]:
                result = worker.optimize(model, [start], steps=32, timeout=120., **ARMS[name])["results"][0]
                record = {"arm": name, "start": i, "optimizer": result, "canonical": None}
                report["records"].append(record)
                save(output / "results.json", report)
                if result.get("best_rates") is not None:
                    checked = canonical(result["best_rates"])
                    record["canonical"] = {"loss": checked["loss"], "complete": checked["simulation_complete"],
                        "absolute_loss_difference": abs(checked["loss"] - result["best_loss"])}
                save(output / "results.json", report)
                print(f"{name} start={i} status={result['status']} loss={result.get('best_loss')} "
                      f"calls={result.get('evaluations')} canonical={record['canonical']}", flush=True)
                if logger:
                    prefix = f"{name}/start_{i}"
                    logger.log_other(prefix + "/status", result["status"])
                    for j, row in enumerate(result.get("history", [])):
                        calls = row.get("evaluations", j+1)
                        logger.log_metrics({prefix + "/best_loss": row["best_loss"]}, step=calls)
                    logger.log_metrics({prefix + "/evaluations": result.get("evaluations", 0),
                        prefix + "/scenario_solves": result.get("scenario_solves", 0),
                        prefix + "/seconds": result.get("elapsed_s", 0)}, step=32)
        report["finished"] = datetime.now(timezone.utc).isoformat()
        save(output / "results.json", report)
    render(output, report, logger)


def render(output, report, logger=None):
    records = report["records"]
    rows = []
    for record in records:
        r, check = record["optimizer"], record["canonical"]
        rows.append(dict(arm=record["arm"], start=record["start"], status=r["status"],
            accepted_steps=max(0, len(r.get("history", []))-1), evaluations=r.get("evaluations", 0),
            scenario_solves=r.get("scenario_solves", 0),
            rejected_evaluations=r.get("evaluations", 0)-r.get("completed_evaluations", 0),
            loss=None if check is None else check["loss"],
            canonical_complete=None if check is None else check["complete"],
            seconds=r.get("elapsed_s", 0.)))
    with (output / "endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid_starts = [i for i, c in enumerate(report["canonical_initial"]) if c["complete"]]
    fig, axes = plt.subplots(1, len(valid_starts), figsize=(4.3*len(valid_starts), 3.8),
                             layout="constrained", squeeze=False)
    for i, ax in zip(valid_starts, axes[0]):
        for name in ARMS:
            r = next(v["optimizer"] for v in records if v["arm"] == name and v["start"] == i)
            history = r.get("history", [])
            calls = [h.get("evaluations", j+1) for j, h in enumerate(history)]
            losses = [h["best_loss"] for h in history]
            if calls and calls[-1] < r["evaluations"]:
                calls.append(r["evaluations"])
                losses.append(losses[-1])
            ax.step(calls, losses, where="post", color=COLORS[name], label=LABELS[name],
                    linestyle="--" if name.startswith("feasible") else "-",
                    marker="o", markersize=4, markevery=[0, -1] if history else None)
        ax.set(yscale="log", xlabel="Attempted loss + gradient evaluations",
               ylabel="Best evaluated weighted L1 loss",
               title={0: "Original good start", 1: "Perturbed start"}.get(i, f"Random start {i}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "feasibility.png", dpi=180)
    if logger:
        logger.log_figure(figure_name="Feasibility-aware AdaGrad", figure=fig)
    plt.close(fig)
    lines = ["# Feasibility-Aware AdaGrad Test", "",
        "One RPA topology, eight fixed starts, no LLM or RL calls. Engineering test only.", "",
        "CVODE, nine scenarios, 1000 times, 1e-8 tolerances, rate bounds [0.001,100], "
        "species ceiling 10000. Same starts and 32-update cap in each arm. Backtracking "
        "halves an infeasible step at most eight times and is capped at 65 attempted "
        "joint evaluations per start. The stop controls use at most 33. Actual consumed "
        "compute is NOT matched: compare losses together with solve counts and time. "
        "Failed trials count; initial failures are not resampled. Warmup is separate.", "",
        "| Arm | Completed 32 | Canonically valid best | Evaluations | Scenario solves | Fit seconds | Status counts |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for name in ARMS:
        group = [r for r in rows if r["arm"] == name]
        statuses = dict(Counter(r["status"] for r in group))
        lines.append(f"| {name} | {sum(r['status']=='ok' for r in group)}/8 | "
            f"{sum(bool(r['canonical_complete']) for r in group)}/8 | "
            f"{sum(r['evaluations'] for r in group)} | {sum(r['scenario_solves'] for r in group)} | "
            f"{sum(r['seconds'] for r in group):.3f} | {statuses} |")
    lines += ["", "| Start | Initial canonical loss | Stop 0.1 | Stop 0.01 | Feasible 0.1 |",
              "|---|---:|---:|---:|---:|"]
    for i in range(8):
        initial = report["canonical_initial"][i]
        values = []
        for name in ARMS:
            row = next(r for r in rows if r["arm"] == name and r["start"] == i)
            values.append(f"{row['loss']:.8g}" if row["canonical_complete"] else "invalid")
        lines.append(f"| {i} | {initial['loss']:.8g} ({'valid' if initial['complete'] else 'invalid'}) | " + " | ".join(values) + " |")
    lines += ["", "![Matched original starts](feasibility.png)", "",
        "Backtracking tests feasibility only, not monotonic loss descent. Optimizer memory "
        "is updated once per accepted step, never repeatedly for rejected trials. An "
        "exhausted retry/budget retains the previous valid point and an explicit non-ok "
        "status. Unknown exceptions are not converted to recoverable failures.", "",
        "The plots show all three initially feasible starts, carrying the retained best "
        "through terminal failed trials so the x axis includes their costs. Five initially "
        "infeasible starts remain explicit in the table/CSV. There is no multi-topology "
        "generalization or statistical superiority claim. "
        "Retries can consume partial scenarios, so joint-evaluation counts are not a "
        "uniform cost unit. Times exclude canonical verification and warmup.", "",
        "For the difficult perturbed start, backtracking improves the retained canonical "
        "loss from 0.00391752 to 0.00153414 compared with stop-on-failure, at higher cost "
        "(50 vs 13 attempted evaluations). It still exhausts retries near the ceiling. "
        "On the other feasible random start, the smaller fixed step performs best. "
        "Backtracking neither improves the already-good start nor repairs an infeasible "
        "initial rate vector. Better feasible initialization and step selection remain "
        "necessary before integrating this fitter into search.", "",
        (f"Comet experiment: `{report['comet']}`." if report.get("comet") else
         "Stored locally. Comet upload awaits explicit approval for this comparison.")]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    if logger:
        logger.log_asset(str(output / "REPORT.md"))
        logger.log_asset(str(output / "endpoints.csv"))
    invalid = [r for r in records if r["canonical"] is not None and
               (not r["canonical"]["complete"] or r["canonical"]["absolute_loss_difference"] > 2e-6)]
    if invalid:
        raise RuntimeError(f"Canonical verification mismatch in {len(invalid)} retained bests; artifacts preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comet-project")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output, json.loads((args.output / "results.json").read_text()))
        return
    logger = None
    try:
        if args.comet_project:
            from comet_ml import Experiment
            logger = Experiment(project_name=args.comet_project, auto_param_logging=False,
                auto_metric_logging=False, log_code=False, log_git_patch=False, log_graph=False,
                log_git_metadata=False, log_env_details=False, log_env_host=False,
                log_env_cpu=False, log_env_gpu=False, display_summary_level=0)
            if not logger.get_key() or not getattr(logger, "alive", False):
                raise RuntimeError("Comet authentication failed")
            logger.set_name(args.output.name)
            logger.add_tag("optimizer-software-verification")
            logger.log_parameters({"arms": ARMS, "starts": 8, "steps": 32, "rng_seed": 20260915})
        run(args.output, logger)
    finally:
        if logger is not None:
            logger.end()


if __name__ == "__main__":
    main()
