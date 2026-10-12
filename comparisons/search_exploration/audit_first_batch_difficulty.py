"""Read-only screening of completed contract-v2 first LLM batches (no new calls)."""

import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "paper/iclr2027_genai_net_llm/generated"


def audit():
    primary = json.loads((GENERATED / "contract_v2_primary_summary.json").read_text())
    thresholds = {r["task"]: r["rl_median"] for r in primary["endpoint"]}
    with (GENERATED / "contract_v2_endpoints.csv").open() as stream:
        controls = {(r["task"], int(r["seed"])): float(r["rl_only_loss"])
                    for r in csv.DictReader(stream) if r["arm"] == "full_duplex"}
    rows = []
    for name in ("contract_v2_full_seed0_audit.json", "contract_v2_full_seeds1to19_audit.json"):
        registry = json.loads((GENERATED / name).read_text())
        for run in registry["runs"]:
            if run["status"] != "pass":
                raise ValueError("Registry includes an unaudited run")
            folder = Path(run["run_root"])
            manifest = json.loads((folder / "run_manifest.json").read_text())
            if manifest.get("model") != "deepseek-v4-flash" or not manifest.get("withhold_initial_hof"):
                raise ValueError("Expected Flash with request-zero withholding")
            workspace = sorted((folder / "harness-workspaces").iterdir())[0]
            state = json.loads((workspace / "CONTEXT/SEARCH_STATE.json").read_text())
            hof = json.loads((workspace / "CONTEXT/HALL_OF_FAME.json").read_text())
            if state["rl_epoch_at_snapshot"] != 0 or state["hall_of_fame_size"] != 0 or hof["entries"]:
                raise ValueError("First batch contains outward HOF conditioning")
            database = folder / "results.sqlite"
            if Path(str(database) + "-wal").exists() and Path(str(database) + "-wal").stat().st_size:
                raise ValueError("Expected finalized database without a live WAL")
            with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                rounds = db.execute("SELECT llm_run_id, launched_epoch, requested FROM llm_runs ORDER BY launched_epoch, created_at").fetchall()
                if not rounds or rounds[0][1] != 0:
                    raise ValueError("Missing first request at epoch zero")
                candidates = db.execute("SELECT loss, valid FROM llm_candidates WHERE llm_run_id=?", (rounds[0][0],)).fetchall()
                first = [loss for loss, valid in candidates if valid and loss is not None and math.isfinite(loss)]
                all_losses = [r[0] for r in db.execute("SELECT loss FROM llm_candidates WHERE valid=1 AND loss IS NOT NULL") if math.isfinite(r[0])]
                failures = db.execute("SELECT COUNT(*) FROM llm_failures").fetchone()[0]
            task, seed = run["task"], run["seed"]
            control = controls[task, seed]
            threshold = thresholds[task]
            rows.append({"task": task, "seed": seed, "first_requested": rounds[0][2],
                "first_valid": len(first), "first_best_loss": min(first, default=None),
                "first_at_or_below_rl_median": sum(v <= threshold for v in first),
                "all_llm_best_loss": min(all_losses, default=None), "llm_rounds": len(rounds),
                "failed_requests": failures, "rl_only_endpoint": control, "rl_median_reference": threshold,
                "first_beats_paired_rl": bool(first and min(first) < control),
                "any_llm_beats_paired_rl": bool(all_losses and min(all_losses) < control),
                "model": manifest["model"], "task_prompt_sha256": manifest["task_prompt_sha256"],
                "initial_hof_entries": len(hof["entries"]),
                "run_root": str(folder)})
    summary = []
    for task in thresholds:
        group = [r for r in rows if r["task"] == task]
        if len(group) != 20 or len({r["seed"] for r in group}) != 20:
            raise ValueError("Expected twenty independent first batches per task")
        summary.append({"task": task, "n": len(group),
            "first_valid": sum(r["first_valid"] for r in group),
            "first_median_best": statistics.median(r["first_best_loss"] for r in group if r["first_best_loss"] is not None),
            "all_llm_median_best": statistics.median(r["all_llm_best_loss"] for r in group if r["all_llm_best_loss"] is not None),
            "first_batch_reaches_rl_median": sum(r["first_at_or_below_rl_median"] > 0 for r in group),
            "first_beats_paired_rl": sum(r["first_beats_paired_rl"] for r in group),
            "any_llm_beats_paired_rl": sum(r["any_llm_beats_paired_rl"] for r in group),
            "rl_median_reference": thresholds[task]})
    return {"summary": summary, "runs": rows,
        "caveat": "Historical contract-v2 prompts, five scheduled batches with later full-duplex conditioning, no AdaGrad. First request has initial HOF withheld. RL endpoint is a search baseline, NOT a functional success criterion. Task losses cannot be compared across tasks. This is exploratory task selection, not confirmation of communication benefit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "runs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(result["runs"])
    lines = ["# Historical First-Batch Difficulty", "", result["caveat"], "",
        "First batch means one Decider/Writer proposal batch of ten CRNs, not one CRN.", "",
        "| Task | Recorded valid / 200 | First-batch median best | All-batch median best | RL endpoint median | Batches reaching RL median / 20 | First beats paired RL / 20 | Any LLM beats paired RL / 20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["summary"]:
        lines.append(f"| {r['task']} | {r['first_valid']} | {r['first_median_best']:.6g} | {r['all_llm_median_best']:.6g} | {r['rl_median_reference']:.6g} | {r['first_batch_reaches_rl_median']} | {r['first_beats_paired_rl']} | {r['any_llm_beats_paired_rl']} |")
    lines += ["", "## Selection Notes", "",
        "Prefer biphasic dose response as a harder follow-up compatible in principle with trajectory-L1 sensitivity fitting. Ultrasensitive dose response is an alternative. These are provisional engineering recommendations, not evidence of a communication advantage.", "",
        "The original five-reaction Logic task also has a difficult first batch. Historical first-batch success does not predict success under the newer workspace-loss prompt; compare modern prompts with a small independent-first-batch screen before committing a full cohort.", "",
        "Frequency-controlled oscillation is a strong harder-task candidate, but its peak-detection/autocorrelation objective is not supported by the current Julia sensitivity-loss adapter. Do not silently replace it with a trajectory surrogate or assume gradients are available.", "",
        "Audit caveat: paper_breadth_tasks.py assigns oscillator_mean osc_w=[0,0.6,0.1,0.3]. oscillation_error multiplies mean_error by w[0]=0, while missing frequency targets give zero frequency error. Thus the recorded objective does not enforce the advertised mean of 1. Do not use this task as mean-tracking evidence without a separately versioned correction and evaluation.", "",
        "The current Julia adapter explicitly accepts only tracking/logic tasks with L1 and one output. Dose response also uses canonical dynamic_tracking_error, but enabling it requires a verified adapter extension; classifier has two outputs and needs additional support.", "",
        "No new model calls, simulations, search campaigns, or manuscript changes. The parity four-arm campaign was not launched while task selection was reconsidered."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
