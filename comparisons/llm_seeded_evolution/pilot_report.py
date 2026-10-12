"""Compare a completed pilot with the frozen RL-only control at shared budgets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from comparisons.llm_seeded_evolution.replay import GENERATED, ROOT, sha256


def report(run: Path) -> dict:
    result = json.loads((run / "completed.json").read_text())
    if result["status"] != "complete" or result["pilot"] is not True:
        raise ValueError("This report requires a completed pilot; use compare.py for full runs.")
    replay = json.loads((run / "replay.json").read_text())
    for filename, key in (("replay.json", "replay_sha256"), ("config.json", "config_sha256")):
        if sha256(run / filename) != result[key]:
            raise ValueError(f"Changed pilot input: {filename}")
    plan = json.loads((GENERATED / "analysis_plan_v2.json").read_text())
    control = next(r for r in plan["control_audit"][result["task"]]["controls"]
                   if int(r["seed"]) == result["seed"])
    control_path = ROOT / control["artifact_path"] / "progress.csv"
    if sha256(control_path) != control["progress_sha256"]:
        raise ValueError("RL-only trajectory no longer matches the frozen analysis plan.")
    with (run / "candidates.csv").open() as stream:
        candidates = list(csv.DictReader(stream))
    with control_path.open() as stream:
        controls = list(csv.DictReader(stream))
    if (len(candidates) != result["candidate_evaluations"]
            or [int(r["candidate_evaluations"]) for r in candidates] != list(range(1, len(candidates) + 1))
            or min(float(r["loss"]) for r in candidates) != result["best_loss"]):
        raise ValueError("Pilot candidate trajectory disagrees with completion summary.")
    by_count = {int(r["candidate_evaluations"]): r for r in candidates}
    checkpoints = []
    for row in controls:
        n = int(float(row["candidate_evaluations"]))
        if n not in by_count:
            continue
        ga, rl = float(by_count[n]["best_so_far_loss"]), float(row["best_so_far_loss"])
        checkpoints.append({"candidate_evaluations": n, "evolution_best_loss": ga,
                            "rl_only_best_loss": rl, "evolution_minus_rl": ga - rl,
                            "llm_evaluations": int(by_count[n]["llm_evaluations"])})
    if not checkpoints:
        raise ValueError("No shared evaluation checkpoints.")
    first_arrival = min(e["after_search_evaluations"] for e in replay["events"] if e["candidates"])
    before_llm = [r for r in candidates if int(r["llm_evaluations"]) == 0]
    direct = [r for r in candidates if r["source"] == "llm"]
    winner = min(candidates, key=lambda r: float(r["loss"]))
    def best(rows):
        return min((float(r["loss"]) for r in rows), default=None)
    summary = {
        "task": result["task"], "seed": result["seed"], "pilot": True,
        "candidate_evaluations": result["candidate_evaluations"],
        "model": replay["model"], "proposal_branch": "none", "new_model_calls": 0,
        "first_proposal_after_search_evaluations": first_arrival,
        "best_before_any_llm": best(before_llm), "best_replayed_proposal": best(direct),
        "best_without_llm_ancestry": best([r for r in candidates if not r["proposal_id"]]),
        "best_loss": result["best_loss"], "qualified_topologies": result["qualified_topologies"],
        "winning_operation": winner["source"], "winning_proposal_lineage": winner["proposal_id"] or None,
        "matched_checkpoints": checkpoints,
        "rl_only_full_budget_loss_context": control["endpoint_loss"],
        "no_communication_full_budget_loss_context": min(r["loss"] for r in replay["source_terminal_archive"]),
        "caveat": "Single seed and a budget prefix. After proposal arrival, search includes LLM seeding; ancestry alone is not a separate GA-only ablation.",
        "candidate_csv_sha256": sha256(run / "candidates.csv"),
        "rl_control_progress_sha256": control["progress_sha256"],
    }
    (run / "pilot_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4.2), layout="constrained")
    axis.plot(list(by_count), [float(r["best_so_far_loss"]) for r in candidates],
              label="Evolution + scheduled LLM proposals", color="#0072B2")
    axis.step([r["candidate_evaluations"] for r in checkpoints],
              [r["rl_only_best_loss"] for r in checkpoints], where="post", label="RL only", color="#D55E00")
    if direct:
        cumulative = []
        lowest = float("inf")
        for row in direct:
            lowest = min(lowest, float(row["loss"]))
            cumulative.append(lowest)
        axis.step([int(r["candidate_evaluations"]) for r in direct] + [len(candidates)],
                  cumulative + [lowest], where="post", label="Replayed proposals alone", color="#009E73", linestyle="--")
    axis.axvline(first_arrival, color="gray", linestyle=":", label="First proposal batch available")
    axis.set(xlabel="Candidate evaluations", ylabel="Best loss (lower is better)", yscale="log",
             title=f"{result['task'].title()}, seed {result['seed']}: exploratory pilot")
    axis.legend(fontsize=8)
    figure.savefig(run / "pilot_comparison.png", dpi=180)
    figure.savefig(run / "pilot_comparison.pdf")
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run), indent=2))


if __name__ == "__main__":
    main()
