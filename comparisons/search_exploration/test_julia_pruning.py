"""Canonical-CVODE pruning verification on the documented selected RPA fixture."""

import argparse
import json
from pathlib import Path

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.engine import atomic_json
from RL4CRN.search_exploration.evaluation import build, evaluate_once
from RL4CRN.search_streams.pruning import run_post_pruning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path("comparisons/rpa_search/configs/rpa_cvode.json").read_text())
    config["rpa"].update(rtol=1e-8, atol=1e-8)
    build_args = ("rpa", config, .001, 100.)
    evaluator, task, _ = build(*build_args)
    candidate = {"reaction_ids": [18, 51, 7, 13, 80],
        "rates": [5.79576795, 11.87837895, 47.32750789, .7406736, 2.34991266]}
    value = evaluate_once(evaluator, candidate)
    parent = {**candidate, **{k: v for k, v in value.items() if k != "task_info"},
        "qualified": value["simulation_complete"] and value["loss"] < .05,
        "origin": "selected_previous_ga_fixture", "source": "software_verification"}
    atomic_json(args.output / "fixture.json", {"parent": parent, "task_config": config,
        "label": "Selected single-topology pruning test, not a search-method comparison",
        "setup_canonical_evaluations": 1})
    result = run_post_pruning([parent], project=args.project, evaluator=evaluator, task=task,
        build_args=build_args, output=args.output, config=ExplorationConfig())
    assert len(result["parents"]) == 1 and result["evaluations"] <= 60
    item = result["parents"][0]
    lines = ["# Selected RPA Pruning Test", "", "Software verification on one previously selected GA design, not a search comparison.", "",
        f"Original canonical loss: {parent['loss']:.9g}.",
        f"Outcome: {item['status']}; {result['evaluations']} pruning evaluations plus one setup verification.", "",
        "| Removed ID | Joint attempts | Canonical loss | Accepted |", "|---|---:|---:|---|"]
    for trial in item["trials"]:
        loss = trial.get("canonical", {}).get("loss")
        lines.append(f"| {trial['removed_reaction_id']} | {trial['fit']['evaluations']} | "
            f"{loss if loss is not None else 'no feasible fit'} | {trial['accepted']} |")
    lines += ["", "No original archive was edited. All failed starts/steps and optimizer checkpoints are retained in `post_pruning/`. "
        "No model calls or Comet uploads were made."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
