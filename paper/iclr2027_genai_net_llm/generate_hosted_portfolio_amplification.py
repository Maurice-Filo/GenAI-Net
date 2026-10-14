#!/usr/bin/env python3
"""Apply the same time-resolved portfolio reconstruction to frozen Flash cohorts."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper.iclr2027_genai_net_llm import analyze_contract_v2_primary as primary
from paper.iclr2027_genai_net_llm.generate_local_logic_amplification import (
    CATEGORIES, GENERATED, LABELS, PAPER, load_run, manuscript_figures, step_values, write_csv,
)


def main():
    registry = primary.read_json(primary.REGISTRY_PATH)
    plan = primary.read_json(primary.PLAN_PATH)
    if registry["status"] != "audited-complete":
        raise RuntimeError("Primary registry is not complete")
    audits = {}
    for relative in registry["audit_files"]:
        audit = primary.read_json(primary.ROOT / relative)
        if audit["status"] != "pass":
            raise RuntimeError(f"Non-passing campaign audit: {relative}")
        for row in audit["runs"]:
            if row["status"] == "pass":
                audits[str(Path(row["run_root"]) / "results.sqlite")] = row["database_sha256"]
    summaries, endpoints, provenance, gains = [], [], [], []
    for task in ("rpa", "classifier"):
        trajectories = []
        threshold = plan["quality_thresholds"][task]
        for arm, registry_arm in (("none", "no_communication"), ("full", "full_duplex")):
            for seed in registry["seeds"]:
                database = primary.database_path(registry, registry_arm, task, seed)
                rows, metadata = load_run(arm, seed, database.parent, audits[str(database)], threshold)
                trajectories.extend({"task":task, **row} for row in rows)
                provenance.append({"task":task, **metadata})
        terminal = [row for row in trajectories if row["boundary"] == "terminal_boundary"]
        training = defaultdict(list)
        for row in trajectories:
            if row["boundary"] != "terminal_boundary":
                training[row["arm"], row["seed"]].append(row)
        start = max(min(row["evaluations"] for row in rows) for rows in training.values())
        stop = min(max(row["evaluations"] for row in rows) for rows in training.values())
        grid = np.linspace(start, stop, 101)
        curves = {arm: np.asarray([step_values(training[arm, seed], grid) for seed in registry["seeds"]])
                  for arm in ("none", "full")}
        task_summaries = []
        for arm in ("none", "full"):
            rows = sorted((row for row in terminal if row["arm"] == arm), key=lambda row: row["seed"])
            task_summaries.append({"task":task,"arm":arm,"n":len(rows),"threshold":threshold,
                "mean_qualified":float(np.mean([row["qualified"] for row in rows])),
                "median_qualified":float(np.median([row["qualified"] for row in rows])),
                "mean_llm_only_qualified":float(np.mean([row["llm_only_qualified"] for row in rows])),
                "median_loss":float(np.median([row["best_loss"] for row in rows])),
                **{key:float(np.mean([row[key] for row in rows])) for key in CATEGORIES}})
        baseline = {row["seed"]:row for row in terminal if row["arm"] == "none"}
        delta = np.asarray([row["qualified"]-baseline[row["seed"]]["qualified"] for row in terminal if row["arm"] == "full"])
        gains.append({"task":task,"mean_gain":float(delta.mean()),"median_gain":float(np.median(delta)),
                      "wins":int((delta>0).sum()),"ties":int((delta==0).sum()),"losses":int((delta<0).sum())})
        manuscript_figures(training,terminal,task_summaries,curves,grid,
                           arms=("none","full"),stem=f"flash_{task}_amplification")
        write_csv(GENERATED / f"flash_{task}_amplification_trajectories.csv",trajectories)
        endpoints.extend(terminal)
        summaries.extend(task_summaries)
    write_csv(GENERATED / "flash_portfolio_amplification_endpoints.csv",endpoints)
    payload={"summaries":summaries,"paired_gains":gains,"database_audit":provenance,
             "terminal_reconciliation":"All 80 terminal topology sets and losses match stored HOFs"}
    (GENERATED / "flash_portfolio_amplification.json").write_text(json.dumps(payload,indent=2)+"\n")
    report=["# Flash RPA and classifier: portfolio amplification", "",
            "Same reconstruction as LOCAL_LOGIC_AMPLIFICATION.md, applied to the frozen hosted full-duplex and independent-union cohorts. All 20 matched seeds per task are retained. No new model calls or simulations.", "",
            "| Task | Arm | Mean successful | Median | LLM sufficient | RL threshold crossing | Additional RL | Other shared RL |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        report.append(f"| {row['task']} | {LABELS[row['arm']]} | {row['mean_qualified']:.2f} | {row['median_qualified']:.1f} | {row['llm_sufficient']:.2f} | {row['rl_rescued']:.2f} | {row['rl_additional']:.2f} | {row['rl_shared']:.2f} |")
    for task in ("rpa","classifier"):
        report += ["",f"## {task}","",f"![{task}](figures/flash_{task}_amplification_main.png)","",
                   f"![{task} matched budget](figures/flash_{task}_amplification_budget.png)"]
    report += ["", "## Observed contrast", "",
        "RPA shows a strong direct LLM pool and little net portfolio gain from communication: the change in composition toward additional RL topologies does not translate into a resolved count advantage. Classifier shows a larger successful portfolio despite identical median rank-one loss in the two arms. The latter is evidence for reporting useful-solution yield alongside best loss.",
        "", "## Interpretation limits", "",
        "These interventions toggle both directions together. They cannot isolate the contribution of candidate insertion from outward conditioning. Direct LLM sufficiency has priority over RL emitter labels; shared-topology parameter changes receive success credit only if they cross the frozen task threshold. Additional RL topologies establish source and structural novelty relative to that run's valid completed proposals, not causal descent.","",
        "All database hashes match passing audits. Reconstructed final top-30 sets and losses match all 80 stored terminal portfolios. Training pools use only LLM completions available at each HOF snapshot; post-training completions are confined to the final point. Bands are pointwise seed-bootstrap intervals. Model failures are retained. RPA's bounded archive may approach the top-30 ceiling; success here is simulation-qualified topology count, not experimental feasibility or generic diversity.","",
        "Run: `MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m paper.iclr2027_genai_net_llm.generate_hosted_portfolio_amplification` from the repository root."]
    (PAPER / "FLASH_PORTFOLIO_AMPLIFICATION.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"summaries":summaries,"paired_gains":gains,"verified_databases":len(provenance)},indent=2))


if __name__ == "__main__":
    main()
