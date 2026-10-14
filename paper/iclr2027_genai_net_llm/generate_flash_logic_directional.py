#!/usr/bin/env python3
"""Analyze only completed, audited Flash Logic communication cohorts."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper.iclr2027_genai_net_llm import analyze_contract_v2_primary as primary
from paper.iclr2027_genai_net_llm.generate_local_logic_amplification import (
    ARMS, CATEGORIES, COLORS, GENERATED, LABELS, PAPER,
    load_run, manuscript_figures, step_values, write_csv,
)

STEM = "flash_logic_directional"
NEW_AUDITS = {
    arm: GENERATED / f"flash_logic_pinned_{arm}_20260908_audit.json"
    for arm in ("llm_to_rl", "rl_to_llm")
}
CONTRASTS = (
    ("full", "none"), ("llm_to_rl", "none"), ("full", "rl_to_llm"),
    ("full", "llm_to_rl"), ("rl_to_llm", "none"), ("llm_to_rl", "rl_to_llm"),
)
TEX_LABELS = {"none": "None", "full": "Full duplex",
              "llm_to_rl": r"LLM$\to$RL", "rl_to_llm": r"RL/HOF$\to$LLM"}


def contrast(left, right, *, higher):
    if len(left) != 20 or len(right) != 20:
        raise ValueError("Directional analysis requires 20 matched seeds")
    left, right = np.asarray(left), np.asarray(right)
    relations = Counter(
        ("win" if a > b else "loss" if a < b else "tie") if higher
        else primary.paired_relation(a, b) for a, b in zip(left, right)
    )
    return {"wins": relations["win"], "ties": relations["tie"],
            "losses": relations["loss"], "median_delta": float(np.median(left-right)),
            "p_unadjusted": primary.wilcoxon_p(left.tolist(), right.tolist())}


def returned_batches(audit):
    if "successful_request_epochs" in audit:
        return len(audit["successful_request_epochs"])
    # Older passing audits recorded only the schedule, with a separate failure count.
    if audit["llm_failures"] != 0:
        raise RuntimeError("A legacy audit with failures needs explicit request accounting")
    return len(audit["request_epochs"])


def sources():
    registry = primary.read_json(primary.REGISTRY_PATH)
    if registry["status"] != "audited-complete" or registry["seeds"] != list(range(20)):
        raise RuntimeError("Frozen primary registry must be complete")
    indexed = {}
    for relative in registry["audit_files"]:
        audit = primary.read_json(primary.ROOT / relative)
        if audit["status"] != "pass":
            raise RuntimeError(f"Non-passing audit: {relative}")
        for row in audit["runs"]:
            if row["task"] == "logic" and row["status"] == "pass":
                indexed[str(Path(row["run_root"]) / "results.sqlite")] = row
    result = {}
    for arm, registered in (("none", "no_communication"), ("full", "full_duplex")):
        for seed in range(20):
            result[arm, seed] = indexed[str(primary.database_path(registry, registered, "logic", seed))]
    for arm, path in NEW_AUDITS.items():
        audit = primary.read_json(path)
        expected_id = f"flash-v2-half-logic-{arm.replace('_', '-')}-seeds0to19-flash-pinned-20260908"
        if audit["status"] != "pass" or audit["campaign_id"] != expected_id:
            raise RuntimeError(f"Wrong or non-passing Flash-pinned audit: {path}")
        for row in audit["runs"]:
            key = arm, int(row["seed"])
            if row["task"] != "logic" or row["status"] != "pass" or key in result:
                raise RuntimeError("Invalid or duplicate audited seed")
            result[key] = row
    if set(result) != {(a, s) for a in ARMS for s in range(20)}:
        raise RuntimeError("Incomplete directional grid")
    prompts = set()
    for row in result.values():
        manifest = primary.read_json(Path(row["run_root"]) / "run_manifest.json")
        if manifest["model"] != "deepseek-v4-flash" or not manifest["withhold_initial_hof"]:
            raise RuntimeError("Wrong model or request-0 policy")
        prompts.add(manifest["task_prompt_sha256"])
    if len(prompts) != 1:
        raise RuntimeError("Task prompt hashes differ across arms")
    return result, prompts.pop()


def endpoint_figure(terminal):
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), layout="constrained")
        rng = np.random.default_rng(20260908)
        for ax, field, title, ylabel in zip(axes, ("qualified", "best_loss"),
                ("A  Successful topologies", "B  Rank-one loss"),
                ("Count in terminal top-30", "Canonical loss (log scale)")):
            for x, arm in enumerate(ARMS):
                values = np.array([r[field] for r in terminal if r["arm"] == arm])
                jitter = rng.uniform(-.16, .16, len(values))
                ax.scatter(x+jitter, values, s=13, alpha=.6, color=COLORS[arm])
                q1, med, q3 = np.quantile(values, [.25, .5, .75])
                ax.plot([x, x], [q1, q3], color="#222222", lw=2)
                ax.plot([x-.14, x+.14], [med, med], color="#222222", lw=2)
            ax.set_xticks(range(4), ["Independent\nunion", "RL/HOF\nto LLM", "LLM\nto RL", "Full\nduplex"])
            ax.set(ylabel=ylabel, title=title)
            if field == "best_loss":
                ax.set_yscale("log")
            ax.grid(axis="y", alpha=.18)
        for extension in ("pdf", "png"):
            fig.savefig(PAPER / "figures" / f"{STEM}_endpoints.{extension}", dpi=300)
        plt.close(fig)


def tables(summaries, contrasts):
    lines = [r"\begin{table*}[t]", r"\centering\small",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Arm & Median loss & Median yield & Mean yield & Accepted & Rejected \\", r"\midrule"]
    for r in summaries:
        lines.append(f"{TEX_LABELS[r['arm']]} & {r['median_loss']:.5f} & {r['median_qualified']:.1f} & {r['mean_qualified']:.2f} & {r['accepted']} & {r['rejected']} " + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Flash Logic: 20 matched seeds per arm, 100 epochs, fixed quality threshold $0.04271268$. Accepted and rejected counts are candidate totals across 100 returned batches per arm.}",
              r"\label{tab:flash-logic-endpoints}\end{table*}",
              r"\begin{table*}[t]\centering\small",
              r"\begin{tabular}{lrrrr}\toprule",
              r"Contrast & Yield W/T/L & Yield $p$ & Loss W/T/L & Loss $p$ \\", r"\midrule"]
    for r in contrasts:
        wtl = lambda v: f"{v['wins']}/{v['ties']}/{v['losses']}"
        lines.append(f"{TEX_LABELS[r['left']]} vs {TEX_LABELS[r['right']]} & {wtl(r['yield'])} & {r['yield']['p_unadjusted']:.4g} & {wtl(r['loss'])} & {r['loss']['p_unadjusted']:.4g} " + r"\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Exploratory Flash Logic contrasts; W/T/L favors the left arm. Two-sided paired Wilcoxon tests (Pratt zeros), unadjusted $p$-values. These comparisons are not added to the primary Holm families.}",
              r"\label{tab:flash-logic-contrasts}\end{table*}"]
    (GENERATED / f"{STEM}_tables.tex").write_text("\n".join(lines)+"\n")


def main():
    audited, prompt_hash = sources()
    threshold = primary.read_json(primary.PLAN_PATH)["quality_thresholds"]["logic"]
    trajectories, provenance = [], []
    for (arm, seed), audit in audited.items():
        rows, metadata = load_run(arm, seed, Path(audit["run_root"]), audit["database_sha256"], threshold)
        trajectories.extend(rows)
        provenance.append(metadata)
    terminal = [r for r in trajectories if r["boundary"] == "terminal_boundary"]
    training = defaultdict(list)
    for r in trajectories:
        if r["boundary"] != "terminal_boundary":
            training[r["arm"], r["seed"]].append(r)
    grid = np.linspace(max(min(r["evaluations"] for r in rows) for rows in training.values()),
                       min(max(r["evaluations"] for r in rows) for rows in training.values()), 101)
    curves = {a: np.array([step_values(training[a, s], grid) for s in range(20)]) for a in ARMS}
    summaries = []
    for arm in ARMS:
        rows = [r for r in terminal if r["arm"] == arm]
        audit_rows = [audited[arm, s] for s in range(20)]
        summaries.append({"arm": arm, "n": len(rows),
            "median_loss": float(np.median([r["best_loss"] for r in rows])),
            "median_qualified": float(np.median([r["qualified"] for r in rows])),
            "mean_qualified": float(np.mean([r["qualified"] for r in rows])),
            "accepted": sum(r["accepted_candidates"] for r in audit_rows),
            "rejected": sum(r["rejected_candidates"] for r in audit_rows),
            "batches_returned": sum(returned_batches(r) for r in audit_rows),
            "model_failures": sum(r["llm_failures"] for r in audit_rows),
            **{key: float(np.mean([r[key] for r in rows])) for key in CATEGORIES}})
    indexed = {(r["arm"], r["seed"]): r for r in terminal}
    contrasts = []
    for a, b in CONTRASTS:
        pair = {"left": a, "right": b}
        for metric, field in (("yield", "qualified"), ("loss", "best_loss")):
            left, right = [[indexed[arm, s][field] for s in range(20)] for arm in (a, b)]
            pair[metric] = contrast(left, right, higher=metric == "yield")
            if metric == "loss":
                pair[metric]["median_right_over_left_ratio"] = float(np.median(np.array(right)/left))
        contrasts.append(pair)
    manuscript_figures(training, terminal, summaries, curves, grid, stem=STEM)
    endpoint_figure(terminal)
    tables(summaries, contrasts)
    write_csv(GENERATED / f"{STEM}_endpoints.csv", terminal)
    write_csv(GENERATED / f"{STEM}_trajectories.csv", trajectories)
    payload = {"task": "logic", "model": "deepseek-v4-flash", "epochs": 100,
               "threshold": threshold, "prompt_sha256": prompt_hash,
               "summaries": summaries, "contrasts": contrasts, "database_audit": provenance,
               "inference": "Exploratory paired contrasts; unadjusted p-values. Hosted cohorts collected on different dates; endpoint identifier does not pin remote weights."}
    (GENERATED / f"{STEM}.json").write_text(json.dumps(payload, indent=2)+"\n")
    report = ["# Completed Flash Logic directional ablation", "",
              "80 audited runs, seeds 0..19, 100 epochs. Frozen full/none controls plus the two completed Flash-pinned 20260908 arms. No parity results or interrupted predecessors included; no simulations or model calls rerun.", "",
              f"Task prompt SHA-256: `{prompt_hash}`. Quality threshold: `{threshold}`.", "",
              "| Arm | Median loss | Median yield | Mean yield | Accepted/rejected |",
              "|---|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| {LABELS[r['arm']]} | {r['median_loss']:.8f} | {r['median_qualified']:.1f} | {r['mean_qualified']:.2f} | {r['accepted']}/{r['rejected']} |")
    report += ["", "| Contrast | Yield W/T/L | p | Loss W/T/L | p |", "|---|---:|---:|---:|---:|"]
    for r in contrasts:
        wtl = lambda v: f"{v['wins']}/{v['ties']}/{v['losses']}"
        report.append(f"| {LABELS[r['left']]} vs {LABELS[r['right']]} | {wtl(r['yield'])} | {r['yield']['p_unadjusted']:.6g} | {wtl(r['loss'])} | {r['loss']['p_unadjusted']:.6g} |")
    report += ["", "![Time-resolved yield and source contributions](figures/flash_logic_directional_main.png)",
               "", "![Endpoints](figures/flash_logic_directional_endpoints.png)",
               "", "![Matched evaluation budgets](figures/flash_logic_directional_budget.png)", "",
               "## Interpretation", "",
               "Insertion-only increases successful-topology yield relative to independent union and outward-only conditioning. Full duplex also beats those controls, but does not show a detectable incremental yield or loss benefit over insertion-only. None of the six paired rank-one contrasts is resolved. Flash therefore supports a useful-portfolio claim, not a directional best-loss or full-duplex-synergy claim. Most insertion-only successful topologies are RL-emitted and absent from the completed LLM proposals.", "",
               "All 80 database hashes match passing audits; terminal topology sets and losses reconcile with stored HOFs. Same task-prompt hash and initial-HOF withholding in all arms. All returned candidates, rejections and failures are retained. Self-feedback is available in every arm. New audit filenames and exact database paths are recorded in the generator and generated JSON. The JSON/CSV use unrounded data; LaTeX tables are generated from them.", "",
               "Tests are exploratory and unadjusted, not a new primary family. No detected difference is not an equivalence result or a proof of redundancy. Timing is asynchronous and cohorts were collected on different dates against a hosted endpoint, whose weights are not pinned. Additional RL topologies do not establish causal ancestry.", "",
               "Reproduce: `MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m paper.iclr2027_genai_net_llm.generate_flash_logic_directional`."]
    (PAPER / "FLASH_LOGIC_DIRECTIONAL.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"summaries": summaries, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
