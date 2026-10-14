"""Verify Logic pruning artifacts and render the appendix tables."""

import csv
import itertools
import json
from collections import defaultdict

import numpy as np

from paper.iclr2027_genai_net_llm.analyze_logic_correctness_pruning import (
    ARMS, EPSILONS, GENERATED, PLAN_PATH, ROOT, SOURCES, digest, seed_summary, select_core,
)
from paper.iclr2027_genai_net_llm.generate_posthoc_checks import table
from paper.iclr2027_genai_net_llm.analyze_contract_v2_primary import holm, wilcoxon_p


def verify(payload):
    if payload["status"] != "complete" or digest(PLAN_PATH) != payload["plan_sha256"]:
        raise ValueError("Missing complete results or stale pruning plan")
    for name, expected in payload["source_hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Pruning analysis source changed: {name}")
    for name, expected in payload["plan"]["input_hashes"].items():
        if digest(GENERATED / name) != expected:
            raise ValueError(f"Pruning input changed: {name}")
    groups = defaultdict(list)
    for row in payload["records"]:
        groups[row["cohort"], row["arm"], row["seed"]].append(row)
        if row["qualified"] != (row["loss"] < payload["plan"]["quality_threshold"]):
            raise ValueError("Qualification disagrees with frozen threshold")
        correct = np.all((2 * np.array(row["targets"]) - 1)
                         * (np.array(row["terminal_outputs"]) - .5) > 1e-6)
        if correct != row["all16_correct"]:
            raise ValueError("Stored correctness disagrees with terminal outputs")
        if row["qualified"] and correct:
            actual = {tuple(s["retained_indices"]) for s in row["subsets"]}
            expected = {s for n in range(6) for s in itertools.combinations(range(5), n)}
            if len(row["subsets"]) != 32 or actual != expected:
                raise ValueError("Incomplete reaction subsets")
            for eps in EPSILONS:
                selected = select_core(row["subsets"], eps, payload["plan"]["quality_threshold"])
                if any(selected[k] != v for k, v in row["selected_cores"][str(eps)].items()):
                    raise ValueError("Stored core does not match frozen selection rule")
    if set(groups) != {(c, a, s) for c in SOURCES for a in ARMS for s in range(20)}:
        raise ValueError("Incomplete cohort/arm/seed grid")
    if any(len(rows) != 30 or len({r["topology_hash"] for r in rows}) != 30 for rows in groups.values()):
        raise ValueError("Terminal archives are incomplete or contain duplicate topologies")
    if len(payload["endpoints"]) != 480 or len(payload["summaries"]) != 24 or len(payload["contrasts"]) != 4:
        raise ValueError("Incomplete endpoint, sensitivity, or contrast tables")
    for row in payload["endpoints"]:
        computed = seed_summary(groups[row["cohort"], row["arm"], row["seed"]], row["epsilon"])
        if any(row[k] != v for k, v in computed.items()):
            raise ValueError("Endpoint count disagrees with replay records")
    for row in payload["summaries"]:
        endpoints = [r for r in payload["endpoints"] if all(r[k] == row[k] for k in ("cohort", "arm", "epsilon"))]
        if len(endpoints) != 20:
            raise ValueError("Summary does not contain 20 seeds")
        for key, value in row.items():
            if key.startswith("mean_") or key.startswith("median_"):
                operation, field = key.split("_", 1)
                np.testing.assert_allclose(value, getattr(np, operation)([r[field] for r in endpoints]))
    for cohort, name, field in (
        ("flash_logic", "flash_logic_directional_endpoints.csv", "qualified"),
        ("local_logic", "local_half_duplex_endpoints.csv", "qualified_topologies"),
    ):
        with (GENERATED / name).open() as stream:
            original = {(r["arm"], int(r["seed"])): int(r[field]) for r in csv.DictReader(stream)
                        if r.get("task", "logic") == "logic"}
        for arm in ARMS:
            for seed in range(20):
                if sum(r["qualified"] for r in groups[cohort, arm, seed]) != original[arm, seed]:
                    raise ValueError("Qualifying counts differ from published endpoints")
    for row in payload["contrasts"]:
        counts = []
        for arm in ("llm_to_rl", "none"):
            values = []
            for seed in range(20):
                valid = [r for r in groups[row["cohort"], arm, seed] if r["qualified"] and r["all16_correct"]]
                values.append(len(valid) if row["metric"] == "qualified_all16_correct" else
                              len({r["selected_cores"]["0.01"]["topology_hash"] for r in valid}))
            counts.append(values)
        np.testing.assert_allclose(row["mean_delta"], np.mean(np.subtract(*counts)))
        np.testing.assert_allclose(row["p_raw"], wilcoxon_p(*counts))
    np.testing.assert_allclose([r["p_holm_4"] for r in payload["contrasts"]],
                               holm([r["p_raw"] for r in payload["contrasts"]]))


def render(payload):
    labels = {"none": "Independent union", "rl_to_llm": "Return only",
              "llm_to_rl": "Insertion only", "full": "Full duplex"}
    rows = []
    for cohort in SOURCES:
        for arm in ARMS:
            index = {r["epsilon"]: r for r in payload["summaries"] if r["cohort"] == cohort and r["arm"] == arm}
            r = index[.01]
            model = "Flash" if cohort == "flash_logic" else "Local"
            rows.append(f"{model} & {labels[arm]} & {r['mean_qualified']:.2f} & "
                        f"{r['mean_qualified_all16_correct']:.2f} & "
                        + " & ".join(f"{index[e]['mean_distinct_pruned_cores']:.2f}" for e in EPSILONS)
                        + f" & {r['mean_prunable_candidates']:.2f}")
    counts = table("llrrrrrr",
        r"Model & Arm & Qualified & All 16 & $\epsilon=.001$ & $.01$ & $.05$ & Prunable",
        rows,
        r"Mean Logic counts per seed (20 seeds per arm). Qualified uses the frozen loss cutoff; "
        r"All 16 also requires every terminal truth-table row to pass the 0.5 threshold with signed "
        r"margin greater than $10^{-6}$. The three $\epsilon$ columns count distinct selected "
        r"pruned structures after enforcing maximum absolute OUT trajectory deviation at most $\epsilon$, "
        r"qualification, and all-row correctness. Prunable counts original eligible candidates losing "
        r"at least one added reaction at $\epsilon=.01$. Rates and fixed reactions are unchanged. "
        r"Only $\epsilon=.01$ is used for inference; other tolerances are descriptive.",
        "tab:logic-pruning-counts")
    rows = []
    for r in payload["contrasts"]:
        model = "Flash" if r["cohort"] == "flash_logic" else "Local"
        metric = "All 16 correct" if r["metric"] == "qualified_all16_correct" else "Pruned structures"
        rows.append(f"{model} & {metric} & {r['mean_delta']:.2f} & "
                    f"$[{r['ci_low']:.2f},{r['ci_high']:.2f}]$ & "
                    f"{r['wins']}/{r['ties']}/{r['losses']} & {r['p_raw']:.3g} & {r['p_holm_4']:.3g}")
    contrasts = table("llrrlrr",
        r"Model & Criterion & Mean gain & 95\% interval & W/T/L & $p$ & $p_{H,4}$",
        rows,
        r"Insertion-only minus independent-union Logic counts at the frozen loss threshold. "
        r"Pruned structures use $\epsilon=.01$. The four two-sided paired Wilcoxon tests "
        r"with Pratt zeros form one new post-hoc Holm family. Intervals use 20,000 paired-seed "
        r"bootstrap resamples and are marginal, not simultaneous. W/T/L denotes insertion wins, "
        r"ties, and losses across 20 matched seeds.", "tab:logic-pruning-contrasts")
    return counts + "\n" + contrasts


def main():
    payload = json.loads((GENERATED / "logic_correctness_pruning.json").read_text())
    verify(payload)
    (GENERATED / "logic_correctness_pruning_table.tex").write_text(render(payload))
    print("Verified 4800 records, 160 published qualifying counts, exhaustive subsets and four contrasts.")


if __name__ == "__main__":
    main()
