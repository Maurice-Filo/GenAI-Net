"""Render verified frozen-data sensitivity analyses; never call a model or solver."""

import csv
import hashlib
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parent
SOURCE = PAPER / "reviews/repo_answers_2026-09-09/results.json"
GENERATED = PAPER / "generated"
ARMS = {"full": "Full", "none": "None", "llm_to_rl": "Insertion",
        "rl_to_llm": "Return"}
FAMILIES = (("directional", "p_holm_22", 22),
            ("overlap_contrasts", "p_holm_4", 4),
            ("classifier", "p_holm_6", 6))


def family_rows(payload, family):
    return payload[family]["contrasts"] if family == "classifier" else payload[family]


def verify(payload):
    for name, expected in payload["input_sha256"].items():
        actual = hashlib.sha256((GENERATED / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Stale post-hoc input: {name}; rerun the analysis")
    for family, field, size in FAMILIES:
        rows = family_rows(payload, family)
        if len(rows) != size or any(r["n"] != 20 for r in rows):
            raise ValueError(f"Unexpected family size or seed count: {family}")
        previous = 0.0
        for rank, row in enumerate(sorted(rows, key=lambda r: r["p_raw"])):
            previous = max(previous, min(1.0, row["p_raw"] * (size - rank)))
            if abs(previous - row[field]) > 1e-12:
                raise ValueError(f"Incorrect Holm value: {family}")


def number(value):
    return f"{value:.3g}"


def effect(row, loss=False):
    precision = 5 if loss else 2
    lo, hi = row["mean_delta_ci95"]
    return (f"${row['mean_delta']:+.{precision}f}$ & "
            f"$[{lo:.{precision}f}, {hi:.{precision}f}]$")


def table(columns, header, rows, caption, label):
    return "\n".join([
        r"\begin{table*}[t]", r"\centering\small",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header + r" \\",
        r"\midrule", *[r + r" \\" for r in rows], r"\bottomrule\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}\end{{table*}}", "",
    ])


def render(payload):
    directional = []
    for row in payload["directional"]:
        model = "Flash" if row["cohort"] == "flash_logic" else "Local"
        loss = row["metric"] in ("best_loss", "rank_one_loss")
        directional.append(
            f"{model} & {ARMS[row['left']]} $-$ {ARMS[row['right']]} & "
            f"{'Loss' if loss else 'Yield'} & {effect(row, loss)} & "
            f"{number(row['p_raw'])} & {number(row['p_holm_22'])}")
    overlap = []
    for row in payload["overlap_contrasts"]:
        model = "Flash" if row["cohort"] == "flash_logic" else "Local"
        overlap.append(
            f"{model} & {ARMS[row['left']]} $-$ {ARMS[row['right']]} & "
            f"{row['right_mean']:.2f} & {row['left_mean']:.2f} & {effect(row)} & "
            f"{number(row['p_raw'])} & {number(row['p_holm_4'])}")
    classifier = []
    for row in payload["classifier"]["contrasts"]:
        metric = "Both binary outputs" if row["metric"] == "binary_correct" else "Winning label"
        classifier.append(
            f"{row['multiplier']:g} & {metric} & {row['right_mean']:.2f} & "
            f"{row['left_mean']:.2f} & {effect(row)} & "
            f"{number(row['p_raw'])} & {number(row['p_holm_6'])}")
    return {
        "posthoc_directional_table.tex": table(
            "lllrrrr", r"Model & Contrast & Metric & Mean $\Delta$ & 95\% interval & $p$ & $p_{H,22}$",
            directional,
            r"Post-hoc correction of the 22 reported Logic directional endpoint tests. "
            r"Differences are left minus right: positive yield and negative loss favor the left arm. "
            r"All contrasts use 20 matched seeds. Intervals are marginal paired-seed "
            r"percentile-bootstrap intervals (20,000 resamples), not simultaneous intervals. "
            r"Insertion and return denote LLM$\to$RL and RL/HOF$\to$LLM respectively; "
            r"none retains both search sources with terminal aggregation.", "tab:posthoc-directional"),
        "posthoc_overlap_table.tex": table(
            "llrrrrrr", r"Model & Contrast & Right & Left & Mean $\Delta$ & 95\% interval & $p$ & $p_{H,4}$",
            overlap,
            r"Direct LLM topologies absent from HOF content available at their first proposal. "
            r"Left and right columns are mean counts, with paired mean differences and "
            r"marginal 95\% bootstrap intervals. Holm correction covers these four post-hoc "
            r"tests only. Exact overlap is not causal copying; absence is not global novelty.",
            "tab:posthoc-overlap"),
        "posthoc_classifier_table.tex": table(
            "rlrrrrrr", r"Cutoff/$\tau$ & All-32 criterion & None & Full & Mean $\Delta$ & 95\% interval & $p$ & $p_{H,6}$",
            classifier,
            r"Mean terminal classifier topology counts satisfying both the loss cutoff and "
            r"the stated criterion on all 32 supplied cases. Differences are full minus none; "
            r"intervals are marginal paired-seed bootstrap intervals. Six post-hoc tests form "
            r"one Holm family. The frozen cutoff $\tau=0.10289627$ remains the primary estimand; "
            r"the correctness diagnostics do not establish generalization or robustness.",
            "tab:posthoc-classifier"),
    }


def main():
    payload = json.loads(SOURCE.read_text())
    verify(payload)
    for name, content in render(payload).items():
        (GENERATED / name).write_text(content)
    rows = []
    for family, field, size in FAMILIES:
        for row in family_rows(payload, family):
            rows.append(dict(family=family, family_size=size,
                cohort=row.get("cohort", "flash_classifier"),
                left=row.get("left", "full"), right=row.get("right", "none"),
                metric=row.get("metric", "first_seen_absent_from_exposed_hof"),
                multiplier=row.get("multiplier", ""), n=row["n"],
                left_mean=row["left_mean"], right_mean=row["right_mean"],
                mean_delta=row["mean_delta"], ci_low=row["mean_delta_ci95"][0],
                ci_high=row["mean_delta_ci95"][1], p_raw=row["p_raw"], p_holm=row[field]))
    with (GENERATED / "posthoc_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rendered {len(rows)} post-hoc tests from fingerprint-verified inputs")


if __name__ == "__main__":
    main()
