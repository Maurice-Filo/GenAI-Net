"""Portfolio-level visualization of frozen Logic-5 perturbation diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.search_exploration.analyze_logic5_four_arm import ARMS, COLORS, write_csv

LABELS = {"none": "No\ncommunication", "llm-to-rl": "LLM to\nsearch",
          "rl-to-llm": "Search to\nLLM", "full": "Full\nduplex"}


def summarize(data):
    rows = data["rows"]
    expected = {(arm, seed) for arm in ARMS for seed in range(1, 21)}
    if (data["status"] != "complete" or len(rows) != 80 or
            {(r["arm"], r["seed"]) for r in rows} != expected or
            data.get("baseline_unresolved", 0) or any(r.get("baseline_unresolved", 0) for r in rows)):
        raise ValueError("Require all 80 resolved diagnostic endpoints")
    endpoints = []
    for r in sorted(rows, key=lambda r: (ARMS.index(r["arm"]), r["seed"])):
        if not 0 <= r["robust_cores"] <= r["cores"] <= r["valid"] or r["cores"] == 0:
            raise ValueError("Invalid portfolio counts or undefined robust fraction")
        endpoints.append({"arm": r["arm"], "seed": r["seed"], "cores": r["cores"],
                          "robust_cores": r["robust_cores"],
                          "robust_percent": 100 * r["robust_cores"] / r["cores"]})
    indices = np.random.default_rng(20260925).integers(0, 20, (20000, 20))
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for metric in ("cores", "robust_cores", "robust_percent"):
            values = np.array([r[metric] for r in endpoints if r["arm"] == arm])
            groups[arm][metric] = {"values": values.tolist(), "mean": float(values.mean()),
                "ci95": np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()}
    contrasts = []
    for other in ARMS[:-1]:
        tests = [t for t in data["tests"] if t["a"] == "full" and t["b"] == other and t["metric"] == "robust_cores"]
        if len(tests) != 1:
            raise ValueError("Missing or repeated pre-existing corrected comparison")
        test = tests[0]
        difference = np.array(groups["full"]["robust_cores"]["values"]) - groups[other]["robust_cores"]["values"]
        if not np.isclose(difference.mean(), test["mean_difference"]):
            raise ValueError("Endpoint counts disagree with saved paired test")
        contrasts.append({**test, "seed_differences": difference.tolist()})
    return {"groups": groups, "contrasts": contrasts, "endpoints": endpoints,
            "ratio_of_mean_robust_counts_full_vs_none": groups["full"]["robust_cores"]["mean"] / groups["none"]["robust_cores"]["mean"],
            "baseline_recovered": data["baseline_recovered"], "baseline_unresolved": data["baseline_unresolved"],
            "diagnostic_plan_sha256": data["plan_sha256"]}


def render(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.5))
    fig.subplots_adjust(left=.08, right=.97, bottom=.155, top=.815, wspace=.31, hspace=.62)
    fig.text(.08, .965, "Parameter robustness across the Logic-5 portfolio", fontsize=18, weight="bold", va="top")
    fig.text(.08, .924, "Full duplex yields more robust cores than uncoupled search, despite a smaller total portfolio.",
             fontsize=11, color="#454545", va="top")
    panels = ((axes[0, 0], "cores", "A  Unique core topologies", "Unique cores per seed"),
              (axes[0, 1], "robust_cores", "B  Unique robust core topologies", "Unique robust cores per seed"),
              (axes[1, 0], "robust_percent", "C  Robust fraction of the portfolio", "Robust cores / all cores (%)"))
    for ax, metric, title, ylabel in panels:
        for i, arm in enumerate(ARMS):
            group = summary["groups"][arm][metric]
            values, mean = np.array(group["values"]), group["mean"]
            violin = ax.violinplot(values, positions=[i], widths=.72, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[arm]); body.set_edgecolor("none"); body.set_alpha(.13)
            jitter = np.random.default_rng(20260925).uniform(-.19, .19, len(values))
            ax.scatter(i + jitter, values, color=COLORS[arm], s=22, alpha=.65, linewidths=0, zorder=3)
            low, high = group["ci95"]
            ax.errorbar(i, mean, yerr=[[mean-low], [high-mean]], color="#202020", fmt="D", markersize=5,
                        capsize=4, elinewidth=1.6, zorder=5)
            ax.text(i, 1.018, f"{mean:.1f}" + ("%" if metric == "robust_percent" else ""),
                    transform=ax.get_xaxis_transform(), ha="center", color=COLORS[arm], weight="bold", fontsize=11)
        ax.set_title(title, loc="left", pad=30, fontsize=12, weight="bold")
        ax.set_xticks(range(4), [LABELS[a] for a in ARMS])
        ax.set(xlim=(-.55, 3.55), ylim=(0, None), ylabel=ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", color="#e6e6e6", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=8)
    ax = axes[1, 1]
    ax.set_title("D  Paired full-duplex advantage", loc="left", pad=30, fontsize=12, weight="bold")
    ax.axvline(0, color="#8a8a8a", ls="--", lw=1)
    for i, contrast in enumerate(summary["contrasts"]):
        mean = contrast["mean_difference"]
        low, high = contrast["mean_difference_ci95"]
        color = COLORS["full"] if i == 0 else "#6b6975"
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="D", color=color, markersize=6, capsize=5, lw=2)
        ax.text(.98, i-.18, f"{mean:+.2f} cores; Holm p = {contrast['holm_p_18']:.3f}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9, color=color)
    ax.set_yticks(range(3), ["vs no\ncommunication", "vs LLM\nto search", "vs search\nto LLM"])
    ax.set(ylim=(2.55, -.6), xlim=(-5, 15), xlabel="Mean paired difference in robust-core count")
    ax.grid(axis="x", color="#e6e6e6", linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.08, .075, "Dots: 20 seeds per arm. Diamonds: means; whiskers: marginal 95% seed-bootstrap intervals. Violin shapes are descriptive.", fontsize=9)
    fig.text(.08, .052, "Robust: at least 29/32 draws remain loss-qualified and correct on all 16 rows; rate factors in [1/1.1, 1.1], no refitting.", fontsize=9)
    fig.text(.08, .029, "Canonical topology identity ignores reaction order and rates. Pruning tolerance: 0.01. Panel D: Holm correction over 18 post-hoc tests.", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"LOGIC5_PORTFOLIO_ROBUSTNESS.{ext}", dpi=220)
    plt.close(fig)


def main(source, output, canonical_audit=None):
    data = json.loads(source.read_text())
    summary = summarize(data)
    summary.update(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if canonical_audit is not None:
        audit = json.loads(canonical_audit.read_text())
        if audit["source_sha256"] != summary["source_sha256"] or audit["discrepancies"]:
            raise ValueError("Canonical audit does not confirm these source counts")
        by_seed = {(r["arm"], r["seed"]): r for r in audit["rows"]}
        for row in summary["endpoints"]:
            if any(row[k] != by_seed[row["arm"], row["seed"]][k] for k in ("cores", "robust_cores")):
                raise ValueError("Canonical counts disagree with figure endpoints")
        summary["canonical_audit_sha256"] = hashlib.sha256(canonical_audit.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(output / "endpoints.csv", summary["endpoints"])
    render(output, summary)
    comparison = summary["contrasts"][0]
    low, high = comparison["mean_difference_ci95"]
    text = f"""# Logic-5 Portfolio Robustness

## Figure Caption

Portfolio-wide parameter robustness after fixed-rate, output-preserving pruning.
Every count uses a canonical multiset of complete reaction structures: reaction
order and rates are ignored; stoichiometric multiplicities, species roles and
input channels are retained. Repeated parameterizations or permutations of the
same topology count once per seed, not as separate designs. This is not
equivalence under arbitrary species renaming or equality of dynamical behavior.
(A) Distinct cores per seed. (B) Cores for which at least 29 of 32 shared
antithetic perturbation draws retain both the frozen loss qualification and all
16 correct terminal classifications. Each rate is multiplied by a factor in
[1/1.1, 1.1], without clipping or refitting. (C) Robust-core fraction, computed
within each seed before averaging, not as a ratio of pooled counts. (D) Paired
full-duplex minus comparator differences in robust-core count. Points in A-C
represent all 20 seeds; violin envelopes describe their distributions. Diamonds
and intervals show means and marginal 95% seed-bootstrap intervals. D reuses
the existing two-sided exact Pratt signed-rank tests and Holm correction across
18 post-hoc comparisons, without creating a smaller test family. Its bootstrap
intervals are marginal, so crossing zero need not agree with corrected p-values.
All communication arms and all three full-duplex contrasts are shown.

## Takeaway

Full duplex yields 16.30 versus 9.50 robust cores per seed for no communication:
an increase of {100*(summary['ratio_of_mean_robust_counts_full_vs_none']-1):.1f}%
in the ratio of group means, or {comparison['mean_difference']:.2f} additional
cores per matched seed (marginal bootstrap 95% CI [{low:.2f}, {high:.2f}];
Holm-adjusted p={comparison['holm_p_18']:.5f}). Its differences versus either
one-way arm are not statistically resolved. The increased number of robust cores
coexists with fewer total distinct cores, rather than merely a larger archive.

This supports enrichment for solutions less dependent on precise parameter
values under this diagnostic, not proof of biochemical mechanisms, distinct
functional motifs, biological robustness, or full-duplex synergy. The label
is a finite-draw, post-hoc threshold, not a guarantee throughout parameter space.
Different pruned cores can have different numbers of perturbed parameters.
A core is counted robust when at least one observed derived rate vector passes;
this is an observed-design criterion, not a statement about every rate vector.
Winner robustness is a different estimand and is not generalized from this figure.
The separate winner analysis found lower 10% pass fractions for full-duplex
winners (73.1%) than uncoupled winners (88.0%), descriptively.

## Numerical Audit

All baselines resolved; {summary['baseline_recovered']} required recorded-order
fallback. Canonical order is attempted first, then recorded orders on numerical
failure; baseline retries must also reproduce saved losses and correctness.
Successful but unfavorable perturbations are never retried. Attempt failures
remain logged. No new simulations, fitting, model calls, or hypothesis tests
were performed to produce this figure. Earlier artifacts remain unchanged.

Source: `{source.resolve()}`
Source SHA256: `{summary['source_sha256']}`
Diagnostic plan SHA256: `{summary['diagnostic_plan_sha256']}`
"""
    (output / "CAPTION_AND_INTERPRETATION.md").write_text(text)
    print(output / "LOGIC5_PORTFOLIO_ROBUSTNESS.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-audit", type=Path)
    args = parser.parse_args()
    main(args.source, args.output, args.canonical_audit)
