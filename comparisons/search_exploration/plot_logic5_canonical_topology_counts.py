"""Audit and plot distinct labelled CRN topologies, not permutations or rate vectors."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.search_exploration.analyze_logic5_four_arm import ARMS, COLORS, write_csv
from comparisons.search_exploration.analyze_rpa_mechanisms import structural_token


def canonical_structure(reactions):
    # Reaction multiplicity and input channels remain structural; IDs/rates do not.
    tokens = sorted(structural_token(r) for r in reactions)
    return json.dumps(tokens, separators=(",", ":"))


def canonical_hash(reactions):
    return hashlib.sha256(canonical_structure(reactions).encode()).hexdigest()


def count_memberships(memberships, raw_map, core_map, cutoff):
    banks = {(arm, seed): {key: set() for key in ("valid", "cores", "robust_cores")}
             for arm in ARMS for seed in range(1, 21)}
    for m in memberships:
        bank = banks[m["arm"], m["seed"]]
        bank["valid"].add(raw_map[m["topology_hash"]])
        core = core_map[m["core_hash"]]
        bank["cores"].add(core)
        if m["core_pass_fraction"] >= cutoff:
            bank["robust_cores"].add(core)
    rows = [{"arm": arm, "seed": seed, **{k: len(v) for k, v in bank.items()}}
            for (arm, seed), bank in banks.items()]
    pooled = [{"arm": arm, **{key: len(set().union(*(banks[arm, s][key] for s in range(1, 21))))
                              for key in ("valid", "cores", "robust_cores")}} for arm in ARMS]
    return rows, pooled


def audit(root):
    source = root / "analysis/summary.json"
    summary = json.loads(source.read_text())
    plan = json.loads((root / "plan.json").read_text())
    plan_hash = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    if (summary["status"] != "complete" or summary["baseline_unresolved"] or
            summary["plan_sha256"] != plan_hash):
        raise ValueError("Require complete, resolved and plan-matched diagnostics")
    with gzip.open(root / "selection.json.gz", "rt") as stream:
        selection = json.load(stream)
    if hashlib.sha256((root / "selection.json.gz").read_bytes()).hexdigest() != plan["input_hashes"][str(root / "selection.json.gz")]:
        raise ValueError("Frozen selection changed")
    raw_map, core_map, digests = {}, {}, hashlib.sha256()
    def register(mapping, old, structure):
        new = canonical_hash(structure)
        if old in mapping and mapping[old] != new:
            raise ValueError("One saved hash refers to inconsistent structures")
        mapping[old] = new
    for identifier, candidate in selection["candidates"].items():
        raw = selection["fixed"] + [selection["catalog"][str(i)] for i in candidate["reaction_ids"]]
        register(raw_map, candidate["topology_hash"], raw)
        path = root / "records" / f"{identifier}.json"
        blob = path.read_bytes()
        digests.update(identifier.encode() + hashlib.sha256(blob).digest())
        record = json.loads(blob)
        if record["id"] != identifier or record["status"] != "complete" or record["plan_sha256"] != plan_hash or record["baseline_unresolved"]:
            raise ValueError(f"Unverified diagnostic record: {identifier}")
        if canonical_hash(raw) != canonical_hash(json.loads(record["baseline"]["structure_json"])):
            raise ValueError("Selection and baseline structures disagree")
        core = record["cores"].get("0.01")
        if core is not None:
            structure = json.loads(core["structure_json"])
            expected = selection["fixed"] + [selection["catalog"][str(candidate["reaction_ids"][i])]
                                              for i in core["retained_indices"]]
            if canonical_hash(structure) != canonical_hash(expected):
                raise ValueError("Core structure disagrees with retained reaction indices")
            register(core_map, core["topology_hash"], structure)
    rows, pooled = count_memberships(summary["memberships"], raw_map, core_map, plan["robust_diagnostic_cutoff"])
    saved = {(r["arm"], r["seed"]): r for r in summary["rows"]}
    if len(saved) != 80 or len(summary["rows"]) != 80:
        raise ValueError("Require 80 unique arm-seed rows")
    discrepancies = [{"arm": r["arm"], "seed": r["seed"], "metric": key,
                      "saved": saved[r["arm"], r["seed"]][key], "canonical": r[key]}
                     for r in rows for key in ("valid", "cores", "robust_cores")
                     if r[key] != saved[r["arm"], r["seed"]][key]]
    global_counts = {"valid": len({raw_map[m["topology_hash"]] for m in summary["memberships"]}),
                     "cores": len({core_map[m["core_hash"]] for m in summary["memberships"]}),
                     "robust_cores": len({core_map[m["core_hash"]] for m in summary["memberships"]
                                           if m["core_pass_fraction"] >= plan["robust_diagnostic_cutoff"]})}
    return {"definition": "Sorted multiset of full reaction structures, including fixed template, reaction type, labelled reactant/product multiplicities and input channels; rates, array order and library IDs excluded. Not species-renaming isomorphism or dynamical equivalence.",
            "rows": rows, "pooled": pooled, "global_counts": global_counts, "discrepancies": discrepancies,
            "raw_hashes": len(raw_map), "canonical_raw_hashes": len(set(raw_map.values())),
            "core_hashes": len(core_map), "canonical_core_hashes": len(set(core_map.values())),
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "plan_sha256": plan_hash, "records_digest": digests.hexdigest(),
            "baseline_recovered": summary["baseline_recovered"]}


def render(output, result):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from comparisons.search_exploration.plot_logic5_portfolio_robustness import LABELS
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=.065, right=.98, bottom=.17, top=.80, wspace=.30, hspace=.72)
    fig.text(.065, .955, "Distinct topologies after canonical structural deduplication", fontsize=18, weight="bold")
    fig.text(.065, .911, "Reaction permutations and different rates never create a new topology; species roles and input channels are retained.", fontsize=11)
    bootstrap = np.random.default_rng(20260925).integers(0, 20, (20000, 20))
    for j, (key, title) in enumerate((("valid", "Valid original topologies"), ("cores", "Distinct pruned cores"),
                                     ("robust_cores", "Distinct robust cores"))):
        for i, arm in enumerate(ARMS):
            vals = np.array([r[key] for r in result["rows"] if r["arm"] == arm])
            mean = vals.mean()
            low, high = np.quantile(vals[bootstrap].mean(axis=1), [.025, .975])
            axes[0, j].bar(i, mean, color=COLORS[arm], alpha=.2, width=.65)
            axes[0, j].scatter(i + np.random.default_rng(42).uniform(-.18, .18, 20), vals,
                               color=COLORS[arm], s=17, alpha=.7, linewidth=0)
            axes[0, j].errorbar(i, mean, yerr=[[mean-low], [high-mean]], fmt="D", color="#222222", capsize=3, markersize=4)
            axes[0, j].text(i, 1.02, f"{mean:.1f}", transform=axes[0, j].get_xaxis_transform(), ha="center", color=COLORS[arm], weight="bold")
            pooled = next(r[key] for r in result["pooled"] if r["arm"] == arm)
            axes[1, j].bar(i, pooled, color=COLORS[arm], alpha=.8, width=.65)
            axes[1, j].text(i, 1.02, str(pooled), transform=axes[1, j].get_xaxis_transform(), ha="center", color=COLORS[arm], weight="bold")
        for row in range(2):
            ax = axes[row, j]
            ax.set_title(f"{'ABC'[j] if row == 0 else 'DEF'[j]}  {title}", loc="left", pad=28, fontsize=11, weight="bold")
            ax.set_xticks(range(4), [LABELS[a] for a in ARMS], fontsize=9)
            ax.set_ylim(0, None)
            ax.set_xlim(-.55, 3.55)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[0, j].set_ylabel("Unique per seed")
        axes[1, j].set_ylabel("Unique across 20 seeds")
    fig.text(.065, .083, "Top: 20 seed-level portfolios; diamonds = means, whiskers = marginal 95% seed-bootstrap intervals. Bottom: union across seeds, not their sum.", fontsize=9)
    fig.text(.065, .055, "Robust: at least one observed core rate vector passes >=29/32 draws at 10% perturbation; all 16 rows correct and loss-qualified. No refitting.", fontsize=9)
    fig.text(.065, .027, "Pooled counts are descriptive, with no seed-level significance test. The same topology may appear in more than one communication arm.", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"LOGIC5_CANONICAL_TOPOLOGY_COUNTS.{ext}", dpi=220)
    plt.close(fig)


def main(root, output):
    result = audit(root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    (output / "canonical_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(output / "canonical_counts_per_seed.csv", result["rows"])
    write_csv(output / "canonical_counts_pooled.csv", result["pooled"])
    render(output, result)
    lines = ["# Canonical Topology Counts", "", result["definition"], "",
             f"Recount discrepancies versus the previous per-seed counts: {len(result['discrepancies'])}.",
             "The pooling unit matters: per-seed plots count a topology once in each independent run; pooled plots count it once across all 20 seeds of an arm.",
             "A topology can appear in multiple arms. Pooled counts are descriptive; they have no 20-seed standard error or paired p-value.", "",
             "| Arm | Pooled valid originals | Pooled cores | Pooled robust cores |", "|---|---:|---:|---:|"]
    for row in result["pooled"]:
        lines.append(f"| {row['arm']} | {row['valid']} | {row['cores']} | {row['robust_cores']} |")
    lines += ["", f"Across every arm and seed together: {result['global_counts']}.", "",
              "Robustness uses the same completed, post-hoc order-fallback diagnostics, not new simulations. A robust topology has at least one observed derived rate vector passing the test; no additional fitting was performed.",
              "Pruning uses output tolerance 0.01 with surviving rates unchanged. The same original topology can yield different reduced cores at different saved rate vectors across seeds, so the pooled core count can exceed the pooled original-topology count. Per-seed counts still use one selected rate vector per original topology.",
              f"Recovered baselines: {result['baseline_recovered']}. No baselines unresolved. Canonical counting is distinct from the numerical order used by CVODE.",
              "Species renaming, functional equivalence and approximate family clustering are not used for this identity. Reaction multiplicity and modulated vs unmodulated reactions remain distinct.",
              f"Source: `{result['source']}`", f"Source SHA256: `{result['source_sha256']}`"]
    (output / "CANONICAL_COUNTS.md").write_text("\n\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("discrepancies", "pooled", "global_counts")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
