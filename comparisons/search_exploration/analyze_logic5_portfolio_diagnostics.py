"""Summarize completed post-hoc pruning, perturbation survival and structural families."""

from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from comparisons.search_exploration.analyze_logic5_four_arm import ARMS, LABELS, COLORS, paired_endpoint, holm, write_csv
from comparisons.search_exploration.analyze_rpa_mechanisms import structural_token, rarefied_richness
from RL4CRN.search_exploration.engine import atomic_json


def pass_fraction(record, mode, radius=1.1, field="valid"):
    rows = [r for r in record["perturbations"] if r["mode"] == mode and r["radius"] == radius]
    if len(rows) != 32 or {r["draw"] for r in rows} != set(range(32)):
        raise ValueError("Missing or repeated perturbation draws")
    return float(np.mean([bool(r[field]) if r["status"] == "ok" else False for r in rows]))


def cluster(structures):
    hashes = sorted(structures)
    if not hashes:
        return {}, {}, []
    tokens = sorted({t for structure in structures.values() for t in structure})
    index = {t: i for i, t in enumerate(tokens)}
    features = np.zeros((len(hashes), max(1, len(tokens))), bool)
    for i, h in enumerate(hashes):
        features[i, [index[t] for t in structures[h]]] = True
    tree = linkage(pdist(features, "jaccard"), method="average") if len(hashes) > 1 else None
    cuts = {}
    for cutoff in (.3, .45, .6, .75):
        raw = fcluster(tree, cutoff, criterion="distance") if tree is not None else np.ones(len(hashes), int)
        sizes = Counter(raw)
        order = {old: i+1 for i, old in enumerate(sorted(sizes, key=lambda x: (-sizes[x], x)))}
        cuts[str(cutoff)] = {h: order[int(v)] for h, v in zip(hashes, raw)}
    labels = cuts["0.6"]
    families = []
    for family in sorted(set(labels.values())):
        members = [h for h in hashes if labels[h] == family]
        selected = features[[hashes.index(h) for h in members]]
        medoid = members[int(np.argmin(squareform(pdist(selected, "jaccard")).mean(axis=1)))] if len(members) > 1 else members[0]
        families.append({"family": family, "unique_structures": len(members), "members": members, "medoid": medoid})
    return labels, cuts, families


def analyze(output):
    from comparisons.search_exploration.run_logic5_portfolio_diagnostics import load_plan, read, sha
    plan, selection = load_plan(output)
    fingerprint = sha(output / "plan.json")
    records = {}
    for file in (output / "records").glob("*.json"):
        r = read(file)
        if r.get("status") != "complete" or r.get("plan_sha256") != fingerprint:
            raise ValueError("Unverified diagnostic result")
        records[r["id"]] = r
    if set(records) != set(selection["candidates"]):
        raise ValueError("No final inference from incomplete diagnostics")
    unresolved = {k for k, r in records.items() if r.get("baseline_unresolved", False)}
    resolved_records = [r for k, r in records.items() if k not in unresolved]
    folder = output / "analysis"
    folder.mkdir(exist_ok=True)
    catalog = selection["catalog"]
    raw_structures, core_structures, representatives, memberships = {}, {}, {}, []
    rows, sensitivities = [], []
    for run in selection["runs"]:
        valid, originals, robust_cores, core_hashes = [], [], set(), set()
        row = {"arm": run["arm"], "seed": run["seed"], "qualified": len(run["members"]),
               "valid": 0, "cores": 0, "prunable": 0, "robust_originals": 0, "robust_cores": 0,
               "baseline_unresolved": 0, "baseline_recovered": 0,
               "expected_surviving_originals": 0., "valid_topologies": [], "core_topologies": [], "robust_core_topologies": []}
        for member in run["members"]:
            record = records[member["candidate"]]
            if member["candidate"] in unresolved:
                row["baseline_unresolved"] += 1
                continue
            row["baseline_recovered"] += len(record["baseline"].get("order_attempts", [])) > 1
            if not (record["baseline"]["qualified"] and record["baseline"]["all16_correct"]):
                continue
            valid.append(record)
            candidate = selection["candidates"][record["id"]]
            core = record["cores"]["0.01"]
            signature = tuple(sorted(structural_token(catalog[str(i)]) for i in candidate["reaction_ids"]))
            core_signature = tuple(sorted(structural_token(catalog[str(candidate["reaction_ids"][i])]) for i in core["retained_indices"]))
            for bank, h, sig in ((raw_structures, candidate["topology_hash"], signature),
                                  (core_structures, core["topology_hash"], core_signature)):
                if h in bank and bank[h] != sig:
                    raise ValueError("Conflicting complete structural identities")
                bank[h] = sig
            fraction, core_fraction = pass_fraction(record, "original"), pass_fraction(record, "core")
            row["expected_surviving_originals"] += fraction
            row["robust_originals"] += fraction >= plan["robust_diagnostic_cutoff"]
            row["prunable"] += core["retained_count"] < 5
            core_hashes.add(core["topology_hash"])
            if core_fraction >= plan["robust_diagnostic_cutoff"]:
                robust_cores.add(core["topology_hash"])
            originals.append(candidate["topology_hash"])
            old = representatives.get(core["topology_hash"])
            if old is None or core["loss"] < old["core"]["loss"]:
                representatives[core["topology_hash"]] = {"candidate": candidate, "core": core}
            memberships.append({"arm": run["arm"], "seed": run["seed"], "candidate": record["id"],
                "topology_hash": candidate["topology_hash"], "core_hash": core["topology_hash"],
                "retained_reactions": core["retained_count"], "original_pass_fraction": fraction,
                "core_pass_fraction": core_fraction, "core_loss": core["loss"], "loss": candidate["loss"]})
        row.update(valid=len(valid), cores=len(core_hashes), robust_cores=len(robust_cores),
                   valid_topologies=originals, core_topologies=sorted(core_hashes), robust_core_topologies=sorted(robust_cores))
        rows.append(row)
        for epsilon in plan["epsilons"]:
            sensitivities.append({"arm": run["arm"], "seed": run["seed"], "epsilon": epsilon,
                "valid": len(valid), "cores": len({r["cores"][str(epsilon)]["topology_hash"] for r in valid})})
    print(f"Clustering {len(raw_structures)} valid topologies and {len(core_structures)} pruned cores", flush=True)
    raw_labels, _, _ = cluster(raw_structures)
    core_labels, cuts, families = cluster(core_structures)
    for row in rows:
        row["raw_families"] = len({raw_labels[h] for h in row["valid_topologies"]})
        row["core_families"] = len({core_labels[h] for h in row["core_topologies"]})
        row["robust_core_families"] = len({core_labels[h] for h in row["robust_core_topologies"]})
        richness = rarefied_richness([core_labels[h] for h in row["core_topologies"]], 10)
        row["rarefied_core_families_k10"] = float(richness) if np.isfinite(richness) else None
    for member in memberships:
        member["core_family"] = core_labels[member["core_hash"]]
    for family in families:
        family["representative"] = representatives[family["medoid"]]
        family["seed_coverage"] = {arm: len({r["seed"] for r in rows if r["arm"] == arm and
            any(core_labels[h] == family["family"] for h in r["core_topologies"])}) for arm in ARMS}
    tests = []
    by = {(r["arm"], r["seed"]): r for r in rows}
    for b, a in itertools.combinations(ARMS, 2):
        if unresolved:
            continue  # Do not test incomplete validated counts as if unresolved meant invalid.
        for metric in ("valid", "cores", "robust_cores"):
            tests.append({"a": a, "b": b, "metric": metric, **paired_endpoint(
                [by[a, s][metric] for s in range(1, 21)], [by[b, s][metric] for s in range(1, 21)], smaller=False)})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["holm_p_18"] = p
    winner_curves = []
    for run in selection["runs"]:
        record = records[run["members"][0]["candidate"]]
        if record.get("baseline_unresolved", False):
            continue
        for radius in plan["winner_radii"]:
            draws = [r for r in record["perturbations"] if r["mode"] == "original" and r["radius"] == radius]
            winner_curves.append({"arm": run["arm"], "seed": run["seed"], "radius": radius,
                "qualified": pass_fraction(record, "original", radius, "qualified"),
                "all16_correct": pass_fraction(record, "original", radius, "all16_correct"),
                "valid": pass_fraction(record, "original", radius, "valid"),
                "failure_fraction": sum(d["status"] != "ok" for d in draws)/32})
    scalar_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
    summary = {arm: {key: float(np.mean([r[key] for r in rows if r["arm"] == arm]))
        for key in ("qualified", "valid", "cores", "prunable", "robust_originals", "robust_cores",
                    "expected_surviving_originals", "raw_families", "core_families", "robust_core_families",
                    "baseline_unresolved", "baseline_recovered")} for arm in ARMS}
    payload = {"plan_sha256": fingerprint, "status": "complete", "summary": summary, "rows": rows,
        "tests": tests, "families": families, "core_labels": core_labels, "core_labels_cut_sensitivity": cuts,
        "memberships": memberships, "winner_curves": winner_curves, "pruning_sensitivity": sensitivities,
        "evaluations": sum(r["evaluator_calls"] for r in records.values()),
        "baseline_unresolved": len(unresolved),
        "unresolved_candidates": sorted(unresolved),
        "baseline_recovered": sum(len(r["baseline"].get("order_attempts", [])) > 1 for r in resolved_records),
        "subset_order_recoveries": sum(len(s.get("order_attempts", [])) > 1 and s["status"] == "ok"
                                      for r in records.values() for s in r["subsets"]),
        "perturbation_order_recoveries": sum(len(d.get("order_attempts", [])) > 1 and d["status"] == "ok"
                                            for r in records.values() for d in r["perturbations"]),
        "replay_strict_tolerance_warnings": sum(not r["baseline"]["strict_prior_tolerance_pass"] for r in resolved_records),
        "maximum_replay_loss_deviation": max((r["baseline"]["max_replay_loss_deviation"] for r in resolved_records), default=None),
        "failed_subset_evaluations": sum(s["status"] != "ok" for r in records.values() for s in r["subsets"]),
        "failed_perturbation_records": sum(d["status"] != "ok" for r in records.values() for d in r["perturbations"])}
    atomic_json(folder / "summary.json", payload)
    atomic_json(folder / "unresolved_baselines.json", {k: records[k]["baseline"] for k in sorted(unresolved)})
    for name, data in (("endpoints", scalar_rows), ("memberships", memberships), ("paired_tests", tests),
                       ("winner_curves", winner_curves), ("pruning_sensitivity", sensitivities)):
        if data:
            write_csv(folder / f"{name}.csv", data)
        else:
            (folder / f"{name}.csv").write_text("")
    render(folder, payload, plan, catalog)
    text = ["# Logic-5 Functional Topology Families", "", "Post-hoc analysis of all four arms and 20 matched seeds. Original archives are unchanged; no fitting or model calls.", "",
        "## Portfolio Counts", "", "Mean counts per seed. Valid = frozen loss qualification AND all 16 terminal output margins >1e-6. Core = minimal output-preserving subset at epsilon0.01. Robust = at least 29/32 draws remain valid under the 10% rate protocol. Unresolved baselines are reported separately, not classified as invalid. Valid/core counts are observed lower bounds if unresolved cases remain; count tests are then suppressed.", "",
        "| Arm | Qualified | Valid | Distinct cores | Robust originals | Robust cores | Core families | Unresolved | Recovered |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        s = summary[arm]
        text.append(f"| {LABELS[arm]} | " + " | ".join(f"{s[k]:.2f}" for k in ("qualified", "valid", "cores", "robust_originals", "robust_cores", "core_families", "baseline_unresolved", "baseline_recovered")) + " |")
    text += ["", "## Definitions and Limits", "",
        "Pruning reproduces the prior Logic rule: all 32 subsets; fixed input template and surviving rates unchanged; loss below the same cutoff; all16 terminal correctness; max OUT deviation across 16x1000 observations <= epsilon. Epsilon0.01 is primary; 0.001 and0.05 are sensitivity checks. Reduced cores are diagnostics outside the exact-five-reaction search contract, not newly admitted results.", "",
        "Shared antithetic directions are indexed by library reaction ID: rates multiply by exp(log(1+p)*z), z in [-1,1], without clipping or refitting. Bounds may be exceeded in this sensitivity diagnostic. The original all16-valid portfolio and selected primary cores receive 32 draws at p=10%; rank-one winners also receive p=1,5,25,50% and one-at-a-time 10% probes. This is not the RPA-specific integral-balance test. Draws failing all recorded-order attempts remain nonpassing. Numerically successful draws are never retried to improve loss or classification. All attempt failures remain logged, including those followed by recovery.", "",
        "Original and core pass fractions use matched parameter directions. A smaller core has fewer varying parameters: improved core robustness is descriptive, not an equal-dimensional contest. Repeated candidate/rate vectors share computations but still belong to their original seed portfolios. A robust core is counted when at least one observed derived rate vector passes; no additional rate optimization is performed.", "",
        plan["replay_audit_note"], "",
        f"Replay audit: {payload['baseline_recovered']} baselines recovered using recorded order; {len(unresolved)} remain unresolved. Among resolved baselines, {payload['replay_strict_tolerance_warnings']} exceeded the older strict tolerance; maximum absolute loss difference {payload['maximum_replay_loss_deviation']}. All resolved baselines passed the documented solver floor and unchanged qualification/classification checks. Original endpoint scores remain unchanged.", "",
        "Families use complete labelled added-reaction structure, including inputs and multiplicity, minus the unchanged fixed template; never subtract bare IDs. Average linkage on reaction-set Jaccard distances uses cutoff0.60, with0.30/0.45/0.75 sensitivity. Species roles are retained, rates ignored. These are structural families, not proven mechanisms. Raw and pruned cluster systems are fit separately, so family numbers are not matched across the two systems. Family rarefaction at10 cores is undefined when a seed has fewer than10, not assigned zero.", "",
        "When no baseline is unresolved, paired inference uses 20 seed-level replicates, exact Pratt signed-rank tests and one Holm family of18 comparisons (six arm pairs x three count metrics). Otherwise all count contrasts are withheld. Bootstrap effect intervals resample seeds, not perturbation draws, and are marginal. Winner plots use only seeds whose winners resolved in all four arms, with that number stated. These conditional curves do not account for unresolved winners. The 29/32 robust threshold is an operational post-hoc label, not a confidence bound or guarantee of biological robustness.", "",
        f"Canonical calls: {payload['evaluations']}; failed subsets: {payload['failed_subset_evaluations']}; failed perturbation records: {payload['failed_perturbation_records']} (core copies of unchanged originals may count twice as records but not calls).", "",
        "## All Paired Count Tests", "", "| First vs second | Metric | Mean change | W/T/L | Holm p (18) |", "|---|---|---:|---:|---:|"]
    for t in tests:
        text.append(f"| {LABELS[t['a']]} vs {LABELS[t['b']]} | {t['metric']} | {t['mean_difference']:+.2f} | {t['wins']}/{t['ties']}/{t['losses']} | {t['holm_p_18']:.5g} |")
    (folder / "REPORT.md").write_text("\n".join(text)+"\n")
    return folder


def render(folder, payload, plan, catalog):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    pdf = PdfPages(folder / "LOGIC5_FUNCTIONAL_FAMILIES.pdf")
    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=16, ha="left", x=.04)
        if payload.get("baseline_unresolved", 0):
            note += f"\nUnresolved baseline jobs: {payload['baseline_unresolved']}; validated portfolio counts are lower bounds, not complete counts."
        fig.text(.04, .015, note, fontsize=9, va="bottom")
        fig.tight_layout(rect=(.02, .10, .99, .93), h_pad=2.5, w_pad=2.5)
        fig.savefig(folder / f"{name}.png", dpi=190)
        pdf.savefig(fig)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, key, title in zip(axes.flat, ("qualified", "valid", "cores", "robust_cores"),
            ("A  Loss-qualified structures", "B  Also correct on all 16 rows", "C  Distinct output-preserving cores", "D  Cores passing >=29/32 perturbations")):
        for i, arm in enumerate(ARMS):
            values = [r[key] for r in payload["rows"] if r["arm"] == arm]
            ax.bar(i, np.mean(values), color=COLORS[arm], alpha=.22, width=.58)
            ax.scatter(i+np.linspace(-.16,.16,20), values, color=COLORS[arm], s=18, alpha=.8)
        ax.set_xticks(range(4), ["None", "LLM to\nsearch", "Search to\nLLM", "Full"])
        ax.set(title=title, ylabel="Count per seed", ylim=(0,None))
        ax.grid(axis="y", alpha=.15)
    save(fig, "01_portfolio_counts", "Logic-5: from raw designs to functionally valid cores",
         "Bars: seed means; dots: 20 matched seeds. All four arms include a terminal union of search and LLM candidates.\nPruning preserves surviving rates and outputs within 0.01. Robustness label uses 10% multiplicative rate perturbations, without refitting.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    matched = sorted(set.intersection(*({r["seed"] for r in payload["winner_curves"] if r["arm"] == arm}
                                        for arm in ARMS)))
    bootstrap = np.random.default_rng(20260924).integers(0,len(matched),(10000,len(matched))) if matched else None
    for ax, metric, title in zip(axes.flat, ("qualified", "all16_correct", "valid", "failure_fraction"),
                                 ("A  Loss qualification", "B  All-16 classification", "C  Both conditions", "D  Failed evaluations")):
        for arm in ARMS:
            if not matched:
                continue
            matrix = np.array([[next(r[metric] for r in payload["winner_curves"] if r["arm"]==arm and
                r["seed"]==seed and r["radius"]==radius) for radius in plan["winner_radii"]] for seed in matched])
            ci = np.quantile(matrix[bootstrap].mean(axis=1), [.025,.975], axis=0)
            x = 100*(np.array(plan["winner_radii"])-1)
            ax.plot(x, matrix.mean(axis=0), "o-", color=COLORS[arm], label=LABELS[arm], markersize=4)
            ax.fill_between(x, ci[0], ci[1], color=COLORS[arm], alpha=.12)
        ax.set(title=title, xlabel="Multiplicative perturbation radius (%)", ylabel="Mean fraction of draws", ylim=(-.03,1.03))
        ax.grid(alpha=.15)
    if matched:
        axes[0,0].legend(fontsize=9, frameon=False)
    save(fig, "02_winner_robustness", "Rank-one winners: kinetic sensitivity across perturbation strengths",
         f"Matched resolved winners: {len(matched)}/20 seeds; 32 draws per radius. Shading: seed-bootstrap 95% intervals conditional on resolution.\nFailed-all-order draws remain failures. No clipping, refitting or replacement. Curves are not portfolio-wide robustness estimates.")
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    for i, arm in enumerate(ARMS):
        rows = [r for r in payload["rows"] if r["arm"]==arm]
        for j,key in enumerate(("raw_families","core_families","robust_core_families")):
            axes[0].bar(i+(j-1)*.22,np.mean([r[key] for r in rows]),width=.21,color=COLORS[arm],alpha=(.3,.6,1)[j])
    axes[0].set_xticks(range(4), ["None","LLM to\nsearch","Search to\nLLM","Full"])
    axes[0].set(ylabel="Mean structural families per seed", title="A  Valid / core / robust-core families")
    top=payload["families"][:12]
    coverage=np.array([[f["seed_coverage"][a] for f in top] for a in ARMS])
    if len(top):
        im=axes[1].imshow(coverage,aspect="auto",vmin=0,vmax=20,cmap="Blues")
        axes[1].set_xticks(range(len(top)),[f"F{f['family']}" for f in top],rotation=45)
        axes[1].set_yticks(range(4),[LABELS[a] for a in ARMS])
        fig.colorbar(im,ax=axes[1],label="Seeds containing family",shrink=.8)
    axes[1].set_title("B  Recurrence of the largest core families")
    save(fig,"03_families","Structural families after removing output-negligible reactions",
         "A: light = valid original structures; medium = pruned cores; dark = robust pruned cores.\nAverage-linkage Jaccard cutoff 0.60; fixed template excluded by full structure, rates ignored. Families are not proven biochemical mechanisms.")
    fig, axes=plt.subplots(2,3,figsize=(13,8))
    def complex_name(labels):
        counts=Counter(labels)
        return " + ".join((str(n)+" " if n>1 else "")+s for s,n in sorted(counts.items())) or "empty"
    for ax,family in zip(axes.flat,payload["families"][:6]):
        representative=family["representative"]; c,core=representative["candidate"],representative["core"]
        lines=[f"F{family['family']} | {family['unique_structures']} cores", ""]
        for i in core["retained_indices"]:
            s=catalog[str(c["reaction_ids"][i])]["structure"]
            lines.append(f"{complex_name(s['reactants'])} -> {complex_name(s['products'])}")
            lines.append(f"    k = {c['rates'][i]:.4g}")
        lines += ["",f"Loss: {core['loss']:.5g}","Seed coverage (none/inward/outward/full):",
                  "/".join(str(family["seed_coverage"][a]) for a in ARMS)]
        ax.text(0,1,"\n".join(lines),va="top",fontsize=9,linespacing=1.45,transform=ax.transAxes)
        ax.axis("off")
    for ax in axes.flat[len(payload["families"][:6]):]:
        ax.axis("off")
    save(fig,"04_representatives","Representative reduced circuits from the largest structural families",
         "Representatives are structural medoids, with their best observed derived rate vector; no fitting was performed here.\nOnly added reactions are shown. The four input-modulated fixed production reactions remain present in every simulation.")
    pdf.close()


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    analyze(parser.parse_args().output)
