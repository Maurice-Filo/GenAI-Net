"""Read-only post-hoc RPA portfolio, structural-family and cached-dynamics analysis."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import textwrap
import zlib

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln

from comparisons.search_exploration.analyze_rpa_statistical import COLORS

ARMS = ("julia", "evolution", "ga")
LABELS = {"julia": "AdaGrad hybrid", "evolution": "Evolution-fit hybrid", "ga": "Standalone GA"}
SHORT = ["AdaGrad\nhybrid", "Evolution-fit\nhybrid", "Standalone\nGA"]
SPECIES = ("X_1", "X_2", "X_3")


def structural_token(reaction):
    """Inputs and stoichiometric multiplicity are part of reaction identity."""
    s = reaction["structure"]
    return json.dumps({"type": s["type"], "reactants": sorted(s["reactants"]),
        "products": sorted(s["products"]), "inputs": s["inputs"]}, sort_keys=True)


def extra_structure(reactions, fixed):
    remaining = Counter(map(structural_token, reactions))
    for token in map(structural_token, fixed):
        if remaining[token] < 1:
            raise ValueError("Missing exact fixed-template structure")
        remaining[token] -= 1
    return tuple(sorted(remaining.elements()))


def set_jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def rarefied_richness(labels, k):
    """Exact expected families in a uniform sample without replacement."""
    n = len(labels)
    if n < k or k < 1:
        return np.nan
    sizes = np.array(list(Counter(labels).values()))
    missing = np.zeros(len(sizes))
    valid = n - sizes >= k
    m = n - sizes[valid]
    missing[valid] = np.exp(gammaln(m + 1) - gammaln(m-k+1)
        - gammaln(n+1) + gammaln(n-k+1))
    return float(np.sum(1-missing))


def stoichiometry(reaction):
    s = reaction["structure"]
    return np.array([s["products"].count(x)-s["reactants"].count(x) for x in SPECIES], float)


def integral_balance(reactions):
    """Test the restricted balance d(X1+a X2)/dt = u1 - gain X3 + residual."""
    groups = defaultdict(lambda: np.zeros(3))
    for r in reactions:
        s = r["structure"]
        if any(x is not None for x in s["inputs"]):
            continue
        groups[tuple(sorted(s["reactants"]))] += float(r["parameters"][0])*stoichiometry(r)
    other = np.array([v for k, v in groups.items() if k != ("X_3",)]).reshape(-1, 3)
    denom = float(other[:, 1] @ other[:, 1])
    alpha = -float(other[:, 0] @ other[:, 1])/denom if denom > 0 else 0.
    residual = other[:, 0] + alpha*other[:, 1]
    gain = -float(groups.get(("X_3",), np.zeros(3)) @ np.array([1., alpha, 0.]))
    scale = float(np.abs(other[:, 0]).sum()+abs(alpha)*np.abs(other[:, 1]).sum()+abs(gain))
    relative = float(np.abs(residual).sum()/max(scale, 1e-15))
    return {"alpha": alpha, "gain": gain, "relative_coefficient_residual": relative,
        "restricted_exact_balance": bool(gain > 0 and relative <= 1e-8),
        "restricted_near_unit_balance": bool(gain > 0 and abs(gain-1) <= .05 and relative <= .01)}


def cached_function(info):
    y = np.asarray(info["outputs"], float)[:, 0, :]
    x = np.asarray(info["trajectories"], float)
    u = np.asarray(info["inputs"], float)
    targets = np.asarray(info["setpoint"], float).ravel()
    sampling = info.get("diagnostic_sampling")
    tail_index = 800
    expected_samples = 1000
    if sampling is not None:
        indices = sampling.get("retained_indices", [])
        if (sampling.get("schema") != "diagnostic-time-stride-v1"
                or sampling.get("original_samples") != 1000
                or indices != sorted(set(indices))
                or not indices or indices[0] != 0 or indices[-1] != 999
                or 800 not in indices):
            raise ValueError("Downsampled diagnostics lack the original endpoint/drift samples")
        expected_samples = len(indices)
        tail_index = indices.index(800)
        if not np.allclose(info["time_horizon"], np.linspace(0, 100, 1000)[indices]):
            raise ValueError("Downsampled diagnostic times do not match retained indices")
    if not (y.shape == (9, expected_samples) and x.shape[0] == 9
            and x.shape[-1] == expected_samples and np.isfinite(x).all()):
        raise ValueError("Unexpected cached trajectory shape or nonfinite values")
    final = y[:, -1]
    disturbance = [np.ptp(final[u[:, 0] == value])/value for value in np.unique(u[:, 0])]
    # Change over the final 20% is observational, not a stability proof.
    drift = np.max(np.abs(x[:, :, -1]-x[:, :, tail_index])/np.maximum(1., np.abs(x[:, :, -1])))
    return {"max_final_relative_error": float(np.max(np.abs(final-targets)/targets)),
        "max_disturbance_spread": float(max(disturbance)), "max_state_tail_change": float(drift),
        "max_terminal_concentration": float(np.max(x[:, :, -1]))}


def dump_json(path, data):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [convert(v) for v in x]
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    path.write_text(json.dumps(convert(data), indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(root, output, plan):
    cache = output / "extracted.json.gz"
    plan_hash = hashlib.sha256((root / "active_plan.json").read_bytes()).hexdigest()
    if cache.exists():
        with gzip.open(cache, "rt") as stream:
            data = json.load(stream)
        if data["plan_sha256"] != plan_hash:
            raise ValueError("Extraction cache belongs to a different plan")
        return data["runs"]
    progress = json.loads((root / plan["progress_file"]).read_text())
    validated = json.loads((root / plan["analysis_dir"] / "summary.json").read_text())
    if not validated["inference_ready"] or any(j["status"] != "complete" for j in progress["jobs"]):
        raise ValueError("Requires all 60 completed, audited endpoints")
    expected = {(r["arm"], r["seed"]): r for r in validated["endpoints"]}
    runs = []
    for arm in ARMS:
        for job in sorted((j for j in plan["jobs"] if j["arm"] == arm), key=lambda j: j["seed"]):
            folder = Path(job.get("output", root / "runs" / job["name"]))
            with sqlite3.connect(f"file:{folder / 'exploration.sqlite'}?mode=ro", uri=True) as db:
                db.execute("PRAGMA temp_store=MEMORY")
                rows = db.execute("""SELECT payload FROM (
                    SELECT payload, row_number() OVER (
                        PARTITION BY json_extract(payload,'$.topology_hash')
                        ORDER BY json_extract(payload,'$.loss'), key) AS rank
                    FROM designs WHERE json_extract(payload,'$.simulation_complete')=1
                    AND json_type(payload,'$.loss') IN ('integer','real')
                    AND json_extract(payload,'$.loss') >= 0
                    AND json_extract(payload,'$.loss') < 0.1
                    ) WHERE rank=1""")
                records = sorted((json.loads(row[0]) for row in rows), key=lambda d: (d["loss"], d["candidate_hash"]))
            qualified = [d for d in records if d["loss"] < plan["threshold"]]
            ref = expected[arm, job["seed"]]
            if len(qualified) != ref["qualified_discovered"] or not np.isclose(records[0]["loss"], ref["loss"], rtol=1e-12):
                raise ValueError(f"Endpoint/count mismatch for {job['name']}")
            for d in records:
                if len(d["reaction_ids"]) != 5 or len(set(d["reaction_ids"])) != 5 or set(d["reaction_ids"]) & {0, 1, 28}:
                    raise ValueError("Reaction-count/template restriction violated")
                if not all(.001 <= k <= 100 for k in d["rates"]):
                    raise ValueError("Rate-bound violation")
            runs.append({"arm": arm, "seed": job["seed"], "folder": str(folder),
                "losses": [d["loss"] for d in records], "qualified": qualified, "best": records[0]})
            print("Extracted", job["name"], "qualified", len(qualified), flush=True)
    with gzip.open(cache, "wt") as stream:
        json.dump({"plan_sha256": plan_hash, "runs": runs}, stream)
    return runs


def reaction_catalog(root):
    from comparisons.rpa_search.src.common.rpa_task import build_rpa_components
    from RL4CRN.utils.results_database import serialize_crn
    crn, parts, _, _ = build_rpa_components(json.loads((root / "configs/task.json").read_text()))
    catalog = {}
    for rid in range(len(parts[0])):
        r = parts[0].get_reaction(rid)
        catalog[rid] = {"reaction_id": rid, "structure": {"type": type(r).__name__,
            "reactants": list(r.reactant_labels), "products": list(r.product_labels), "inputs": list(r.input_channels)}}
    fixed = json.loads(serialize_crn(crn)["parameters_json"])
    return catalog, fixed


def reactions_for(record, catalog, fixed):
    return fixed + [{**catalog[rid], "parameters": [rate]} for rid, rate in zip(record["reaction_ids"], record["rates"])]


def diagnostics(run, record):
    with sqlite3.connect(f"file:{run['folder']}/exploration.sqlite?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM diagnostics WHERE key=?", (record["evaluation_cache_key"],)).fetchone()
    return json.loads(zlib.decompress(row[0])) if row else None


def family_analysis(runs, catalog, fixed, output):
    by_hash = {}
    for run in runs:
        for record in run["qualified"]:
            signature = extra_structure(reactions_for(record, catalog, fixed), fixed)
            h = record["topology_hash"]
            if h in by_hash and by_hash[h] != signature:
                raise ValueError("Topology hash maps to conflicting structures")
            by_hash[h] = signature
    hashes = sorted(by_hash)
    if len(set(by_hash.values())) != len(hashes):
        raise ValueError("Different hashes collapse to identical labelled structures")
    tokens = sorted(set(t for s in by_hash.values() for t in s))
    token_index = {v: i for i, v in enumerate(tokens)}
    features = np.zeros((len(hashes), len(tokens)), dtype=bool)
    for i, h in enumerate(hashes):
        features[i, [token_index[t] for t in by_hash[h]]] = True
    print("Clustering", len(hashes), "distinct qualified structures", flush=True)
    condensed = pdist(features, metric="jaccard")
    tree = linkage(condensed, method="average")
    distances = squareform(condensed).astype(np.float32)
    index = {h: i for i, h in enumerate(hashes)}
    raw_labels = fcluster(tree, t=.6, criterion="distance")
    counts = Counter(raw_labels)
    families = {old: new+1 for new, old in enumerate(sorted(counts, key=lambda k: (-counts[k], int(k))))}
    labels = np.array([families[k] for k in raw_labels])
    sensitivity = []
    for threshold in (.3, .45, .6, .75):
        lab = fcluster(tree, t=threshold, criterion="distance")
        for arm in ARMS:
            subset = [r for r in runs if r["arm"] == arm]
            richness = [len({int(lab[index[d["topology_hash"]]]) for d in r["qualified"]}) for r in subset]
            sensitivity.append({"cut": threshold, "arm": arm, "global_families": len(set(lab)),
                "mean_families_per_seed": np.mean(richness)})
    rows = []
    for run in runs:
        for record in run["qualified"]:
            record["posthoc_balance"] = integral_balance(reactions_for(record,catalog,fixed))
        ix = [index[d["topology_hash"]] for d in run["qualified"]]
        run["indices"] = ix
        lab = labels[ix]
        sub = distances[np.ix_(ix, ix)]
        p = np.array(list(Counter(lab).values()), float)
        p = p/p.sum() if len(p) else p
        rows.append({"arm": run["arm"], "seed": run["seed"], "best_loss": run["best"]["loss"],
            "qualified_topologies": len(ix), "families": len(p),
            "effective_families": float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
            "mean_pairwise_jaccard_distance": float(sub[np.triu_indices(len(ix), 1)].mean()) if len(ix)>1 else np.nan,
            "rarefied_families_k10": rarefied_richness(lab, 10),
            "median_rate": float(np.median([k for d in run["qualified"] for k in d["rates"]])) if ix else np.nan,
            "upper_bound_rate_fraction": np.mean([k >= 100-1e-6 for d in run["qualified"] for k in d["rates"]]) if ix else np.nan})
        rows[-1]["exact_balance_fraction"] = np.mean([d["posthoc_balance"]["restricted_exact_balance"] for d in run["qualified"]]) if ix else np.nan
        rows[-1]["near_unit_balance_fraction"] = np.mean([d["posthoc_balance"]["restricted_near_unit_balance"] for d in run["qualified"]]) if ix else np.nan
    exact = np.full((len(runs), len(runs)), np.nan)
    nearest = exact.copy()
    for i, a in enumerate(runs):
        ai = a["indices"]
        for j, b in enumerate(runs):
            bi = b["indices"]
            if ai and bi:
                exact[i, j] = set_jaccard(set(ai), set(bi))
                d = distances[np.ix_(ai, bi)]
                nearest[i, j] = 1 - .5*(d.min(axis=0).mean()+d.min(axis=1).mean())
    write_csv(output / "per_run.csv", rows)
    write_csv(output / "cluster_cut_sensitivity.csv", sensitivity)
    names = [f"{r['arm']}-{r['seed']:02d}" for r in runs]
    for name, matrix in (("exact_overlap", exact), ("nearest_structure_similarity", nearest)):
        write_csv(output / (name+".csv"), [{"run": names[i], **dict(zip(names, row))} for i, row in enumerate(matrix)])
    entries = [{"topology_hash": h, "family": int(labels[i]), "reaction_structure": list(by_hash[h])} for i, h in enumerate(hashes)]
    dump_json(output / "families.json", entries)
    np.savez_compressed(output / "clustering.npz", linkage=tree, labels=labels, topology_hashes=hashes)
    return hashes, labels, distances, rows, exact, nearest, sensitivity


def equation(r, latex=False):
    def side(items):
        if not items:
            return r"\varnothing" if latex else "empty"
        counts = Counter(items)
        return " + ".join((str(n) if n>1 else "")+(s.replace("_", "_{")+"}" if latex else s) for s,n in sorted(counts.items()))
    s = r["structure"]
    return side(s["reactants"]) + (r" \rightarrow " if latex else " -> ") + side(s["products"])


def render(root, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "pdf.fonttype": 42, "savefig.dpi": 190, "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "active_plan.json").read_text())
    runs = extract(root, output, plan)
    catalog, fixed = reaction_catalog(root)
    hashes, labels, distances, rows, exact, nearest, sensitivity = family_analysis(runs, catalog, fixed, output)
    pdf = PdfPages(output / "RPA_PORTFOLIO_ATLAS.pdf")
    figures = []
    def save(fig, name, caption):
        wrapped = textwrap.fill(caption, 160)
        footer = .025 + .16*(wrapped.count("\n")+1)/fig.get_figheight()
        fig.get_layout_engine().set(rect=(0, footer, 1, 1-footer))
        fig.text(.025,.012,wrapped,fontsize=8,va="bottom",linespacing=1.3)
        fig.savefig(output / (name+".png"), bbox_inches="tight")
        fig.savefig(output / (name+".pdf"), bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        figures.append((name, caption))
        plt.close(fig)
    def dots(ax, key, ylabel, log=False, seeds=None):
        rng = np.random.default_rng(47)
        for i, arm in enumerate(ARMS):
            values = np.array([r[key] for r in rows if r["arm"] == arm and (seeds is None or r["seed"] in seeds)], float)
            values = values[np.isfinite(values)]
            jitter = rng.uniform(-.15,.15,len(values)) if seeds is None else np.zeros(len(values))
            ax.scatter(i+jitter, values, color=COLORS[arm], s=25, alpha=.75)
            if len(values):
                ax.hlines(np.median(values), i-.23, i+.23, color=COLORS[arm], lw=3)
            ax.text(i, 1.01, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set_xticks(range(3), SHORT)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=.15)
    legend = [Line2D([0],[0],color=COLORS[a],lw=2,label=LABELS[a]) for a in ARMS]
    fig, axes = plt.subplots(1,3,figsize=(13,4.6), constrained_layout=True)
    dots(axes[0], "best_loss", "Best canonical loss", True)
    axes[0].axhline(plan["threshold"],ls="--",color=".4",lw=1)
    dots(axes[1], "qualified_topologies", "Qualifying topologies discovered")
    thresholds = np.unique(np.r_[np.geomspace(.0002,.1,200), plan["threshold"]])
    curves = []
    for arm in ARMS:
        values = np.array([np.searchsorted(r["losses"],thresholds,side="left") for r in runs if r["arm"]==arm])
        mean, sd = values.mean(0), values.std(0,ddof=1)
        axes[2].plot(thresholds,mean,color=COLORS[arm])
        axes[2].fill_between(thresholds,np.maximum(0,mean-sd),mean+sd,color=COLORS[arm],alpha=.12)
        curves.extend({"arm":arm,"threshold":t,"mean_count":m,"sample_sd":s} for t,m,s in zip(thresholds,mean,sd))
    axes[2].axvline(plan["threshold"],ls="--",color=".4",lw=1)
    axes[2].set(xscale="log",yscale="symlog",xlabel="Loss threshold",ylabel="Mean qualifying topology count")
    axes[2].legend(handles=legend,fontsize=8,loc="upper left",frameon=False)
    for ax,title in zip(axes,("A  Endpoint quality", "B  Useful design count", "C  Threshold sensitivity")):
        ax.set_title(title,loc="left",pad=24)
    save(fig,"01_quality", "Twenty matched seeds per workflow. Dots are seeds, bars medians. Dashed lines mark the frozen threshold. Shading is one sample SD across seeds, not a confidence interval or significance test. Counts include historical discoveries, not only retained elites.")
    write_csv(output / "threshold_curves.csv",curves)

    fig, axes = plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    dots(axes[0],"mean_pairwise_jaccard_distance","Mean reaction-set Jaccard distance")
    dots(axes[1],"families","Structural families per seed")
    matched_rarefaction = {seed for seed in plan["seeds"] if all(
        any(r["arm"]==arm and r["seed"]==seed and np.isfinite(r["rarefied_families_k10"]) for r in rows) for arm in ARMS)}
    for seed in sorted(matched_rarefaction):
        axes[2].plot(range(3),[next(r["rarefied_families_k10"] for r in rows if r["arm"]==a and r["seed"]==seed) for a in ARMS],color=".85",lw=.6,zorder=0)
    dots(axes[2],"rarefied_families_k10","Expected families in 10 qualifying designs",seeds=matched_rarefaction)
    for ax,title in zip(axes,("A  Within-run separation", "B  Family coverage", "C  Equal-size portfolios")):
        ax.set_title(title,loc="left",pad=24)
    save(fig,"02_within_run_diversity", "Families use average-linkage Jaccard clustering cut at 0.60. A requires at least two qualifying structures. B includes all 20 seeds, with zero families for unsuccessful seeds. C uses only matched seeds with at least ten qualifying designs in all three methods; gray lines join seeds. Rarefaction is the exact expectation without replacement and controls portfolio size, not search effort. Missing seeds are not zero diversity.")

    fig, axes = plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    for ax,matrix,title in zip(axes,(exact,nearest),("A  Exact topology overlap", "B  Nearest structural similarity")):
        cmap=plt.get_cmap("viridis").copy(); cmap.set_bad("#e2e2e2")
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap=cmap,interpolation="nearest")
        ax.set_xticks([9.5,29.5,49.5],SHORT); ax.set_yticks([9.5,29.5,49.5],[LABELS[a] for a in ARMS])
        for v in (19.5,39.5):
            ax.axhline(v,color="white",lw=1); ax.axvline(v,color="white",lw=1)
        ax.set_title(title,loc="left"); fig.colorbar(im,ax=ax,shrink=.8,label="Similarity")
    save(fig,"03_between_run_similarity", "Each row/column is one seed, ordered 1-20 within method. A: Jaccard similarity between discovered topology sets. B: symmetric mean of nearest reaction-set matches between the two portfolios. Gray means at least one run has no qualified structures; empty-empty is not perfect agreement. The diagonal is self-comparison. Pairwise cells are dependent and are not independent statistical replicates.")

    family_rows=[]
    for family in sorted(set(labels)):
        members=np.flatnonzero(labels==family)
        row={"family":int(family),"distinct_topologies":len(members)}
        for arm in ARMS:
            counts=[sum(labels[ix]==family for ix in r["indices"]) for r in runs if r["arm"]==arm]
            row[arm+"_seed_prevalence"]=sum(n>0 for n in counts)
            row[arm+"_mean_topologies"]=float(np.mean(counts))
        family_rows.append(row)
    family_rows.sort(key=lambda r: (-sum(r[a+"_seed_prevalence"] for a in ARMS),-r["distinct_topologies"]))
    write_csv(output / "family_summary.csv",family_rows)
    top=family_rows[:12]
    fig,axes=plt.subplots(1,3,figsize=(13,5.6),constrained_layout=True)
    prevalence=np.array([[r[a+"_seed_prevalence"] for a in ARMS] for r in top])
    im=axes[0].imshow(prevalence,vmin=0,vmax=20,cmap="YlGnBu",aspect="auto")
    axes[0].set_yticks(range(len(top)),[f"F{r['family']} ({r['distinct_topologies']} nets)" for r in top]); axes[0].set_xticks(range(3),SHORT)
    for i in range(len(top)):
        for j in range(3): axes[0].text(j,i,str(prevalence[i,j]),ha="center",va="center",color="white" if prevalence[i,j]>12 else "black",fontsize=9)
    axes[0].set_title("A  Recurring families: seeds / 20",loc="left")
    for arm in ARMS:
        s=[r for r in sensitivity if r["arm"]==arm]
        axes[1].plot([r["cut"] for r in s],[r["mean_families_per_seed"] for r in s],"o-",color=COLORS[arm],label=LABELS[arm])
        rr=[r for r in rows if r["arm"]==arm]
        axes[2].scatter([r["qualified_topologies"] for r in rr],[r["best_loss"] for r in rr],color=COLORS[arm],alpha=.8,s=28)
    axes[1].set(xlabel="Average-linkage distance cut",ylabel="Mean families per seed",title="B  Clustering sensitivity")
    axes[1].axvline(.6,color=".5",ls="--",lw=1); axes[1].legend(fontsize=8,frameon=False)
    axes[2].set(xlabel="Qualifying topologies",ylabel="Best loss",yscale="log",title="C  Quality versus breadth")
    save(fig,"04_structural_families", "Families cluster distinct labelled five-reaction structures, counting each global topology once. F labels are size-ranked; displayed families are ranked by total seed prevalence. Repeated discovery is measured across seeds, not repeated parameter fits. Clusters are structural groupings, not validated biochemical mechanisms. Panel B exposes sensitivity to the arbitrary post-hoc cut; panel C shows both quality and breadth.")

    reaction_ids=sorted(set(rid for run in runs for d in run["qualified"] for rid in d["reaction_ids"]))
    usage={}
    for arm in ARMS:
        rr=[r for r in runs if r["arm"]==arm and r["qualified"]]
        usage[arm]=np.array([[sum(rid in d["reaction_ids"] for d in r["qualified"])/len(r["qualified"]) for rid in reaction_ids] for r in rr]).mean(0)
    chosen=np.argsort(np.mean([usage[a] for a in ARMS],axis=0))[-16:][::-1]
    fig,axes=plt.subplots(1,3,figsize=(14,6),constrained_layout=True,gridspec_kw={"width_ratios":[1.6,1,1]})
    matrix=np.array([usage[a][chosen] for a in ARMS]).T
    im=axes[0].imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    axes[0].set_yticks(range(len(chosen)),[f"{reaction_ids[i]}: $"+equation(catalog[reaction_ids[i]],True)+"$" for i in chosen],fontsize=8)
    axes[0].set_xticks(range(3),SHORT); fig.colorbar(im,ax=axes[0],shrink=.6,label="Mean within-run frequency")
    dots(axes[1],"median_rate","Median rate among qualifying designs",True)
    dots(axes[2],"upper_bound_rate_fraction","Fraction of rates at upper bound")
    for ax,title in zip(axes,("A  Common added reactions", "B  Rate scale", "C  Bound saturation")):
        ax.set_title(title,loc="left",pad=24)
    save(fig,"05_reactions_rates", "Reaction usage is averaged within each successful seed, then across successful seeds, so large archives do not dominate. Failure seeds are omitted here, explicitly 20/20/13 successful seeds for AdaGrad/evolution/GA. Fixed input-modulated reactions are excluded by full structural identity. Each topology uses its best recorded parameterization. Bound saturation is not evidence that a reaction is functionally irrelevant.")
    write_csv(output / "reaction_usage.csv",[{"reaction_id":rid,"equation":equation(catalog[rid]),**{a:usage[a][i] for a in ARMS}} for i,rid in enumerate(reaction_ids)])

    functional=[]
    for run in runs:
        info=diagnostics(run,run["best"])
        if info is None:
            functional.append({"arm":run["arm"],"seed":run["seed"],"loss":run["best"]["loss"],"cached":False})
        else:
            functional.append({"arm":run["arm"],"seed":run["seed"],"loss":run["best"]["loss"],"cached":True,**cached_function(info)})
    dump_json(output / "best_network_function.json",functional)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    rng=np.random.default_rng(81)
    for ax,key,title in zip(axes,("max_final_relative_error","max_disturbance_spread","max_state_tail_change"),
            ("A  Final tracking error", "B  Disturbance sensitivity", "C  Hidden-state drift")):
        for i,arm in enumerate(ARMS):
            vals=[d[key] for d in functional if d["arm"]==arm and d["cached"]]
            ax.scatter(i+rng.uniform(-.15,.15,len(vals)),vals,color=COLORS[arm],s=25,alpha=.8)
            ax.hlines(np.median(vals),i-.23,i+.23,color=COLORS[arm],lw=3)
        ax.set_xticks(range(3),SHORT); ax.set_yscale("symlog",linthresh=1e-5); ax.set_title(title,loc="left")
    axes[0].set_ylabel("Worst-case |OUT - target| / target")
    axes[1].set_ylabel("Max OUT range over u2 / u1")
    axes[2].set_ylabel("Max |x(100)-x(80)| / max(1, |x(100)|)")
    save(fig,"06_cached_function", "Best network per seed, including GA failures. Cached trajectories only; no re-simulation. Final error and disturbance spread use the nine training scenarios. Drift considers all species, not just output. Small drift is not a proof of equilibrium or stability; low training-grid error is not held-out robustness. Symmetric-log axes retain exact zeros. Missing cached diagnostics are recorded, never recomputed silently.")

    motif=[]
    for run in runs:
        designs=run["qualified"]
        motif.append({"arm":run["arm"],"seed":run["seed"],"qualified":len(designs),
            "exchange_pair_fraction":np.mean([{51,80} <= set(d["reaction_ids"]) for d in designs]) if designs else np.nan,
            "best_has_exchange_pair":{51,80} <= set(run["best"]["reaction_ids"])})
    write_csv(output / "exchange_motif.csv",motif)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    rng=np.random.default_rng(41)
    for i,arm in enumerate(ARMS):
        m=[d["exchange_pair_fraction"] for d in motif if d["arm"]==arm and d["qualified"]]
        axes[0].scatter(i+rng.uniform(-.15,.15,len(m)),m,color=COLORS[arm],s=25)
        axes[0].hlines(np.mean(m),i-.23,i+.23,color=COLORS[arm],lw=3)
        axes[0].text(i,1.01,f"n={len(m)}",transform=axes[0].get_xaxis_transform(),ha="center",fontsize=9)
        ff=[d for d in functional if d["arm"]==arm and d["cached"]]
        for qualified,marker in ((True,"o"),(False,"x")):
            selected=[d for d in ff if (d["loss"]<plan["threshold"])==qualified]
            axes[1].scatter([d["max_state_tail_change"] for d in selected],[d["max_final_relative_error"] for d in selected],color=COLORS[arm],marker=marker,s=28)
        axes[2].scatter(i+rng.uniform(-.15,.15,len(ff)),[d["max_terminal_concentration"] for d in ff],color=COLORS[arm],s=25)
    axes[0].set_xticks(range(3),SHORT); axes[0].set_ylabel("Fraction containing reactions 51 and 80")
    axes[0].set_title("A  Recurrent exchange motif",loc="left",pad=24)
    axes[1].set(xlabel="All-state relative tail change",ylabel="Worst final relative output error",xscale="symlog",yscale="symlog")
    axes[1].set_xscale("symlog",linthresh=1e-5); axes[1].set_yscale("symlog",linthresh=1e-5)
    axes[1].set_title("B  Output accuracy is not state boundedness",loc="left",fontsize=10,pad=24)
    axes[1].legend(handles=legend,fontsize=8,frameon=False)
    axes[2].set_xticks(range(3),SHORT); axes[2].set_yscale("log"); axes[2].set_ylabel("Largest terminal species concentration")
    axes[2].set_title("C  Internal concentration scale",loc="left",pad=24)
    save(fig,"07_mechanistic_cautions", "The exchange pair is X1+X2 -> X1+X3 and X2+X3 -> 2X2. Panel A averages within successful seeds; bars are means. B-C use each seed's best network; crosses in B indicate failure of the frozen loss threshold. A small output error can coexist with internal accumulation. The observed finite tail change does not prove unbounded growth; longer-horizon and perturbation tests remain necessary.")

    fig,axes=plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    dots(axes[0],"exact_balance_fraction","Fraction with restricted exact integral balance")
    dots(axes[1],"near_unit_balance_fraction","Fraction with near-unit integral balance")
    axes[0].set_title("A  Algebraic cancellation",loc="left",pad=24)
    axes[1].set_title("B  Approximate reference tracking",loc="left",pad=24)
    save(fig,"08_integral_balance", "For each best-per-topology parameterization, fit d(X1+a X2)/dt = u1-g X3 plus residual polynomial terms. A requires g>0 and relative coefficient residual <=1e-8. B permits residual <=0.01 and |g-1|<=0.05. Fractions are within successful seeds; bars are medians. These diagnostics are not mutually exclusive with exchange motifs. Algebraic balance alone does not prove convergence, stability or biological robustness.")
    cases=[(entry,None) for entry in family_rows[:6]]
    exact_candidates=sorted([(r,d) for r in runs for d in r["qualified"] if d["posthoc_balance"]["restricted_exact_balance"]],key=lambda pair:pair[1]["loss"])
    for run,record in exact_candidates:
        info=diagnostics(run,record)
        if info is not None:
            family=int(labels[hashes.index(record["topology_hash"])])
            cases.append(({"family":family},(run,record,info)))
            break
    representatives=[]
    for entry,forced in cases:
        family=entry["family"]; members=np.flatnonzero(labels==family)
        ranked_members=members[np.argsort(distances[np.ix_(members,members)].mean(1),kind="stable")]
        medoid=int(ranked_members[0])
        if forced is not None:
            run,record,info=forced
            selection="best cached exact-balance example"
        else:
            info=None
            # Prefer the medoid; otherwise the nearest-central structure with saved dynamics.
            for selected_index in ranked_members:
                instances=sorted([(r,d) for r in runs for d in r["qualified"] if d["topology_hash"]==hashes[selected_index]],key=lambda pair:pair[1]["loss"])
                for run,record in instances:
                    info=diagnostics(run,record)
                    if info is not None:
                        break
                if info is not None:
                    break
            selection="medoid" if selected_index==medoid else "nearest-central cached structure"
        reactions=reactions_for(record,catalog,fixed)
        balance=integral_balance(reactions)
        representative={"family":family,"run":f"{run['arm']}-{run['seed']}","record":record,"reactions":reactions,
            "integral_balance":balance,"function":cached_function(info) if info else None,"cache_available":info is not None,
            "selection":selection,"medoid_topology_hash":hashes[medoid]}
        representatives.append(representative)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        fig.suptitle(f"Family F{family}: {selection} | {LABELS[run['arm']]} seed {run['seed']} | loss {record['loss']:.4g}",fontsize=12)
        axes[0,0].axis("off")
        for i,r in enumerate(reactions):
            channel="".join(str(v) for v in r["structure"]["inputs"] if v is not None)
            text="$"+equation(r,True)+"$"+f"     k={r['parameters'][0]:.4g}"+(f" x {channel}" if channel else "")
            axes[0,0].text(.02,.91-i*.115,text,transform=axes[0,0].transAxes,fontsize=11)
        axes[0,0].set_title("A  Mass-action reactions (first two fixed)",loc="left")
        if info is not None:
            from RL4CRN.search_exploration.evaluation import build, materialize
            from RL4CRN.utils.results_database import serialize_crn
            evaluator,_,_=build("rpa",json.loads((root/"configs/task.json").read_text()),.001,100)
            state=materialize(evaluator,record).state
            if serialize_crn(state)["topology_hash"] != record["topology_hash"]:
                raise ValueError("Representative reconstruction does not match saved topology")
            order=[state.species_labels.index(s) for s in SPECIES]
            t=np.asarray(info["time_horizon"]); u=np.asarray(info["inputs"])
            y=np.asarray(info["outputs"])[:,0,:]; x=np.asarray(info["trajectories"])[:,order,:]
            palette=["#176f97","#d68919","#8c4799"]; styles=["-","--",":"]
            for case,(u1,u2) in enumerate(u):
                axes[0,1].plot(t,y[case],color=palette[list(np.unique(u[:,0])).index(u1)],ls=styles[list(np.unique(u[:,1])).index(u2)],lw=1.2)
            for val in np.unique(u[:,0]): axes[0,1].axhline(val,color=".65",lw=.7,alpha=.5)
            axes[0,1].set(xlabel="Time",ylabel="OUT",title="B  All nine cached output trajectories")
            axes[0,1].legend(handles=[Line2D([0],[0],color=c,label=f"u1={v}") for c,v in zip(palette,np.unique(u[:,0]))]+[Line2D([0],[0],color=".35",ls=s,label=f"u2={v}") for s,v in zip(styles,np.unique(u[:,1]))],fontsize=8,ncol=3,frameon=False)
            nominal=int(np.flatnonzero(np.all(u==1,axis=1))[0])
            for k,species in enumerate(SPECIES): axes[1,0].plot(t,x[nominal,k],label=species,color=palette[k])
            axes[1,0].set(xlabel="Time",ylabel="Concentration",yscale="log",title="C  Internal states at u1=u2=1")
            axes[1,0].legend(fontsize=9,frameon=False)
            flux=[]
            for reaction in reactions:
                s=reaction["structure"]
                v=np.full(9,reaction["parameters"][0])
                for species in s["reactants"]: v*=x[:,SPECIES.index(species),-1]
                for channel in s["inputs"]:
                    if channel is not None: v*=u[:,["u_1","u_2"].index(channel)]
                flux.append(v)
            flux=np.asarray(flux).T
            d_out=np.array([stoichiometry(r)[2] for r in reactions])
            production=flux@np.maximum(d_out,0); consumption=flux@np.maximum(-d_out,0)
            xp=np.arange(9)
            axes[1,1].bar(xp-.18,production,width=.36,color="#168577",label="Production")
            axes[1,1].bar(xp+.18,consumption,width=.36,color="#bd5b71",label="Consumption")
            axes[1,1].set_xticks(xp,[f"{a:g}\n{b:g}" for a,b in u],fontsize=8)
            axes[1,1].set(xlabel="u1 (top), u2 (bottom)",ylabel="Terminal OUT flux",title="D  Terminal production and consumption")
            axes[1,1].legend(fontsize=9,frameon=False)
            representative["terminal_fluxes"]=flux.tolist()
            representative["terminal_state_order"]=list(SPECIES)
            representative["terminal_states"]=x[:,:,-1].tolist()
            representative["output_production"]=production.tolist()
            representative["output_consumption"]=consumption.tolist()
        else:
            for ax in (axes[0,1],axes[1,0],axes[1,1]): ax.text(.5,.5,"No cached diagnostics",ha="center",va="center",transform=ax.transAxes)
        if forced is None:
            save(fig,f"family_{family:03d}", "Selection: the structural medoid if cached dynamics exist, otherwise the structure minimizing mean distance to its family among those with cached dynamics; best cached parameterization of that structure. Not the family's best-loss network. Reactions and terminal mass-action fluxes are reconstructed without integrating ODEs. These are training-grid dynamics, not perturbation or deletion experiments.")
        else:
            save(fig,"integral_balance_example", f"Illustrative selection: lowest-loss qualifying candidate with both an exact restricted integral balance and cached trajectories. Not a random sample or a family medoid. Here a={balance['alpha']:.6g}, g={balance['gain']:.6g}, so d(X1+a X2)/dt=u1-g OUT within the stated numerical coefficient tolerance. A steady integral coordinate implies OUT=u1/g; the balance alone does not establish stability or held-out robustness. No new simulation was run.")
    dump_json(output / "representatives.json",representatives)
    method_summary=[]
    for arm in ARMS:
        rr=[r for r in rows if r["arm"]==arm]
        method_summary.append({"arm":arm,"successes":sum(r["qualified_topologies"]>0 for r in rr),
            "median_best_loss":np.median([r["best_loss"] for r in rr]),
            "mean_qualified":np.mean([r["qualified_topologies"] for r in rr]),
            "mean_families":np.mean([r["families"] for r in rr]),
            "mean_pairwise_distance_successful":np.nanmean([r["mean_pairwise_jaccard_distance"] for r in rr]),
            "mean_rarefied_families_k10":np.nanmean([r["rarefied_families_k10"] for r in rr]),
            "mean_rarefied_families_matched":np.mean([r["rarefied_families_k10"] for r in rr if r["seed"] in matched_rarefaction]),
            "mean_exact_balance_fraction_successful":np.nanmean([r["exact_balance_fraction"] for r in rr]),
            "rarefaction_eligible_seeds":sum(np.isfinite(r["rarefied_families_k10"]) for r in rr)})
    summary={"post_hoc":True,"new_simulations":0,"new_model_calls":0,"threshold":plan["threshold"],
        "unique_qualified_topologies":len(hashes),"families_at_cut_0_6":len(set(labels)),"methods":method_summary,
        "diagnostics_available":sum(d["cached"] for d in functional),"matched_rarefaction_seeds":sorted(matched_rarefaction),"figures":figures}
    summary["functional_summary"]=[{"arm":arm,
        "mean_exchange_pair_fraction_successful":np.nanmean([m["exchange_pair_fraction"] for m in motif if m["arm"]==arm]),
        "best_network_tail_change_above_1pct":sum(d["max_state_tail_change"]>.01 for d in functional if d["arm"]==arm and d["cached"]),
        "median_max_final_relative_error":np.median([d["max_final_relative_error"] for d in functional if d["arm"]==arm and d["cached"]])} for arm in ARMS]
    dump_json(output / "summary.json",summary)
    notes=["All analyses are post hoc and descriptive. The independent unit is the seed, not a topology or pairwise matrix cell.",
        "Only the 60 completed final attempts enter the analysis. Failed attempts and development seed 0 are excluded; their recovery overhead remains in the primary report.",
        "A topology is counted once per run, using its best complete recorded candidate. Ledger discoveries include evicted candidates but not optimizer trials never submitted as designs. Reaction order and rates are not structural identity; input channels and repeated stoichiometric labels are.",
        "Species labels are retained: X1 receives the reference input and X3 is the output with disturbance-modulated degradation. No arbitrary species permutation is claimed to preserve the task.",
        "Clustering uses all distinct qualified structures, no sampling: average linkage on reaction-set Jaccard distance. The 0.60 cut and its sensitivity grid are exploratory. Cluster count and effective family count are not numbers of independent biochemical mechanisms.",
        "Higher topology count need not imply larger pairwise distance. Rarefaction controls portfolio size conditionally on having at least ten qualifying designs; it excludes unsuccessful seeds and must be read alongside success rates.",
        "Function panels reuse cached trajectories. This task tests constant u1/u2 combinations from one initial condition, not a within-trajectory disturbance step or held-out conditions. Finite-horizon tracking is not proof of robust perfect adaptation.",
        "The restricted balance diagnostic asks whether d(X1 + a X2)/dt = u1 - g X3, up to residual polynomial coefficients. Even an exact balance requires convergence and stability to imply adaptation; absence does not exclude nonlinear feedback or other integral coordinates.",
        "Flux balance and recurring reactions identify hypotheses, not causal mechanisms. Reaction-removal tests, parameter perturbations, longer integration and new input values would be separate experiments. No such experiments were performed.",
        "This is a workflow comparison, not an isolated RL effect: archive management, fitting, and consumed canonical/joint computation differ. No new significance tests are inferred from the descriptive plots."]
    lines=["# RPA Portfolio and Mechanism Atlas", "", "60 completed runs, 20 matched seeds per workflow. No new simulations or model calls.","",
        f"Frozen threshold: {plan['threshold']:.12g}. Global qualified structures: {len(hashes)}; structural families at cut 0.60: {len(set(labels))}.","",
        "| Workflow | Successes | Median best loss | Mean qualified | Mean families | Mean within-run distance | Rarefied families (n eligible) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in method_summary:
        lines.append(f"| {LABELS[s['arm']]} | {s['successes']}/20 | {s['median_best_loss']:.5g} | {s['mean_qualified']:.2f} | {s['mean_families']:.2f} | {s['mean_pairwise_distance_successful']:.3f} | {s['mean_rarefied_families_k10']:.2f} ({s['rarefaction_eligible_seeds']}) |")
    lines += ["", "## Main Observations", "",
        "Both hybrids show greater within-run structural separation and broader family coverage than GA descriptively. The same ordering remains under equal-size rarefaction, although its eligible-seed subset excludes failures and very small portfolios. AdaGrad has fewer qualifying structures than evolution fitting on average, but more families at the displayed cut: raw topology count is not the whole diversity story.", "",
        "A recurrent pair is X1+X2 -> X1+X3 (rate k51) and X2+X3 -> 2X2 (rate k80). Its contribution to output dynamics is X2*(k51*X1 - k80*X3). When this exchange dominates and X2 is large, it tends to enforce X3 approximately (k51/k80)*X1. The explicit disturbance removal term -u2*X3 is then relatively weak. This is a mechanistic hypothesis from the executable mass-action equations, not a reaction-deletion result or proof of robust perfect adaptation.", "",
        "In some representatives the output is nearly constant while X2 continues to accumulate. Increasing internal abundance can strengthen exchange and suppress disturbance effects over the tested window without convergence of the whole state. Therefore low finite-horizon loss must not be equated with a bounded steady-state biochemical controller. Other networks do settle; the finding is not universal.", ""]
    lines += ["The portfolios also include networks with a restricted exact integral balance. Within successful seeds, mean fractions of qualifying structures with this balance are: " + "; ".join(f"{LABELS[s['arm']]} {s['mean_exact_balance_fraction_successful']:.1%}" for s in method_summary) +
        ". These use best-per-topology rates; exact means relative polynomial-coefficient residual <=1e-8. The dedicated integral-balance example is selected by lowest loss among candidates with cached dynamics and is not representative of frequency. A balance yielding OUT=u1/g at equilibrium is stronger algebraic evidence than reaction overlap, but still needs convergence and stability for an adaptation guarantee.", ""]
    for fs in summary["functional_summary"]:
        lines.append(f"- {LABELS[fs['arm']]}: mean exchange-pair frequency among qualifying structures within successful seeds {fs['mean_exchange_pair_fraction_successful']:.1%}; best-network all-state tail change above 1% in {fs['best_network_tail_change_above_1pct']}/20 seeds; median worst-case final relative output error {fs['median_max_final_relative_error']:.3g}.")
    lines += ["", f"Matched rarefaction uses seeds {sorted(matched_rarefaction)}. Mean expected families among ten qualifying structures: " +
        "; ".join(f"{LABELS[s['arm']]} {s['mean_rarefied_families_matched']:.2f}" for s in method_summary) + ". The table above also reports each method's full eligible subset; panel C deliberately uses the matched subset.", ""]
    lines += ["", "## Figures", "", "[Combined PDF atlas](RPA_PORTFOLIO_ATLAS.pdf)", ""]
    for name,caption in figures:
        lines += [f"### {name}","",f"![{name}]({name}.png)","",caption,""]
    lines += ["## Representative Balance Diagnostics", ""]
    for r in representatives:
        b=r["integral_balance"]; f=r["function"]
        lines += [f"- F{r['family']}, {r['run']}: restricted integral coordinate a={b['alpha']:.4g}, g={b['gain']:.4g}, relative coefficient residual={b['relative_coefficient_residual']:.3g}; exact balance: {b['restricted_exact_balance']}. " +
            (f"Worst endpoint relative error={f['max_final_relative_error']:.3g}; disturbance spread={f['max_disturbance_spread']:.3g}; all-state tail change={f['max_state_tail_change']:.3g}." if f else "No cached dynamics.")]
    lines += ["", "## Definitions and Limits", ""]+[f"- {n}" for n in notes]
    lines += ["", "## Reproduce", "", "```bash", f"MPLCONFIGDIR=/local0/tmp/mpl .venv/bin/python -m comparisons.search_exploration.analyze_rpa_mechanisms {root} --output {output}","```", "",
        "Data: per_run.csv, threshold_curves.csv, family_summary.csv, families.json, cluster_cut_sensitivity.csv, exact_overlap.csv, nearest_structure_similarity.csv, reaction_usage.csv, best_network_function.json, representatives.json. The extraction cache is keyed to the frozen active-plan hash.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))
    fig=plt.figure(figsize=(11.7,8.3))
    fig.text(.06,.93,"RPA portfolio atlas: main observations",fontsize=18)
    observations=[f"60 completed runs; 20 seeds per workflow. {len(hashes)} distinct qualifying structures grouped into {len(set(labels))} exploratory structural families."]
    observations += [f"{LABELS[s['arm']]}: {s['successes']}/20 successful seeds; mean {s['mean_qualified']:.1f} qualifying topologies and {s['mean_families']:.1f} structural families per seed." for s in method_summary]
    observations += [f"Equal-size comparison ({len(matched_rarefaction)} matched eligible seeds, ten structures): "+"; ".join(f"{LABELS[s['arm']]} {s['mean_rarefied_families_matched']:.2f} families" for s in method_summary)+".",
        "A recurrent exchange motif contributes X2*(k51*X1-k80*X3) to output dynamics. Large internal X2 can suppress relative disturbance influence. This is an equation-based hypothesis, not a perturbation-tested mechanism.",
        "Low output error can coexist with continuing internal accumulation. These results establish finite-horizon training-grid behavior, not bounded-state robust perfect adaptation. All new analyses are post hoc and descriptive; no new simulations were performed."]
    yy=.85
    for observation in observations:
        wrapped=textwrap.fill(observation,108); fig.text(.06,yy,wrapped,fontsize=11,va="top",linespacing=1.4); yy-=.035*(wrapped.count("\n")+1)+.035
    pdf.savefig(fig); plt.close(fig)
    for start in range(0,len(notes),5):
        fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.94,"RPA atlas: interpretation and limits",fontsize=17)
        yy=.86
        for note in notes[start:start+5]:
            wrapped=textwrap.fill(note,105); fig.text(.06,yy,wrapped,fontsize=11,va="top",linespacing=1.5); yy-=.036*(wrapped.count("\n")+1)+.035
        pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print(json.dumps(summary,default=float,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    render(args.root.resolve(),args.output.resolve())
