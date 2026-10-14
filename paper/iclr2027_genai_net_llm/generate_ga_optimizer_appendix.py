"""Generate the supplemental RPA workflow comparison from completed artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from comparisons.search_exploration.analyze_rpa_statistical import holm
from comparisons.search_exploration.rpa_statistical_campaign import check_frozen

PAPER=Path(__file__).resolve().parent
ARMS=("julia","evolution","ga")
LABELS={"julia":"AdaGrad hybrid","evolution":"Evolution-fit hybrid","ga":"Standalone GA"}
ROW_END = r"\\"


def exact_pratt_pvalue(differences):
    d=np.asarray(differences,dtype=float)
    if not d.size or not np.isfinite(d).all():
        raise ValueError("Nonempty finite paired differences required")
    # Twice the average ranks are integers even with ties. Zeros retain Pratt ranks
    # but contribute no signed weight; dynamic programming enumerates their exact null.
    ranks=np.rint(2*rankdata(np.abs(d))).astype(int)
    weights=ranks[d!=0]
    total=int(weights.sum()); observed=int(ranks[d>0].sum())
    distribution=[0]*(total+1); distribution[0]=1
    for w in weights:
        for i in range(total,int(w)-1,-1):
            distribution[i]+=distribution[i-int(w)]
    tail=sum(distribution[:min(observed,total-observed)+1])
    return min(1.,2*tail/(2**len(weights)))


def perturbation_contrasts(rows):
    selected=[r for r in rows if r["mode"]=="all_rates" and r["arm"] in ARMS]
    table={(r["arm"],int(r["seed"]),float(r["radius"])):float(r["qualified_fraction"]) for r in selected}
    radii=(1.01,1.05,1.1,1.25,1.5)
    expected={(a,s,r) for a in ARMS for s in range(1,21) for r in radii}
    if set(table)!=expected or len(selected)!=len(expected):
        raise ValueError("Need all 20 matched seeds in each method/radius, without duplicates")
    if any(int(r["draws"])!=32 or not 0<=float(r["qualified_fraction"])<=1 for r in selected):
        raise ValueError("Unexpected perturbation count/fraction")
    indices=np.random.default_rng(20260917).integers(0,20,size=(20000,20))
    contrasts=[]
    for radius in radii:
        for arm,control in (("julia","evolution"),("julia","ga"),("evolution","ga")):
            a=np.array([table[arm,s,radius] for s in range(1,21)])
            b=np.array([table[control,s,radius] for s in range(1,21)])
            d=a-b
            contrasts.append({"arm":arm,"control":control,"radius":radius,"n":20,
                "mean_a":float(a.mean()),"mean_b":float(b.mean()),
                "wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),
                "mean_difference":float(d.mean()),
                "mean_difference_ci95":np.quantile(d[indices].mean(1),[.025,.975]).tolist(),
                "p":exact_pratt_pvalue(d)})
    for r,p in zip(contrasts,holm([r["p"] for r in contrasts])):
        r["p_holm_15"]=p
    return contrasts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_figures(rows, pairs):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    from comparisons.search_exploration.analyze_rpa_perturbations import COLORS
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"legend.fontsize":7,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    radii=(1.01,1.05,1.1,1.25,1.5)
    x=[100*(r-1) for r in radii]
    fig,axes=plt.subplots(1,2,figsize=(6.2,2.8),constrained_layout=True)
    rng=np.random.default_rng(20260917)
    for arm in ARMS:
        means=[]; lower=[]; upper=[]; spread=[]
        for radius in radii:
            rr=[r for r in rows if r["arm"]==arm and r["mode"]=="all_rates" and float(r["radius"])==radius]
            v=np.array([float(r["qualified_fraction"]) for r in rr])
            means.append(20*v.mean())
            lo,hi=np.quantile(v[rng.integers(0,20,(20000,20))].mean(1)*20,[.025,.975])
            lower.append(lo); upper.append(hi)
            spread.append(np.mean([float(r["disturbance_spread_below_5pct_fraction"]) for r in rr]))
        axes[0].plot(x,means,"o-",ms=3,color=COLORS[arm],label=LABELS[arm])
        axes[0].fill_between(x,lower,upper,color=COLORS[arm],alpha=.13)
        axes[1].plot(x,spread,"o-",ms=3,color=COLORS[arm])
    for ax in axes:
        ax.set_xscale("log"); ax.set_xticks([1,5,10,25,50]); ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off(); ax.set_xlabel("Multiplicative radius (%)"); ax.grid(alpha=.15)
    axes[0].set(title="A  Useful perturbed winners",ylabel="Mean qualifying count (of 20)",ylim=(0,20.5))
    axes[1].set(title="B  Disturbance rejection",ylabel="Fraction with spread <5%",ylim=(0,1.05))
    axes[0].legend(frameon=False)
    fig.savefig(PAPER/"figures/rpa_optimizer_robustness.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.2,2.8),constrained_layout=True)
    for p in pairs:
        for ax,keys in zip(axes,(("all_rates_ratio","preserving_ratio"),("all_rates_final_error","preserving_final_error"))):
            ax.plot([0,1],[p[k] for k in keys],"o-",ms=2,color=COLORS[p["arm"]],alpha=.5,lw=.8)
    for ax in axes:
        ax.set_xticks([0,1],["All five rates","Balance-preserving\nrates only"])
    axes[0].set(yscale="log",title="A  Loss inflation",ylabel="Median perturbed / baseline loss")
    axes[1].set(title="B  Setpoint accuracy",ylabel="Median worst-case relative error")
    axes[1].set_yscale("symlog",linthresh=1e-8); axes[1].set_ylim(bottom=0)
    labels={"legacy_rl":"Historical RL-only",**LABELS}
    axes[0].legend(handles=[Line2D([0],[0],color=COLORS[a],label=labels[a]) for a in labels],frameon=False,loc="upper right")
    fig.savefig(PAPER/"figures/rpa_optimizer_balance.pdf"); plt.close(fig)


def generate(campaign,perturbations):
    plan=json.loads((campaign/"active_plan.json").read_text())
    check_frozen(campaign,plan)
    summary_path=campaign/plan["analysis_dir"]/"summary.json"
    primary=json.loads(summary_path.read_text())
    if not primary["inference_ready"] or len(primary["endpoints"])!=60:
        raise ValueError("Completed validated primary cohort required")
    counts_path=perturbations/"analysis/per_network.csv"
    with counts_path.open() as stream:
        rows=list(csv.DictReader(stream))
    contrasts=perturbation_contrasts(rows)
    perturbation_summary_path=perturbations/"analysis/summary.json"
    perturbation_summary=json.loads(perturbation_summary_path.read_text())
    frozen=json.loads((perturbations/"plan.json").read_text())
    if sha(perturbations/"plan.json")!=(perturbations/"plan.sha256").read_text().strip():
        raise ValueError("Changed perturbation plan")
    if frozen["threshold"]!=plan["threshold"] or not (perturbations/"completed.json").exists():
        raise ValueError("Incomplete study or threshold mismatch")
    out=PAPER/"generated"
    payload={"primary":primary,"posthoc_perturbation_contrasts":contrasts,
        "analysis":"Two-sided exact Pratt signed-rank; 15-test Holm family; 20 paired seeds. All winners included. Bootstrap CIs are marginal, conditional on fixed perturbation directions.",
        "inputs":{str(p):sha(p) for p in (summary_path,counts_path,perturbations/"plan.json",perturbation_summary_path)}}
    (out/"rpa_optimizer_comparison.json").write_text(json.dumps(payload,indent=2)+"\n")
    with (out/"rpa_optimizer_perturbation_tests.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(contrasts[0])); writer.writeheader(); writer.writerows(contrasts)

    lines=[r"% Generated by generate_ga_optimizer_appendix.py; do not edit numbers manually.",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"Workflow & Median loss & Mean $N_q$ & Qualifying runs & Attempts \\\midrule"]
    for arm in ARMS:
        rr=[r for r in primary["endpoints"] if r["arm"]==arm]
        lo=min(r["evaluations"] for r in rr); hi=max(r["evaluations"] for r in rr)
        attempts=f"{lo:,}" if lo==hi else f"{lo:,}--{hi:,}"
        lines.append(f"{LABELS[arm]} & {np.median([r['loss'] for r in rr]):.6f} & "
            f"{np.mean([r['qualified_discovered'] for r in rr]):.2f} & "
            f"{sum(r['loss']<plan['threshold'] for r in rr)}/20 & {attempts} " + ROW_END)
    lines += [r"\bottomrule\end{tabular}",
        r"\caption{Complementary no-LLM RPA comparison. $N_q$ counts qualifying discovered topology hashes in the complete ledger, not just retained archive entries. Attempts include canonical and, for AdaGrad, joint loss/gradient evaluations; they are not equal units of compute.}",
        r"\label{tab:rpa-optimizer-endpoints}\end{table}",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{@{}llrr@{}}\toprule",
        r"AdaGrad compared with & Endpoint & W/T/L & Holm $p$ \\\midrule"]
    for r in primary["contrasts"]:
        metric="Lower loss" if r["metric"]=="loss" else "Larger $N_q$"
        lines.append(f"{LABELS[r['control']]} & {metric} & {r['wins']}/{r['ties']}/{r['losses']} & {r['p_holm']:.6g} " + ROW_END)
    lines += [r"\bottomrule\end{tabular}",
        r"\caption{Planned paired workflow comparisons, corrected within the four-test family. Wins favor AdaGrad. These tests compare complete recipes, not isolated optimizer components.}",
        r"\label{tab:rpa-optimizer-primary-tests}\end{table}",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{@{}rrrrr@{}}\toprule",
        r"Radius & Mean qualifying winners: A/E/G & A vs. G $p$ & A vs. E $p$ & E vs. G $p$ \\\midrule"]
    for radius in frozen["radii"]:
        ag=next(r for r in contrasts if r["radius"]==radius and r["arm"]=="julia" and r["control"]=="ga")
        ae=next(r for r in contrasts if r["radius"]==radius and r["arm"]=="julia" and r["control"]=="evolution")
        eg=next(r for r in contrasts if r["radius"]==radius and r["arm"]=="evolution")
        lines.append(f"{100*(radius-1):.0f}\\% & {20*ag['mean_a']:.2f}/{20*ae['mean_b']:.2f}/{20*ag['mean_b']:.2f} & "
            f"{ag['p_holm_15']:.4f} & {ae['p_holm_15']:.4f} & {eg['p_holm_15']:.4f} " + ROW_END)
    lines += [r"\bottomrule\end{tabular}",
        r"\caption{Post-hoc perturbed qualification yield. A: AdaGrad hybrid; E: evolution-fit hybrid; G: standalone GA. Counts average over 32 directions, each perturbing one saved winner from each of 20 runs. Tests use per-run qualifying fractions, not individual perturbations; all 20 winners per arm are included. All three pairs and five radii form one 15-test Holm family.}",
        r"\label{tab:rpa-optimizer-perturbation-tests}\end{table}"]
    (out/"rpa_optimizer_tables.tex").write_text("\n".join(lines)+"\n")
    render_figures(rows,perturbation_summary["balance_pairs"])
    print(json.dumps({"contrasts":len(contrasts),"output":str(out/"rpa_optimizer_tables.tex")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("perturbations",type=Path)
    args=parser.parse_args()
    generate(args.campaign,args.perturbations)
