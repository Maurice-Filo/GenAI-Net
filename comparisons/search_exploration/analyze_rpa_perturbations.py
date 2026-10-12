"""Report the separate RPA perturbation study; no simulations or fitting here."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import textwrap
import time

import numpy as np

from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json, write_csv
from comparisons.search_exploration.run_rpa_perturbations import digest, jobs

ARMS=("legacy_rl","julia","evolution","ga")
LABELS={"legacy_rl":"Historical RL-only","julia":"AdaGrad hybrid","evolution":"Evolution-fit hybrid","ga":"Standalone GA"}
COLORS={"legacy_rl":"#6c588d","julia":"#168577","evolution":"#376ab3","ga":"#bd5b71"}


def bootstrap_mean(values, rng):
    v=np.asarray(values,float)
    if not len(v):
        return np.nan,np.nan,np.nan
    samples=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
    lo,hi=np.quantile(samples,[.025,.975])
    return float(v.mean()),float(lo),float(hi)


def aggregate(plan, data):
    by_id={d["id"]:d for d in plan["designs"]}
    groups=defaultdict(list)
    baselines={}
    for r in data:
        if r["mode"]=="baseline": baselines[r["design"]]=r
        else: groups[r["design"],r["mode"],r["radius"]].append(r)
    rows=[]
    for (name,mode,radius),samples in sorted(groups.items()):
        baseline=baselines[name]; design=by_id[name]
        valid=baseline["status"]=="ok"
        loss=baseline["loss"] if valid else np.nan
        finite=[s for s in samples if s["status"]=="ok"]
        losses=np.array([s["loss"] if s["status"]=="ok" else np.inf for s in samples])
        row={"design":name,"arm":design["arm"],"seed":design["seed"],"mode":mode,"radius":radius,
            "baseline_loss":loss,"stored_loss":design["candidate"]["loss"],
            "baseline_qualified":valid and loss<plan["threshold"],
            "baseline_valid":valid,"draws":len(samples),"failures":len(samples)-len(finite),
            "qualified_fraction":sum(s["status"]=="ok" and s["loss"]<plan["threshold"] for s in samples)/len(samples),
            "median_loss":float(np.median(losses)),
            "p90_loss":float(np.quantile(losses,.9,method="inverted_cdf")),
            "max_loss":float(losses.max()),
            "median_loss_ratio":float(np.median(losses)/loss) if valid and loss>0 else np.nan,
            "fraction_within_2x_baseline":float(np.mean(losses<=2*loss)) if valid else np.nan,
            "final_error_below_5pct_fraction":sum(s["status"]=="ok" and s["max_final_relative_error"]<.05 for s in samples)/len(samples),
            "disturbance_spread_below_5pct_fraction":sum(s["status"]=="ok" and s["max_disturbance_spread"]<.05 for s in samples)/len(samples),
            "median_final_error_finite":float(np.median([s["max_final_relative_error"] for s in finite])) if finite else np.nan,
            "median_disturbance_spread_finite":float(np.median([s["max_disturbance_spread"] for s in finite])) if finite else np.nan,
            "aif_like_exact":design["aif_like_exact"],"exact_balance":design["balance"]["restricted_exact_balance"],
            "varied_parameters":len(samples[0]["changed_ids"]) if mode!="oat" else 1}
        rows.append(row)
    audit=[]
    for name,b in baselines.items():
        stored=by_id[name]["candidate"]["loss"]
        drift=abs(b["loss"]-stored)/max(abs(stored),1e-15) if b["status"]=="ok" else np.nan
        audit.append({"design":name,"stored_loss":stored,"reevaluated_loss":b["loss"],"status":b["status"],
            "relative_drift":drift,"drift_above_1pct":bool(drift>.01)})
    return rows,audit,baselines


def analyze(root):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plan=json.loads((root/"plan.json").read_text())
    if digest(root/"plan.json")!=(root/"plan.sha256").read_text().strip():
        raise ValueError("Frozen perturbation plan hash mismatch")
    with sqlite3.connect(f"file:{root/'results.sqlite'}?mode=ro",uri=True) as db:
        data=[json.loads(r[0]) for r in db.execute("SELECT payload FROM results ORDER BY id")]
    if len(data)!=plan["expected_evaluations"] or not (root/"completed.json").exists():
        raise ValueError("No final inference from an incomplete perturbation study")
    expected={j["id"]:j for j in jobs(plan)}
    if {r["id"] for r in data}!=set(expected):
        raise ValueError("Unexpected perturbation result inventory")
    for r in data:
        if any(r[k]!=v for k,v in expected[r["id"]].items()):
            raise ValueError(f"Result does not match frozen job: {r['id']}")
    rows,audit,baselines=aggregate(plan,data)
    out=root/"analysis"; out.mkdir(exist_ok=True)
    write_csv(out/"per_network.csv",rows)
    write_csv(out/"baseline_audit.csv",audit)
    dump_json(out/"failures.json",[d for d in data if d["status"]!="ok"])
    oat=[]
    for d in data:
        if d["mode"]=="oat":
            b=baselines[d["design"]]
            oat.append({"design":d["design"],"reaction_id":d["changed_ids"][0],"sign":d["draw"],"loss":d["loss"],
                "status":d["status"],"loss_ratio":d["loss"]/b["loss"] if d["status"]=="ok" and b["status"]=="ok" else np.inf})
    write_csv(out/"one_at_a_time.csv",oat)
    summary=[]; rng=np.random.default_rng(20260917)
    for arm in ARMS:
        for radius in plan["radii"]:
            rr=[r for r in rows if r["arm"]==arm and r["mode"]=="all_rates" and r["radius"]==radius]
            eligible=[r for r in rr if r["baseline_qualified"]]
            mean,lo,hi=bootstrap_mean([r["qualified_fraction"] for r in eligible],rng)
            summary.append({"arm":arm,"radius":radius,"networks":len(rr),"baseline_qualified_networks":len(eligible),
                "mean_retention":mean,"retention_ci95":[lo,hi],
                "median_network_loss_ratio":float(np.median([r["median_loss_ratio"] for r in rr if r["baseline_valid"]])),
                "mean_tracking_pass_fraction":float(np.mean([r["final_error_below_5pct_fraction"] for r in rr])),
                "mean_disturbance_pass_fraction":float(np.mean([r["disturbance_spread_below_5pct_fraction"] for r in rr])),
                "failed_draws":sum(r["failures"] for r in rr),
                "failed_draws_baseline_qualified":sum(r["failures"] for r in eligible)})
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    pdf=PdfPages(out/"RPA_PERTURBATIONS.pdf"); figures=[]
    def save(fig,name,caption):
        wrapped=textwrap.fill(caption,150)
        footer=.03+.17*(wrapped.count("\n")+1)/fig.get_figheight()
        fig.get_layout_engine().set(rect=(0,footer,1,1-footer))
        fig.text(.025,.012,wrapped,fontsize=8,va="bottom",linespacing=1.3)
        fig.savefig(out/(name+".png"),dpi=190,bbox_inches="tight")
        fig.savefig(out/(name+".pdf"),bbox_inches="tight")
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig); figures.append((name,caption))
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    for arm in ARMS:
        ss=[s for s in summary if s["arm"]==arm]; xx=[0]+[100*(s["radius"]-1) for s in ss]
        bb=[b for b in baselines.values() if b["arm"]==arm]
        tracking=np.mean([b["status"]=="ok" and b["max_final_relative_error"]<.05 for b in bb])
        spread=np.mean([b["status"]=="ok" and b["max_disturbance_spread"]<.05 for b in bb])
        axes[0].plot(xx,[1]+[s["mean_retention"] for s in ss],"o-",color=COLORS[arm],label=f"{LABELS[arm]} (n={ss[0]['baseline_qualified_networks']})")
        axes[0].fill_between(xx,[1]+[s["retention_ci95"][0] for s in ss],[1]+[s["retention_ci95"][1] for s in ss],color=COLORS[arm],alpha=.12)
        axes[1].plot(xx,[tracking]+[s["mean_tracking_pass_fraction"] for s in ss],"o-",color=COLORS[arm],label=LABELS[arm])
        axes[2].plot(xx,[spread]+[s["mean_disturbance_pass_fraction"] for s in ss],"o-",color=COLORS[arm],label=LABELS[arm])
    titles=("A  Retaining qualification", "B  Output accuracy", "C  Disturbance rejection")
    for ax,title in zip(axes,titles):
        ax.set(xlabel="Multiplicative radius (%)",ylim=(-.02,1.02),title=title); ax.grid(alpha=.15)
    axes[0].set_ylabel("Mean fraction passing frozen loss threshold"); axes[0].legend(fontsize=8,frameon=False)
    axes[1].set_ylabel("Fraction with final relative error <5%"); axes[1].legend(fontsize=8,frameon=False)
    axes[2].set_ylabel("Fraction with disturbance spread <5%")
    save(fig,"01_robustness", "All five added rates are perturbed without clipping or refitting. Radius p means factors in [1/(1+p),1+p], log-uniform, with 32 shared antithetic directions. A conditions on freshly reevaluated baseline qualification; shading is a seed-level bootstrap 95% interval. B-C include all eligible networks, including poor baselines. Historical RL has 19 networks after the explicit null-reaction exclusion. Failed solves count as failures, never passes.")
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    target=[r for r in rows if r["mode"]=="all_rates" and r["radius"]==1.1]
    ratio_labels=["Historical\nRL-only","AdaGrad\nhybrid","Evolution\nhybrid","GA"]
    for i,arm in enumerate(ARMS):
        rr=[r for r in target if r["arm"]==arm and r["baseline_valid"]]
        vals=[r["median_loss_ratio"] for r in rr]
        nonfinite=sum(not np.isfinite(v) for v in vals)
        if nonfinite:
            ratio_labels[i]+=f"\n({nonfinite} nonfinite)"
        axes[0].scatter(i+rng.uniform(-.13,.13,len(vals)),vals,color=COLORS[arm],s=25)
        axes[0].hlines(np.median(vals),i-.23,i+.23,color=COLORS[arm],lw=3)
        for aif,marker in ((True,"s"),(False,"o")):
            subset=[r for r in rr if r["aif_like_exact"]==aif]
            axes[1].scatter([r["baseline_loss"] for r in subset],[r["median_loss_ratio"] for r in subset],color=COLORS[arm],marker=marker,s=30)
            axes[2].scatter([r["baseline_loss"] for r in subset],[r["median_loss"]-r["baseline_loss"] for r in subset],color=COLORS[arm],marker=marker,s=30)
    axes[0].set_xticks(range(4),ratio_labels)
    axes[0].set(yscale="log",ylabel="Median perturbed / baseline loss",title="A  Loss inflation at radius 10%")
    axes[0].axhline(1,color=".5",ls="--",lw=1)
    axes[1].set(xscale="log",yscale="log",xlabel="Reevaluated baseline loss",ylabel="Median loss ratio",title="B  Baseline quality matters")
    axes[1].legend(handles=[Line2D([0],[0],marker="s",ls="",color=".35",label="AIF-like balance"),Line2D([0],[0],marker="o",ls="",color=".35",label="Other")],fontsize=8,frameon=False)
    axes[2].set(xscale="log",xlabel="Reevaluated baseline loss",ylabel="Median absolute loss change",title="C  Absolute degradation")
    axes[2].set_yscale("symlog",linthresh=.001)
    changes=[r["median_loss"]-r["baseline_loss"] for r in target if r["baseline_valid"] and np.isfinite(r["median_loss"])]
    if changes and min(changes)>=0:
        axes[2].set_ylim(0,max(.001,1.3*max(changes)))
    save(fig,"02_quality_conditioning", "Each dot is one saved rank-one network, not a perturbation draw. Very small baseline losses can produce large ratios from modest absolute changes; both quantities are shown. Squares indicate the post-hoc AIF-like algebraic/motif screen. Nonfinite network medians cannot be plotted; their counts appear on the left axis, and they remain in summaries and CSV. This is not a matched-proposal or equal-search-budget comparison; no cross-method significance tests are implied.")
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    paired=[]
    for r in rows:
        if r["mode"]!="balance_preserving" or r["radius"]!=1.1:
            continue
        allrow=next(s for s in target if s["design"]==r["design"])
        paired.append({"design":r["design"],"arm":r["arm"],"varied_parameters":r["varied_parameters"],
            "all_rates_ratio":allrow["median_loss_ratio"],"preserving_ratio":r["median_loss_ratio"],
            "all_rates_final_error":allrow["median_final_error_finite"],"preserving_final_error":r["median_final_error_finite"],
            "all_rates_spread":allrow["median_disturbance_spread_finite"],"preserving_spread":r["median_disturbance_spread_finite"]})
    for ax,(left,right),title in zip(axes,(("all_rates_ratio","preserving_ratio"),("all_rates_final_error","preserving_final_error"),("all_rates_spread","preserving_spread")),
            ("A  Loss inflation", "B  Output tracking error", "C  Disturbance spread")):
        for p in paired:
            ax.plot([0,1],[p[left],p[right]],color=COLORS[p["arm"]],alpha=.35,lw=1)
            ax.scatter([0,1],[p[left],p[right]],color=COLORS[p["arm"]],s=12,alpha=.8)
        ax.set_xticks([0,1],["All added\nrates","Balance-preserving\nrates only"])
        ax.set_title(title)
        if left=="all_rates_ratio":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog",linthresh=1e-8)
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Median perturbed / baseline loss")
    axes[1].set_ylabel("Median worst-case relative error")
    axes[2].set_ylabel("Median disturbance spread / reference")
    axes[0].legend(handles=[Line2D([0],[0],color=COLORS[a],label=LABELS[a]) for a in ARMS],fontsize=8,frameon=False)
    save(fig,"03_balance_preserving", f"Paired within-network comparison at radius 10%, restricted to {len(paired)} exact-balance networks. Preserving perturbations vary only reactions whose stoichiometry is orthogonal to the original integral coordinate: sensor/calibration terms are held fixed. They vary fewer rates, so this is a mechanistic control, not an equal-dimensional robustness contest. Final-error/spread medians condition on successful solves; failures remain in per_network.csv and robustness probabilities.")
    write_csv(out/"balance_preserving_pairs.csv",paired)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    oat_labels=["Historical\nRL-only","AdaGrad\nhybrid","Evolution\nhybrid","GA"]
    for i,arm in enumerate(ARMS):
        names=[d["id"] for d in plan["designs"] if d["arm"]==arm]
        worst=[max(r["loss_ratio"] for r in oat if r["design"]==name) for name in names]
        axes[0].scatter(i+rng.uniform(-.13,.13,len(worst)),worst,color=COLORS[arm],s=24)
        axes[0].hlines(np.median(worst),i-.23,i+.23,color=COLORS[arm],lw=3)
        failed=sum(not np.isfinite(v) for v in worst)
        if failed:
            oat_labels[i]+=f"\n({failed} incomplete)"
        ss=[s for s in summary if s["arm"]==arm]
        axes[1].plot([100*(s["radius"]-1) for s in ss],[s["failed_draws"]/(s["networks"]*32) for s in ss],"o-",color=COLORS[arm],label=LABELS[arm])
    axes[0].set_xticks(range(4),oat_labels)
    axes[0].set(yscale="log",ylabel="Worst one-rate loss / baseline loss",title="A  Single-rate fragility (radius 10%)")
    axes[1].set(xlabel="Multiplicative radius (%)",ylabel="Incomplete evaluation fraction",title="B  Canonical noncompletion")
    axes[1].legend(fontsize=8,frameon=False)
    save(fig,"04_single_rate_failures", "Each one-rate test changes exactly one added rate by a factor 1.1 or 1/1.1, leaving the remaining rates fixed. Ten evaluations per design; the worst is shown. Infinite ratios from failed solves are retained in CSV data rather than assigned an invented finite loss. Panel B counts all failed/incomplete all-rate evaluations, including those belonging to nonqualifying baseline designs.")
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    small=[r for r in rows if r["mode"]=="all_rates" and r["radius"]==1.01]
    headroom=[]
    for r in small:
        b=baselines[r["design"]]
        if b["status"]!="ok":
            continue
        headroom.append({"design":r["design"],"arm":r["arm"],"baseline_loss":b["loss"],
            "baseline_qualified":r["baseline_qualified"],"max_terminal_concentration":b["max_terminal_concentration"],
            "max_state_tail_change":b["max_state_tail_change"],"incomplete_fraction_at_1pct":r["failures"]/r["draws"]})
    for arm in ARMS:
        rr=[r for r in headroom if r["arm"]==arm]
        for qualified,marker in ((True,"o"),(False,"x")):
            ss=[r for r in rr if r["baseline_qualified"]==qualified]
            axes[0].scatter([r["max_terminal_concentration"] for r in ss],[r["incomplete_fraction_at_1pct"] for r in ss],s=35,color=COLORS[arm],marker=marker)
            axes[1].scatter([r["baseline_loss"] for r in ss],[r["max_state_tail_change"] for r in ss],s=35,color=COLORS[arm],marker=marker)
    axes[0].axvline(1e4,color=".35",ls="--",lw=1,label="State ceiling")
    axes[0].set(xscale="log",xlabel="Largest baseline terminal concentration",ylabel="Incomplete fraction at radius 1%",title="A  State-ceiling headroom")
    axes[1].set(xscale="log",xlabel="Reevaluated baseline loss",ylabel="Largest relative state change in final 20%",title="B  Low loss is not equilibration")
    axes[1].set_yscale("symlog",linthresh=1e-8)
    axes[1].set_ylim(0,max(.01,1.5*max(r["max_state_tail_change"] for r in headroom)))
    axes[0].legend(handles=[Line2D([0],[0],marker="o",ls="",color=".35",label="Baseline qualifies"),Line2D([0],[0],marker="x",ls="",color=".35",label="Baseline does not qualify"),Line2D([0],[0],color=".35",ls="--",label="State ceiling: 10,000")],fontsize=8,frameon=False)
    axes[1].legend(handles=[Line2D([0],[0],color=COLORS[a],label=LABELS[a]) for a in ARMS],fontsize=8,frameon=False)
    save(fig,"05_state_headroom", "Unperturbed diagnostics for all eligible saved winners. Panel A links baseline terminal concentration to incomplete evaluations under tiny rate changes. Several networks operate near the existing 10,000 all-state ceiling. Panel B shows that low output loss can coexist with internal-state drift. Tail change is absolute state change over the final 20%, divided by max(1, absolute terminal state), maximized across species and scenarios. Neither panel proves mathematical divergence.")
    write_csv(out/"state_headroom.csv",headroom)
    failure_audit=None
    if (out/"failure_diagnostics.json").exists():
        failure_data=json.loads((out/"failure_diagnostics.json").read_text())
        if {r["id"] for r in failure_data["records"]}!={r["id"] for r in data if r["status"]!="ok"}:
            raise ValueError("Failure audit does not cover current incomplete/error inventory")
        failure_audit=failure_data["summary"]
    balance_summary={"networks":len(paired),"varied_parameters_min":min(p["varied_parameters"] for p in paired),
        "varied_parameters_max":max(p["varied_parameters"] for p in paired)}
    for key in ("all_rates_ratio","preserving_ratio","all_rates_final_error","preserving_final_error","all_rates_spread","preserving_spread"):
        balance_summary[key]=float(np.median([p[key] for p in paired]))
    output={"study_complete":True,"evaluations":len(data),"failed_or_incomplete":sum(d["status"]!="ok" for d in data),
        "baseline_drift_above_1pct":sum(d["drift_above_1pct"] for d in audit),"summary":summary,"balance_pairs":paired,
        "balance_summary_at_10pct":balance_summary,
        "status_counts":{s:sum(d["status"]==s for d in data) for s in ("ok","incomplete","error")},
        "diagnostic_failure_replay":failure_audit,
        "exclusions":plan["exclusions"],"figures":figures}
    dump_json(out/"summary.json",output)
    lines=["# RPA Rate-Perturbation Analysis", "",f"{len(plan['designs'])} saved rank-one networks; {len(data)} canonical CVODE evaluations. No refitting, clipping, or LLM calls.","",
        f"Failed/incomplete evaluations: {output['failed_or_incomplete']}. Baseline discrepancies above 1%: {output['baseline_drift_above_1pct']} (see baseline_audit.csv).", "",
        "## Protocol", "",
        "Added rates are multiplied by exp(log(radius)*z), z uniform on [-1,1]. Sixteen draws and their negatives give 32 directions, shared by reaction ID across networks and nested across radii. Fixed input production/degradation coefficients stay unchanged. One-at-a-time tests use factors 1.1 and 1/1.1. No scientific wall-clock cutoff is applied.","",
        "Diagnostic rates deliberately may cross the original search bounds, with no clipping: generated rates lie within [0.001/1.5,150], admitted by separate diagnostic validator bounds [0.0005,200]. The CVODE task, nine input scenarios, horizon 100, 1000 points and tolerances 1e-8 are unchanged. Original experiments and evaluator code are not modified.","",
        "For a verified identity d(X1+a X2)/dt = u1-g OUT, a steady state requires OUT = u1/g. Changing sensor gain g can therefore shift the setpoint without destroying disturbance rejection. Balance-preserving perturbations hold those terms fixed and vary only individually orthogonal stoichiometric directions. The number of perturbed parameters is reported; this control is not equal-dimensional to perturbing all five rates.","",
        "Historical RL-only seed 10 is explicitly excluded because its winner contains a null reaction forbidden by the current contract. All other saved winners are included regardless of baseline quality. Historical seeds/settings differ from the new workflows, so this is not a controlled causal comparison of optimizers.","",
        "## All-Rate Results", "",
        "Retention means passing the original frozen loss threshold among networks whose freshly reevaluated baseline passes it. Failure draws are never dropped from probabilities. Loss ratios use freshly reevaluated baseline losses, not historical scores. Each network/seed is the aggregation unit; 32 draws are not 32 independent search runs.","",
        "| Method | Radius | Baseline-qualified n | Mean retention | Median network loss ratio | Failures |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {LABELS[s['arm']]} | {100*(s['radius']-1):.0f}% | {s['baseline_qualified_networks']} | {s['mean_retention']:.1%} | {s['median_network_loss_ratio']:.3g} | {s['failed_draws']} |")
    lines += ["", "## Interpretation", "",
        "The strict loss score is sensitive to simultaneous rate changes in every workflow. At a 10% multiplicative radius, qualification retention is low even when output disturbance rejection remains strong. Calibration sensitivity and loss of disturbance rejection are different outcomes; neither should be substituted for the other.", "",
        f"In the {len(paired)} exact-balance networks, the median of network-median loss ratios at radius 10% is {balance_summary['all_rates_ratio']:.3g} when all rates vary, versus {balance_summary['preserving_ratio']:.5g} when only balance-preserving rates vary. Corresponding worst-case final relative output errors are {100*balance_summary['all_rates_final_error']:.3g}% and {100*balance_summary['preserving_final_error']:.3g}%. These controls vary {balance_summary['varied_parameters_min']}-{balance_summary['varied_parameters_max']} rather than five rates. They support sensitivity concentrated in calibration/balance terms, not a claim that arbitrary kinetic perturbations are harmless.", "",
        "### Functional checks at radius 10%", "",
        "The following fractions include all saved eligible winners, not just those passing the original loss threshold. Qualification retention in the table above uses a different, explicitly conditional denominator.", "",
        "| Method | Final output error <5% | Disturbance spread <5% | Incomplete/error draws |",
        "|---|---:|---:|---:|"]
    for s in summary:
        if s["radius"]==1.1:
            lines.append(f"| {LABELS[s['arm']]} | {s['mean_tracking_pass_fraction']:.1%} | {s['mean_disturbance_pass_fraction']:.1%} | {s['failed_draws']}/{s['networks']*32} |")
    lines += ["", "### Failure disclosure", "",
        f"Recorded statuses: {output['status_counts']}. An incomplete status means the canonical output array was absent, nonfinite or contained its failure sentinel; it is not evidence by itself of mathematical instability or a biological mechanism. The canonical simulator also stops at its existing all-state ceiling of 10,000. These records count as non-passes in every probability and as infinite loss in loss-order summaries. No timeout or retry filter removed them.", "",
        "| Method | Radius 1% incomplete/error draws | Of these, from qualifying baselines |",
        "|---|---:|---:|"]
    for s in summary:
        if s["radius"]==1.01:
            lines.append(f"| {LABELS[s['arm']]} | {s['failed_draws']}/{s['networks']*32} | {s['failed_draws_baseline_qualified']} |")
    if failure_audit:
        lines += ["",f"A separate diagnostic audit replayed {failure_audit['diagnostic_replays']} incomplete/error evaluations, without replacing any outcomes. It reproduced {failure_audit['same_status']} statuses and {failure_audit['same_loss']} losses (rtol 1e-8, atol 1e-10). Classified at the evaluation level: {failure_audit['ceiling_only']} state-ceiling events only, {failure_audit['solver_only']} solver failures only, {failure_audit['both']} both, {failure_audit['neither']} neither. See failure_diagnostics.json for every scenario's raw status, message, last time and terminal maximum. These replays are additional diagnostic compute, not extra robustness draws. Crossing the ceiling is a canonical task failure, not proof of unbounded mathematical growth.",""]
    lines += ["", "## Figures", "", "[Combined PDF](RPA_PERTURBATIONS.pdf)", ""]
    for name,caption in figures: lines += [f"![{name}]({name}.png)","",caption,""]
    lines += ["## Limits", "",
        "This post-hoc study tests kinetic fragility of selected winners, not every discovered topology or biological implementation. The grid has constant inputs and one initial condition; it is not a step-disturbance or held-out-input experiment. Better retention can reflect more baseline headroom. Very low baseline losses inflate ratios. The figures expose both effects; no adjusted causal or significance claims are made.","",
        "Finite-horizon output accuracy can coexist with accumulating internal species. These tests do not prove stability, boundedness or robust perfect adaptation. All nonfinite/solver/exception outcomes remain explicit in failures.json. Process crashes stop the controller rather than being silently classified as biological failure. Bootstrap intervals condition on the sampled directions and do not constitute a between-method significance test.","",
        "Progress is status.json; results.sqlite contains every persisted result. Execution can resume missing results after infrastructure interruption without repeating already recorded successes or failures. Plan and input/source hashes prevent silent protocol changes.",""]
    (out/"REPORT.md").write_text("\n".join(lines))
    fig=plt.figure(figsize=(11.7,8.3)); fig.text(.06,.93,"RPA perturbations: protocol and interpretation",fontsize=17)
    notes=[lines[2], lines[4], lines[8], lines[10], lines[12], lines[14],
        "All analyses are post hoc. Survival is conditional on baseline qualification. Bootstrap bands resample networks/seeds, not individual perturbation draws. No new significance tests or causal attribution to RL are asserted."]
    yy=.85
    for note in notes:
        wrapped=textwrap.fill(note,110); fig.text(.06,yy,wrapped,fontsize=10,va="top",linespacing=1.35); yy-=.023*(wrapped.count("\n")+1)+.018
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print(json.dumps({k:v for k,v in output.items() if k in ("evaluations","failed_or_incomplete","baseline_drift_above_1pct","summary")},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--watch",action="store_true")
    args=parser.parse_args()
    if args.watch:
        while not (args.root/"completed.json").exists():
            if (args.root/"controller_failed.json").exists():
                raise RuntimeError("Perturbation controller failed; inspect before analyzing")
            time.sleep(20)
    analyze(args.root)
