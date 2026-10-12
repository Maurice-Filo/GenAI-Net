"""Absolute surviving-design counts from saved perturbations; no new simulations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json, write_csv
from comparisons.search_exploration.analyze_rpa_perturbations import ARMS, LABELS
from comparisons.search_exploration.run_rpa_perturbations import digest, jobs


def summarize_counts(plan, data):
    baselines={r["design"]:r for r in data if r["mode"]=="baseline"}
    groups=defaultdict(lambda: defaultdict(list))
    for r in data:
        if r["mode"] in ("all_rates","balance_preserving"):
            groups[r["arm"],r["mode"],r["radius"]][r["design"]].append(r)
    passes=lambda r: r["status"]=="ok" and r["loss"]<plan["threshold"]
    ndraws=len(plan["directions"])
    rows=[]
    for (arm,mode,radius),designs in sorted(groups.items()):
        counts={}
        for name,rr in designs.items():
            if len(rr)!=ndraws or {r["draw"] for r in rr}!=set(range(ndraws)):
                raise ValueError("Incomplete or duplicate perturbation directions")
            counts[name]=sum(passes(r) for r in rr)
        qualified=[name for name in designs if passes(baselines[name])]
        survivors=[sum(passes(r) for name in qualified for r in designs[name] if r["draw"]==draw)
                   for draw in range(ndraws)]
        qdraws=sum(counts[name] for name in qualified)
        rows.append({"arm":arm,"mode":mode,"radius":radius,"tested_winners":len(designs),
            "baseline_qualified_winners":len(qualified),"draws_per_winner":ndraws,
            "qualifying_draws_from_qualified_baselines":qdraws,
            "tested_draws_from_qualified_baselines":len(qualified)*ndraws,
            "qualifying_draws_all_baselines":sum(counts.values()),
            "tested_draws_all_baselines":len(designs)*ndraws,
            "new_qualifying_draws_from_nonqualifying_baselines":sum(counts.values())-qdraws,
            "baseline_qualified_winners_passing_any":sum(counts[n]>0 for n in qualified),
            "baseline_qualified_winners_passing_at_least_half":sum(2*counts[n]>=ndraws for n in qualified),
            "baseline_qualified_winners_passing_all":sum(counts[n]==ndraws for n in qualified),
            "mean_remaining_qualified_winners_per_direction":sum(survivors)/ndraws,
            "min_remaining_qualified_winners_per_direction":min(survivors),
            "max_remaining_qualified_winners_per_direction":max(survivors)})
    return rows


def report(plan, rows):
    lines=["# Perturbation Follow-up: Absolute Numbers Remaining", "",
        "Companion to RPA_PORTFOLIO_ATLAS.pdf. The separate completed perturbation study is at "
        "[the report](../../rpa-rate-perturbations-20260917/analysis/REPORT.md) and "
        "[six-page figure PDF](../../rpa-rate-perturbations-20260917/analysis/RPA_PERTURBATIONS.pdf).", "",
        "## Scope and interpretation", "",
        f"The frozen qualifying criterion is loss < {plan['threshold']:.17g}. This study perturbed 79 saved rank-one winners: "
        "20 AdaGrad hybrid, 20 evolution-fit hybrid, 20 standalone GA, and 19 historical RL-only. "
        "Historical RL-only seed 10 was explicitly excluded because its saved winner contains a null reaction. "
        "Initially qualifying winners number 20, 20, 13 and 9 respectively.", "",
        "The atlas contains entire portfolios; these perturbations cover only one winner per run. "
        "They do NOT measure how many of all archived topologies remain qualifying. "
        "Do not multiply these retention percentages by full-archive counts: rank-one networks are a selected subset. "
        "Counts below are seed-level saved designs, not globally deduplicated structures.", "",
        "Each winner has 32 antithetic perturbations per radius. Passing evaluations are repeated kinetic tests, "
        "not newly discovered topologies. The mean remaining winner count averages over directions, with one "
        "perturbation per winner in each direction. It can be fractional. All-rate changes use factors in "
        "[1/(1+p),1+p] and no clipping or refitting. Incomplete evaluations count as non-passes.", "",
        "## At 10%: Initially Qualifying Winners", "",
        "| Search | Initially qualifying | Passing evaluations / tested | Mean winners remaining per direction | Winners passing any / at least half / all 32 |",
        "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        r=next(r for r in rows if r["arm"]==arm and r["mode"]=="all_rates" and r["radius"]==1.1)
        lines.append(f"| {LABELS[arm]} | {r['baseline_qualified_winners']} | "
            f"{r['qualifying_draws_from_qualified_baselines']}/{r['tested_draws_from_qualified_baselines']} | "
            f"{r['mean_remaining_qualified_winners_per_direction']:.3f} | "
            f"{r['baseline_qualified_winners_passing_any']} / {r['baseline_qualified_winners_passing_at_least_half']} / {r['baseline_qualified_winners_passing_all']} |")
    lines += ["", "Similar retention percentages need not mean similar absolute yield because the number of initially "
        "qualifying winners differs. Conversely, passing at least once is weak evidence of robustness; "
        "the half/all columns make that limitation explicit.", "",
        "## All Radii", "",
        "This table includes initially nonqualifying winners, so it also counts improvements caused by perturbation. "
        "The gain column separates these from genuine retention.", "",
        "| Search | Radius | Passing evaluations, all winners | Passing evaluations from initially nonqualifying winners | Mean initially qualifying winners retained |",
        "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for r in rows:
            if r["arm"]==arm and r["mode"]=="all_rates":
                lines.append(f"| {LABELS[arm]} | {100*(r['radius']-1):.0f}% | "
                    f"{r['qualifying_draws_all_baselines']}/{r['tested_draws_all_baselines']} | "
                    f"{r['new_qualifying_draws_from_nonqualifying_baselines']} | "
                    f"{r['mean_remaining_qualified_winners_per_direction']:.3f} |")
    lines += ["", "## Balance-preserving Control at 10%", "",
        "Only exact-balance networks are eligible. Sensing/calibration coefficients are held fixed; "
        "1-4 rates vary rather than all five. This is a mechanistic control, not an equal-dimensional comparison.", "",
        "| Search | Eligible / initially qualifying winners | Passing evaluations from qualifying baselines | Winners passing all 32 |",
        "|---|---:|---:|---:|"]
    for arm in ARMS:
        for r in rows:
            if r["arm"]==arm and r["mode"]=="balance_preserving" and r["radius"]==1.1:
                lines.append(f"| {LABELS[arm]} | {r['tested_winners']} / {r['baseline_qualified_winners']} | "
                    f"{r['qualifying_draws_from_qualified_baselines']}/{r['tested_draws_from_qualified_baselines']} | "
                    f"{r['baseline_qualified_winners_passing_all']} |")
    lines += ["", "## Mechanistic Takeaway", "",
        "The strict nominal score is rate-sensitive in all workflows, but loss of calibration is not synonymous "
        "with loss of disturbance rejection. The balance-preserving control separates these effects. "
        "All 651 incomplete outcomes in the sweep were independently replayed and reproduced; every one "
        "hit the existing all-state ceiling of 10,000, with no CVODE solver failure. They remain counted as failures. "
        "A ceiling event does not prove mathematical divergence.", "",
        "This is descriptive, post-hoc evidence. Historical search settings differ; no new significance test "
        "or causal superiority claim is made. No additional simulations were performed for this absolute-count note.", "",
        "Machine-readable counts for every arm, mode and radius are in "
        "[CSV](../../rpa-rate-perturbations-20260917/analysis/absolute_counts.csv) and "
        "[JSON](../../rpa-rate-perturbations-20260917/analysis/absolute_counts.json).", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--atlas",type=Path,required=True)
    args=parser.parse_args(); root=args.root
    plan=json.loads((root/"plan.json").read_text())
    if digest(root/"plan.json")!=(root/"plan.sha256").read_text().strip():
        raise ValueError("Plan hash mismatch")
    with sqlite3.connect(f"file:{root/'results.sqlite'}?mode=ro",uri=True) as db:
        data=[json.loads(r[0]) for r in db.execute("SELECT payload FROM results ORDER BY id")]
    expected={j["id"]:j for j in jobs(plan)}
    if len(data)!=len(expected) or {r["id"] for r in data}!=set(expected):
        raise ValueError("Incomplete or unexpected result inventory")
    for r in data:
        if any(r[k]!=v for k,v in expected[r["id"]].items()):
            raise ValueError("Result differs from frozen job")
    rows=summarize_counts(plan,data)
    write_csv(root/"analysis/absolute_counts.csv",rows)
    dump_json(root/"analysis/absolute_counts.json",{"plan_sha256":digest(root/"plan.json"),"counts":rows})
    note=args.atlas/"PERTURBATION_FOLLOWUP.md"
    note.write_text(report(plan,rows))
    print(note)
    for r in rows:
        if r["mode"]=="all_rates" and r["radius"]==1.1:
            print(json.dumps(r))


if __name__=="__main__":
    main()
