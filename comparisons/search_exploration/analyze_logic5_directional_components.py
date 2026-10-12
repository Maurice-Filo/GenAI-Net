"""Post-hoc paired directional checks of robust LLM-lineage yield and RL-emitted loss."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.search_exploration.analyze_logic5_four_arm import ARMS, paired_endpoint, holm, write_csv
from comparisons.search_exploration.analyze_rpa_llm_provenance import connect

EDGES = (("rl-to-llm", "none"), ("full", "llm-to-rl"),
         ("llm-to-rl", "none"), ("full", "rl-to-llm"))


def best_rl_events(db):
    result = {}
    for metric, allow_fit in (("rl_sampled_loss", False), ("rl_with_fit_loss", True)):
        query = """SELECT e.id,e.payload FROM events e
            JOIN designs d ON d.key=json_extract(e.payload,'$.candidate_hash')
            WHERE e.kind='archive_insertion' AND json_extract(e.payload,'$.source')='rl'
            AND json_extract(d.payload,'$.simulation_complete')=1"""
        if not allow_fit:
            query += " AND coalesce(json_extract(e.payload,'$.stream'),'')!='optimization'"
        query += " ORDER BY json_extract(e.payload,'$.loss'),e.id LIMIT 1"
        found = db.execute(query).fetchone()
        if found is None:
            raise ValueError(f"No complete RL event for {metric}")
        eid, payload = found
        event = json.loads(payload)
        if not np.isfinite(event["loss"]) or event["loss"] <= 0:
            raise ValueError("Invalid RL loss")
        result[metric] = event["loss"]
        result[metric + "_event"] = {"id": eid, **event}
    return result


def analyze(root, provenance, output):
    prov = json.loads(provenance.read_text())
    diagnostics = json.loads((root / "analysis/summary.json").read_text())
    if prov["status"] != "complete" or prov["plan_sha256"] != diagnostics["plan_sha256"]:
        raise ValueError("Provenance and diagnostics do not match")
    with gzip.open(root / "selection.json.gz", "rt") as stream:
        selection = json.load(stream)
    rows = []
    for run in selection["runs"]:
        arm, seed = run["arm"], run["seed"]
        cores = [c for c in prov["cores"] if c["arm"] == arm and c["seed"] == seed]
        count = sum(bool(set(c["routes"]) & {"llm_direct", "llm_fit", "ga_llm"}) for c in cores)
        with connect(Path(run["output"])) as db:
            rl = best_rl_events(db)
        rows.append({"arm": arm, "seed": seed, "llm_supported_robust_cores": count,
                     **rl, "folder": run["output"]})
        print(f"{len(rows)}/80 {arm}/{seed}: LLM-supported cores={count}, RL={rl['rl_sampled_loss']:.6g}, RL+fit={rl['rl_with_fit_loss']:.6g}", flush=True)
    by = {(r["arm"], r["seed"]): r for r in rows}
    if len(rows) != 80 or set(by) != {(a,s) for a in ARMS for s in range(1,21)}:
        raise ValueError("Expected all 80 matched endpoints")
    tests = []
    for metric in ("llm_supported_robust_cores", "rl_sampled_loss", "rl_with_fit_loss"):
        for a,b in EDGES:
            tests.append({"a": a, "b": b, "metric": metric, **paired_endpoint(
                [by[a,s][metric] for s in range(1,21)], [by[b,s][metric] for s in range(1,21)],
                smaller=metric.endswith("loss"))})
    for t,p in zip(tests,holm([t["p"] for t in tests])):
        t["holm_p_12_components"] = p
    extended = holm([t["p"] for t in diagnostics["tests"]] + [t["p"] for t in tests])
    for t,p in zip(tests,extended[len(diagnostics["tests"]):]):
        t["holm_p_30_with_portfolio_tests"] = p
    summary = {arm: {metric: float(np.mean([r[metric] for r in rows if r["arm"]==arm]))
        if metric == "llm_supported_robust_cores" else float(np.median([r[metric] for r in rows if r["arm"]==arm]))
        for metric in ("llm_supported_robust_cores", "rl_sampled_loss", "rl_with_fit_loss")} for arm in ARMS}
    output.mkdir(parents=True,exist_ok=True)
    result = {"rows":rows,"tests":tests,"summary":summary,
              "provenance_source":str(provenance),"provenance_sha256":hashlib.sha256(provenance.read_bytes()).hexdigest(),
              "diagnostic_plan_sha256":diagnostics["plan_sha256"],"new_simulations":0}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_csv(output/'paired_tests.csv',tests)
    write_csv(output/'endpoints.csv',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows])
    write_report(output, result)


def write_report(output, result):
    summary, tests = result['summary'], result['tests']
    lines = ["# Paired Directional Component Checks", "", "Exploratory post-hoc decomposition; no new simulations or model calls.", "",
        "| Arm | Mean LLM-supported robust cores | Median RL-sampled loss | Median RL + fitting loss |",
        "|---|---:|---:|---:|"]
    for arm,s in summary.items():
        lines.append(f"| {arm} | {s['llm_supported_robust_cores']:.2f} | {s['rl_sampled_loss']:.6g} | {s['rl_with_fit_loss']:.6g} |")
    lines += ["", "| Comparison | Metric | W/T/L | Mean difference | Raw p | Holm (12 new) | Holm (30 extended) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for t in tests:
        lines.append(f"| {t['a']} vs {t['b']} | {t['metric']} | {t['wins']}/{t['ties']}/{t['losses']} | {t['mean_difference']:.6g} | {t['p']:.6g} | {t['holm_p_12_components']:.6g} | {t['holm_p_30_with_portfolio_tests']:.6g} |")
    lines += ["", "LLM-supported robust cores count each canonical core once per seed if any audited robust witness has an LLM-direct, LLM-fit or explicit LLM-mutation route. Shared LLM/RL routes are included, so these numbers differ from exclusive-source bars. This is a robust-portfolio endpoint, not all LLM proposals or an estimate of proposal success probability.",
        "Return-only versus none isolates enabling outward search/HOF context without inserting or fitting LLM proposals. Full versus insertion-only adds that same outward channel while insertion/fitting is active; this second contrast measures downstream LLM-lineage yield, not only direct LLM proposal quality. Context includes the mixed search state, not exclusively RL samples.",
        "RL-sampled loss is the best complete source=rl event excluding optimization-stream events. RL + fitting additionally permits AdaGrad-refined RL samples. Both exclude direct LLM and GA-emitted candidates. Minima are reconstructed from event journals, not mutable final source labels. No exclusion is made based on whether the candidate led the joint archive.",
        "Four one-channel edges x three endpoints form the 12 new tests, including both outward and inward edges for every metric. Holm across these 12 and an extended family including the previous 18 portfolio tests are both reported. The latter does not cover every exploratory test ever performed on this cohort. Original adjusted p-values are not silently replaced.",
        "W/T/L favors more cores and lower loss. Mean differences are first minus second (negative is better for loss). Paired loss ratios and marginal seed-bootstrap intervals are in the JSON/CSV. These asynchronous system comparisons do not equalize actual proposal banks, completed request counts or insertion schedules; event provenance is not a mechanistic mediation analysis."]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.root.resolve(),args.provenance.resolve(),args.output)
