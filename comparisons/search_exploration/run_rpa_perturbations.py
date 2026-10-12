"""Frozen, resumable rate perturbations of saved RPA winners; never refit designs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import csv
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import time

import numpy as np

from RL4CRN.search_exploration.engine import atomic_json
from comparisons.search_exploration.analyze_rpa_mechanisms import (
    cached_function, integral_balance, reaction_catalog, reactions_for, stoichiometry,
)

ROOT = Path(__file__).resolve().parents[2]
RADII = (1.01, 1.05, 1.10, 1.25, 1.50)
EVALUATOR = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directions(seed=20260917, pairs=16):
    z = np.random.default_rng(seed).uniform(-1., 1., (pairs, 91))
    return np.concatenate((z, -z))


def perturb(ids, rates, radius, direction, selected=None):
    selected = set(ids) if selected is None else set(selected)
    return [float(k*np.exp(np.log(radius)*direction[rid])) if rid in selected else float(k)
            for rid,k in zip(ids,rates)]


def balance_preserving_ids(record, catalog, balance):
    if not balance["restricted_exact_balance"]:
        return []
    q = np.array([1., balance["alpha"], 0.])
    result = []
    for rid in record["reaction_ids"]:
        s = stoichiometry(catalog[rid])
        if abs(q@s) <= 1e-10*max(1.,np.linalg.norm(q)*np.linalg.norm(s)):
            result.append(rid)
    return result


def identity(arm, seed):
    return f"{arm}-seed{seed:02d}"


def prepare(campaign, output, workers):
    from comparisons.search_exploration.rpa_statistical_campaign import check_frozen
    active = json.loads((campaign/"active_plan.json").read_text())
    check_frozen(campaign, active)
    summary = json.loads((campaign/active["analysis_dir"]/"summary.json").read_text())
    if not summary["inference_ready"]:
        raise ValueError("Completed audited cohort required")
    atlas = campaign/"analysis-mechanisms-20260917/extracted.json.gz"
    with gzip.open(atlas,"rt") as stream:
        data = json.load(stream)
    if data["plan_sha256"] != digest(campaign/"active_plan.json"):
        raise ValueError("Atlas does not match frozen campaign")
    config = json.loads((campaign/"configs/task.json").read_text())
    catalog,fixed = reaction_catalog(campaign)
    candidates = [{"arm":r["arm"],"seed":r["seed"],"candidate":r["best"],"source":r["folder"]} for r in data["runs"]]
    inputs = {str(atlas):digest(atlas), str(campaign/"active_plan.json"):digest(campaign/"active_plan.json")}
    for seed in range(20):
        folder=ROOT/"comparisons/rpa_search/data/raw/rl4crn"/f"rpa_full102400_seed{seed}_cvode"
        n=json.loads((folder/"best_network.json").read_text())
        old_config=json.loads((folder/"config.json").read_text())
        if old_config["rpa"] != config["rpa"] or n["reaction_ids"][:2]!=[1,28] or n["rate_constants"][:2]!=[1.,1.]:
            raise ValueError("Historical task/template mismatch")
        with (folder/"progress.csv").open() as stream:
            history=list(csv.DictReader(stream))
        if len(history)!=100 or int(history[-1]["candidate_evaluations"])!=102400:
            raise ValueError("Historical run is incomplete")
        candidates.append({"arm":"legacy_rl","seed":seed,"candidate":{"reaction_ids":n["reaction_ids"][2:],
            "rates":n["rate_constants"][2:],"loss":float(history[-1]["best_so_far_loss"])},"source":str(folder)})
        for name in ("best_network.json","config.json","progress.csv"):
            inputs[str(folder/name)]=digest(folder/name)
    designs=[]; exclusions=[]
    for item in candidates:
        d=item["candidate"]; ids=d["reaction_ids"]
        item["id"]=identity(item["arm"],item["seed"])
        if len(ids)!=5 or len(set(ids))!=5 or set(ids)&{0,1,28}:
            exclusions.append({**item,"reason":"Violates current five-distinct-added-reaction contract (null/template/duplicate)"})
            continue
        if not all(.001<=k<=100 for k in d["rates"]):
            raise ValueError("Unexpected original rate bounds")
        order=np.argsort(ids)
        item["candidate"]={"reaction_ids":[int(ids[i]) for i in order],"rates":[float(d["rates"][i]) for i in order],"loss":d["loss"]}
        b=integral_balance(reactions_for(item["candidate"],catalog,fixed))
        item["balance"]=b
        item["balance_preserving_ids"]=balance_preserving_ids(item["candidate"],catalog,b)
        rr=reactions_for(item["candidate"],catalog,fixed)
        anti=any(r["structure"]["reactants"]==["X_1","X_2"] and np.all(stoichiometry(r)[:2]<0) for r in rr)
        sensor=any(r["structure"]["reactants"]==["X_3"] and stoichiometry(r)[1]>0 for r in rr)
        item["aif_like_exact"]=bool(anti and sensor and b["alpha"]<0 and b["restricted_exact_balance"])
        designs.append(item)
    if len(designs)!=79 or [e["id"] for e in exclusions]!=["legacy_rl-seed10"]:
        raise ValueError("Unexpected selection/exclusion; inspect rather than silently omit")
    output.mkdir(parents=True,exist_ok=False)
    for name in ("diagnostics","analysis"):
        (output/name).mkdir()
    sources=[Path(__file__),ROOT/"comparisons/search_exploration/analyze_rpa_mechanisms.py",
        ROOT/"RL4CRN/search_exploration/evaluation.py",ROOT/"RL4CRN/llm/candidate_evaluator.py"]
    plan={"schema":"rpa-rate-perturbation-v1","created":time.time(),"campaign":str(campaign),
        "workers":workers,"task_config":config,"threshold":active["threshold"],"designs":designs,"exclusions":exclusions,
        "radii":list(RADII),"directions":directions().tolist(),"oat_radius":1.10,
        "diagnostic_bounds":[.0005,200.],"refit":False,"clip":False,"scientific_timeout":None,
        "source_hashes":{str(p):digest(p) for p in sources},"input_hashes":inputs,
        "interpretation":"Post-hoc sensitivity of rank-one designs, not equal-budget search or a portfolio-wide causal comparison"}
    plan["expected_evaluations"]=sum(1+len(RADII)*32+10+(len(RADII)*32 if d["balance_preserving_ids"] else 0) for d in designs)
    atomic_json(output/"plan.json",plan)
    (output/"plan.sha256").write_text(digest(output/"plan.json")+"\n")
    atomic_json(output/"status.json",{"phase":"prepared","expected":plan["expected_evaluations"],"completed":0,"updated":time.time()})
    print(f"Prepared {len(designs)} designs, {plan['expected_evaluations']} evaluations",flush=True)
    return plan


def jobs(plan):
    z=np.asarray(plan["directions"])
    for d in plan["designs"]:
        ids,rates=d["candidate"]["reaction_ids"],d["candidate"]["rates"]
        base={"design":d["id"],"arm":d["arm"],"seed":d["seed"],"reaction_ids":ids}
        yield {**base,"id":d["id"]+"/baseline","mode":"baseline","radius":1.,"draw":0,"rates":rates,"changed_ids":[]}
        for mode,selected in (("all_rates",ids),("balance_preserving",d["balance_preserving_ids"])):
            if not selected:
                continue
            for radius in plan["radii"]:
                for draw,direction in enumerate(z):
                    yield {**base,"id":f"{d['id']}/{mode}/{radius:g}/{draw}","mode":mode,"radius":radius,"draw":draw,
                        "rates":perturb(ids,rates,radius,direction,selected),"changed_ids":selected}
        for rid in ids:
            for sign in (-1,1):
                direction=np.zeros(91); direction[rid]=sign
                yield {**base,"id":f"{d['id']}/oat/{rid}/{sign}","mode":"oat","radius":plan["oat_radius"],"draw":sign,
                    "rates":perturb(ids,rates,plan["oat_radius"],direction,[rid]),"changed_ids":[rid]}


def initialize(config, bounds):
    global EVALUATOR
    import torch
    torch.set_num_threads(1)
    from RL4CRN.search_exploration.evaluation import build
    EVALUATOR,_,_=build("rpa",config,*bounds)


def evaluate(job):
    from RL4CRN.search_exploration.evaluation import evaluate_once
    start=time.monotonic()
    result=dict(job)
    try:
        value=evaluate_once(EVALUATOR,job)
        result.update(status="ok" if value["simulation_complete"] else "incomplete",loss=value["loss"],
            simulation_complete=value["simulation_complete"],topology_hash=value["topology_hash"])
        if value["simulation_complete"]:
            result.update(cached_function(value["task_info"]))
        if job["mode"]=="baseline":
            result["diagnostics"]=value["task_info"]
    except Exception as error:
        result.update(status="error",loss=None,simulation_complete=False,error_type=type(error).__name__,error=str(error))
    result["elapsed_seconds"]=time.monotonic()-start
    return result


def execute(output):
    plan=json.loads((output/"plan.json").read_text())
    if digest(output/"plan.json")!=(output/"plan.sha256").read_text().strip():
        raise ValueError("Plan hash mismatch")
    for path,sha in {**plan["input_hashes"],**plan["source_hashes"]}.items():
        if digest(path)!=sha:
            raise ValueError(f"Frozen input/source changed: {path}")
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[k]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.pop("COMET_API_KEY",None)
    db=sqlite3.connect(output/"results.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)")
    done={r[0] for r in db.execute("SELECT id FROM results")}
    todo=[job for job in jobs(plan) if job["id"] not in done]
    if len(done)+len(todo)!=plan["expected_evaluations"]:
        raise ValueError("Unexpected job inventory")
    error_count=sum(json.loads(r[0])["status"]!="ok" for r in db.execute("SELECT payload FROM results"))
    started=time.time(); pending={}; iterator=iter(todo); stopped=False
    def progress(phase):
        atomic_json(output/"status.json",{"phase":phase,"updated":time.time(),"started":started,
            "completed":len(done),"expected":plan["expected_evaluations"],"failed_or_incomplete":error_count,
            "workers":plan["workers"],"in_flight":len(pending),"pid":os.getpid()})
    with ProcessPoolExecutor(max_workers=plan["workers"],mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,initargs=(plan["task_config"],plan["diagnostic_bounds"])) as pool:
        last_progress=0.
        while pending or not stopped:
            while not stopped and len(pending)<2*plan["workers"]:
                job=next(iterator,None)
                if job is None:
                    stopped=True
                else:
                    pending[pool.submit(evaluate,job)]=job
            ready,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in ready:
                result=future.result()  # Infrastructure/worker crashes are not scientific failures.
                del pending[future]
                diagnostics=result.pop("diagnostics",None)
                if diagnostics is not None:
                    with gzip.open(output/"diagnostics"/(result["design"]+".json.gz"),"wt") as stream:
                        json.dump(diagnostics,stream,allow_nan=False)
                db.execute("INSERT INTO results VALUES (?,?)",(result["id"],json.dumps(result,allow_nan=False)))
                done.add(result["id"])
                error_count+=result["status"]!="ok"
            db.commit()
            if time.monotonic()-last_progress>=10:
                progress("running"); last_progress=time.monotonic()
                print(f"{len(done)}/{plan['expected_evaluations']} persisted; {error_count} failed/incomplete",flush=True)
    progress("complete")
    db.close()
    atomic_json(output/"completed.json",{"completed":len(done),"failed_or_incomplete":error_count,"finished":time.time()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=64)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare",action="store_true")
    action.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        prepare(args.campaign.resolve(),args.output.resolve(),args.workers)
    else:
        with (args.output/"run.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                execute(args.output)
            except BaseException as error:
                atomic_json(args.output/"controller_failed.json",{"time":time.time(),"type":type(error).__name__,"message":str(error)})
                raise


if __name__=="__main__":
    main()
