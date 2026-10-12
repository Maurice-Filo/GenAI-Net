"""Replay incomplete perturbations for diagnostics only; never replace results."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch

import numpy as np

from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json
from comparisons.search_exploration.run_rpa_perturbations import digest, initialize, evaluate


def audit(saved):
    from RL4CRN.iocrns.iocrn import IOCRN
    original=IOCRN.solve_with_cvode
    solves=[]

    def observed(self, x0, time_horizon, u, nonneg_idx, stop_fn):
        result=original(self,x0,time_horizon,u,nonneg_idx,stop_fn)
        yy=np.asarray(result.y); tt=np.asarray(result.t)
        early=not len(tt) or tt[-1]<time_horizon[-1]
        terminal=yy[:,-1] if yy.size else np.array([])
        at_ceiling=bool(terminal.size and np.isclose(np.max(np.abs(terminal)),1e4,rtol=1e-6))
        solves.append({"input":list(u),"status":int(result.status),"success":bool(result.raw.success),
            "message":str(result.message),"last_time":float(tt[-1]) if len(tt) else None,
            "max_terminal_state":float(np.max(np.abs(terminal))) if terminal.size else None,
            "early_stop":bool(early),"state_ceiling_event":early and at_ceiling and bool(result.raw.success),
            "solver_failure":result.status<0 or not bool(result.raw.success)})
        return result

    job={k:saved[k] for k in ("id","design","arm","seed","reaction_ids","rates","mode","radius","draw","changed_ids")}
    with patch.object(IOCRN,"solve_with_cvode",observed):
        result=evaluate(job)
    return {"id":saved["id"],"arm":saved["arm"],"original_status":saved["status"],
        "replay_status":result["status"],"original_loss":saved["loss"],"replay_loss":result["loss"],
        "same_status":result["status"]==saved["status"],
        "same_loss":bool(np.isclose(result["loss"],saved["loss"],rtol=1e-8,atol=1e-10)) if result["loss"] is not None and saved["loss"] is not None else result["loss"]==saved["loss"],
        "has_ceiling_event":any(s["state_ceiling_event"] for s in solves),
        "has_solver_failure":any(s["solver_failure"] for s in solves),"solves":solves}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args(); root=args.root
    plan=json.loads((root/"plan.json").read_text())
    if digest(root/"plan.json")!=(root/"plan.sha256").read_text().strip():
        raise ValueError("Changed plan")
    for path,sha in plan["source_hashes"].items():
        if digest(path)!=sha:
            raise ValueError(f"Changed source: {path}")
    with sqlite3.connect(f"file:{root/'results.sqlite'}?mode=ro",uri=True) as db:
        failed=[json.loads(r[0]) for r in db.execute("SELECT payload FROM results ORDER BY id")]
    failed=[r for r in failed if r["status"]!="ok"]
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[key]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.pop("COMET_API_KEY",None)
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,initargs=(plan["task_config"],plan["diagnostic_bounds"])) as pool:
        rows=list(pool.map(audit,failed))
    counts=Counter((r["has_ceiling_event"],r["has_solver_failure"]) for r in rows)
    summary={"diagnostic_replays":len(rows),"same_status":sum(r["same_status"] for r in rows),
        "same_loss":sum(r["same_loss"] for r in rows),
        "ceiling_only":counts[True,False],"solver_only":counts[False,True],
        "both":counts[True,True],"neither":counts[False,False],
        "interpretation":"Diagnostic replays never replace original outcomes or remove failures. The existing all-state ceiling is 10000; a ceiling event does not prove mathematical divergence."}
    dump_json(root/"analysis/failure_diagnostics.json",{"summary":summary,"records":rows})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
