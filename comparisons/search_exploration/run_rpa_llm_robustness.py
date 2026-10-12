"""Reuse the established winner-rate perturbation protocol for LLM interaction arms."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

OLD = Path('/local1/rossin/search-exploration/rpa-rate-perturbations-20260917')
ARMS = ('control','flash','llm-to-rl','rl-to-llm','independent-union')
LABELS = ('AdaGrad/RL','Full duplex','Insertion only','Return only','Independent union')
COLORS = ('#168577','#a64e85','#bc7838','#376ab3','#6b6975')


def read(path):
    return json.loads(path.read_text())


def rate_key(candidate):
    return tuple(sorted((int(i),float(k).hex()) for i,k in zip(candidate['reaction_ids'],candidate['rates'])))


def prepare(winners, output, workers=8):
    from comparisons.search_exploration import run_rpa_perturbations as protocol
    from comparisons.search_exploration.analyze_rpa_mechanisms import reaction_catalog, integral_balance, reactions_for
    from RL4CRN.search_exploration.engine import atomic_json
    old = read(OLD/'plan.json')
    if protocol.digest(OLD/'plan.json') != (OLD/'plan.sha256').read_text().strip():
        raise ValueError('Old perturbation plan changed')
    for path,h in old['source_hashes'].items():
        if protocol.digest(path)!=h: raise ValueError('Cached evaluator/protocol changed')
    if not (OLD/'completed.json').exists(): raise ValueError('Old study incomplete')
    selected = read(winners)
    catalog,fixed = reaction_catalog(Path(old['campaign']))
    designs,aliases,unique = [],[],{}
    for w in selected['designs']:
        d=w['candidate']; ids=d['reaction_ids']; rates=d['rates']
        if len(ids)!=5 or len(set(ids))!=5 or set(ids)&{0,1,28} or len(rates)!=5 or not all(.001<=k<=100 for k in rates):
            raise ValueError('Invalid selected winner')
        order=np.argsort(ids)
        candidate={'reaction_ids':[int(ids[i]) for i in order], 'rates':[float(rates[i]) for i in order], 'loss':d['loss']}
        key=rate_key(candidate)
        logical_id=protocol.identity(w['arm'],w['seed'])
        if key not in unique:
            unique[key]=logical_id
            balance=integral_balance(reactions_for(candidate,catalog,fixed))
            designs.append({'id':logical_id,'arm':w['arm'],'seed':w['seed'],'candidate':candidate,
                'source':w['source'], 'balance':balance,
                'balance_preserving_ids':protocol.balance_preserving_ids(candidate,catalog,balance),
                'aif_like_exact':None})
        aliases.append({'id':logical_id,'arm':w['arm'],'seed':w['seed'],'physical_design':unique[key],
                        'candidate':candidate,'source':w['source']})
    output.mkdir(parents=True,exist_ok=False)
    for sub in ('diagnostics','analysis'): (output/sub).mkdir()
    plan={**old,'schema':'rpa-llm-winner-perturbations-v1','created':time.time(),'workers':workers,
        'designs':designs,'aliases':aliases,'exclusions':[],
        'directional_matched_seeds':selected['directional_matched_seeds'],
        'input_hashes':{str(winners):protocol.digest(winners),str(OLD/'plan.json'):protocol.digest(OLD/'plan.json')},
        'source_hashes':{**old['source_hashes'],str(Path(__file__).resolve()):protocol.digest(__file__)},
        'interpretation':'Rank-one rate sensitivity only; equal seed weighting. Partial directional matched subset is frozen, not a final 20-seed ablation. No refitting or clipping.'}
    scheduled=list(protocol.jobs(plan)); plan['expected_evaluations']=len(scheduled)
    # Cache only exact rate vectors under the already verified identical task/protocol.
    path=OLD/'results.sqlite'
    wal=path.with_name(path.name+'-wal')
    if wal.exists() and wal.stat().st_size: raise ValueError('Old perturbation database not checkpointed')
    with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        cached={}
        for (payload,) in db.execute('SELECT payload FROM results ORDER BY id'):
            r=json.loads(payload); cached.setdefault(rate_key(r),r)
    reused=0
    with sqlite3.connect(output/'results.sqlite') as db:
        db.execute('CREATE TABLE results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        for job in scheduled:
            found=cached.get(rate_key(job))
            if found is not None:
                record={**found,**job,'reused_from':str(OLD),'reused_result_id':found['id']}
                db.execute('INSERT INTO results VALUES (?,?)',(job['id'],json.dumps(record,allow_nan=False)))
                reused+=1
    plan['reused_evaluations']=reused
    atomic_json(output/'plan.json',plan)
    (output/'plan.sha256').write_text(protocol.digest(output/'plan.json')+'\n')
    atomic_json(output/'status.json',{'phase':'prepared','completed':reused,'expected':len(scheduled),'workers':workers})
    print(json.dumps({'logical_winners':len(aliases),'unique_winners':len(designs),
                      'cached_evaluations':reused,'new_evaluations':len(scheduled)-reused}),flush=True)


def analyze(root):
    from comparisons.search_exploration.run_rpa_perturbations import jobs,digest
    from comparisons.search_exploration.analyze_rpa_perturbations import aggregate
    from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json,write_csv
    p=read(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip(): raise ValueError('Plan changed')
    with sqlite3.connect(root/'results.sqlite') as db:
        actual={r['id']:r for (text,) in db.execute('SELECT payload FROM results') for r in [json.loads(text)]}
    expected={j['id']:j for j in jobs(p)}
    if set(actual)!=set(expected): raise ValueError('Incomplete perturbation inventory')
    for key,j in expected.items():
        if any(actual[key][k]!=v for k,v in j.items()): raise ValueError('Result/job mismatch')
    expanded,logical=[],[]
    physical={d['id']:d for d in p['designs']}
    for a in p['aliases']:
        logical.append({**physical[a['physical_design']],**a})
        for r in actual.values():
            if r['design']==a['physical_design']:
                expanded.append({**r,'design':a['id'],'arm':a['arm'],'seed':a['seed']})
    rows,audit,baselines=aggregate({**p,'designs':logical},expanded)
    out=root/'analysis'
    write_csv(out/'per_network.csv',rows); write_csv(out/'baseline_audit.csv',audit)
    dump_json(out/'failed_draws.json',[r for r in expanded if r['status']!='ok'])
    subsets={'Full cohort':list(range(1,21)),'Matched directional subset':p['directional_matched_seeds']}
    summaries=[]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    rng=np.random.default_rng(20260918)
    for col,(subset,seeds) in enumerate(subsets.items()):
        arms=ARMS[:2] if col==0 else ARMS
        for arm in arms:
            i=ARMS.index(arm); values=[]; ratios=[]; low=[]; high=[]
            for radius in p['radii']:
                rr=sorted((r for r in rows if r['arm']==arm and r['seed'] in seeds and r['mode']=='all_rates' and r['radius']==radius),key=lambda r:r['seed'])
                if len(rr)!=len(seeds): raise ValueError('Missing logical winner')
                v=np.array([r['qualified_fraction'] for r in rr]); ratio=np.array([r['median_loss_ratio'] for r in rr])
                bs=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
                lo,hi=np.quantile(bs,[.025,.975])
                values.append(v.mean());low.append(lo);high.append(hi)
                ratios.append(float(np.median(ratio)))
                summaries.append({'subset':subset,'arm':arm,'radius':radius,'n':len(v),
                    'mean_qualification_survival':float(v.mean()),'ci95':[float(lo),float(hi)],
                    'median_relative_loss':float(np.median(ratio)),
                    'failed_draws':sum(r['failures'] for r in rr),
                    'baseline_qualified':sum(r['baseline_qualified'] for r in rr)})
            x=[100*(r-1) for r in p['radii']]
            axes[0,col].plot(x,values,'o-',color=COLORS[i],label=LABELS[i],ms=4)
            axes[0,col].fill_between(x,low,high,color=COLORS[i],alpha=.10)
            finite=np.isfinite(ratios)
            axes[1,col].plot(np.array(x)[finite],np.array(ratios)[finite],'o-',color=COLORS[i],label=LABELS[i],ms=4)
        axes[0,col].set(title=f'{subset} (n={len(seeds)} per arm)',ylabel='Fraction of draws qualifying',ylim=(-.02,1.02))
        axes[0,col].legend(frameon=False,fontsize=8)
        axes[1,col].set(yscale='log',ylabel='Median network loss inflation',xlabel='Multiplicative perturbation radius (%)')
        for ax in axes[:,col]: ax.grid(alpha=.12)
    fig.suptitle('RPA | Are the winning networks robust to rate perturbations?',fontsize=14)
    fig.text(.04,.015,'32 shared antithetic directions per radius; rates varied jointly, without clipping or refitting. Failures never count as passes.\n'
        'Top: unconditional survival, 95% seed-bootstrap intervals. Bottom: relative to reevaluated baseline; nonfinite values remain in CSV.\n'
        'Only rank-one winners, not whole portfolios. Independent union is post hoc; directional subset is provisional.',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.96))
    fig.savefig(out/'RPA_LLM_ROBUSTNESS.pdf');fig.savefig(out/'robustness.png',dpi=190);plt.close(fig)
    dump_json(out/'summary.json',{'summaries':summaries,'baseline_audit':audit,
        'physical_draws':len(actual),'logical_winners':len(logical),'reused_draws':p['reused_evaluations']})
    lines=['# RPA LLM Winner Robustness','',
        'Same protocol as the previous winner study: multiplicative radii 1%, 5%, 10%, 25%, 50%; 32 shared antithetic directions, CVODE, no refitting/clipping. Added rates vary; fixed input-template rates stay fixed. Diagnostic bounds [0.0005,200] deliberately permit perturbations outside search bounds.','',
        f"{len(logical)} logical winners, {len(p['designs'])} distinct reaction/rate vectors, {p['reused_evaluations']} reused prior evaluations. Identical vectors across arms are evaluated only once. Failed solves remain failures. No new LLM calls.",'',
        '| Subset | Arm | 10% survival | Median 10% loss inflation | Failed draws |','|---|---|---:|---:|---:|']
    for s in summaries:
        if s['radius']==1.1: lines.append(f"| {s['subset']} | {s['arm']} | {s['mean_qualification_survival']:.3f} | {s['median_relative_loss']:.3g} | {s['failed_draws']} |")
    lines+=['','Absolute qualification survival combines initial quality margin and sensitivity. Relative loss inflation helps separate these, but is not a causal robustness comparison. Designs were selected for low loss, not robustness. Whole-portfolio robustness and mechanistic diversity are not established. The 32 draws are not 32 independent search runs.','',
        'Baseline score drift, failed/incomplete solves, and one-at-a-time and balance-preserving results remain in the CSV/JSON artifacts; no solver fallback or silent failure removal. Directional inference is deferred until the full recovery cohort is available.','',
        '![Winner robustness](robustness.png)','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def run(root):
    os.environ.pop('COMET_API_KEY',None)
    from comet_ml import Experiment
    experiment=Experiment(project_name='genai-net-exploration',auto_param_logging=False,auto_metric_logging=False,
        log_code=False,log_git_patch=False,log_graph=False,log_git_metadata=False,log_env_details=False,
        log_env_host=False,log_env_cpu=False,log_env_gpu=False,display_summary_level=0)
    if not experiment.get_key() or not getattr(experiment,'alive',False): raise RuntimeError('Comet unavailable')
    experiment.set_name(root.name); experiment.add_tags(['rpa','posthoc','winner-robustness','no-refitting'])
    from comparisons.search_exploration.run_rpa_perturbations import execute
    try:
        with (root/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(root)
            analyze(root)
        experiment.log_image(str(root/'analysis/robustness.png'),name='Winner robustness')
        for s in read(root/'analysis/summary.json')['summaries']:
            experiment.log_metric(f"{s['subset']}/{s['arm']}/survival",s['mean_qualification_survival'],step=round(100*(s['radius']-1)))
    finally:
        experiment.end()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--winners',type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.winners,args.root)
    if args.run: run(args.root)
    if args.analyze: analyze(args.root)
