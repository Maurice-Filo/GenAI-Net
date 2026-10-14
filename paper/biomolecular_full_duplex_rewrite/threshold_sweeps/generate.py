"""Threshold sensitivity of distinct topology counts, using event-time provenance.

No simulations. Counts cover all saved completed evaluations, not the bounded
final archive. A topology may have witnesses from more than one source.
"""
from pathlib import Path
import sys,json,sqlite3,math,gzip,csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import orjson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from generate_dose_overviews import TASKS, select_runs

ROUTES=['llm_lineage','rl_lineage','llm_direct','rl_direct','other','all']
MAX_THRESHOLD=.2
THRESHOLDS=np.unique(np.r_[np.geomspace(1e-4,MAX_THRESHOLD,301),[.001,.005,.01,.025,.05,.1,.2]])

def update(mapping,topology,loss):
    mapping[topology]=min(mapping.get(topology,math.inf),loss)


def extract(item):
    task,seed,run,done,replacement=item
    path=(run/'exploration.sqlite').resolve()
    wal=Path(str(path)+'-wal');assert not wal.exists() or wal.stat().st_size==0
    db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    best={k:{} for k in ROUTES};events=Counter();n=0
    sql="""SELECT e.payload,json_extract(d.payload,'$.simulation_complete'),
        json_extract(d.payload,'$.topology_hash') FROM events e
        LEFT JOIN designs d ON d.key=json_extract(e.payload,'$.candidate_hash')
        WHERE e.kind='archive_insertion'"""
    for payload,complete,topology in db.execute(sql):
        event=orjson.loads(payload);n+=1
        if topology is None:raise ValueError(f'Missing candidate linkage: {run}')
        if not complete:continue
        loss=event['loss']
        if not isinstance(loss,(int,float)) or not math.isfinite(loss) or loss<0:
            raise ValueError(f'Invalid completed loss: {run}')
        if loss>=MAX_THRESHOLD:continue
        source=event['source'];origin=event['origin'];fitted=event.get('stream')=='optimization'
        if source=='llm' or origin=='llm':route='llm_lineage'
        elif source=='rl' or origin=='rl':route='rl_lineage'
        else:route='other'
        update(best[route],topology,loss);update(best['all'],topology,loss);events[route]+=1
        if source=='llm' and not fitted:update(best['llm_direct'],topology,loss)
        if source=='rl' and not fitted:update(best['rl_direct'],topology,loss)
    db.close()
    # Independent direct-proposal ledger checks the unfitted LLM curve.
    direct={};f=run/'direct_llm_proposals.jsonl'
    for line in f.read_text().splitlines() if f.exists() else []:
        row=orjson.loads(line);record=row.get('record')
        if row['valid'] and record and record['simulation_complete'] and record['loss']<MAX_THRESHOLD:
            update(direct,record['topology_hash'],record['loss'])
    assert direct.keys()==best['llm_direct'].keys(),(task,seed,'direct LLM topology mismatch')
    for topology,loss in direct.items():
        assert math.isclose(loss,best['llm_direct'][topology],rel_tol=1e-9,abs_tol=1e-12)
    total_at_005=sum(v<.05 for v in best['all'].values())
    union=json.loads((run/'terminal_union.json').read_text())
    assert total_at_005==union['union_qualified'],(task,seed,total_at_005,union)
    for direct_route,lineage in [('llm_direct','llm_lineage'),('rl_direct','rl_lineage')]:
        assert best[direct_route].keys()<=best[lineage].keys()
        assert all(best[lineage][k]<=v for k,v in best[direct_route].items())
    result=dict(task=task,seed=seed,run=str(run.resolve()),replacement=replacement,
                events_scanned=n,best_by_topology=best,counts_at_005={k:sum(v<.05 for v in d.values()) for k,d in best.items()})
    print(task,seed,'verified',result['counts_at_005'],flush=True)
    return result


def count_curve(losses,thresholds):
    # Strict qualification: loss < threshold, not <=.
    return np.searchsorted(np.sort(list(losses)),thresholds,side='left')


def plot_task(task,runs,pdf):
    arrays={route:np.array([count_curve(r['best_by_topology'][route].values(),THRESHOLDS) for r in runs]) for route in ROUTES}
    for a in arrays.values():assert np.all(np.diff(a,axis=1)>=0)
    title=TASKS[task][0];n=len(runs)
    fig,ax=plt.subplots(figsize=(8.8,5.8))
    styles=[('llm_lineage','LLM lineage + numerical refinement / mutation','#a64e85','-'),
            ('rl_lineage','RL lineage + numerical refinement / mutation','#168577','-'),
            ('llm_direct','Direct LLM proposals','#a64e85','--'),
            ('rl_direct','Direct RL proposals (before refinement)','#168577','--')]
    for route,label,color,style in styles:
        values=arrays[route];mean=values.mean(axis=0)
        if style=='-':
            q=np.quantile(values,[.25,.75],axis=0);ax.fill_between(THRESHOLDS,q[0],q[1],color=color,alpha=.12)
        ax.plot(THRESHOLDS,mean,color=color,ls=style,lw=2 if style=='-' else 1.4,label=label)
    ax.axvline(.05,color='#555555',ls=':',lw=1.2,label='Original threshold: 0.05')
    ax.set(xscale='log',yscale='symlog',xlim=(THRESHOLDS[0],THRESHOLDS[-1]),
           xlabel='Loss threshold τ (a topology qualifies when loss < τ)',
           ylabel='Mean distinct topologies per run',
           title='')
    fig.suptitle(f'{title} · {n}/20 completed runs',fontsize=13,y=.99)
    ax.set_yscale('symlog',linthresh=1,linscale=.55)
    ax.set_xticks([.0001,.001,.01,.05,.2],['0.0001','0.001','0.01','0.05','0.2'])
    ymax=max(a.mean(axis=0).max() for k,a in arrays.items() if k in ['llm_lineage','rl_lineage'])
    ticks=[0,1,10,100,1000,10000];ax.set_yticks([t for t in ticks if t<=max(1,ymax*1.3)])
    ax.yaxis.set_major_formatter(ScalarFormatter());ax.set_ylim(bottom=0)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='upper center',bbox_to_anchor=(.52,.935),ncol=2,frameon=False)
    ax.grid(alpha=.15)
    at={route:float(np.mean([r['counts_at_005'][route] for r in runs])) for route in ROUTES}
    fig.text(.09,.14,f"At τ = 0.05: LLM lineage {at['llm_lineage']:.2f}; RL lineage {at['rl_lineage']:.2f}; "
             f"direct LLM {at['llm_direct']:.2f}; direct RL {at['rl_direct']:.2f} topologies/run.",fontsize=9)
    fig.text(.09,.095,'Solid: explicit origin, including descendants. Dashed: direct proposals at supplied / initial rates.\n'
             'Shading: 25–75% across runs. Counts can overlap between routes; they must not be added.',fontsize=8.5)
    fig.text(.09,.047,'Both axes use logarithmic scaling (counts from 0 to 1 are linear). Species labels are preserved.\n'
             'Post-hoc threshold sweep; search used τ = 0.05. No pruning or perturbation robustness filter.',fontsize=8)
    fig.tight_layout(rect=(0,.19,1,.82))
    fig.savefig(OUT/f'{task}_threshold_sweep.png',dpi=180);fig.savefig(OUT/f'{task}_threshold_sweep.pdf');pdf.savefig(fig);plt.close(fig)
    table=[]
    for i,threshold in enumerate(THRESHOLDS):
        for route,a in arrays.items():
            table.append(dict(task=task,runs=n,threshold=float(threshold),route=route,mean=float(a[:,i].mean()),
                q25=float(np.quantile(a[:,i],.25)),median=float(np.median(a[:,i])),q75=float(np.quantile(a[:,i],.75))))
    return table,dict(task=task,runs=n,counts_at_005=at)


def main():
    tasks=['hill','ultrasensitive','biphasic','classifier','parity']
    if '--plot-only' in sys.argv:
        previous=json.loads((OUT/'summary.json').read_text())
        selections={k:([],previous['excluded'][k]) for k in tasks}
        with gzip.open(OUT/'topology_route_minima.json.gz','rt') as f:results=json.load(f)
    else:
        selections={k:select_runs(k) for k in tasks}
        items=[(k,*item) for k in tasks for item in selections[k][0]]
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(extract,items))
        with gzip.open(OUT/'topology_route_minima.json.gz','wt') as f:json.dump(results,f)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    table=[];summaries=[]
    with PdfPages(OUT/'llm_rl_threshold_sweeps.pdf') as pdf:
        for task in tasks:
            runs=[r for r in results if r['task']==task]
            if not runs:continue
            part,summary=plot_task(task,runs,pdf);table.extend(part);summaries.append(summary)
    with (OUT/'threshold_counts.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    manifest=dict(snapshot=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),summaries=summaries,
        excluded={k:v[1] for k,v in selections.items()},
        attribution='Event-time source and origin. RL lineage includes explicit RL-origin GA descendants and AdaGrad fits. LLM lineage likewise includes mutation and fitting. Direct curves exclude refinement/mutation. Random-origin designs are retained as other in data; all-route union is retained separately. Routes overlap, never sum to a unique total. RL can learn from LLM elites without carrying explicit LLM origin.',
        counting='Minimum complete canonical loss per recorded topology hash, per route, per run; strict loss < threshold. No permutation-of-species quotient or pruning. Mean/IQR across completed runs, not pooled cross-run union. Original quality threshold fixed at .05; post-hoc relaxation does not reproduce a changed search.',
        validation='All source links resolved; complete finite losses; direct LLM minima match independent proposal JSONL; all-route counts at .05 match terminal_union.json for every run; direct route subsets and monotone counts verified.')
    (OUT/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
