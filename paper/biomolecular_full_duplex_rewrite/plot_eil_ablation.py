"""Read-only, paired Logic EIL ablation analysis; plots and review text are separate.
Run from the repository root with .venv/bin/python. No simulations or model calls.
"""
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from RL4CRN.search_exploration.archive import descriptor
from count_output_categories import Reaction
from proposed_results_figures import style, axes_style

ROOT = Path('/local1/rossin/search-exploration/logic-flash-full-eil0-20seed-20261012-r2')
OUT = HERE / 'eil_ablation'
CATALOG = Path('/local1/rossin/search-exploration/logic5-portfolio-diagnostics-order-fallback-20260924/selection.json.gz')
COLORS = {'eil': '#943C87', 'zero': '#737980'}
LABELS = {'eil': 'Full duplex', 'zero': 'Full duplex, EIL = 0'}

def read(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def extract(item):
    job, arm, library = item
    path = Path(job['reference' if arm == 'eil' else 'output'])
    manifest = read(path/'manifest.json')
    completed = read(path/'completed.json')
    terminal = read(path/'terminal_union.json')
    assert completed['status'] == 'complete' and completed['epochs'] == 100
    assert completed['termination'] == 'epochs_and_all_streams_drained'
    assert manifest['communication_mode'] == 'full'
    assert manifest['config']['imitation_weight'] == (0.1 if arm == 'eil' else 0.0)
    assert completed['metrics']['LLM/pending_rounds'] == 0
    threshold = manifest['config']['quality_threshold']
    metrics = lines(path/'metrics.jsonl')
    by_epoch = {int(m['epoch']):m for m in metrics if int(m['epoch']) < 100}
    assert sorted(by_epoch) == list(range(100))
    records, first = {}, {}
    topologies, categories, first_topology = {}, {}, {}
    wal = path/'exploration.sqlite-wal'
    assert not wal.exists() or wal.stat().st_size == 0
    with sqlite3.connect(f'file:{path}/exploration.sqlite?mode=ro&immutable=1', uri=True) as db:
        for (payload,) in db.execute("SELECT payload FROM designs WHERE json_extract(payload,'$.simulation_complete')=1 AND json_extract(payload,'$.loss') < ?", (threshold,)):
            v = json.loads(payload)
            records[v['candidate_hash']] = v
        for (payload,) in db.execute("SELECT payload FROM events WHERE kind='archive_insertion' AND json_extract(payload,'$.loss') < ? ORDER BY id", (threshold,)):
            v = json.loads(payload)
            if v['candidate_hash'] in records:
                first.setdefault(v['candidate_hash'], int(v['epoch']))
    def add(v, epoch):
        ids = tuple(sorted(v['reaction_ids']))
        assert len(ids) == len(set(ids)) == 5
        th = v['topology_hash']
        assert th not in topologies or topologies[th] == ids
        topologies[th] = ids
        cat = descriptor(ids, library, ['OUT'], 'reaction_roles_v1', '')
        assert 'region' not in v or v['region'] == cat
        categories[th] = cat
        first_topology[th] = min(first_topology.get(th, 100), epoch)
    assert set(records) <= set(first), 'Qualifying record lacks a first-insertion event'
    for key, v in records.items():
        add(v, first[key])
    for row in lines(path/'direct_llm_proposals.jsonl'):
        v = row.get('record')
        if v and v.get('simulation_complete') and v['loss'] < threshold:
            add(v, int(row['completed_epoch']))
    assert len(set(topologies.values())) == len(topologies), 'Different hashes encode the same reaction set'
    assert len(topologies) == terminal['union_qualified'] == completed['metrics']['Union/qualified_topologies']
    assert min([v['loss'] for v in records.values()] + [terminal['llm_best_loss']]) == terminal['union_best_loss']
    curve = []
    for epoch in range(100):
        m = by_epoch[epoch]
        curve.append(dict(arm=arm, seed=job['seed'], completed_epochs=epoch+1,
                          loss=m['Archive/best_loss'], qualified=sum(e <= epoch for e in first_topology.values()),
                          evaluations=m['Compute/expensive_evaluations']))
    endpoint = dict(arm=arm, seed=job['seed'], loss=terminal['union_best_loss'],
                    qualified=len(topologies), categories=len(set(categories.values())),
                    evaluations=completed['metrics']['Compute/expensive_evaluations'],
                    merged=completed['metrics']['LLM/merged_candidates'])
    members = [dict(arm=arm, seed=job['seed'], reaction_ids=';'.join(map(str,ids)),
                    category=categories[th], first_recorded_epoch=first_topology[th]) for th,ids in sorted(topologies.items())]
    evidence = dict(arm=arm, seed=job['seed'], path=str(path), threshold=threshold,
                    hashes={name:sha(path/name) for name in ['manifest.json','completed.json','terminal_union.json','metrics.jsonl','direct_llm_proposals.jsonl']},
                    endpoint_count_verified=True, descriptors_verified=True,
                    request_errors=[h.get('error') for h in lines_or_json(path/'llm_history.json') if h.get('error')])
    print(f"Verified {arm} seed {job['seed']:02d}: {len(topologies)} topologies, {endpoint['categories']} categories", flush=True)
    return endpoint, curve, members, evidence

def lines_or_json(p):
    return read(p)

def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)

def exact_pratt_p(a, b):
    # Exact paired sign permutation of Pratt signed ranks, including tied ranks.
    d = np.asarray(a)-np.asarray(b)
    ranks = np.rint(2*rankdata(abs(d))).astype(int)
    nonzero = ranks[d != 0]
    if not len(nonzero):
        return 1.0
    counts = np.zeros(int(nonzero.sum())+1, dtype=np.int64)
    counts[0] = 1
    for rank in nonzero:
        counts[rank:] += counts[:-rank].copy()
    positive = int(ranks[d > 0].sum())
    tail = min(positive, int(nonzero.sum())-positive)
    return min(1.0, 2*int(counts[:tail+1].sum())/2**len(nonzero))


def summarize(endpoints):
    summary={}
    for key in ['loss','qualified','categories','evaluations','merged','common_loss']:
        a=np.array([r[key] for r in endpoints if r['arm']=='eil']);b=np.array([r[key] for r in endpoints if r['arm']=='zero'])
        summary[key]={arm:dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),max=float(v.max())) for arm,v in [('eil',a),('zero',b)]}
        rng=np.random.default_rng(20261013);idx=rng.integers(0,20,(20000,20))
        d=a-b
        summary[key]['paired_mean_difference_ci95']=np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist()
        summary[key]['eil_lower']=int(sum(a<b));summary[key]['eil_higher']=int(sum(a>b))
        if key in ['loss','qualified','categories']:
            summary[key]['paired_wilcoxon_p']=exact_pratt_p(a,b)
    # Holm correction across the three displayed final endpoints.
    tests=sorted((summary[k]['paired_wilcoxon_p'],k) for k in ['loss','qualified','categories'])
    bound=0
    for i,(p,k) in enumerate(tests):
        bound=max(bound,min(1,(3-i)*p));summary[k]['holm_p']=bound
    return summary

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=200,facecolor='white')
    plt.close(fig)

def plot(endpoints,curves):
    style()
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.55),gridspec_kw={'width_ratios':[1.1,1.1,.9]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.77,wspace=.38)
    x=np.arange(1,101)
    for i,arm in enumerate(['eil','zero']):
        color=COLORS[arm];rows=[r for r in endpoints if r['arm']==arm]
        for ax,key,curvekey in [(axes[0],'loss','loss'),(axes[1],'qualified','qualified')]:
            a=np.array([[v[curvekey] for v in curves if v['arm']==arm and v['seed']==seed] for seed in range(1,21)])
            lo,med,hi=np.quantile(a,[.25,.5,.75],axis=0)
            ax.step(x,med,where='post',color=color,lw=1.8)
            ax.fill_between(x,lo,hi,step='post',color=color,alpha=.12,lw=0)
            low,mid,high=np.quantile([v[key] for v in rows],[.25,.5,.75])
            ax.errorbar(108+(i-.5)*2,mid,yerr=[[mid-low],[high-mid]],fmt='o',color=color,ms=4,capsize=2,lw=1.2)
        values=np.array([v['categories'] for v in rows]);idx=np.random.default_rng(20261013).integers(0,20,(20000,20))
        low,high=np.quantile(values[idx].mean(axis=1),[.025,.975])
        bar_x = 1-i  # Ablation first, full system second.
        axes[2].bar(bar_x,values.mean(),width=.6,color=color,alpha=.2,edgecolor=color,linewidth=.8)
        jitter=np.random.default_rng(42).uniform(-.17,.17,20)
        axes[2].scatter(bar_x+jitter,values,s=18,color=color,alpha=.75,linewidths=.3,edgecolors='white',zorder=3)
        axes[2].errorbar(bar_x,values.mean(),yerr=[[values.mean()-low],[high-values.mean()]],fmt='none',color=color,capsize=4,lw=1.5,zorder=4)
    for ax in axes[:2]:
        ax.set(xlim=(0,114),xticks=[0,25,50,75,100,108],xticklabels=['0','25','50','75','100','T'],xlabel='Completed epochs')
        ax.tick_params(axis='x',labelsize=8)
    axes[0].set(yscale='log',ylabel='Best loss')
    axes[1].set(ylabel='Cumulative qualifying topologies',ylim=(0,None))
    ratios = {key: np.mean([r[key] for r in endpoints if r['arm']=='eil']) /
              np.mean([r[key] for r in endpoints if r['arm']=='zero'])
              for key in ['qualified','categories']}
    lower, upper = [np.mean([r['qualified'] for r in endpoints if r['arm']==arm])
                    for arm in ['zero','eil']]
    axes[1].plot([114,116,116,114],[lower,lower,upper,upper],color='#51565C',lw=.9)
    axes[1].text(119,(lower+upper)/2,f"{ratios['qualified']:.1f}×",
                 ha='left',va='center',color=COLORS['eil'],fontsize=10,fontweight='bold')
    axes[1].set_xlim(0,132)
    bracket_y = max(r['categories'] for r in endpoints)+3
    axes[2].plot([0,0,1,1],[bracket_y-1,bracket_y,bracket_y,bracket_y-1],
                 color='#51565C',lw=.9)
    axes[2].text(.5,bracket_y+1,f"{ratios['categories']:g}×",ha='center',va='bottom',
                 color=COLORS['eil'],fontsize=10,fontweight='bold')
    axes[2].set(xticks=[0,1],xticklabels=['EIL = 0','Full duplex'],
                ylabel='Qualifying categories',ylim=(0,bracket_y+7))
    for ax,title in zip(axes,['A  Convergence','B  Structural discoveries','C  Category coverage']):
        axes_style(ax,title)
        for spine in ax.spines.values():spine.set_linewidth(.7)
    fig.legend([Line2D([],[],color=COLORS[a],lw=2) for a in COLORS],[LABELS[a] for a in COLORS],loc='upper center',bbox_to_anchor=(.5,1),ncol=2,frameon=False)
    save(fig,'eil_ablation')
    # SI: same best-loss trajectories aligned to the largest common allowance.
    cap=min(r['evaluations'] for r in endpoints)//1000*1000
    grid=np.linspace(0,cap,301)
    fig,ax=plt.subplots(figsize=(5.6,3.4));fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.9)
    for arm in COLORS:
        ys=[]
        for seed in range(1,21):
            rs=[r for r in curves if r['arm']==arm and r['seed']==seed]
            costs=np.array([r['evaluations'] for r in rs]);loss=np.array([r['loss'] for r in rs]);ix=np.searchsorted(costs,grid,side='right')-1
            ys.append(np.where(ix>=0,loss[np.maximum(ix,0)],np.nan))
        a=np.array(ys);mask=np.all(np.isfinite(a),axis=0);lo,med,hi=np.quantile(a[:,mask],[.25,.5,.75],axis=0)
        ax.step(grid[mask]/1000,med,where='post',color=COLORS[arm],lw=1.8,label=LABELS[arm]);ax.fill_between(grid[mask]/1000,lo,hi,step='post',color=COLORS[arm],alpha=.12,lw=0)
    ax.set(yscale='log',xlabel='Charged evaluations (thousands)',ylabel='Best loss',xlim=(0,cap/1000));axes_style(ax,'Logic: common simulation allowance');ax.legend(frameon=False,fontsize=9)
    save(fig,'eil_ablation_common_allowance')

def main():
    OUT.mkdir(exist_ok=True)
    jobs=read(ROOT/'plan.json')['jobs']
    assert [j['seed'] for j in jobs]==list(range(1,21))
    for j in jobs:
        a=read(Path(j['reference'])/'manifest.json');b=read(Path(j['output'])/'manifest.json')
        assert {k:(a['config'].get(k),b['config'].get(k)) for k in a['config'].keys()|b['config'].keys() if a['config'].get(k)!=b['config'].get(k)}=={'imitation_weight':(.1,0.)}
        for k in ['task_config','julia_recipe','model','system_prompt_sha256','task_prompt_sha256','search_workers','optimization_workers','llm_every','llm_candidates','llm_timeout','policy']:
            assert a[k]==b[k],k
    with gzip.open(CATALOG,'rt') as f:catalog=json.load(f)['catalog']
    library={int(k):Reaction(v['structure']) for k,v in catalog.items()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        data=list(pool.map(extract,[(j,arm,library) for arm in ['eil','zero'] for j in jobs]))
    endpoints=[v[0] for v in data];curves=[r for v in data for r in v[1]];members=[r for v in data for r in v[2]]
    cap=min(r['evaluations'] for r in endpoints)//1000*1000
    for row in endpoints:
        eligible=[v for v in curves if v['arm']==row['arm'] and v['seed']==row['seed'] and v['evaluations']<=cap]
        row['common_loss']=eligible[-1]['loss'];row['common_allowance']=cap
    write_csv('endpoints.csv',endpoints);write_csv('curves.csv',curves);write_csv('topology_membership.csv',members)
    (OUT/'audit.json').write_text(json.dumps({'source':str(ROOT),'catalog_sha256':sha(CATALOG),'paired_config_check':'Only imitation_weight differs; task, prompts, model, worker settings and policy match','epoch_mapping':'Recorded epoch 0..99 maps to completed epochs 1..100. Recorded epoch 100 is terminal draining, plotted separately at T. Topology discovery uses first archive-insertion or direct-proposal receipt, regardless of archive admission.','statistics':'Exact paired Pratt signed-rank sign-permutation tests; Holm correction across final loss, qualifying topology count and category count. Paired seed bootstrap (20,000 resamples) gives marginal mean-difference intervals. Panel C shows marginal bootstrap mean intervals.', 'counts':'Complete, finite, below-threshold designs deduplicated by sorted reaction IDs; species labels preserved. No pruning or robustness filter.','runs':[v[3] for v in data]},indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(summarize(endpoints),indent=2)+'\n')
    plot(endpoints,curves)
    print('Saved',OUT)

if __name__=='__main__':main()
