"""Matched prompt-ablation convergence and the paper's unchanged RPA motif screens.

Read-only SQLite transactions include WAL data from active runs. Preliminary
portfolio comparisons use the same seeds and common completed epoch. Final
portfolios include terminal synchronization and are checked against the paper.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from comparisons.search_exploration.analyze_rpa_communication_families import antithetic_motif, aif_like
from comparisons.search_exploration.analyze_rpa_mechanisms import integral_balance, reactions_for, dump_json, write_csv

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/biomolecular_full_duplex_rewrite'
COLORS = {'reference': '#9A4C89', 'unspecified': '#26835B'}
LABELS = {'reference': 'Original prompt', 'unspecified': 'Mechanism-unspecified prompt'}


def read(path):
    return json.loads(path.read_text())


def metrics(folder):
    raw = (folder / 'metrics.jsonl').read_bytes()
    rows = []
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise
    return sorted(rows, key=lambda m:m['epoch']), hashlib.sha256(raw).hexdigest()


def records(folder, cutoff):
    # Do not use immutable mode on a live WAL database, or inspect diagnostics.
    with sqlite3.connect((folder / 'exploration.sqlite').as_uri() + '?mode=ro', uri=True, timeout=30) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('BEGIN')
        threshold = read(folder / 'manifest.json')['config']['quality_threshold']
        seen = '' if cutoff is None else """WITH seen AS (
            SELECT json_extract(payload,'$.candidate_hash') AS key,
                   min(json_extract(payload,'$.epoch')) AS epoch
            FROM events WHERE kind='archive_insertion' GROUP BY key)
            """
        join = '' if cutoff is None else ' JOIN seen s ON d.key=s.key '
        condition = '' if cutoff is None else ' AND s.epoch <= ? '
        sql = seen + """SELECT payload FROM (SELECT d.payload,
            row_number() OVER (PARTITION BY json_extract(d.payload,'$.topology_hash')
              ORDER BY json_extract(d.payload,'$.loss'), d.key) AS rank
            FROM designs d """ + join + """ WHERE json_extract(d.payload,'$.simulation_complete')=1
            AND json_type(d.payload,'$.loss') IN ('integer','real')
            AND json_extract(d.payload,'$.loss') >= 0
            AND json_extract(d.payload,'$.loss') < ? """ + condition + ') WHERE rank=1'
        result = [json.loads(r[0]) for r in db.execute(sql, (threshold,) if cutoff is None else (threshold, cutoff))]
        db.commit()
    return sorted(result, key=lambda r:(r['loss'],r['candidate_hash']))


def catalog(config):
    from comparisons.rpa_search.src.common.rpa_task import build_rpa_components
    from RL4CRN.utils.results_database import serialize_crn
    crn, parts, _, _ = build_rpa_components(config)
    mapping = {}
    for rid in range(len(parts[0])):
        r = parts[0].get_reaction(rid)
        mapping[rid] = dict(reaction_id=rid, structure=dict(type=type(r).__name__,
            reactants=list(r.reactant_labels), products=list(r.product_labels), inputs=list(r.input_channels)))
    return mapping, json.loads(serialize_crn(crn)['parameters_json'])


def curve(run, axis):
    rows = [m for m in run['metrics'] if m['epoch'] < 100]
    x = np.array([m['epoch']+1 if axis=='epoch' else m['Compute/expensive_evaluations'] for m in rows])
    y = np.minimum.accumulate([m['Archive/best_loss'] for m in rows])
    assert np.all(np.diff(x)>=0) and np.all(np.isfinite(y)) and np.all(y>0)
    return x, y


def render(data, out):
    plt.rcParams.update({'font.size':10, 'font.family':'DejaVu Sans', 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,3.8))
    fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.80,wspace=.26)
    for ax, axis, title in zip(axes, ('epoch','cost'), ('A  Search epochs','B  Charged evaluations')):
        lo = max(curve(r,axis)[0][0] for r in data['runs'])
        hi = min(curve(r,axis)[0][-1] for r in data['runs'])
        grid = np.unique([lo, hi] + [float(v) for r in data['runs'] for v in curve(r,axis)[0] if lo<=v<=hi])
        for arm in COLORS:
            group = [r for r in data['runs'] if r['arm']==arm]
            values=[]
            for r in group:
                x,y=curve(r,axis)
                values.append(y[np.searchsorted(x,grid,side='right')-1])
            low,med,high=np.quantile(values,[.25,.5,.75],axis=0)
            scaled=grid if axis=='epoch' else grid/1000
            ax.step(scaled,med,where='post',color=COLORS[arm],lw=2,label=LABELS[arm])
            ax.fill_between(scaled,low,high,step='post',color=COLORS[arm],alpha=.14,lw=0)
            data['convergence_endpoints'][axis][arm]=dict(allowance=float(hi),median=float(med[-1]),iqr=[float(low[-1]),float(high[-1])])
        ax.set(title=title,xlabel='Completed epochs' if axis=='epoch' else 'Charged evaluations (thousands)',ylabel='Best loss',yscale='log')
        ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    for ext in ('pdf','png'):fig.savefig(out/f'convergence.{ext}',dpi=190)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.8,1]})
    fig.subplots_adjust(left=.07,right=.985,bottom=.20,top=.80,wspace=.25)
    fields=('motif_fraction','balance_fraction','combined_fraction')
    offsets=(-.19,.19)
    for i,arm in enumerate(COLORS):
        rows=[r for r in data['per_run'] if r['arm']==arm]
        for k,field in enumerate(fields):
            vals=np.array([r[field] for r in rows])*100
            x=k+offsets[i]
            axes[0].bar(x,vals.mean(),.34,color=COLORS[arm],alpha=.82)
            axes[0].scatter(x+np.linspace(-.09,.09,len(vals)),vals,s=12,c='black',alpha=.45,zorder=3)
            axes[0].text(x,vals.mean()+2.5,f'{vals.mean():.1f}%',ha='center',fontsize=9)
        vals=np.array([r['qualified_topologies'] for r in rows])
        axes[1].bar(i,vals.mean(),.55,color=COLORS[arm],alpha=.82)
        axes[1].scatter(i+np.linspace(-.12,.12,len(vals)),vals,s=16,c='black',alpha=.5,zorder=3)
        axes[1].text(i,vals.mean()+max(vals)*.03,f'{vals.mean():.1f}',ha='center',fontsize=10)
    axes[0].set(xticks=range(3),xticklabels=['Antithetic-like\nmotif','Integral\nbalance','Motif + balance\n'+r'($\alpha<0$)'],ylabel='Qualifying topologies (%)',ylim=(0,113),title='A  Structural and algebraic screens')
    axes[1].set(xticks=range(2),xticklabels=['Original\nprompt','Mechanism-\nunspecified'],ylabel='Qualifying topologies per run',title='B  Portfolio size')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    for ext in ('pdf','png'):fig.savefig(out/f'motif_and_balance.{ext}',dpi=190)
    plt.close(fig)


def analyze(root, out, final=False):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing analysis snapshot: '+str(out))
    plan=read(root/'plan.json');progress=read(root/'progress.json')
    if final:
        assert (root/'finished.json').exists()
        assert len(progress['jobs'])==20 and all(j['status']=='complete' and j.get('storage_exit_code')==0 for j in progress['jobs'])
    runs=[]
    for j in progress['jobs']:
        folder=Path(j['output'])
        if j['status'] not in ('running','complete') or not (folder/'metrics.jsonl').exists():continue
        new,digest=metrics(folder)
        if not new:continue
        reference=Path(j['reference']);ref,refdigest=metrics(reference)
        for arm,path,ms,d in [('unspecified',folder,new,digest),('reference',reference,ref,refdigest)]:
            runs.append(dict(arm=arm,seed=j['seed'],folder=str(path),metrics=ms,metrics_sha256=d,status=j['status'] if arm=='unspecified' else 'complete'))
    assert runs,'No completed epoch available'
    cutoff=None if final else min(max(m['epoch'] for m in r['metrics'] if m['epoch']<100) for r in runs)
    task=read(root/'configs/task-01.json')
    cats,fixed=catalog(task)
    paper_rows=list(csv.DictReader((PAPER/'results_figure_proposal/rpa_integral_screen.csv').open()))
    paper={int(r['seed']):r for r in paper_rows if r['arm']=='full'}
    per_run=[];memberships=[]
    for run in runs:
        folder=Path(run['folder']);manifest=read(folder/'manifest.json')
        assert manifest['task_config']==task
        bank=records(folder,cutoff)
        assert bank,'No qualifying structures at common epoch'
        assert len({r['topology_hash'] for r in bank})==len(bank)
        result=[]
        for rec in bank:
            assert len(rec['reaction_ids'])==len(set(rec['reaction_ids']))==5 and not set(rec['reaction_ids'])&{0,1,28}
            rr=reactions_for(rec,cats,fixed);balance=integral_balance(rr)
            item=dict(arm=run['arm'],seed=run['seed'],topology_hash=rec['topology_hash'],candidate_hash=rec['candidate_hash'],loss=rec['loss'],
                reaction_ids=rec['reaction_ids'],rates=rec['rates'],motif=antithetic_motif(rr),combined=aif_like(rr,balance),**balance)
            memberships.append(item);result.append(item)
        row=dict(arm=run['arm'],seed=run['seed'],qualified_topologies=len(bank),best_qualifying_loss=bank[0]['loss'],
                 motif_fraction=float(np.mean([r['motif'] for r in result])),
                 balance_fraction=float(np.mean([r['restricted_exact_balance'] for r in result])),
                 combined_fraction=float(np.mean([r['combined'] for r in result])))
        if final and run['arm']=='reference':
            check=paper[run['seed']]
            assert row['qualified_topologies']==int(check['qualified_topologies'])
            for a,b in [('motif_fraction','antithetic_motif_fraction'),('balance_fraction','exact_balance_fraction'),('combined_fraction','aif_like_fraction')]:
                assert np.isclose(row[a],float(check[b]),rtol=1e-12,atol=1e-12),(run['seed'],a,row[a],check[b])
        per_run.append(row)
        print(run['arm'],run['seed'],'qualifying',len(bank),'combined',row['combined_fraction'],flush=True)
    data=dict(timestamp=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),campaign=str(root),final=final,
        seeds=sorted({r['seed'] for r in runs}),completed_epochs=None if final else cutoff+1,
        threshold=manifest['config']['quality_threshold'],runs=runs,per_run=per_run,
        convergence_endpoints={'epoch':{},'cost':{}},summary={},
        procedure='Best saved complete parameterization per topology and seed, identical paper motif and integral_balance functions. Equal run weighting.',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
            REPO/'comparisons/search_exploration/analyze_rpa_mechanisms.py',
            REPO/'comparisons/search_exploration/analyze_rpa_communication_families.py',root/'plan.json']})
    for arm in COLORS:
        rows=[r for r in per_run if r['arm']==arm]
        data['summary'][arm]={k:float(np.mean([r[k] for r in rows])) for k in ('qualified_topologies','motif_fraction','balance_fraction','combined_fraction')}
    render(data,out)
    dump_json(out/'snapshot.json',data)
    dump_json(out/'memberships.json',memberships)
    write_csv(out/'per_run.csv',per_run)
    scope=('All 20 seeds after 100 epochs and terminal synchronization.' if final else
           f'Preliminary snapshot: paired seeds {data["seeds"]}, screened through {cutoff+1} completed epochs in both campaigns; later discoveries excluded.')
    (out/'captions.txt').write_text(scope+'\n\nConvergence: median best loss and interquartile range across the same seeds. Each panel stops at the largest allowance reached by every included run; no extrapolation. Epoch curves exclude terminal draining.\n\nMotif and balance: bars show the mean fraction across runs; points show individual runs. The combined screen requires the structural motif, restricted exact integral balance, and alpha < 0. Portfolio counts use the original task threshold and count each topology once per run.\n')
    print(json.dumps(dict(output=str(out),final=final,seeds=data['seeds'],completed_epochs=data['completed_epochs'],summary=data['summary'],convergence=data['convergence_endpoints'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    analyze(args.root.resolve(),args.output.resolve(),args.final)
