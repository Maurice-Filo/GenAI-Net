"""Parity truth-table correctness and topology yield from saved simulations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from generate_dose_overviews import select_runs
from proposed_results_figures import style, axes_style, PURPLE


def extract():
    selected, excluded = select_runs('parity')
    assert len(selected) == 20 and not excluded
    topologies, runs = [], []
    for seed, folder, done, replaced in selected:
        assert done['status']=='complete' and done['epochs']==100 and not replaced
        path = (folder/'exploration.sqlite').resolve()
        wal = Path(str(path)+'-wal')
        assert not wal.exists() or wal.stat().st_size==0
        before=path.stat()
        db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        best, correct_witnesses = {}, set()
        digest=hashlib.sha256()
        for key, raw in db.execute("""SELECT key,payload FROM designs
                WHERE json_extract(payload,'$.simulation_complete')=1
                  AND json_extract(payload,'$.loss')<0.05 ORDER BY key"""):
            digest.update(key.encode()+b'\0'+raw.encode()+b'\n')
            d=json.loads(raw)
            assert 0 <= d['loss'] < .05 and d['simulation_complete']
            identity=tuple(sorted(d['reaction_ids']))
            assert len(identity)==len(set(identity))==12
            if d['all_rows_correct']:correct_witnesses.add(identity)
            if identity not in best or (d['loss'],key)<(best[identity]['loss'],best[identity]['candidate_hash']):
                assert key==d['candidate_hash']
                best[identity]=d
        union=json.loads((folder/'terminal_union.json').read_text())
        assert len(best)==union['union_qualified']
        assert len(best)==len({d['topology_hash'] for d in best.values()})
        records=[]
        for ids,d in sorted(best.items()):
            assert isinstance(d['all_rows_correct'],bool)
            blob=db.execute('SELECT payload FROM diagnostics WHERE key=?',(d['evaluation_cache_key'],)).fetchone()
            diagnostic={}
            correct_rows=16 if d['all_rows_correct'] else None
            if blob is not None:
                info=json.loads(zlib.decompress(blob[0]))
                inputs=np.asarray(info['inputs']); target=np.asarray(info['setpoint'])[:,0]
                outputs=np.asarray(info['outputs'])[:,0,-1]
                assert inputs.shape==(16,4) and outputs.shape==target.shape==(16,)
                assert len({tuple(row) for row in inputs})==16
                np.testing.assert_array_equal(target,inputs.sum(axis=1)%2)
                assert np.isfinite(outputs).all() and info['time_horizon'][-1]==100
                margins=(2*target-1)*(outputs-.5)
                correct=margins>1e-6
                assert bool(correct.all())==d['all_rows_correct']
                correct_rows=int(correct.sum())
                diagnostic=dict(inputs=inputs.tolist(),targets=target.tolist(),terminal_outputs=outputs.tolist(),
                                diagnostics_sha256=hashlib.sha256(blob[0]).hexdigest())
            row=dict(seed=seed,topology_hash=d['topology_hash'],candidate_hash=d['candidate_hash'],
                     reaction_ids=list(ids),loss=d['loss'],correct_rows=correct_rows,
                     all_rows_correct=d['all_rows_correct'],diagnostics_available=blob is not None,**diagnostic)
            records.append(row)
        db.close()
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        topologies.extend(records)
        runs.append(dict(seed=seed,folder=str(folder),qualifying_topologies=len(records),
                         fully_correct_topologies=sum(r['all_rows_correct'] for r in records),
                         any_fully_correct_qualifying_parameterization=len(correct_witnesses),
                         qualifying_records_sha256=digest.hexdigest(),
                         terminal_union_sha256=hashlib.sha256((folder/'terminal_union.json').read_bytes()).hexdigest(),
                         database_size=after.st_size,database_mtime_ns=after.st_mtime_ns))
        print(seed,len(records),runs[-1]['fully_correct_topologies'],flush=True)
    snapshot=dict(runs=runs,topologies=topologies,loss_threshold=.05,classification_threshold=.5,
                  margin=1e-6,time=100,representative='Lowest recorded canonical loss per sorted added-reaction set within each run; ties by candidate hash. All complete discoveries through terminal drain, not final archive only. Species labels are preserved.',
                  validation='Counts match terminal_union for every seed; correctness uses canonical all_rows_correct and is independently recomputed from each available stored terminal response; missing diagnostics never exclude candidates; no simulations.')
    (HERE/'qualification_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    with (HERE/'qualification_per_run.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
    return snapshot


def convergence_panel(ax):
    with (HERE/'best_loss_vs_epochs.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    lookup={(int(r['seed']),r['completed_epoch']):float(r['loss']) for r in rows}
    assert len(rows)==len(lookup)==20*101
    values=np.array([[lookup[seed,str(epoch)] for epoch in range(1,101)] for seed in range(1,21)])
    finals=np.array([lookup[seed,'T'] for seed in range(1,21)])
    assert np.isfinite(values).all() and np.all(values>0)
    low,median,high=np.quantile(values,[.25,.5,.75],axis=0)
    flo,fmed,fhi=np.quantile(finals,[.25,.5,.75])
    x=np.arange(1,101)
    for y in values:
        ax.step(x,y,where='post',color=PURPLE,alpha=.12,lw=.6)
    ax.fill_between(x,low,high,step='post',color=PURPLE,alpha=.18,lw=0)
    ax.step(x,median,where='post',color=PURPLE,lw=1.7)
    ax.scatter(np.linspace(110,114,20),finals,color=PURPLE,alpha=.25,s=8)
    ax.errorbar(112,fmed,yerr=[[fmed-flo],[fhi-fmed]],color=PURPLE,fmt='o',ms=3,lw=1,capsize=2)
    ax.axhline(.05,color='#555555',ls=':',lw=1)
    ax.axvline(105,color='#AAAAAA',ls=':',lw=.7)
    ax.set(yscale='log',xlim=(0,118),xticks=[0,50,100,112],xticklabels=['0','50','100','T'],
           xlabel='Completed search epochs',ylabel='Best loss')
    axes_style(ax,'A  Convergence')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
    data=extract() if args.refresh else json.loads((HERE/'qualification_snapshot.json').read_text())
    runs=data['runs'];topologies=data['topologies']
    assert all(r['correct_rows'] is not None for r in topologies), 'Incomplete classification without row-level diagnostics'
    style();fig,panels=plt.subplots(1,3,figsize=(11.4,3.0),gridspec_kw={'width_ratios':[1.25,1,1.4]},constrained_layout=True)
    convergence_panel(panels[0])
    axes=panels[1:]
    bins=list(range(17))
    fractions=np.array([[sum(r['correct_rows']==k for r in topologies if r['seed']==s)/
                          next(r['qualifying_topologies'] for r in runs if r['seed']==s)*100
                          for k in bins] for s in range(1,21)])
    observed=[k for k in bins if fractions[:,k].sum()>0]
    # Keep categories that occur; correctness is discrete, not an interpolated scale.
    means=fractions[:,observed].mean(axis=0)
    bars=axes[0].bar(range(len(observed)),means,color=PURPLE,alpha=.65,width=.55)
    axes[0].bar_label(bars,labels=[f'{value:.2f}%' for value in means],padding=3,fontsize=9)
    axes[0].set(xticks=range(len(observed)),xticklabels=[f'{k}/16' for k in observed],
                xlabel='Correct truth-table rows',ylabel='Qualifying topologies (%)',ylim=(0,110),
                xlim=(-.7,len(observed)-.3))
    axes_style(axes[0],'B  Classification correctness')
    seeds=[r['seed'] for r in runs];counts=[r['qualifying_topologies'] for r in runs]
    correct=[r['fully_correct_topologies'] for r in runs]
    axes[1].bar(seeds,correct,color=PURPLE,label='All 16 rows correct')
    axes[1].bar(seeds,np.array(counts)-correct,bottom=correct,color='#BBC1C6',label='Fewer than 16 rows correct')
    axes[1].set(xlabel='Run / seed',ylabel='Qualifying topologies',xticks=[1,5,10,15,20],xlim=(.2,20.8))
    axes[1].legend(frameon=False,fontsize=7,loc='upper left')
    axes[1].set_ylim(0,88)
    axes_style(axes[1],'C  Number of solutions')
    for ext in ['pdf','png','svg']:fig.savefig(HERE/f'parity_qualification.{ext}',dpi=180)
    plt.close(fig)
    summary=dict(n=20,total_run_topology_pairs=len(topologies),
                 fully_correct_run_topology_pairs=sum(r['all_rows_correct'] for r in topologies),
                 observed_correct_row_counts=observed,mean_qualifying_topologies=float(np.mean(counts)),
                 mean_fully_correct_topologies=float(np.mean(correct)),min_count=min(counts),max_count=max(counts),
                 runs_with_fully_correct_solution=sum(v>0 for v in correct),
                 representatives_with_diagnostics=sum(r['diagnostics_available'] for r in topologies),
                 source_sha256=hashlib.sha256((HERE/'qualification_snapshot.json').read_bytes()).hexdigest())
    (HERE/'qualification_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
