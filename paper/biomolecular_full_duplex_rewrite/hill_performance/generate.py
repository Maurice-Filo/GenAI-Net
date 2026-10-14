"""Descriptive Hill full-duplex results from frozen completed runs; no new simulations."""
from pathlib import Path
import json,sqlite3,zlib,csv
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'comparisons/search_exploration/reports/dose-hill-full-20seed-20261001'
OUT=Path(__file__).resolve().parent
rows=[];curves=[];responses=[];all_topologies=set();winners=[]
for seed in range(1,21):
 run=BASE/'runs'/f'dose-hill-full-seed{seed:02}'
 done=json.loads((run/'completed.json').read_text());assert done['status']=='complete' and done['epochs']==100
 m=done['metrics'];union=json.loads((run/'terminal_union.json').read_text())
 metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
 by_epoch={int(x['epoch'])+1:x for x in metrics if 'epoch' in x}
 losses=np.array([by_epoch[e]['Archive/best_loss'] for e in range(1,101)])
 losses[-1]=min(losses[-1],m['Archive/best_loss']);losses=np.minimum.accumulate(losses);curves.append(losses)
 first=int(np.flatnonzero(losses<.05)[0])+1
 direct=[json.loads(x) for x in (run/'direct_llm_proposals.jsonl').read_text().splitlines()]
 direct_losses=[x['record']['loss'] for x in direct if x.get('valid') and isinstance(x.get('record',{}).get('loss'),(int,float))]
 db=sqlite3.connect(f'file:{run}/exploration.sqlite?mode=ro&immutable=1',uri=True)
 winner=json.loads(db.execute("SELECT payload FROM designs WHERE json_extract(payload,'$.loss') IS NOT NULL ORDER BY json_extract(payload,'$.loss') LIMIT 1").fetchone()[0])
 assert np.isclose(winner['loss'],m['Archive/best_loss'],rtol=1e-9,atol=1e-12)
 all_topologies.update(x[0] for x in db.execute("SELECT DISTINCT json_extract(payload,'$.topology_hash') FROM designs WHERE json_extract(payload,'$.qualified')=1"))
 blob=db.execute('SELECT payload FROM diagnostics WHERE key=?',(winner['evaluation_cache_key'],)).fetchone()
 assert blob is not None
 info=json.loads(zlib.decompress(blob[0]));assert 'diagnostic_sampling' not in info
 inputs=np.asarray(info['inputs']).ravel();target=np.asarray(info['setpoint']).ravel();outputs=np.asarray(info['outputs'])[:,0,-1]
 responses.append(outputs);winners.append(dict(seed=seed,**winner))
 db.close()
 row=dict(seed=seed,best_loss=m['Archive/best_loss'],first_qualifying_epoch=first,
          final_archive_topologies=m['Archive/qualified_topologies'],full_portfolio_topologies=union['union_qualified'],
          best_direct_llm_loss=min(direct_losses),qualifying_direct_llm_proposals=sum(x<.05 for x in direct_losses),
          winning_source=winner['source'],winning_origin=winner.get('origin'),
          evaluation_calls=done['counters']['expensive_evaluations'],minutes=done['elapsed_seconds']/60)
 rows.append(row)
with (OUT/'per_run.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for key in ['best_loss','first_qualifying_epoch','final_archive_topologies','full_portfolio_topologies','best_direct_llm_loss','evaluation_calls','minutes']:
 a=np.array([r[key] for r in rows]);summary[key]=dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)))
summary.update(successes=sum(r['best_loss']<.05 for r in rows),below_001=sum(r['best_loss']<.01 for r in rows),
               direct_llm_successful_runs=sum(r['best_direct_llm_loss']<.05 for r in rows),
               winning_sources=dict(Counter(r['winning_source'] for r in rows)),
               winning_origins=dict(Counter(r['winning_origin'] for r in rows)),
               pooled_qualifying_topologies=len(all_topologies),
               interpretation='Single full-duplex cohort; descriptive association only. Qualification means nominal loss <0.05, not perturbation robustness. Portfolio counts use recorded topology hashes, not symmetry-canonical/pruned motifs. Response panel shows stored final time points; reported loss is the task dynamic tracking objective.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'winning_designs.json').write_text(json.dumps(winners,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(10.5,7.4),constrained_layout=True)
a=axes[0,0];curves=np.array(curves);x=np.arange(1,101)
for c in curves:a.plot(x,c,color='#4188ae',alpha=.17,lw=.7)
q=np.quantile(curves,[.25,.5,.75],axis=0);a.fill_between(x,q[0],q[2],color='#4188ae',alpha=.23,label='25–75% of runs');a.plot(x,q[1],color='#15557b',lw=2,label='Median')
a.axhline(.05,color='#b34936',ls='--',lw=1,label='Qualification: loss <0.05');a.set(yscale='log',xlabel='Completed epochs',ylabel='Best loss so far',title='A  Convergence across 20 runs');a.legend(fontsize=8)
a=axes[0,1];a.step(x,np.mean(curves<.05,axis=0)*100,where='post',color='#15557b',lw=2);a.set(xlabel='Completed epochs',ylabel='Runs with loss <0.05 (%)',ylim=(-2,105),title='B  Search reliability');a.axhline(100,color='.6',ls=':',lw=.8)
a=axes[1,0]
for y in responses:a.plot(inputs,y,color='#4188ae',alpha=.25,lw=1)
u=np.linspace(0,1,300);a.plot(u,2*u**2/(.25**2+u**2),'k--',lw=1.6,label='Target Hill curve')
best=int(np.argmin([r['best_loss'] for r in rows]));a.plot(inputs,responses[best],'o-',ms=4,color='#cf6c32',lw=1.5,label=f'Best run (seed {best+1})');a.set(xlabel='Input concentration',ylabel='Output at final simulation time',title='C  Winning networks: dose response');a.legend(fontsize=8)
a=axes[1,1];seeds=np.arange(1,21);a.bar(seeds,[r['full_portfolio_topologies'] for r in rows],color='#a9c9dd',label='All qualifying topologies found');a.bar(seeds,[r['final_archive_topologies'] for r in rows],color='#15557b',label='Retained in final archive');a.set(xlabel='Run / seed',ylabel='Distinct qualifying topology hashes',title='D  Nominal structural diversity',xticks=[1,5,10,15,20]);a.legend(fontsize=8)
fig.suptitle('Hill dose response · full duplex · 20 runs × 100 epochs',fontsize=14)
fig.savefig(OUT/'hill_performance.pdf');fig.savefig(OUT/'hill_performance.png',dpi=180)
print(json.dumps(summary,indent=2))
