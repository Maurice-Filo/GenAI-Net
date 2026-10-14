"""Four-panel snapshots of completed breadth runs.

Uses the explicit LLM-recovery replacements irrespective of achieved performance.
Active and queued runs are excluded, keeping convergence denominators fixed.
"""
from pathlib import Path
import json, sqlite3, zlib, csv
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'comparisons/search_exploration/reports'
OUT=Path(__file__).resolve().parent
TASKS={
 'hill':('Hill dose response','dose-hill-full-20seed-20261001','dose-hill-full-seed',lambda u:2*u**2/(.25**2+u**2)),
 'ultrasensitive':('Ultrasensitive dose response','dose-ultrasensitive-full-20seed-20261001','dose-ultrasensitive-full-seed',lambda u:2*u**8/(.5**8+u**8)),
 'classifier':('Classification','classifier-full-20seed-20261001','classifier-full-seed',None),
 'parity':('Four-input parity','parity12-breadth-full-20seed-20261001','parity12-breadth-full-seed',None),
 'biphasic':('Biphasic dose response','dose-biphasic-full-20seed-20261001','dose-biphasic-full-seed',lambda u:8*u/(1+u)/(1+(u/.55)**4)),
}

def read(p):return json.loads(p.read_text())

def select_runs(key):
 title,campaign,prefix,target=TASKS[key]
 root=REPORTS/campaign;progress={j['seed']:j for j in read(root/'progress.json')['jobs']}
 replacements={}
 if key=='ultrasensitive':
  rr=REPORTS/'dose-ultrasensitive-llm-recovery-20261005'
  if (rr/'replacement_manifest.json').exists():
   manifest=read(rr/'replacement_manifest.json')
   replacements={seed:rr for seed in manifest['seeds']}
 selected=[];excluded=[]
 for seed in range(1,21):
  base=replacements.get(seed,root);run=base/'runs'/f'{prefix}{seed:02}'
  jobs=progress if base==root else {j['seed']:j for j in read(base/'progress.json')['jobs']}
  status=jobs[seed]['status']
  if status!='complete':excluded.append(dict(seed=seed,status=status));continue
  done=read(run/'completed.json');assert done['status']=='complete' and done['epochs']==100
  marker=run/'diagnostic_downsampling.json'
  if marker.exists() and read(marker)['status']!='complete':
   excluded.append(dict(seed=seed,status='storage_migration_in_progress'));continue
  selected.append((seed,run,done,seed in replacements))
 return selected,excluded

def generate(key,selection):
 title,campaign,prefix,target_fn=TASKS[key]
 selected,excluded=selection
 if not selected:return
 dest=OUT/f'{key}_performance';dest.mkdir(exist_ok=True)
 curves=[];responses=[];rows=[];sources=[]
 for seed,run,done,replaced in selected:
  metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
  by_epoch={int(x['epoch'])+1:x for x in metrics if 'epoch' in x}
  losses=np.array([by_epoch[e]['Archive/best_loss'] for e in range(1,101)])
  m=done['metrics'];losses[-1]=min(losses[-1],m['Archive/best_loss'])
  assert np.all(np.isfinite(losses)) and np.all(losses>0)
  curves.append(np.minimum.accumulate(losses))
  path=(run/'exploration.sqlite').resolve();wal=Path(str(path)+'-wal');assert not wal.exists() or wal.stat().st_size==0
  db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
  winner=read_winner(db)
  assert np.isclose(winner['loss'],m['Archive/best_loss'],rtol=1e-9,atol=1e-12)
  blob=db.execute('SELECT payload FROM diagnostics WHERE key=?',(winner['evaluation_cache_key'],)).fetchone();assert blob
  info=json.loads(zlib.decompress(blob[0]));db.close()
  target=np.asarray(info['setpoint']);endpoints=np.asarray(info['outputs'])[:,:,-1]
  if key=='classifier':
   initial=np.asarray(info['initial_conditions']);inputs=initial[:,0]-initial[:,1]
   assert target.shape==(32,2) and endpoints.shape==(32,2)
   np.testing.assert_array_equal(target.argmax(axis=1),(inputs<0).astype(int))
   outputs=endpoints[:,0]-endpoints[:,1]
  elif key=='parity':
   patterns=np.asarray(info['inputs']);inputs=np.arange(len(patterns))
   assert patterns.shape==(16,4) and endpoints.shape==(16,1)
   np.testing.assert_array_equal(target[:,0],patterns.sum(axis=1)%2)
   outputs=endpoints[:,0]
  else:
   inputs=np.asarray(info['inputs']).ravel();target=target.ravel()
   np.testing.assert_allclose(target,target_fn(inputs),rtol=1e-6,atol=1e-8)
   outputs=endpoints[:,0]
  responses.append(outputs)
  union=read(run/'terminal_union.json')
  hits=np.flatnonzero(curves[-1]<.05)
  rows.append(dict(seed=seed,best_loss=m['Archive/best_loss'],first_qualifying_epoch=int(hits[0])+1 if len(hits) else None,
      final_archive_topologies=m['Archive/qualified_topologies'],full_portfolio_topologies=union['union_qualified'],replacement=replaced))
  sources.append(dict(seed=seed,run=str(run.resolve()),replacement=replaced,winner=winner))
 a=np.array(curves);x=np.arange(1,101);n=len(rows)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(2,2,figsize=(10.5,7.4),constrained_layout=True)
 ax=axes[0,0]
 for c in a:ax.plot(x,c,color='#4188ae',alpha=.2,lw=.7)
 q=np.quantile(a,[.25,.5,.75],axis=0);ax.fill_between(x,q[0],q[2],color='#4188ae',alpha=.23,label='25–75% of runs')
 ax.plot(x,q[1],color='#15557b',lw=2,label='Median');ax.axhline(.05,color='#b34936',ls='--',lw=1,label='Qualification: loss <0.05')
 ax.set(yscale='log',xlabel='Completed epochs',ylabel='Best loss so far',title=f'A  Convergence across {n} completed runs');ax.legend(fontsize=8)
 ax=axes[0,1];ax.step(x,np.mean(a<.05,axis=0)*100,where='post',color='#15557b',lw=2)
 ax.set(xlabel='Completed epochs',ylabel='Runs with loss <0.05 (%)',ylim=(-2,105),title=f'B  Search reliability ({sum(r["best_loss"]<.05 for r in rows)}/{n} qualify)');ax.axhline(100,color='.6',ls=':',lw=.8)
 ax=axes[1,0];best=int(np.argmin([r['best_loss'] for r in rows]))
 best_label=f'Best run (seed {rows[best]["seed"]})'
 if key=='classifier':
  for y in responses:ax.scatter(inputs,y,color='#4188ae',alpha=.15,s=10)
  lim=max(abs(inputs))*1.1
  ax.plot([-lim,0,0,lim],[-1,-1,1,1],'k--',lw=1.4,label='Target output difference')
  ax.scatter(inputs,responses[best],color='#cf6c32',s=22,zorder=3,label=best_label)
  ax.set(xlabel='Initial concentration difference (species 1 − species 2)',ylabel='Final output difference (output 1 − output 2)',title='C  Winning networks: classification')
  ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
 elif key=='parity':
  for y in responses:ax.plot(inputs,y,color='#4188ae',alpha=.25,lw=1)
  ax.plot(inputs,target[:,0],'ks',ms=5,mfc='none',label='Target parity')
  ax.plot(inputs,responses[best],'o-',ms=3,color='#cf6c32',lw=1,label=best_label)
  ax.axhline(.5,color='.6',ls=':',lw=.8)
  ax.set(xlabel='Input combination',ylabel='Output at final simulation time',title='C  Winning networks: parity truth table',xticks=inputs)
  ax.set_xticklabels([''.join(str(int(v)) for v in row) for row in patterns],rotation=45,ha='right',fontsize=8)
 else:
  for y in responses:ax.plot(inputs,y,color='#4188ae',alpha=.25,lw=1)
  u=np.linspace(0,1,300);ax.plot(u,target_fn(u),'k--',lw=1.6,label='Target response')
  ax.plot(inputs,responses[best],'o-',ms=4,color='#cf6c32',lw=1.5,label=best_label)
  ax.set(xlabel='Input concentration',ylabel='Output at final simulation time',title='C  Winning networks: dose response')
 ax.legend(fontsize=8)
 ax=axes[1,1];seeds=[r['seed'] for r in rows]
 ax.bar(seeds,[r['full_portfolio_topologies'] for r in rows],color='#a9c9dd',label='All qualifying topologies found')
 ax.bar(seeds,[r['final_archive_topologies'] for r in rows],color='#15557b',label='Retained in final archive')
 ax.set(xlabel='Run / seed',ylabel='Distinct qualifying topology hashes',title='D  Nominal structural diversity',xticks=seeds if n<16 else [1,5,10,15,20]);ax.legend(fontsize=8)
 ax.set_ylim(0,max(1,max(r['full_portfolio_topologies'] for r in rows))*1.35)
 label=f'{n}/20 completed runs' if n<20 else '20 runs'
 suffix=' · replacement seeds included' if any(r['replacement'] for r in rows) else ''
 fig.suptitle(f'{title} · full duplex\n{label} × 100 epochs{suffix}',fontsize=13)
 fig.savefig(dest/f'{key}_performance.pdf');fig.savefig(dest/f'{key}_performance.png',dpi=180);plt.close(fig)
 with (dest/'overview_per_run.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=dict(task=key,snapshot=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(),n_completed=n,successes=sum(r['best_loss']<.05 for r in rows),
    best_final_loss=min(r['best_loss'] for r in rows),median_final_loss=float(np.median([r['best_loss'] for r in rows])),
    mean_qualifying_topologies=float(np.mean([r['full_portfolio_topologies'] for r in rows])),excluded=excluded,sources=sources,
    interpretation='Completed runs only, fixed denominator. Replacement attempts selected for LLM service failures irrespective of loss; superseded attempts excluded. Epoch 100 includes final stream drain. Panel C shows stored terminal outputs: ten tested concentrations for dose tasks, 32 initial-condition examples for classification, or 16 input combinations for parity; no new dense simulations. Classification uses output differences with class chosen by argmax; parity threshold is 0.5. Qualification is nominal loss <0.05, not a robustness guarantee; topology hashes are not pruned mechanistic equivalence classes.')
 (dest/'overview_snapshot.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(key,{k:v for k,v in summary.items() if k not in ['sources','interpretation','excluded']},flush=True)

def read_winner(db):
 return json.loads(db.execute("SELECT payload FROM designs WHERE json_extract(payload,'$.simulation_complete')=1 ORDER BY json_extract(payload,'$.loss'),key LIMIT 1").fetchone()[0])

if __name__=='__main__':
 selections={key:select_runs(key) for key in TASKS}
 for key,selection in selections.items():generate(key,selection)
