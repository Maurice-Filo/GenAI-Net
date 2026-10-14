"""Best-so-far Hill loss versus epochs; each curve retains the lowest observed loss."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
BASE=OUT.parents[2]/'comparisons/search_exploration/reports/dose-hill-full-20seed-20261001'
curves=[]
for seed in range(1,21):
 run=BASE/'runs'/f'dose-hill-full-seed{seed:02}'
 records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
 epochs={int(r['epoch'])+1:r['Archive/best_loss'] for r in records if 'epoch' in r}
 loss=np.array([epochs[e] for e in range(1,101)])
 loss[-1]=min(loss[-1],json.loads((run/'completed.json').read_text())['metrics']['Archive/best_loss'])
 curves.append(np.minimum.accumulate(loss))
a=np.array(curves);x=np.arange(1,101)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained')
for i,c in enumerate(a):ax.plot(x,c,color='#6791af',alpha=.22,lw=.8,label='Individual runs' if i==0 else None)
ax.plot(x,np.median(a,axis=0),color='#165b85',lw=2.2,label='Median best-so-far loss (20 runs)')
ax.plot(x,np.min(a,axis=0),color='#cb6c2e',lw=2.2,label='Best loss across all 20 runs')
ax.axhline(.05,color='#aa4040',ls='--',lw=1.2,label='Qualification threshold (0.05)')
ax.set(xlabel='Completed epochs',ylabel='Best-so-far loss (log scale)',yscale='log',xlim=(1,100),title='Hill dose response — full-duplex search')
ax.legend(fontsize=9,loc='upper right');ax.grid(axis='y',alpha=.15)
fig.savefig(OUT/'hill_best_loss_vs_epochs.pdf');fig.savefig(OUT/'hill_best_loss_vs_epochs.png',dpi=180)
