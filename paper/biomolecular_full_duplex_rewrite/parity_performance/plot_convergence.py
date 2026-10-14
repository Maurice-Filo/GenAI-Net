"""Four-input parity convergence for the complete 20-run Flash cohort."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from generate_dose_overviews import select_runs
from proposed_results_figures import style, axes_style, PURPLE


def main():
    selected, excluded = select_runs('parity')
    assert len(selected) == 20 and not excluded
    curves, finals, rows, sources = [], [], [], []
    for seed, folder, done, replacement in selected:
        assert not replacement and done['epochs'] == 100 and done['status'] == 'complete'
        path = folder / 'metrics.jsonl'
        raw = path.read_bytes()
        metrics = {int(m['epoch']): m for m in map(json.loads, raw.splitlines())}
        assert set(metrics) == set(range(101))
        losses = [metrics[e]['Archive/best_loss'] for e in range(100)]
        final = done['metrics']['Archive/best_loss']
        np.testing.assert_allclose(metrics[100]['Archive/best_loss'], final, rtol=1e-12)
        assert np.isfinite(losses).all() and min(losses) > 0
        assert np.all(np.diff(losses) <= 1e-12) and 0 < final <= losses[-1]+1e-12
        curves.append(losses); finals.append(final)
        sources.append(dict(seed=seed, folder=str(folder), metrics_sha256=hashlib.sha256(raw).hexdigest(),
                            completed_sha256=hashlib.sha256((folder/'completed.json').read_bytes()).hexdigest()))
        for e in range(101):
            m = metrics[e]
            rows.append(dict(seed=seed, completed_epoch=e+1 if e<100 else 'T',
                             loss=m['Archive/best_loss'], charged_evaluations=m['Compute/expensive_evaluations']))
    values = np.asarray(curves)
    low, median, high = np.quantile(values, [.25,.5,.75], axis=0)
    final_low, final_median, final_high = np.quantile(finals, [.25,.5,.75])
    style()
    fig, ax = plt.subplots(figsize=(8.4,4.8), constrained_layout=True)
    x = np.arange(1,101)
    for y in values:
        ax.step(x,y,where='post',color=PURPLE,alpha=.12,lw=.65)
    ax.fill_between(x,low,high,step='post',color=PURPLE,alpha=.18,lw=0,label='Interquartile range')
    ax.step(x,median,where='post',color=PURPLE,lw=2,label='Full duplex: Flash (median, 20 runs)')
    ax.scatter(np.linspace(104.5,107.5,20), finals, color=PURPLE,alpha=.25,s=14)
    ax.errorbar(106,final_median,yerr=[[final_median-final_low],[final_high-final_median]],
                color=PURPLE,fmt='o',ms=5,lw=1.5,capsize=3)
    ax.axhline(.05,color='#555555',ls=':',lw=1.1,label='Qualification threshold')
    ax.axvline(103,color='#AAAAAA',ls=':',lw=.7)
    ax.set(yscale='log',xlim=(0,110),xticks=[0,25,50,75,100,106],
           xticklabels=['0','25','50','75','100','T'],xlabel='Completed search epochs',ylabel='Best loss')
    axes_style(ax,'Four-input parity')
    handles,labels=ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in [1,0,2]],[labels[i] for i in [1,0,2]],fontsize=9,frameon=False)
    for ext in ['pdf','png','svg']:
        fig.savefig(HERE/f'parity_best_loss_vs_epochs.{ext}',dpi=180)
    plt.close(fig)
    with (HERE/'best_loss_vs_epochs.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(n=20,successes=sum(v<.05 for v in finals),threshold=.05,
                 median_epoch100=float(median[-1]),median_final=float(final_median),
                 final_iqr=[float(final_low),float(final_high)],best_final=min(finals),
                 sources=sources,definition='Median/IQR over all 20 completed runs; faint curves are individuals. Epochs 1–100 exclude terminal drain; T shows completed-run endpoints after pending work finishes. Qualification denotes canonical loss <0.05, not perturbation robustness.')
    (HERE/'convergence_snapshot.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))


if __name__ == '__main__':
    main()
