"""Separate setpoint calibration from disturbance rejection using saved perturbations."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from comparisons.search_exploration.run_rpa_llm_robustness import ARMS,LABELS,COLORS


def analyze(root):
    out=root/'analysis'
    plan=json.loads((root/'plan.json').read_text())
    with (out/'per_network.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fields=('final_error_below_5pct_fraction','disturbance_spread_below_5pct_fraction')
    summaries=[]
    for col,seeds in enumerate((list(range(1,21)),plan['directional_matched_seeds'])):
        for arm in ARMS[:2] if col==0 else ARMS:
            i=ARMS.index(arm)
            for ax,field in zip(axes[:,col],fields):
                yy=[]
                for radius in plan['radii']:
                    values=[float(r[field]) for r in rows if r['arm']==arm and int(r['seed']) in seeds
                            and r['mode']=='all_rates' and float(r['radius'])==radius]
                    if len(values)!=len(seeds): raise ValueError('Unmatched functional sample')
                    yy.append(np.mean(values))
                    summaries.append({'subset':'full' if col==0 else 'directional','arm':arm,'n':len(seeds),
                                      'radius':radius,'metric':field,'mean':float(np.mean(values))})
                ax.plot([100*(r-1) for r in plan['radii']],yy,'o-',color=COLORS[i],label=LABELS[i],ms=4)
        axes[0,col].set_title(f"{'Full cohort' if col==0 else 'Matched directional subset'} (n={len(seeds)})")
        axes[0,col].legend(frameon=False,fontsize=8)
        for ax in axes[:,col]:
            ax.set(ylim=(-.02,1.02),xlabel='Multiplicative perturbation radius (%)')
            ax.grid(alpha=.15)
        axes[0,col].set_ylabel('Draws with final target error <5%')
        axes[1,col].set_ylabel('Draws with disturbance spread <5%')
    fig.suptitle('RPA | Parameter sensitivity is not the same as lost adaptation',fontsize=14)
    fig.text(.035,.014,'Top: worst final relative output error over the nine cases. Bottom: worst final output spread across disturbances at fixed input,\n'
        'normalized by the target. All draws count; failed solves fail both criteria. Winner-only, finite-horizon diagnostics, not a stability proof.\n'
        'A shifted setpoint can fail the original loss threshold while retaining disturbance rejection. Lines are seed means; no significance claim.',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.96))
    fig.savefig(out/'02_functional_robustness.pdf');fig.savefig(out/'functional_robustness.png',dpi=190);plt.close(fig)
    original=out/'01_loss_robustness.pdf'
    if not original.exists(): shutil.copy2(out/'RPA_LLM_ROBUSTNESS.pdf',original)
    temp=out/'RPA_LLM_ROBUSTNESS.tmp.pdf'
    subprocess.run(['pdfunite',str(original),str(out/'02_functional_robustness.pdf'),str(temp)],check=True)
    temp.replace(out/'RPA_LLM_ROBUSTNESS.pdf')
    difference={}
    for field in ('qualified_fraction',*fields):
        values={arm:{int(r['seed']):float(r[field]) for r in rows if r['arm']==arm and
                r['mode']=='all_rates' and float(r['radius'])==1.1} for arm in ('control','flash')}
        delta=np.array([values['flash'][s]-values['control'][s] for s in range(1,21)])
        ix=np.random.default_rng(20260918).integers(0,20,(10000,20))
        difference[field]={'mean_difference':float(delta.mean()),
                          'paired_bootstrap_ci95':np.quantile(delta[ix].mean(1),[.025,.975]).tolist()}
    (out/'functional_summary.json').write_text(json.dumps({'summaries':summaries,
        'posthoc_full_cohort_10pct_paired_differences':difference},indent=2)+'\n')
    report=out/'REPORT.md'
    text=report.read_text().split('## Functional Interpretation')[0]
    text+='\n## Functional Interpretation\n\nAt a 10% perturbation radius, the full 20-seed cohorts show:\n\n'
    text+='| Workflow | Target accuracy <5% | Disturbance spread <5% |\n|---|---:|---:|\n'
    for arm,label in zip(ARMS[:2],LABELS[:2]):
        v=[next(r['mean'] for r in summaries if r['subset']=='full' and r['arm']==arm and r['radius']==1.1 and r['metric']==field) for field in fields]
        text+=f'| {label} | {v[0]:.3%} | {v[1]:.3%} |\n'
    text+='\nThus sensitivity of the optimized loss does not imply loss of disturbance adaptation: the setpoint can shift while disturbance rejection remains. These are finite-horizon numerical diagnostics, not formal stability or biological robustness guarantees. Relative loss inflation is also sensitive to the much smaller unperturbed Flash loss.\n\n'
    text+='Post-hoc paired seed-bootstrap intervals at the 10% radius are in functional_summary.json; no simultaneous or multiplicity-adjusted significance inference is claimed.\n\n![Accuracy and disturbance rejection](functional_robustness.png)\n'
    report.write_text(text)
    print(json.dumps(difference,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
