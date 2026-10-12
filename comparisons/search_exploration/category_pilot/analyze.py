"""Candidate-pool-matched pilot analysis; incomplete runs remain explicitly provisional."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np


def lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass  # A running writer can have an unfinished last line.
    return result


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0}
    return {'n':len(a),'mean':float(a.mean()),'std':float(a.std()),
        **{k:float(v) for k,v in zip(('min','p01','p25','median','p75','p99','max'),np.quantile(a,[0,.01,.25,.5,.75,.99,1]))}}


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(folder,job):
    manifest=json.loads((folder/'manifest.json').read_text())
    threshold=manifest['config']['quality_threshold']
    db=sqlite3.connect(f'file:{folder / "exploration.sqlite"}?mode=ro',uri=True)
    records=[json.loads(p) for p, in db.execute("SELECT payload FROM events WHERE kind='pilot_candidate' ORDER BY id")]
    summaries=[json.loads(p) for p, in db.execute('SELECT payload FROM evaluation_summaries')]
    fatal=db.execute("SELECT count(*) FROM events WHERE kind='evaluation_failed'").fetchone()[0]
    disagreements=[json.loads(p) for p, in db.execute("SELECT payload FROM events WHERE kind='julia_loss_agreement' ORDER BY id")]
    db.close()
    metrics=lines(folder/'metrics.jsonl'); learning=lines(folder/'learning.jsonl')
    complete=[r for r in records if r['simulation_complete']]
    supplied=[r for r in complete if r['stream']=='search']
    fitted=[r for r in complete if r['stream']=='optimization']
    search=[r for r in records if r['stream']=='search']
    qual={r['topology_hash'] for r in complete if r['loss']<threshold}
    all_correct={r['topology_hash'] for r in complete if r['loss']<threshold and r.get('all_rows_correct') is True}
    done=json.loads((folder/'completed.json').read_text()) if (folder/'completed.json').exists() else None
    counters=done['counters'] if done else {k.removeprefix('Compute/'):v for k,v in (metrics[-1] if metrics else {}).items() if k.startswith('Compute/')}
    best=lambda rs:min((r['loss'] for r in rs),default=None)
    endpoint={**{k:job[k] for k in ('name','task','condition','seed')},'status':'complete' if done else 'failed' if (folder/'failed.json').exists() else 'running',
        'epochs':done['epochs'] if done else len(learning),'best_supplied_loss':best(supplied),'best_fitted_loss':best(fitted),
        'best_pool_loss':best(complete),'qualified':bool(qual),'qualifying_topologies':len(qual),
        'qualifying_all_rows_correct_topologies':len(all_correct),'threshold':threshold,
        'search_commits':len(search),'search_incomplete':sum(not r['simulation_complete'] for r in search),
        'search_completion_rate':sum(r['simulation_complete'] for r in search)/len(search) if search else None,
        'unique_canonical_evaluations':len(summaries),'unique_canonical_incomplete':sum(not r['simulation_complete'] for r in summaries),
        'fatal_evaluator_errors':fatal,'recorded_loss_disagreements':len(disagreements),
        'large_loss_disagreements':sum(r['large_disagreement'] for r in disagreements),
        'max_loss_disagreement_relative':max((r['relative_difference'] for r in disagreements),default=0.),'actual_numerical_evaluations':counters.get('expensive_evaluations',0),
        'canonical_search_evaluations':counters.get('evaluations/search',0),
        'canonical_fit_evaluations':counters.get('evaluations/optimization',0)-counters.get('julia/joint_evaluations',0)-counters.get('julia/reserved',0),
        'joint_loss_gradient_evaluations':counters.get('julia/joint_evaluations',0),
        'scenario_solves_fitting':counters.get('julia/scenario_solves',0),
        'fitting_jobs':counters.get('optimization/submitted',0),'llm_evaluations':counters.get('evaluations/llm',0),
        'elapsed_seconds':done.get('elapsed_seconds') if done else None}
    rewards=[];advantages=[];failed_rewards=[];failed_advantages=[];cohorts=[]
    for row in learning:
        byid={r['proposal_id']:r for r in row['search_records']}
        failures=[not byid[i]['simulation_complete'] for i in row['rl_proposal_ids']]
        rewards.extend(row['rewards']);advantages.extend(row['advantages'])
        failed_rewards.extend(r for r,failed in zip(row['rewards'],failures) if failed)
        failed_advantages.extend(r for r,failed in zip(row['advantages'],failures) if failed)
        m=row['metrics']
        cohorts.append({'name':job['name'],'epoch':row['epoch'],'baseline':row['baseline_before'],
            'reward_mean':float(np.mean(row['rewards'])),'advantage_mean':float(np.mean(row['advantages'])),
            'entropy':m['Policy/entropy'],'policy_gradient_loss':m['Policy/loss'],
            'imitation_loss':m['Policy/imitation'],'weighted_imitation_loss':manifest['config']['imitation_weight']*m['Policy/imitation'],
            'weighted_entropy':manifest['config']['entropy_weight']*m['Policy/entropy'],
            'gradient_norm_before_clipping':m.get('Policy/gradient_norm'),
            'failure_fraction':float(np.mean(failures)),
            'pg_abs_over_weighted_imitation':abs(m['Policy/loss'])/max(1e-300,manifest['config']['imitation_weight']*abs(m['Policy/imitation']))})
    diagnostics={'name':job['name'],'loss_disagreements':disagreements,'reward':distribution(rewards),'advantage':distribution(advantages),
        'failure_reward':distribution(failed_rewards),'failure_advantage':distribution(failed_advantages),
        'failure_absolute_reward_fraction':sum(abs(v) for v in failed_rewards)/max(1e-300,sum(abs(v) for v in rewards)),
        'failure_positive_advantages':sum(v>0 for v in failed_advantages),
        'failure_note':'Nonpositive raw rewards do not guarantee nonpositive centered advantages; the baseline update is unchanged.'}
    trajectories=[]
    for m in metrics:
        # Epoch 0 is the first completed cohort; epoch 100 is the post-drain boundary.
        epoch=m['epoch']; pool=[r for r in complete if r['epoch']<=epoch]
        trajectories.append({'name':job['name'],'task':job['task'],'condition':job['condition'],'seed':job['seed'],
            'epoch':min(epoch+1,100) if epoch<100 else 100,'terminal_drain':epoch==100,
            'actual_numerical_evaluations':m.get('Compute/expensive_evaluations'),
            'search_worker_seconds':m.get('Julia/search_worker_seconds'),'fit_worker_seconds':m.get('Julia/fit_worker_seconds'),
            'best_pool_loss':best(pool),'best_supplied_loss':best([r for r in pool if r['stream']=='search']),
            'qualifying_topologies':len({r['topology_hash'] for r in pool if r['loss']<threshold})})
    fits=json.loads((folder/'optimization_results.json').read_text()) if (folder/'optimization_results.json').exists() else []
    diagnostics['fitting']={'jobs':len(fits),'improved':sum(r['best_loss'] is not None and r['best_loss']<r['initial_loss'] for r in fits),
        'without_completed_best':sum(r['best_loss'] is None for r in fits),'failure_records':sum(len(r['failures']) for r in fits)}
    return endpoint,trajectories,diagnostics,cohorts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args()
    root=a.campaign.resolve();out=root/'analysis';out.mkdir(exist_ok=True)
    meta=json.loads((root/'campaign.json').read_text())
    endpoints=[];trajectories=[];diagnostics=[];cohorts=[]
    for job in meta['jobs']:
        folder=root/'runs'/job['name']
        if not (folder/'exploration.sqlite').exists():continue
        e,t,d,c=summarize(folder,job);endpoints.append(e);trajectories.extend(t);diagnostics.append(d);cohorts.extend(c)
    write_csv(out/'endpoints.csv',endpoints);write_csv(out/'trajectories.csv',trajectories);write_csv(out/'learning_scales.csv',cohorts)
    (out/'reward_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    checks=[]
    for task in ('rpa','logic'):
        for seed in (1,2,3):
            paths=[root/'runs'/f'{task}-{condition}-seed{seed:02d}'/'initialization.json' for condition in ('baseline','ablation')]
            if all(p.exists() for p in paths):
                initial=[json.loads(p.read_text()) for p in paths]
                checks.append({'task':task,'seed':seed,'initialization_identical':initial[0]==initial[1]})
    (out/'paired_initialization.json').write_text(json.dumps(checks,indent=2)+'\n')
    if endpoints:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors={'baseline':'#2166ac','ablation':'#b2182b'}
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for j,task in enumerate(('rpa','logic')):
            for condition in colors:
                for seed in (1,2,3):
                    ts=[r for r in trajectories if (r['task'],r['condition'],r['seed'])==(task,condition,seed)]
                    for col,key in enumerate(('epoch','actual_numerical_evaluations')):
                        for losskey,style in [('best_pool_loss','-'),('best_supplied_loss','--')]:
                            valid=[r for r in ts if r[losskey] is not None]
                            axes[j,col].plot([r[key] for r in valid],[r[losskey] for r in valid],style,color=colors[condition],alpha=.75,lw=1,
                                label=f'{condition} seed {seed}'+(' supplied' if style=='--' else ' pool'))
                        axes[j,col].set(xlabel='Search epoch (100 includes drain)' if col==0 else 'Canonical + joint evaluations',ylabel=task+' canonical loss',yscale='log')
            for seed in (1,2,3):
                es=[next((r for r in endpoints if (r['task'],r['condition'],r['seed'])==(task,c,seed) and r['status']=='complete'),None) for c in colors]
                if all(es):
                    axes[j,2].plot([0,1],[r['best_pool_loss'] for r in es],'o-',label=f'seed {seed}')
            axes[j,2].set(xticks=[0,1],xticklabels=list(colors),ylabel=task+' final pool loss',yscale='log')
            handles,labels=axes[j,2].get_legend_handles_labels()
            if handles:axes[j,2].legend(fontsize=8)
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=7)
        fig.savefig(out/'convergence.png',dpi=180);fig.savefig(out/'convergence.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for job in meta['jobs']:
            rows=[r for r in cohorts if r['name']==job['name']]
            j=0 if job['task']=='rpa' else 1
            for col,key in enumerate(('reward_mean','entropy','pg_abs_over_weighted_imitation')):
                axes[j,col].plot([r['epoch']+1 for r in rows],[r[key] for r in rows],color=colors[job['condition']],alpha=.6)
                axes[j,col].set(xlabel='Epoch',ylabel=job['task']+' '+key)
                if col==2:axes[j,col].set_yscale('log')
        fig.savefig(out/'learning_scales.png',dpi=180);plt.close(fig)
    fmt=lambda x:'pending' if x is None else f'{x:.6g}'
    report=['# Combined category and reward pilot','',f"Completed {sum(r['status']=='complete' for r in endpoints)}/12 fresh runs. Partial trajectories are provisional; endpoint pairs require both runs to complete.",'',
        'Pool: all complete committed numerical search proposals plus completed canonically reevaluated fitting results; excludes internal Julia trials. Supplied-rate losses and fitted results are reported separately. Repeated topology identities count once for qualification. No occupied-region diversity comparison is made.','',
        '| Task | Seed | Condition | Status | Supplied loss | Pool loss | Qualifying topologies | Completion rate | Numerical evaluations |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for e in endpoints:report.append(f"| {e['task']} | {e['seed']} | {e['condition']} | {e['status']} | {fmt(e['best_supplied_loss'])} | {fmt(e['best_pool_loss'])} | {e['qualifying_topologies']} | {fmt(e['search_completion_rate'])} | {e['actual_numerical_evaluations']} |")
    report+=['', 'Numerical loss disagreements are recorded under the user-authorized record-only protocol in both conditions. Canonical loss remains authoritative; no solver tolerance or loss is changed. Detailed checks are in reward_diagnostics.json.', '']
    report+=['','Paired observations (ablation / baseline final pool loss):']
    for task in ('rpa','logic'):
        ratios=[]
        for seed in (1,2,3):
            pair=[next((r for r in endpoints if (r['task'],r['condition'],r['seed'])==(task,c,seed) and r['status']=='complete'),None) for c in ('baseline','ablation')]
            if all(pair) and pair[0]['best_pool_loss']:
                ratio=pair[1]['best_pool_loss']/pair[0]['best_pool_loss'];ratios.append(ratio)
                report.append(f'- {task}, seed {seed}: {ratio:.3g}.')
        if len(ratios)==3:report.append(f'- {task}: ablation lower loss in {sum(r<1 for r in ratios)}/3 seeds; median paired ratio {np.median(ratios):.3g}.')
    report+=['','Raw reward and failure contribution:']
    for d in diagnostics:
        if d['reward']['n']:report.append(f"- {d['name']}: reward range [{d['reward']['min']:.6g}, {d['reward']['max']:.6g}], mean {d['reward']['mean']:.6g}; incomplete RL proposals {d['failure_reward']['n']}/{d['reward']['n']}, contributing {100*d['failure_absolute_reward_fraction']:.2f}% of absolute reward mass; {d['failure_positive_advantages']} positive centered failure advantages.")
    report+=['','Failures with a finite canonical loss receive its negative in the ablation. The simulator’s 1e4 sentinel enters the existing weighted loss; it is not a universal scalar loss of 1e4. The unchanged moving baseline can yield positive advantages even when the raw failure reward is negative. Exceptions lacking a finite canonical loss stop the run and are preserved, not imputed or silently dropped.','',
        'This pilot changes category grouping, reward shaping, retention and sampling together. Observed paired differences concern the combined algorithm and cannot establish a causal effect of speciation alone. Three seeds support descriptive comparisons, not strong significance claims. Actual fitting work varies under the unchanged asynchronous measured-compute controller. Concurrent host timing is descriptive; no controlled wall-clock speed claim is made.','',
        'Artifacts: endpoints.csv, trajectories.csv, reward_diagnostics.json, learning_scales.csv, paired_initialization.json, convergence.png/pdf, learning_scales.png.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(f'{out}: {sum(r["status"]=="complete" for r in endpoints)}/12 complete')

if __name__=='__main__':main()
