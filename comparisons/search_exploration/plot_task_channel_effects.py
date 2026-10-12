"""Separate channel contrasts for current Logic and RPA hybrid workflows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from comparisons.search_exploration.analyze_logic5_directional_components import best_rl_events
from comparisons.search_exploration.analyze_logic5_four_arm import paired_endpoint, write_csv
from comparisons.search_exploration.analyze_rpa_llm_provenance import connect
from comparisons.search_exploration.analyze_rpa_flash_adagrad import FULL

RECOVERY = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-directional-balance-recovery-20260918')
LOGIC = Path('comparisons/search_exploration/reports/logic5-directional-components-20260925/summary.json')
OUTWARD = (('rl-to-llm', 'none'), ('full', 'llm-to-rl'))
INWARD = (('llm-to-rl', 'none'), ('full', 'rl-to-llm'))


def read(path):
    return json.loads(path.read_text())


def direct_topologies(db, threshold):
    rows = db.execute("""SELECT DISTINCT json_extract(d.payload,'$.topology_hash')
        FROM events e JOIN designs d ON d.key=json_extract(e.payload,'$.candidate_hash')
        WHERE e.kind='archive_insertion' AND json_extract(e.payload,'$.source')='llm'
        AND coalesce(json_extract(e.payload,'$.stream'),'')!='optimization'
        AND json_extract(d.payload,'$.simulation_complete')=1
        AND json_extract(e.payload,'$.loss')>=0 AND json_extract(e.payload,'$.loss')<?""", (threshold,))
    return {r[0] for r in rows}


def extract_rpa():
    plan, audited = read(RECOVERY/'plan.json'), read(RECOVERY/'analysis/summary.json')
    if audited['audit_errors'] or audited['matched_seeds'] != list(range(1,21)):
        raise ValueError('Require reconciled RPA cohort')
    jobs = {(j['arm'],j['seed']): Path(j['output']) for j in plan['jobs']}
    jobs.update({('full',j['seed']): Path(j.get('output',FULL/'runs'/j['name'])) for j in read(FULL/'plan.json')['jobs']})
    unions = {r['seed']:r for r in audited['independent_unions']}
    jobs.update({('none',s):Path(r['control']) for s,r in unions.items()})
    rows=[]
    for (arm,seed),folder in sorted(jobs.items()):
        threshold=read(folder/'manifest.json')['config']['quality_threshold']
        with connect(folder) as db:
            rl=best_rl_events(db)
            topologies=direct_topologies(db,threshold) if arm=='full' else set()
        bank=Path(unions[seed]['bank'])/'direct_llm_proposals.jsonl' if arm=='none' else folder/'direct_llm_proposals.jsonl'
        if bank.exists():
            for line in bank.read_text().splitlines():
                entry=json.loads(line); record=entry.get('record')
                if not entry['direct_supplied_rate_only']:
                    raise ValueError('Unexpected fitted proposal bank')
                if record and record['simulation_complete'] and 0<=record['loss']<threshold:
                    topologies.add(record['topology_hash'])
        if arm=='none' and len(topologies)!=unions[seed]['llm_qualified']:
            raise ValueError('Reused LLM bank count mismatch')
        if arm in ('llm-to-rl','rl-to-llm'):
            expected=next(r['llm_qualified'] for r in audited['endpoints'] if r['arm']==arm and r['seed']==seed)
            if len(topologies)!=expected:
                raise ValueError('Directional direct-LLM count mismatch')
        rows.append({'task':'rpa','arm':arm,'seed':seed,'llm_yield':len(topologies),
            'yield_definition':'Direct supplied-rate qualifying LLM topologies (not robustness)',
            'folder':str(folder),'llm_bank':str(bank) if bank.exists() else 'unfitted LLM event journal',
            'reused_none':arm=='none',**rl})
        print(f'RPA {len(rows)}/80 {arm}/{seed}: LLM yield {len(topologies)}',flush=True)
    if len(rows)!=80 or len({(r['arm'],r['seed']) for r in rows})!=80:
        raise ValueError('Expected 80 RPA endpoints')
    return rows


def contrasts(rows, logic_tests):
    results=[]
    by={(r['task'],r['arm'],r['seed']):r for r in rows}
    for task in ('logic','rpa'):
        for metric, edges in (('llm_yield',OUTWARD),('rl_with_fit_loss',INWARD),('rl_sampled_loss',INWARD)):
            for a,b in edges:
                first=np.array([by[task,a,s][metric] for s in range(1,21)])
                second=np.array([by[task,b,s][metric] for s in range(1,21)])
                stats=paired_endpoint(first,second,smaller=metric!='llm_yield')
                if metric=='llm_yield':
                    effect=stats['mean_difference']; ci=stats['mean_difference_ci95']
                else:
                    effect=100*(1-stats['median_paired_ratio'])
                    ci=[100*(1-stats['median_ratio_ci95'][1]),100*(1-stats['median_ratio_ci95'][0])]
                old_metric='llm_supported_robust_cores' if metric=='llm_yield' else metric
                old=next((t for t in logic_tests if t['a']==a and t['b']==b and t['metric']==old_metric),None) if task=='logic' else None
                results.append({'task':task,'metric':metric,'a':a,'b':b,'effect':effect,'ci95':ci,
                    'mean_a':float(first.mean()),'mean_b':float(second.mean()),
                    'median_a':float(np.median(first)),'median_b':float(np.median(second)),
                    'holm_p_30_logic':None if old is None else old['holm_p_30_with_portfolio_tests'],
                    'inference':'previous Logic 30-test sensitivity family' if task=='logic' else 'descriptive historical/reused RPA contrast',
                    **stats})
    return results


def render(output, result):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'logic':'#a64e85','rpa':'#168577'}
    with PdfPages(output/'CHANNEL_EFFECTS_LOGIC_RPA.pdf') as pdf:
        for allow_fit in (True,False):
            fig,axes=plt.subplots(2,2,figsize=(13.8,9.3))
            fig.subplots_adjust(left=.15,right=.96,top=.79,bottom=.19,wspace=.49,hspace=.74)
            fig.text(.06,.965,'Separate effects of the two communication channels',fontsize=18,weight='bold',va='top')
            subtitle='RL-emitted loss includes subsequent AdaGrad fitting' if allow_fit else 'Sensitivity view: RL-emitted loss before parameter fitting'
            fig.text(.06,.918,subtitle+'; no direct LLM or GA candidate is credited to the RL endpoint.',fontsize=10)
            for i,task in enumerate(('logic','rpa')):
                for j,metric in enumerate(('llm_yield','rl_with_fit_loss' if allow_fit else 'rl_sampled_loss')):
                    ax=axes[i,j]
                    tests=[t for t in result['contrasts'] if t['task']==task and t['metric']==metric]
                    ax.axvline(0,ls='--',color='#888888',lw=1)
                    for y,t in enumerate(tests):
                        lo,hi=t['ci95']; effect=t['effect']
                        ax.plot([lo,hi],[y,y],color=colors[task],lw=2)
                        ax.plot([lo,hi],[y,y],'|',color=colors[task],markersize=8)
                        ax.plot(effect,y,'D',color=colors[task],markersize=6)
                        suffix='cores' if task=='logic' and j==0 else 'topologies' if j==0 else '%'
                        line=f"{effect:+.2f} {suffix}"
                        if task=='logic':
                            line+=f"; Holm p={t['holm_p_30_logic']:.3f}"
                        else:
                            line+='; descriptive'
                        ax.text(.98,y-.22,line,transform=ax.get_yaxis_transform(),ha='right',fontsize=9,color=colors[task])
                    if j==0:
                        labels=['Insertion OFF\nReturn-only vs none'+('*' if task=='rpa' else ''),
                                'Insertion ON\nFull vs insertion-only']
                        xlabel='Additional LLM-supported robust cores / seed' if task=='logic' else 'Additional direct-LLM qualifying topologies / seed'
                        title='Search / HOF to LLM'
                    else:
                        labels=['Return OFF\nInsertion-only vs none'+('*' if task=='rpa' else ''),
                                'Return ON\nFull vs return-only']
                        xlabel='Median paired RL-emitted loss reduction (%)'
                        title='LLM insertion into search'
                    extent=[0]+[v for t in tests for v in t['ci95']]
                    low,high=min(extent),max(extent); pad=max((high-low)*.13,1)
                    ax.set_xlim(low-pad,high+pad)
                    ax.set(ylim=(1.55,-.65),xlabel=xlabel,yticks=[0,1],yticklabels=labels)
                    ax.set_title(f"{'ABCD'[2*i+j]}  {task.upper()}: {title}",loc='left',weight='bold',fontsize=11,pad=20)
                    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
            fig.text(.06,.115,'Positive values favor enabling the channel. All comparisons pair seed labels (n=20); whiskers are marginal 95% seed-bootstrap intervals.',fontsize=9)
            fig.text(.06,.088,'Logic: executed four-arm cohort; robust-core endpoint. RPA: direct-LLM loss qualification, not whole-portfolio robustness.',fontsize=9)
            fig.text(.06,.061,'*RPA none = historical search control plus reused insertion-only LLM bank; not a newly executed no-communication arm.',fontsize=9)
            fig.text(.06,.034,'Both panels use current RL / mutation / AdaGrad workflows. Older no-GA trials are not pooled. Logic p-values retain the extended 30-test correction.',fontsize=9)
            name='01_fitted_channel_effects' if allow_fit else '02_unfitted_rl_sensitivity'
            fig.savefig(output/(name+'.png'),dpi=210)
            fig.savefig(output/(name+'.pdf'))
            pdf.savefig(fig);plt.close(fig)


def report(output,result):
    lines=['# Per-Channel Effects: Logic and RPA','','## Scope','',
        'These are current hybrid-workflow results, not older RL-without-GA trials. The latter remain separate; this figure does not claim that their effects transfer to the current implementation.',
        'Logic uses LLM-supported robust canonical cores from the audited provenance study. RPA has only a partial winner-robustness study, not a completed whole-portfolio robustness study; its outward endpoint is therefore the number of distinct direct supplied-rate LLM topologies meeting the task loss threshold. Do not compare the magnitudes of these different yield endpoints across tasks.',
        'LLM-supported Logic cores include direct proposals, fitted LLM structures and explicit LLM-lineage mutations, counting shared-route cores once. RPA direct yield excludes fitting and mutation descendants. Both metrics deduplicate order-independent full structural topology identities.',
        'The main page isolates RL-emitted candidates, allowing their AdaGrad refinements but excluding direct LLM and GA emitters. Page two excludes fitting too. The comparison is still between hybrid systems: the policy can be influenced by shared elites and the optimization schedule. It is not a standalone-RL or fixed-proposal experiment.',
        'Search/HOF-to-LLM is tested with insertion absent (return-only vs none) and present (full vs insertion-only). LLM-to-search is tested with RL/HOF feedback absent (insertion-only vs none) and present (full vs return-only). LLM self-feedback remains available in every arm. Thus each contrast changes one channel, rather than only comparing full duplex to none.',
        'RPA none reuses the historical no-LLM hybrid search and the insertion-only direct-LLM proposal bank. Its RL endpoint is the historical control, not the joint union loss. Bank request counts, completion schedules, worker settings and actual numerical costs differ, and one historical control retained finite timeouts. RPA is shown descriptively; seed pairing and bootstrap intervals do not eliminate these limitations.',
        'Logic annotations reuse the extended 30-test correction from the component analysis, not a newly selected favorable testing family. Marginal bootstrap intervals need not agree with adjusted p-values. RPA raw paired statistics are archived for audit but not promoted to controlled significance claims. No new experiments or model calls were made.','',
        '| Task | Endpoint | Comparison | Effect | 95% interval | W/T/L | Logic Holm p (30) |',
        '|---|---|---|---:|---|---|---:|']
    for t in result['contrasts']:
        p='descriptive' if t['holm_p_30_logic'] is None else f"{t['holm_p_30_logic']:.5g}"
        interval=f"[{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]"
        lines.append(f"| {t['task']} | {t['metric']} | {t['a']} vs {t['b']} | {t['effect']:+.3f} | {interval} | {t['wins']}/{t['ties']}/{t['losses']} | {p} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        result=read(args.output/'summary.json')
    else:
        logic=read(LOGIC)
        rows=[{**r,'task':'logic','llm_yield':r['llm_supported_robust_cores'],
               'yield_definition':'LLM-supported robust cores','reused_none':False} for r in logic['rows']]
        rows+=extract_rpa()
        result={'rows':rows,'contrasts':contrasts(rows,logic['tests']),
                'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (LOGIC,RECOVERY/'plan.json',RECOVERY/'analysis/summary.json',FULL/'plan.json')}}
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        write_csv(args.output/'contrasts.csv',result['contrasts'])
    render(args.output,result);report(args.output,result)
