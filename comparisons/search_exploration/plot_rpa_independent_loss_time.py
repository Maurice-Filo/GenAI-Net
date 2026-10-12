"""Observed source loss histories; post-hoc alignment is not a concurrent replay."""

import argparse
import json
from pathlib import Path
import textwrap

import numpy as np

from comparisons.search_exploration.analyze_rpa_llm_provenance import RECOVERY, read
from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json, write_csv
from comparisons.search_exploration.analyze_rpa_communication_families import digest


def proposal_prefix(rows):
    records = [r['record'] for r in rows if r['record'] is not None]
    losses = [r['loss'] if r['simulation_complete'] and r['loss'] is not None
              and np.isfinite(r['loss']) and r['loss'] >= 0 else np.inf for r in records]
    return np.r_[np.inf, np.minimum.accumulate(losses)]


def previous_values(times, values, query):
    times, values, query = np.asarray(times), np.asarray(values), np.asarray(query)
    indices = np.searchsorted(times, query, side='right')-1
    return np.where(indices >= 0, values[np.maximum(indices, 0)], np.inf)


def upper_summary(values):
    ordered = np.sort(values, axis=0)
    return ordered[len(values)//2], ordered[int(np.ceil(.25*(len(values)-1)))], ordered[int(np.ceil(.75*(len(values)-1)))]


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    refs = read(RECOVERY/'analysis/summary.json')
    if refs['audit_errors'] or refs['matched_seeds'] != list(range(1,21)):
        raise ValueError('Need twenty complete audited seeds')
    histories, exported, files = [], [], {}
    for ref in sorted(refs['independent_unions'], key=lambda r:r['seed']):
        bank, control = Path(ref['bank']), Path(ref['control'])
        ledger = [json.loads(line) for line in (bank/'direct_llm_proposals.jsonl').read_text().splitlines()]
        if any(not r['direct_supplied_rate_only'] for r in ledger):
            raise ValueError('Non-direct proposal in bank')
        prefix = proposal_prefix(ledger)
        item = {'seed': ref['seed']}
        for source, folder in [('search',control),('llm',bank)]:
            path = folder/'metrics.jsonl'; files[str(path)] = digest(path)
            metrics = [json.loads(line) for line in path.read_text().splitlines()]
            times = np.array([m['Timing/elapsed_seconds']/60 for m in metrics])
            if np.any(np.diff(times)<0):
                raise ValueError('Metric clock is not monotonic')
            if source == 'search':
                losses = np.array([np.inf if m['Archive/best_loss'] is None else m['Archive/best_loss'] for m in metrics])
            else:
                counts = np.array([m['LLM/direct_evaluated_records'] for m in metrics],int)
                if np.any(np.diff(counts)<0) or counts[-1] != len(prefix)-1:
                    raise ValueError('Proposal counter does not reconstruct complete ledger')
                losses = prefix[counts]
            losses = np.minimum.accumulate(losses)
            target = ref['search_best_loss'] if source=='search' else ref['llm_best_loss']
            if not np.isclose(losses[-1],target,rtol=1e-12,atol=0):
                raise ValueError('Final source loss mismatch')
            epoch = np.full(102,np.inf)
            for m,t,loss in zip(metrics,times,losses):
                e = m['epoch']
                if not 0 <= e <= 100:
                    raise ValueError('Unexpected epoch')
                # Epoch 0 is the first completed training batch; epoch 100 drains.
                step = e+1 if e<100 else 101
                epoch[step] = min(epoch[step],loss)
                exported.append({'seed':ref['seed'],'source':source,'recorded_epoch':e,
                    'elapsed_minutes':float(t),'best_loss':float(loss) if np.isfinite(loss) else None,
                    'direct_records_received':m.get('LLM/direct_evaluated_records') if source=='llm' else None})
            epoch = np.minimum.accumulate(epoch)
            item[source] = {'times':times,'losses':losses,'epoch':epoch}
        files[str(bank/'direct_llm_proposals.jsonl')] = digest(bank/'direct_llm_proposals.jsonl')
        histories.append(item)
    epoch = {s:np.array([r[s]['epoch'] for r in histories]) for s in ('search','llm')}
    epoch['union'] = np.minimum(epoch['search'],epoch['llm'])
    longest = max(r[s]['times'][-1] for r in histories for s in ('search','llm'))
    grid = np.linspace(0,longest,1001)
    wall = {s:np.array([previous_values(r[s]['times'],r[s]['losses'],grid) for r in histories]) for s in ('search','llm')}
    wall['union'] = np.minimum(wall['search'],wall['llm'])
    write_csv(output/'observed_snapshots.csv',exported)
    rows=[]
    for kind,xx,curves in [('completed_epoch',list(range(101))+['T'],epoch),('elapsed_minutes',grid,wall)]:
        for s in curves:
            median,lo,hi=upper_summary(curves[s])
            rows.extend({'alignment':kind,'position':x,'source':s,
                'upper_median':float(m) if np.isfinite(m) else None,
                'q25':float(l) if np.isfinite(l) else None,'q75':float(h) if np.isfinite(h) else None,
                'seeds_with_result':int(n)} for x,m,l,h,n in zip(xx,median,lo,hi,np.isfinite(curves[s]).sum(0)))
    write_csv(output/'aggregate_curves.csv',rows)
    checkpoints=[]
    for e in (1,5,10,20,25,50,75,100,101):
        s,l=epoch['search'][:,e],epoch['llm'][:,e]
        checkpoints.append({'completed_epochs':e if e<=100 else 'T',
            'search_upper_median':float(np.sort(s)[10]),
            'llm_upper_median':float(np.sort(l)[10]) if np.isfinite(np.sort(l)[10]) else None,
            'llm_available':int(np.isfinite(l).sum()),
            'llm_strictly_better':int(((l<s)&~np.isclose(l,s,rtol=1e-9,atol=1e-12)&np.isfinite(l)).sum())})
    dump_json(output/'summary.json',{'n':20,'checkpoints':checkpoints,
        'median_terminal_minutes':{s:float(np.median([r[s]['times'][-1] for r in histories])) for s in ('search','llm')},
        'new_evaluations':0,'new_model_calls':0,'interpretation':'Sources aligned post hoc; not observed independent-union execution'})
    files[str(RECOVERY/'analysis/summary.json')]=digest(RECOVERY/'analysis/summary.json')
    files[str(Path(__file__))]=digest(Path(__file__))
    dump_json(output/'inputs.json',files)
    np.savez_compressed(output/'curves.npz',epoch_search=epoch['search'],epoch_llm=epoch['llm'],
        epoch_union=epoch['union'],minutes=grid,wall_search=wall['search'],wall_llm=wall['llm'],wall_union=wall['union'])
    plot(output,epoch,wall,grid)
    (output/'REPORT.md').write_text('# Independent Union: Loss Through Progress and Time\n\n'
        'Twenty matched seeds. Search is the existing no-LLM AdaGrad/RL hybrid; LLM is the direct supplied-rate proposal bank from insertion-only. '
        'The archive score of the LLM bank\'s donor search is never substituted for direct-proposal losses.\n\n'
        'LLM availability is reconstructed using the cumulative evaluated-record counter at each saved metric snapshot and the corresponding prefix of the direct proposal ledger. '
        'No result is backdated to request launch. Incomplete evaluations have infinite loss. Snapshot times are observation times, not exact inference completion times.\n\n'
        'Training log epochs 0-99 correspond to completed epochs 1-100. T is terminal draining, displayed separately without inventing additional training epochs. '
        'Missing-source losses remain infinite before aggregation; the upper median is the 11th ordered seed, with empirical 25th/75th percentile order statistics. '
        'Curves and bands are absent where their statistics are still infinite. They do not condition on early-returning seeds.\n\n'
        'Elapsed time is measured independently from each source run\'s start, using the last available snapshot without interpolation or looking ahead. '
        'Completed sources retain their final result. The virtual union takes the paired pointwise minimum, as if sources had started together; this is an illustrative reconstruction, not an observed concurrent experiment. '
        'Worker allocations, machine load, completion schedules and consumed budgets differ, so the elapsed-time view is descriptive, not a controlled speed comparison.\n\n'
        'The purple LLM curve may overlap the dashed virtual-union curve once LLM losses dominate. The availability panel keeps request latency visible. '
        'No new simulations or model calls. All 40 source endpoint losses reconstruct the audited union components.\n')
    print(json.dumps(checkpoints,indent=2),flush=True)


def plot(output,epoch,wall,grid):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5.5),layout='constrained')
    colors={'search':'#168577','llm':'#a64e85','union':'#6b6975'}
    labels={'search':'AdaGrad/RL search','llm':'Direct LLM','union':'Virtual union'}
    for ax,data,x in [(axes[0],epoch,np.arange(101)),(axes[1],wall,grid)]:
        for source in ('search','llm','union'):
            median,lo,hi=upper_summary(data[source])
            if ax is axes[0]:
                ax.scatter([106],[median[-1]],color=colors[source],s=24,marker='x' if source=='union' else 'o')
                median,lo,hi=median[:101],lo[:101],hi[:101]
            finite=np.isfinite(median)
            ax.step(x,np.where(finite,median,np.nan),where='post',color=colors[source],
                label=labels[source],ls='--' if source=='union' else '-',lw=1.5 if source=='union' else 2)
            if source!='union':
                valid=np.isfinite(lo)&np.isfinite(hi)
                ax.fill_between(x,lo,hi,where=valid,step='post',color=colors[source],alpha=.12)
        ax.set_yscale('log');ax.set_ylabel('Best-so-far loss (upper median)')
    axes[0].set(xlabel='Completed epochs; T = terminal drain',xticks=[0,25,50,75,100,106],xticklabels=['0','25','50','75','100','T'])
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_xlabel('Elapsed minutes from each source run\'s start')
    available=np.isfinite(epoch['llm']).sum(0)
    better=((epoch['llm']<epoch['search'])&~np.isclose(epoch['llm'],epoch['search'],rtol=1e-9,atol=1e-12)&np.isfinite(epoch['llm'])).sum(0)
    for values,label,color in [(available,'LLM result available','#a64e85'),(better,'LLM lower than search','#376ab3')]:
        axes[2].step(np.arange(101),values[:101],where='post',label=label,color=color)
        axes[2].scatter([106],[values[-1]],color=color,s=22)
    axes[2].set(xlabel='Completed epochs; T = terminal drain',ylabel='Number of matched seeds',
        xticks=[0,25,50,75,100,106],xticklabels=['0','25','50','75','100','T'],ylim=(-.5,20.5))
    axes[2].legend(frameon=False,fontsize=8,loc='lower right')
    for ax,title in zip(axes,('A  Epoch-aligned source progress','B  Independent source clocks','C  Availability and loss leadership')):
        ax.set_title(title,loc='left',fontsize=11)
    fig.get_layout_engine().set(rect=(0,.18,1,.77))
    fig.text(.02,.025,textwrap.fill('20 matched seeds. Bands are empirical interquartile ranges, not confidence intervals. Unavailable LLM results remain infinite before aggregation. Virtual union = paired minimum of separately obtained source histories; it is not an observed concurrent run. Elapsed times use source-specific snapshot clocks and differ in worker allocation/load; no controlled speedup claim. Final points include terminal draining.',175),fontsize=8,va='bottom')
    fig.savefig(output/'RPA_INDEPENDENT_LOSS_TIME.pdf')
    fig.savefig(output/'RPA_INDEPENDENT_LOSS_TIME.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    analyze(parser.parse_args().output)
