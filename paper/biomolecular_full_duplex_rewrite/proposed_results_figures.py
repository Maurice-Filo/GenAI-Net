"""Review figures for the proposed Results narrative; saved data only.

Does not edit the manuscript, original figures, or campaign outputs.
Explanatory text belongs in the companion captions, not inside the figures.
"""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'results_figure_proposal'
OUT.mkdir(exist_ok=True)
SOURCES, SUMMARY = {}, {}
ARMS = ['none', 'llm-to-rl', 'rl-to-llm', 'full']
COLORS = dict(zip(ARMS, ['#737980', '#D58B28', '#167CAB', '#943C87']))
LABELS = ['Independent', 'Insertion only', 'Return only', 'Full duplex']
SHORT = ['Independent', 'Insertion', 'Return', 'Full duplex']
BLUE, GRAY, PURPLE, GREEN = '#167CAB', '#737980', '#943C87', '#26845C'


def read(path, csv_rows=False):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return list(csv.DictReader(raw.decode().splitlines())) if csv_rows else json.loads(raw)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 10, 'axes.titlesize': 11,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def axes_style(ax, title):
    ax.set_title(title, loc='left', pad=10)
    ax.grid(axis='y', color='#DDE2E6', lw=.55)
    ax.set_axisbelow(True)


def save(fig, name, bundle):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, facecolor='white')
    if bundle is not None:
        bundle.savefig(fig)
    plt.close(fig)


def sample(run, grid):
    x, y = np.asarray(run['cost']), np.asarray(run['loss'])
    assert np.isfinite(y).all() and (y > 0).all()
    assert (np.diff(x) >= 0).all() and (np.diff(y) <= 1e-10).all()
    assert grid[0] >= x[0] and grid[-1] <= x[-1]
    return np.minimum.accumulate(y)[np.searchsorted(x, grid, side='right') - 1]


def terminal_summary(group):
    """Verify full-run endpoints against source records before plotting them."""
    for run in group:
        folder = Path(run['folder'])
        if run['arm'] == 'original':
            endpoint = read(folder/'progress.csv', True)[-1]
            assert int(endpoint['step']) == 100
            final_loss = float(endpoint['best_so_far_loss'])
            final_cost = float(endpoint['ode_simulations'])
        else:
            endpoint = read(folder/'completed.json')
            assert endpoint['status'] == 'complete' and endpoint['epochs'] == 100
            final_loss = endpoint['metrics']['Archive/best_loss']
            final_cost = endpoint['counters']['expensive_evaluations']
            if run.get('cost_accounting') == 'numerical_only_excluding_independent_llm':
                manifest = read(folder/'manifest.json')
                assert manifest['communication_mode'] == 'none'
                excluded = endpoint['counters'].get('evaluations/llm', 0)
                assert run['raw_cost'][-1] == final_cost
                assert run['excluded_llm_evaluations'][-1] == excluded
                final_cost -= excluded
        np.testing.assert_allclose(run['loss'][-1], final_loss, rtol=1e-12, atol=1e-15)
        assert run['cost'][-1] == final_cost
        assert np.isfinite(final_loss) and final_loss > 0
    return np.quantile([r['loss'][-1] for r in group], [.25, .5, .75])


def terminal_marker(ax, group, allowance, index, count, color, hollow=False):
    low, median, high = terminal_summary(group)
    x = allowance/1000 * (1.15 + (index-(count-1)/2)*.016)
    ax.errorbar(x, median, yerr=[[median-low], [high-median]],
                fmt='o', ms=3.4, lw=1, color=color,
                mfc='white' if hollow else color, zorder=5)
    return dict(median=float(median), q25=float(low), q75=float(high),
                charged_min=min(r['cost'][-1] for r in group),
                charged_max=max(r['cost'][-1] for r in group))


def allowance_axis_with_terminal(ax, allowance):
    end = allowance/1000
    ticks = [x for x in [0,25,50,75,100] if x <= end]
    ax.axvline(end*1.05, color='#AAAAAA', ls=':', lw=.7)
    ax.set(xlim=(0,end*1.24), xticks=ticks+[end*1.15],
           xticklabels=[str(x) for x in ticks]+['T'])


def numerical(bundle, full=False):
    data = read(HERE / 'numerical_comparison/frozen_runs.json')
    rows = read(HERE / 'numerical_comparison/aggregate_curves.csv', True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.75))
    fig.subplots_adjust(left=.085, right=.98, top=.77, bottom=.18, wspace=.27)
    styles = [('julia', BLUE, 'Hybrid numerical search'), ('ga', GRAY, 'Standalone GA')]
    fig.legend([Line2D([], [], color=c, lw=2) for _, c, _ in styles],
               [label for _, _, label in styles], loc='upper center', ncol=2, frameon=False)
    for arm, color, label in styles:
        r = [v for v in rows if v['arm'] == arm]
        x = np.array([float(v['budget']) for v in r])
        median, lo, hi = [np.array([float(v[key]) for v in r]) for key in ('median','q25','q75')]
        stop = min(v['evaluations'][-1] for v in data['runs'] if v['arm'] == arm)
        cut = np.searchsorted(x, stop, side='right')
        axes[0].step(x[:cut]/1000, median[:cut], where='post', color=color, lw=2)
        if cut < len(x):
            axes[0].step(x[cut-1:]/1000, median[cut-1:], where='post', color=color, lw=2, ls='--')
        axes[0].fill_between(x/1000, lo, hi, step='post', color=color, alpha=.13, lw=0)
        axes[1].step(x/1000, [int(v['successful']) for v in r], where='post', color=color, lw=2)
    axes[0].axhline(data['threshold'], color='#333333', ls=':', lw=1)
    axes[0].set(yscale='log', ylabel='Best loss')
    axes[1].set(ylabel='Runs with a qualifying design', ylim=(-.5,21), yticks=[0,5,10,15,20])
    for ax,title in zip(axes,['A  Convergence','B  Search success']):
        axes_style(ax,title)
        ax.set(xlim=(0,data['ceiling']/1000 if full else 100),
               xticks=[0,50,100,150,200] if full else [0,25,50,75,100],
               xlabel=r'Charged-evaluation allowance ($\times 10^3$)')
    SUMMARY['numerical_full' if full else 'numerical'] = {
        'displayed_allowance':data['ceiling'] if full else 100000,
        'threshold':data['threshold'], 'ceiling':data['ceiling'],
        'successes':{a:sum(r['successful'] for r in data['runs'] if r['arm']==a) for a,_,_ in styles}}
    if not full:
        assert all(r['evaluations'][-1] > 100000 for r in data['runs'])
        assert all(r['first_success_checkpoint'] <= 100000 for r in data['runs'] if r['successful'])
    save(fig, 'figureS_numerical_search_full' if full else 'figure2_numerical_search', bundle)


def communication(bundle):
    rows = read(HERE / 'convergence/snapshots.csv',True)
    references = read(HERE / 'genainet_comparison/audit.json')['runs']
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.9))
    fig.subplots_adjust(left=.085,right=.98,top=.77,bottom=.17,wspace=.27)
    fig.legend([Line2D([],[],color=COLORS[a],lw=2) for a in ARMS]
               + [Line2D([],[],color='#222222',lw=1.8,ls='--')],LABELS+['GenAI-Net'],
               loc='upper center',ncol=5,frameon=False,columnspacing=1.3)
    SUMMARY['communication_reference'] = {}
    for ax,task,title in zip(axes,['rpa','logic'],['A  RPA','B  Logic']):
        reference = sorted([r for r in references if r['task']==task and r['arm']=='original'],
                           key=lambda r:r['seed'])
        assert len(reference)==20 and [r['seed'] for r in reference]==list(range(20))
        assert all(r['epoch']==list(range(1,101)) for r in reference)
        reference_values=np.asarray([r['loss'] for r in reference])
        assert np.isfinite(reference_values).all() and (reference_values>0).all()
        assert (np.diff(reference_values,axis=1)<=1e-10).all()
        ordered=np.sort(reference_values,axis=0)
        median=(ordered[9]+ordered[10])/2
        low,high=ordered[4],ordered[14]
        ax.step(np.arange(1,101),median,where='post',color='#222222',ls='--',lw=1.8)
        ax.fill_between(np.arange(1,101),low,high,step='post',color='#222222',alpha=.07,lw=0)
        SUMMARY['communication_reference'][task]=dict(n=20,epoch_range=[1,100],
            median_final=float(median[-1]),alignment='Completed epochs; no terminal-drain point',
            folders=[r['folder'] for r in reference])
        for arm in ARMS:
            group=[r for r in rows if r['task']==task and r['arm']==arm]
            assert len(group)==2020
            values=[]
            for seed in range(1,21):
                selected=[r for r in group if int(r['seed'])==seed]
                selected.sort(key=lambda r:101 if r['completed_epochs']=='T' else int(r['completed_epochs']))
                values.append([float(r['union']) for r in selected])
            values=np.sort(values,axis=0)
            median=(values[9]+values[10])/2
            low,high=values[4],values[14]
            x=np.arange(1,101)
            ax.step(x,median[:100],where='post',color=COLORS[arm],lw=1.8,
                    ls='--' if task=='rpa' and arm=='none' else '-')
            ax.fill_between(x,low[:100],high[:100],step='post',color=COLORS[arm],alpha=.10,lw=0)
            terminal=106+(ARMS.index(arm)-1.5)*.8
            ax.errorbar(terminal,median[-1],yerr=[[median[-1]-low[-1]],[high[-1]-median[-1]]],
                        color=COLORS[arm],fmt='o',ms=3.5,lw=1)
        ax.axvline(103,color='#AAAAAA',ls=':',lw=.7)
        ax.set(yscale='log',xlim=(0,110),xticks=[0,25,50,75,100,106],
               xticklabels=['0','25','50','75','100','T'],xlabel='Search epoch',ylabel='Best loss')
        axes_style(ax,title)
    save(fig,'figure3_communication',bundle)


def portfolios(bundle):
    canonical=read(HERE.parent/'jmlr_hybrid_crn/generated/canonical_audit.json')
    provenance=read(HERE.parent/'jmlr_hybrid_crn/generated/logic_provenance.json')
    screening=read(OUT/'rpa_integral_screen.csv',True)
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.8))
    fig.subplots_adjust(left=.085,right=.98,top=.94,bottom=.14,hspace=.44,wspace=.28)
    jitter=np.random.default_rng(42).uniform(-.16,.16,20)
    def distribution(ax,values,x,color):
        parts=ax.violinplot(values,positions=[x],widths=.74,showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.12)
        ax.scatter(x+jitter,values,s=18,color=color,alpha=.68,linewidths=.4,
                   edgecolors=color,zorder=3)
        ax.hlines(np.median(values),x-.20,x+.20,color=color,lw=2.6,zorder=4)

    SUMMARY['logic_portfolios']={}
    for ax,key,title in [(axes[0,0],'valid','A  Logic: qualifying structures')]:
        for i,arm in enumerate(ARMS):
            vals=np.array([r[key] for r in canonical['rows'] if r['arm']==arm])
            assert len(vals)==20
            mean=vals.mean()
            distribution(ax,vals,i,COLORS[arm])
            SUMMARY['logic_portfolios'].setdefault(arm,{})[key]=float(mean)
        ax.set(xticks=range(4),xticklabels=SHORT,ylabel='Distinct structures per run',ylim=(0,None),xlim=(-.45,3.45))
        axes_style(ax,title)
    ax=axes[1,0]
    route_keys=['llm_direct','llm_fit','rl_fit','remaining']
    route_labels=['LLM proposal','LLM + AdaGrad','RL + AdaGrad','Other / shared routes']
    route_colors=['#D58B28',PURPLE,BLUE,'#BBC1C6']
    for i,arm in enumerate(ARMS):
        counts=[r for r in provenance['counts'] if r['arm']==arm]
        for r in counts:
            target=next(t for t in canonical['rows'] if t['arm']==arm and t['seed']==r['seed'])
            assert sum(v for k,v in r.items() if k not in ('arm','seed'))==target['robust_cores']
        means={k:np.mean([r[k] for r in counts]) for k in counts[0] if k not in ('arm','seed')}
        means['remaining']=sum(v for k,v in means.items() if k not in route_keys)
        base=0
        for k,label,color in zip(route_keys,route_labels,route_colors):
            ax.bar(i,means[k],bottom=base,width=.60,color=color,edgecolor='white',linewidth=.35,label=label if i==0 else None)
            base+=means[k]
    ax.set(xticks=range(4),xticklabels=SHORT,ylabel='Mean robust structures per run',ylim=(0,19),yticks=[0,5,10,15])
    axes_style(ax,'C  Logic: recorded discovery routes')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False,fontsize=8.5,columnspacing=1)
    rpa_arms=['control','independent-union','llm-to-rl','rl-to-llm','full']
    rpa_labels=['No LLM','Independent','Insertion','Return','Full duplex']
    rpa_colors=['#168577',GRAY,COLORS['llm-to-rl'],BLUE,PURPLE]
    SUMMARY['rpa_integral_screen']={}
    for ax,keys,title in [
            (axes[0,1],['exact_balance_fraction'],'B  RPA: integral-balance signatures'),
            (axes[1,1],['antithetic_motif_fraction','aif_like_fraction'],'D  RPA: antithetic-like motifs')]:
        for i,(arm,color) in enumerate(zip(rpa_arms,rpa_colors)):
            rows=sorted([r for r in screening if r['arm']==arm],key=lambda r:int(r['seed']))
            assert [int(r['seed']) for r in rows]==list(range(1,21))
            values={key:np.array([float(r[key]) for r in rows])*100 for key in keys}
            for key,vals in values.items():
                assert np.isfinite(vals).all() and ((vals>=0)&(vals<=100)).all()
                SUMMARY['rpa_integral_screen'].setdefault(arm,{})[key]=float(vals.mean()/100)
            if len(keys)==1:
                distribution(ax,values[keys[0]],i,color)
            else:
                structural,balanced=values[keys[0]],values[keys[1]]
                assert (balanced<=structural+1e-9).all()
                x=i+jitter
                ax.plot(np.c_[x,x].T,np.array([balanced,structural]),color=color,alpha=.22,lw=.7)
                ax.scatter(x,structural,facecolors='none',edgecolors=color,s=26,
                           marker='D',linewidths=1,alpha=.85,zorder=3)
                ax.scatter(x,balanced,color=color,s=17,linewidths=0,alpha=.85,zorder=4)
        ax.set(xticks=range(5),xticklabels=rpa_labels,ylabel='Qualifying topologies (%)',
               ylim=(-4,104),yticks=[0,25,50,75,100],xlim=(-.5,4.5))
        ax.tick_params(axis='x',labelsize=8)
        axes_style(ax,title)
    axes[1,1].legend([Line2D([],[],marker='D',mfc='white',color='#222222',ls='none',ms=5),
                      Line2D([],[],marker='o',color='#222222',ls='none',ms=5)],
                     ['Structural motif','Motif + integral balance'],
                     loc='upper center',bbox_to_anchor=(.5,-.22),ncol=1,frameon=False,fontsize=8.5)
    for ax in axes.flat:
        ax.grid(axis='y',color='#E8EBEE',lw=.55)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
        ax.tick_params(width=.7,length=3.5)
    save(fig,'figure4_portfolios_and_robustness',bundle)


def rpa_provenance():
    rows=read(OUT/'rpa_provenance_snapshot.json')
    assert sorted(r['seed'] for r in rows)==list(range(1,21))
    keys=['llm_direct','llm_fit','llm_mutation','rl_direct','rl_fit','other_search','other_fit']
    labels=['LLM proposal','LLM + AdaGrad','LLM-lineage mutation','RL at initial rates',
            'RL + AdaGrad','Other search','Other search + AdaGrad']
    colors=[COLORS['llm-to-rl'],PURPLE,'#B39C75',BLUE,'#72ADC7',GRAY,'#BBC1C6']
    data=np.array([[next(g['first_qualified'] for g in r['groups'] if g['group']==k)
                    for k in keys] for r in rows])
    np.testing.assert_array_equal(data.sum(axis=1),[r['qualified_total'] for r in rows])
    weights=np.array([r['winner_weights'] for r in rows])
    np.testing.assert_allclose(weights.sum(axis=1),1)
    bootstrap=np.random.default_rng(20260925).integers(0,20,(20000,20))
    mean=data.mean(axis=0);lo,hi=np.quantile(data[bootstrap].mean(axis=1),[.025,.975],axis=0)
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),sharey=True)
    fig.subplots_adjust(left=.20,right=.98,bottom=.17,top=.88,wspace=.18)
    axes[0].barh(range(7),mean,xerr=[mean-lo,hi-mean],color=colors,capsize=3)
    axes[1].barh(range(7),weights.sum(axis=0),color=colors)
    axes[0].set(yticks=range(7),yticklabels=labels,xlabel='Mean first-qualified topologies per run')
    axes[0].invert_yaxis()
    axes[1].set(xlabel='Winning-rate credit across 20 runs',xlim=(0,20),xticks=[0,5,10,15,20])
    for ax,title in zip(axes,['A  RPA: portfolio contributions','B  RPA: final best-loss contributions']):
        axes_style(ax,title);ax.grid(False,axis='y');ax.grid(axis='x',color='#DDE2E6',lw=.55)
    SUMMARY['rpa_provenance']={'mean_first_qualified':dict(zip(keys,mean.tolist())),
                              'winning_rate_credit':dict(zip(keys,weights.sum(axis=0).tolist()))}
    save(fig,'figureS_rpa_provenance',None)


def response_snapshots():
    """Freeze terminal outputs of the same 20 run winners used in the overviews."""
    cache=OUT/'response_snapshots.json'
    if cache.exists():
        data=read(cache)
        for key,item in data.items():
            path=HERE/f'{key}_performance/overview_snapshot.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['overview_sha256']
        return data
    data={}
    for key in ('hill','ultrasensitive','biphasic','classifier'):
        path=HERE/f'{key}_performance/overview_snapshot.json'
        snapshot=read(path)
        assert snapshot['n_completed']==20 and len(snapshot['sources'])==20
        rows=[]
        for source in snapshot['sources']:
            run=Path(source['run']); winner=source['winner']
            done=read(run/'completed.json')
            assert done['status']=='complete' and done['epochs']==100
            assert np.isclose(winner['loss'],done['metrics']['Archive/best_loss'],rtol=1e-9,atol=1e-12)
            dbpath=run/'exploration.sqlite';wal=Path(str(dbpath)+'-wal')
            assert not wal.exists() or wal.stat().st_size==0
            with sqlite3.connect(dbpath.as_uri()+'?mode=ro&immutable=1',uri=True) as db:
                blob=db.execute('SELECT payload FROM diagnostics WHERE key=?',(winner['evaluation_cache_key'],)).fetchone()
                assert blob is not None
                info=json.loads(zlib.decompress(blob[0]))
            target=np.asarray(info['setpoint']);endpoint=np.asarray(info['outputs'])[:,:,-1]
            if key=='classifier':
                initial=np.asarray(info['initial_conditions']);x=initial[:,0]-initial[:,1]
                assert target.shape==endpoint.shape==(32,2)
                np.testing.assert_array_equal(target.argmax(axis=1),(x<0).astype(int))
                y=endpoint[:,0]-endpoint[:,1];target=target[:,0]-target[:,1]
            else:
                x=np.asarray(info['inputs']).ravel();y=endpoint[:,0];target=target.ravel()
                assert len(x)==10
            assert np.isfinite(y).all()
            rows.append(dict(seed=source['seed'],loss=winner['loss'],candidate_hash=winner['candidate_hash'],
                evaluation_cache_key=winner['evaluation_cache_key'],run=str(run),
                diagnostics_sha256=hashlib.sha256(blob[0]).hexdigest(),time_horizon=info['time_horizon'],
                inputs=x.tolist(),outputs=y.tolist(),targets=target.tolist()))
        rows.sort(key=lambda r:r['seed'])
        assert [r['seed'] for r in rows]==list(range(1,21))
        best=min(rows,key=lambda r:(r['loss'],r['seed'],r['candidate_hash']))
        data[key]=dict(overview_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,
                       best_seed=best['seed'],best_loss=best['loss'])
    cache.write_text(json.dumps(data,indent=2)+'\n')
    return read(cache)


def breadth(bundle):
    data=read(HERE/'genainet_comparison/audit.json')
    qwen=read(HERE/'qwen_comparison/qwen_preliminary_convergence_snapshot.json')
    assert len(qwen['qwen_runs'])==40 and all(r['status']=='complete' for r in qwen['qwen_runs'])
    runs=data['runs']+qwen['qwen_runs']
    styles={'original':(GRAY,'--','GenAI-Net'),'hybrid':(BLUE,'-','Hybrid numerical search'),
            'full':(PURPLE,'-','Full duplex: Flash'),'qwen':(GREEN,'-','Full duplex: Qwen')}
    tasks=[('rpa','RPA'),('logic','Logic'),('dose_hill','Hill dose response'),
           ('dose_ultrasensitive','Ultrasensitivity'),('dose_biphasic','Biphasic response'),('classifier','Classification')]
    responses=response_snapshots()
    fig=plt.figure(figsize=(11.6,9.7))
    gs=fig.add_gridspec(3,1,height_ratios=[1.10,1,1],left=.075,right=.985,bottom=.12,top=.89,hspace=.67)
    top=gs[0].subgridspec(1,2,wspace=.23);middle=gs[1].subgridspec(1,4,wspace=.43)
    bottom=gs[2].subgridspec(1,4,wspace=.43)
    fig.legend([Line2D([],[],color=c,ls=ls,lw=1.9) for c,ls,_ in styles.values()],
               [s[2] for s in styles.values()],loc='upper center',bbox_to_anchor=(.53,.99),
               ncol=2,frameon=False,columnspacing=2.5)
    SUMMARY['common_allowance_endpoints']=[]
    SUMMARY['terminal_endpoints']=[]
    for k,(task,title) in enumerate(tasks):
        ax=fig.add_subplot(top[0,k] if k<2 else middle[0,k-2])
        selected=[r for r in runs if r['task']==task]
        lo=max(r['cost'][0] for r in selected);hi=min(102400,min(r['cost'][-1] for r in selected))
        grid=np.unique([lo,hi]+[c for r in selected for c in r['cost'] if lo<=c<=hi])
        present=[a for a in styles if any(r['arm']==a for r in selected)]
        for arm,(color,ls,label) in styles.items():
            group=[r for r in selected if r['arm']==arm]
            if not group:continue
            assert len(group)==20 and len({r['seed'] for r in group})==20
            values=np.array([sample(r,grid) for r in group]);q25,median,q75=np.quantile(values,[.25,.5,.75],axis=0)
            ax.step(grid/1000,median,where='post',color=color,ls=ls,lw=1.7)
            ax.fill_between(grid/1000,q25,q75,step='post',color=color,alpha=.11,lw=0)
            SUMMARY['common_allowance_endpoints'].append(dict(task=task,arm=arm,allowance=hi,median=float(median[-1])))
            terminal=terminal_marker(ax,group,hi,present.index(arm),len(present),color)
            assert terminal['median'] <= median[-1]+1e-12
            SUMMARY['terminal_endpoints'].append(dict(task=task,arm=arm,**terminal))
        ax.set(yscale='log')
        allowance_axis_with_terminal(ax,hi)
        axes_style(ax,f'{chr(65+k)}  {title}')
        ax.set_xlabel(r'Charged allowance ($\times 10^3$)',fontsize=9)
        if k>=2:ax.tick_params(labelsize=8)
    fig.text(.017,.69,'Best loss',rotation=90,va='center',fontsize=11)
    targets={'hill':lambda u:2*u**2/(.25**2+u**2),
             'ultrasensitive':lambda u:2*u**8/(.5**8+u**8),
             'biphasic':lambda u:8*u/(1+u)/(1+(u/.55)**4)}
    for k,(key,title) in enumerate(zip(('hill','ultrasensitive','biphasic','classifier'),
                                      [t[1] for t in tasks[2:]])):
        ax=fig.add_subplot(bottom[0,k]);item=responses[key]
        best=next(r for r in item['rows'] if r['seed']==item['best_seed'])
        for row in item['rows']:
            if row['seed']==item['best_seed']:continue
            if key=='classifier':
                ax.scatter(row['inputs'],row['outputs'],s=9,color=GRAY,alpha=.18,linewidths=0)
            else:
                np.testing.assert_allclose(row['targets'],targets[key](np.asarray(row['inputs'])),rtol=1e-6,atol=1e-8)
                order=np.argsort(row['inputs'])
                ax.plot(np.asarray(row['inputs'])[order],np.asarray(row['outputs'])[order],color=GRAY,alpha=.25,lw=.8)
        x=np.asarray(best['inputs']);y=np.asarray(best['outputs']);order=np.argsort(x)
        if key=='classifier':
            lim=max(abs(x))*1.08
            ax.plot([-lim,0,0,lim],[-1,-1,1,1],color='#222222',ls='--',lw=1.3)
            ax.scatter(x,y,s=15,color=PURPLE,zorder=4,linewidths=.3,edgecolors='white')
            ax.axhline(0,color='#AAAAAA',lw=.5);ax.axvline(0,color='#AAAAAA',lw=.5)
            ax.set(xlabel='Initial input difference',ylabel='Final output difference')
        else:
            u=np.linspace(0,1,300)
            ax.plot(u,targets[key](u),color='#222222',ls='--',lw=1.3)
            ax.plot(x[order],y[order],color=PURPLE,marker='o',ms=3,lw=1.5,zorder=4)
            ax.set(xlabel='Input concentration')
            if k==0:ax.set_ylabel('Final output')
        axes_style(ax,f'{chr(71+k)}  {title}')
        ax.tick_params(labelsize=8);ax.xaxis.label.set_size(9);ax.yaxis.label.set_size(9)
    fig.legend([Line2D([],[],color='#222222',ls='--',lw=1.4),
                Line2D([],[],color=PURPLE,marker='o',lw=1.6,ms=4),
                Line2D([],[],color=GRAY,alpha=.5,lw=1)],
               ['Target','Best final design','Other run winners'],loc='lower center',
               bbox_to_anchor=(.53,.025),ncol=3,frameon=False,columnspacing=2)
    SUMMARY['response_panels']={k:dict(best_seed=v['best_seed'],best_loss=v['best_loss'],
                                     n=len(v['rows']),selection='Minimum final canonical loss, including terminal drain')
                               for k,v in responses.items()}
    save(fig,'figure5_tasks_and_models',bundle)


if __name__=='__main__':
    style()
    numerical(None, full=True)
    rpa_provenance()
    with PdfPages(OUT/'proposed_results_figures.pdf') as bundle:
        numerical(bundle)
        communication(bundle)
        portfolios(bundle)
        breadth(bundle)
    (OUT/'audit.json').write_text(json.dumps(dict(sources=SOURCES,summary=SUMMARY,
        note='Proposal only. All loss curves use medians and IQR; Qwen recomputed from its complete frozen trajectories. No new simulations or inference.'),indent=2)+'\n')
    print(json.dumps(SUMMARY,indent=2))
