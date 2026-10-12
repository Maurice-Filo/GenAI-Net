"""Event-time provenance, optimizer interaction and winner extraction; no simulations."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess

import numpy as np

from comparisons.search_exploration.analyze_rpa_mechanisms import dump_json, write_csv
from comparisons.search_exploration.analyze_rpa_flash_adagrad import FULL, CONTROL

GROUPS = ('llm_direct', 'llm_fit', 'llm_mutation', 'rl_direct', 'rl_fit', 'other_search', 'other_fit')
LABELS = ('Direct LLM', 'LLM + AdaGrad', 'LLM-lineage mutation', 'RL at initial rates',
          'RL + AdaGrad', 'Other search', 'Other search + AdaGrad')
COLORS = ('#a64e85', '#d08dae', '#bb7130', '#168577', '#68b3a1', '#617889', '#a5b2bc')
RECOVERY = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-directional-balance-recovery-20260918')


def read(path):
    return json.loads(path.read_text())


def connect(folder):
    path = (folder / 'exploration.sqlite').resolve()
    wal = path.with_name(path.name+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Only checkpointed completed databases can be inspected immutably')
    if read(folder/'completed.json')['status'] != 'complete':
        raise ValueError('Incomplete search')
    return sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)


def group(event):
    fitted = event.get('stream') == 'optimization'
    source, origin = event['source'], event['origin']
    if source == 'llm':
        return 'llm_fit' if fitted else 'llm_direct'
    if origin == 'llm':
        if source != 'ga':
            raise ValueError('Unexpected explicit LLM descendant emitter')
        return 'llm_mutation'
    if source == 'rl':
        return 'rl_fit' if fitted else 'rl_direct'
    return 'other_fit' if fitted else 'other_search'


def winner_weights(losses):
    losses = np.asarray(losses, float)
    best = losses.min()
    if not np.isfinite(best):
        return np.zeros(len(losses))
    ties = np.isclose(losses, best, rtol=1e-9, atol=1e-12)
    return ties / ties.sum()


def best_record(folder):
    with connect(folder) as db:
        row = db.execute("""SELECT payload FROM designs WHERE
            json_extract(payload,'$.simulation_complete')=1 ORDER BY
            json_extract(payload,'$.loss'),key LIMIT 1""").fetchone()
    return json.loads(row[0])


def extract(folder, seed):
    threshold = read(folder/'manifest.json')['config']['quality_threshold']
    curves = np.full((101, len(GROUPS)), np.inf)
    first, qualified, counts, interactions = {}, defaultdict(set), Counter(), Counter()
    llm_topologies = set()
    with connect(folder) as db:
        events = db.execute("""SELECT e.payload,
            json_extract(d.payload,'$.simulation_complete'),json_extract(d.payload,'$.topology_hash'),
            json_extract(p.payload,'$.loss'),json_extract(p.payload,'$.simulation_complete'),
            json_extract(p.payload,'$.topology_hash')
            FROM events e LEFT JOIN designs d ON d.key=json_extract(e.payload,'$.candidate_hash')
            LEFT JOIN designs p ON p.key=json_extract(e.payload,'$.parent_hash')
            WHERE e.kind='archive_insertion' ORDER BY e.id""")
        for payload, complete, topology, parent_loss, parent_complete, parent_topology in events:
            e = json.loads(payload)
            if topology is None:
                raise ValueError('Missing exact candidate record')
            g = group(e)
            counts[g] += 1
            if g == 'llm_direct':
                llm_topologies.add(topology)
            if not complete:
                continue
            loss, epoch = e['loss'], e['epoch']
            if not np.isfinite(loss) or loss < 0 or not 0 <= epoch <= 100:
                raise ValueError('Invalid event loss or epoch')
            index = GROUPS.index(g)
            curves[epoch, index] = min(curves[epoch,index], loss)
            if loss < threshold:
                qualified[g].add(topology)
                first.setdefault(topology, {'group':g, 'epoch':epoch})
            if g in ('llm_fit', 'llm_mutation'):
                interactions[g+'/complete_events'] += 1
                if e.get('parent_hash') is None or parent_loss is None:
                    interactions[g+'/parent_unavailable'] += 1
                    continue
                interactions[g+'/improved_complete_parent'] += int(bool(parent_complete) and loss < parent_loss)
                interactions[g+'/qualified_events'] += int(loss < threshold)
                interactions[g+'/rescued_incomplete_parent'] += int(not parent_complete and loss < threshold)
                if g == 'llm_mutation':
                    interactions[g+'/different_parent_topology'] += int(topology != parent_topology)
    curves = np.minimum.accumulate(curves, axis=0)
    endpoint = read(folder/'completed.json')['metrics']['Archive/best_loss']
    if not np.isclose(curves[-1].min(), endpoint, rtol=1e-12, atol=0):
        raise ValueError('Reconstructed best loss disagrees with endpoint')
    seen = set().union(*qualified.values())
    first_counts = Counter(x['group'] for x in first.values())
    record = best_record(folder)
    return {'seed': seed, 'folder':str(folder), 'threshold':threshold, 'curves':curves,
        'winner_weights': np.array([winner_weights(row) for row in curves]),
        'groups':[{'group':g, 'events':counts[g], 'first_qualified':first_counts[g],
            'seen_qualified':len(qualified[g]), 'qualified_not_direct_llm':len(qualified[g]-llm_topologies),
            'best_loss':curves[-1,GROUPS.index(g)]} for g in GROUPS],
        'qualified_total':len(seen), 'direct_llm_topologies':len(llm_topologies),
        'qualified_absent_from_direct_llm':len(seen-llm_topologies),
        'first_qualified':first, 'interactions':dict(interactions), 'winner':record}


def plot(root, runs):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,'pdf.fonttype':42})
    curves = np.array([r['curves'] for r in runs])
    weights = np.array([r['winner_weights'] for r in runs])
    pdf = PdfPages(root/'RPA_PROVENANCE_INTERACTION.pdf')
    def save(fig, name, caption):
        fig.text(.04,.02,caption,fontsize=8,va='bottom')
        fig.tight_layout(rect=(0,.12,1,.95),w_pad=2.5)
        fig.savefig(root/(name+'.png'),dpi=190)
        fig.savefig(root/(name+'.pdf'))
        pdf.savefig(fig); plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(13,5.5))
    for i, (label,color) in enumerate(zip(LABELS,COLORS)):
        # Keep missing groups at infinity: do not condition medians on early arrivals.
        med = np.sort(curves[:,:,i],axis=0)[len(runs)//2]
        med[~np.isfinite(med)] = np.nan
        ax[0].plot(np.arange(100),med[:100],label=label,color=color,lw=1.8)
        if np.isfinite(med[-1]): ax[0].scatter(105,med[-1],s=20,color=color)
    ax[0].set(yscale='log',ylabel='Best-so-far loss: upper median across seeds',
              xlabel='Recorded insertion epoch; T = drained endpoint',title='A  Which path supplies the low losses?',
              xticks=[0,25,50,75,99,105],xticklabels=['0','25','50','75','99','T'])
    ax[0].legend(frameon=False,fontsize=8,ncol=2)
    ax[1].stackplot(np.arange(100),weights[:,:100,:].mean(0).T,labels=LABELS,colors=COLORS,alpha=.9)
    ax[1].set(xlabel='Recorded insertion epoch',ylabel='Fraction of seeds supplying rank-one loss',
              ylim=(0,1),xlim=(0,99),title='B  Loss leadership during training')
    save(fig,'01_loss_dominance','20 completed full-duplex seeds. Curves include only results available at each event epoch; T includes terminal draining.\n'
         'Tied best sources share leadership credit. Missing-source losses count as infinity before aggregation. Attribution is not a causal ablation.')
    fig, ax = plt.subplots(1,3,figsize=(14,5.5))
    data=np.array([[g['first_qualified'] for g in r['groups']] for r in runs])
    ax[0].barh(np.arange(7),data.mean(0),xerr=data.std(0,ddof=1),color=COLORS,capsize=3)
    ax[0].set(yticks=np.arange(7),yticklabels=LABELS,xlabel='Mean first-qualified topologies (+/- SD)',title='A  Who expands the qualifying set?')
    ax[0].invert_yaxis()
    remaining=np.array([r['qualified_absent_from_direct_llm'] for r in runs])
    direct=np.array([r['qualified_total'] for r in runs])-remaining
    ax[1].bar(np.arange(1,21),remaining,color=COLORS[0+3],label='Not in direct LLM proposal set')
    ax[1].bar(np.arange(1,21),direct,bottom=remaining,color=COLORS[0],label='Also directly proposed by LLM')
    ax[1].set(xlabel='Seed',ylabel='Distinct qualifying topologies',title='B  Beyond the direct LLM proposals',xticks=[1,5,10,15,20])
    ax[1].set_ylim(0,float(np.max(remaining+direct))*1.24)
    ax[1].legend(frameon=False,fontsize=8)
    end=weights[:,-1,:].sum(0)
    ax[2].barh(np.arange(7),end,color=COLORS)
    ax[2].set(yticks=np.arange(7),yticklabels=LABELS,xlabel='Winning-rate seed credit (total 20)',title='C  Winning candidate path')
    ax[2].invert_yaxis()
    save(fig,'02_portfolio_provenance','First qualification uses recorded merge order, with one credit per topology. Rate fitting can qualify a structure without inventing it.\n'
         'B: retrospective structural overlap, not mechanistic novelty. C: emitter/fitter of winning rates, NOT first inventor of the structure.')
    pdf.close()


def classify_exposure(topology, hof, prior, loss):
    matches=[x for x in hof if x['topology_hash']==topology]
    rl=[x for x in matches if x.get('source','').lower()=='rl']
    return {'prior_llm_match':topology in prior,'hof_match':bool(matches),
        'rl_labelled_hof_match':bool(rl),'rl_hof_match_new_to_llm':bool(rl) and topology not in prior,
        'matching_hof_loss':min(x['loss'] for x in matches) if matches else None,
        'rl_hof_loss':min(x['loss'] for x in rl) if rl else None,
        'improves_matching_hof':bool(matches) and loss<min(x['loss'] for x in matches)}


def exposure(output):
    """Exact matches against the HOF actually exposed before each request."""
    rows=[]
    for job in read(FULL/'plan.json')['jobs']:
        folder=FULL/'runs'/job['name']
        mapping={}
        with connect(folder) as db:
            for (payload,) in db.execute("""SELECT DISTINCT d.payload FROM events e JOIN designs d
                ON d.key=json_extract(e.payload,'$.candidate_hash') WHERE e.kind='archive_insertion'
                AND json_extract(e.payload,'$.source')='llm'
                AND COALESCE(json_extract(e.payload,'$.stream'),'')!='optimization'"""):
                d=json.loads(payload); key=tuple(sorted(d['reaction_ids']))
                if key in mapping and mapping[key]!=d['topology_hash']: raise ValueError('Ambiguous structure identity')
                mapping[key]=d['topology_hash']
        prior=set()
        for workspace in sorted((folder/'harness-workspaces').iterdir()):
            file=workspace/'evaluations.jsonl'
            if not file.exists(): continue
            hof=read(workspace/'CONTEXT/HALL_OF_FAME.json')['entries']
            epoch=read(workspace/'CONTEXT/SEARCH_STATE.json')['rl_epoch_at_snapshot']
            newly=set()
            for text in file.read_text().splitlines():
                e=json.loads(text)
                if not e['valid']: continue
                key=tuple(sorted(e['candidate']['reaction_ids']))
                if key not in mapping: raise ValueError('Valid workspace proposal missing from host ledger')
                h=mapping[key]
                rows.append({'seed':job['seed'],'request_epoch':epoch,'workspace':str(workspace),
                    'topology_hash':h,'loss':e['loss'],**classify_exposure(h,hof,prior,e['loss'])})
                newly.add(h)
            prior.update(newly)
    write_csv(output/'return_channel_matches.csv',rows)
    summary={'valid_proposal_members':len(rows),**{k:sum(r[k] for r in rows) for k in
        ('prior_llm_match','hof_match','rl_labelled_hof_match','rl_hof_match_new_to_llm','improves_matching_hof')}}
    dump_json(output/'return_channel_matches.json',summary)
    (output/'RETURN_CHANNEL.md').write_text('# Saved Return-Channel Exposure\n\n'+
        '\n'.join(f'- {k}: {v}' for k,v in summary.items())+'\n\n'+
        'Counts are valid proposal members, not independent networks or seeds. Exact topology matches use the saved HOF at request launch, before the response. Prior-LLM membership uses only earlier requests. RL-labelled HOF entries identify their emitter, not independence from LLM-influenced policy training. Matching does not prove copying or causal use; absence does not rule out inspiration or non-exact structural edits.\n')
    winner_lineage(output,rows)
    return summary


def winner_lineage(output, proposals):
    rows=[]
    for r in read(output/'provenance.json'):
        h=r['winner']['topology_hash']
        with connect(Path(r['folder'])) as db:
            event=json.loads(db.execute("""SELECT e.payload FROM events e JOIN designs d
                ON d.key=json_extract(e.payload,'$.candidate_hash') WHERE e.kind='archive_insertion'
                AND json_extract(d.payload,'$.topology_hash')=? ORDER BY e.id LIMIT 1""",(h,)).fetchone()[0])
        hits=[p for p in proposals if p['seed']==r['seed'] and p['topology_hash']==h and p['rl_hof_match_new_to_llm']]
        first=hits[0] if hits else None
        rows.append({'seed':r['seed'],'topology_hash':h,'first_emitter':event['source'],
            'first_origin':event['origin'],'first_epoch':event['epoch'],'first_loss':event['loss'],
            'winner_loss':r['winner']['loss'],'exposed_rl_hof_before_first_llm':bool(hits),
            'exposed_hof_loss':first['rl_hof_loss'] if first else None,
            'first_matching_llm_loss':first['loss'] if first else None,
            'workspace':first['workspace'] if first else None})
    write_csv(output/'winner_structure_lineage.csv',rows)
    counts=Counter(r['first_emitter'] for r in rows)
    selected=[r for r in rows if r['exposed_rl_hof_before_first_llm']]
    fresh=[p for p in proposals if p['rl_hof_match_new_to_llm']]
    summary={'winner_structure_first_emitters':dict(counts),
        'winner_seeds_exposed_rl_hof_before_first_llm':[r['seed'] for r in selected],
        'new_to_llm_rl_hof_matching_members':len(fresh),
        'new_to_llm_rl_hof_matching_unique_seed_topologies':len({(p['seed'],p['topology_hash']) for p in fresh}),
        'new_to_llm_matches_improving_exposed_hof':sum(p['improves_matching_hof'] for p in fresh)}
    dump_json(output/'winner_structure_lineage.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,5.5))
    axes[0].bar(['RL first','LLM first'],[counts['rl'],counts['llm']],color=['#168577','#a64e85'])
    axes[0].set(ylabel='Final winning topologies (20 seeds)',title='A  Where did the structure appear first?',ylim=(0,20))
    axes[1].bar(['Matches','Improve HOF'],[len(fresh),summary['new_to_llm_matches_improving_exposed_hof']],color=['#376ab3','#a64e85'])
    axes[1].set(ylabel='Valid LLM proposal members',title='B  RL-labelled HOF matches\nnot in prior LLM requests')
    for r in selected:
        values=[r['first_loss'],r['exposed_hof_loss'],r['first_matching_llm_loss'],r['winner_loss']]
        axes[2].plot(range(4),values,'o-',ms=4,label=f"Seed {r['seed']}")
    axes[2].set(yscale='log',xticks=range(4),xticklabels=['First RL','Exposed\nHOF','First LLM\nmatch','Final'],
                ylabel='Canonical loss (same topology)',title='C  Four winning-structure histories')
    axes[2].legend(frameon=False,fontsize=8)
    fig.text(.04,.02,'A: first recorded structural occurrence, not provenance of the final rates. RL sampling may already reflect LLM-influenced policy updates.\n'
        'B: exact matches to HOF state exposed BEFORE each request, excluding earlier LLM matches; repeated members are not independent samples.\n'
        'C: temporal evidence consistent with reuse/refinement, not proof of copying or a causal advantage of the return channel.',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,1),w_pad=2.5)
    fig.savefig(output/'03_return_channel.png',dpi=190);fig.savefig(output/'03_return_channel.pdf');plt.close(fig)
    # Merge our own generated vector pages; no external documents are processed.
    temp=output/'RPA_PROVENANCE_INTERACTION.tmp.pdf'
    subprocess.run(['pdfunite',*[str(output/(n+'.pdf')) for n in
        ('01_loss_dominance','02_portfolio_provenance','03_return_channel')],str(temp)],check=True)
    temp.replace(output/'RPA_PROVENANCE_INTERACTION.pdf')
    report=output/'REPORT.md'
    text=report.read_text().split('## Structure versus Winning Rates')[0]
    text+='\n## Structure versus Winning Rates\n\n'
    text+=f"First recorded emitter of the final winning topology: {dict(counts)}. All winning rate vectors nevertheless come through direct LLM proposals or their AdaGrad fits. These statements describe different provenance axes.\n\n"
    text+=f"For seeds {[r['seed'] for r in selected]}, that winning structure was in an RL-labelled HOF entry exposed before its first LLM proposal. Across all requests, {len(fresh)} valid members match exposed RL-labelled HOF structures absent from prior LLM requests; {summary['new_to_llm_matches_improving_exposed_hof']} improve the matching HOF loss. There are {summary['new_to_llm_rl_hof_matching_unique_seed_topologies']} distinct seed/topology pairs among those matches.\n\n"
    text+='This is evidence consistent with RL-to-LLM reuse and subsequent refinement, not proof that the model copied a structure or that the return channel improves the endpoint distribution. First occurrence is not an independent causal origin: RL can already have learned from LLM-influenced replay. Numerical trajectories in the figure are chronological, not controlled interventions.\n\n![Structural and return-channel chronology](03_return_channel.png)\n'
    report.write_text(text)


def analyze(output):
    from comparisons.search_exploration.recover_rpa_directional_balance import request_audit
    from comparisons.search_exploration.analyze_rpa_directional_streams import direct_records
    output.mkdir(parents=True,exist_ok=False)
    runs=[]
    for j in read(FULL/'plan.json')['jobs']:
        folder=FULL/'runs'/j['name']
        if request_audit(folder)['balance_affected']: raise ValueError('Balance-affected full-duplex record')
        r=extract(folder,j['seed']); runs.append(r)
        print('Provenance seed',j['seed'],'winner',[(GROUPS[i],float(w)) for i,w in enumerate(r['winner_weights'][-1]) if w],flush=True)
    runs.sort(key=lambda r:r['seed'])
    dump_json(output/'provenance.json',runs)
    write_csv(output/'source_summary.csv',[{'seed':r['seed'],**g} for r in runs for g in r['groups']])
    winners=[{'arm':'flash','seed':r['seed'],'candidate':r['winner'],'source':r['folder']} for r in runs]
    cp=read(CONTROL/'active_plan.json')
    for j in cp['jobs']:
        if j['arm']=='julia':
            f=Path(j.get('output',CONTROL/'runs'/j['name']))
            winners.append({'arm':'control','seed':j['seed'],'candidate':best_record(f),'source':str(f)})
    rp=read(RECOVERY/'plan.json'); selected=read(RECOVERY/'analysis/summary.json')['matched_seeds']
    controls={w['seed']:w for w in winners if w['arm']=='control'}
    for j in rp['jobs']:
        if j['seed'] not in selected: continue
        f=Path(j['output'])
        if request_audit(f)['balance_affected']: raise ValueError('Invalid directional source')
        records=[best_record(f), *[d for d in direct_records(f) if d['simulation_complete']]]
        best=min(records,key=lambda d:(d['loss'],d['candidate_hash']))
        winners.append({'arm':j['arm'],'seed':j['seed'],'candidate':best,'source':str(f)})
        if j['arm']=='llm-to-rl':
            pool=[controls[j['seed']]['candidate'],*[d for d in direct_records(f) if d['simulation_complete']]]
            winners.append({'arm':'independent-union','seed':j['seed'],
                            'candidate':min(pool,key=lambda d:(d['loss'],d['candidate_hash'])),
                            'source':str(f),'control_source':controls[j['seed']]['source']})
    dump_json(output/'winners.json',{'directional_matched_seeds':selected,'designs':winners})
    plot(output,runs)
    weights=np.array([r['winner_weights'][-1] for r in runs]).sum(0)
    summary={'rank_one_credit':dict(zip(GROUPS,weights.tolist())),
        'mean_qualified':float(np.mean([r['qualified_total'] for r in runs])),
        'mean_qualified_absent_from_direct_llm':float(np.mean([r['qualified_absent_from_direct_llm'] for r in runs])),
        'first_qualified_means':{g:float(np.mean([r['groups'][i]['first_qualified'] for r in runs])) for i,g in enumerate(GROUPS)},
        'interactions':dict(sum((Counter(r['interactions']) for r in runs),Counter()))}
    dump_json(output/'summary.json',summary)
    lines=['# RPA Provenance and Interaction','',
        'Twenty completed full-duplex seeds; no model calls or simulations in this analysis.','',
        '| Path | Endpoint rank-one seed credit | Mean first-qualified topologies |','|---|---:|---:|']
    lines += [f'| {label} | {weights[i]:.2f} | {summary["first_qualified_means"][g]:.2f} |' for i,(g,label) in enumerate(zip(GROUPS,LABELS))]
    lines += ['',f'Mean qualifying structures absent from every direct LLM proposal in the same run: {summary["mean_qualified_absent_from_direct_llm"]:.2f} of {summary["mean_qualified"]:.2f}.','',
        'Origin follows explicit mutation parentage, not neural-policy ancestry. An RL emitter can have learned from LLM-influenced replay. AdaGrad is the rate optimizer, not the neural policy. LLM-lineage mutation can include subsequent AdaGrad fitting. Other search includes random proposals and mutations of non-LLM ancestors.','',
        'Best-loss histories use event-time provenance, not the final mutable record label. Parent-improvement counts are repeated events, not independent networks. First qualification is assigned once in merge order; rediscovery and delayed arrivals can change attribution. Structural absence from direct LLM proposals does not establish mechanistic novelty or an LLM-independent policy.','',
        'The distinction between excellent rank-one candidates and contributions to the wider qualifying set is descriptive, not evidence of causal full-duplex synergy. The directional interventions are the relevant ablations.','',
        '![Loss paths](01_loss_dominance.png)','', '![Portfolio paths](02_portfolio_provenance.png)','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    exposure(output)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
