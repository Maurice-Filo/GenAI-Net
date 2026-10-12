"""Read-only source decomposition of the audited post-hoc independent RPA union."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import textwrap

import numpy as np

from comparisons.search_exploration.analyze_rpa_communication_families import (
    THRESHOLD, aif_like, digest, union_records,
)
from comparisons.search_exploration.analyze_rpa_directional_streams import (
    audit_self_feedback_bank, direct_records,
)
from comparisons.search_exploration.analyze_rpa_llm_provenance import (
    RECOVERY, extract as provenance, read,
)
from comparisons.search_exploration.analyze_rpa_mechanisms import (
    dump_json, integral_balance, rarefied_richness, reaction_catalog,
    reactions_for, write_csv,
)
from comparisons.search_exploration.analyze_rpa_flash_adagrad import CONTROL

ATLAS = Path('/local1/rossin/search-exploration/rpa-llm-interaction-20260918/topology-families')
COLORS = ('#168577', '#a64e85', '#6b6975')
LABELS = ('AdaGrad/RL search', 'Direct LLM', 'Independent union')


def partition(search, llm):
    """Partition qualifying topology sets, not reaction sets or proposal occurrences."""
    return {'search_only': len(search-llm), 'shared': len(search & llm),
            'llm_only': len(llm-search), 'search': len(search), 'llm': len(llm),
            'union': len(search | llm)}


def winning_source(search_loss, llm_loss):
    if np.isclose(search_loss, llm_loss, rtol=1e-9, atol=1e-12):
        return 'tie'
    return 'llm' if llm_loss < search_loss else 'search'


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    recovered = read(RECOVERY/'analysis/summary.json')
    if recovered['audit_errors'] or recovered['matched_seeds'] != list(range(1, 21)):
        raise ValueError('Requires twenty audited matched seeds')
    inputs = read(ATLAS/'inputs.json')
    for file, sha in inputs['files'].items():
        if digest(Path(file)) != sha:
            raise ValueError('Atlas input has changed')
    for files in inputs['runs'].values():
        for file, sha in files.items():
            if digest(Path(file)) != sha:
                raise ValueError('Completed source artifact changed')
    with gzip.open(ATLAS/'extracted.json.gz', 'rt') as stream:
        extracted = json.load(stream)
    controls = {r['seed']: r for r in extracted if r['arm']=='control'}
    unions = {r['seed']: r for r in extracted if r['arm']=='independent-union'}
    families = {r['topology_hash']: r['family'] for r in read(ATLAS/'topology_structures.json')}
    catalog, fixed = reaction_catalog(CONTROL)
    per_seed, sources, first_rows, memberships, curves = [], [], [], [], []
    provenance_rows, audit_rows = [], []
    thresholds = np.unique(np.r_[np.geomspace(.0002, .08, 160), THRESHOLD])
    for ref in sorted(recovered['independent_unions'], key=lambda r: r['seed']):
        seed = ref['seed']
        control, bank = Path(ref['control']), Path(ref['bank'])
        if control.resolve() != Path(controls[seed]['folder']).resolve():
            raise ValueError('Wrong no-LLM source')
        if read(control/'manifest.json')['harness'] is not False:
            raise ValueError('Control contains LLM integration')
        audited = audit_self_feedback_bank(bank)
        raw = direct_records(bank)
        if digest(bank/'direct_llm_proposals.jsonl') != inputs['runs'][str(bank)][str(bank/'direct_llm_proposals.jsonl')]:
            raise ValueError('Wrong direct proposal bank')
        sr = controls[seed]['qualified']
        lr = [r for r in union_records(raw) if r['loss'] < THRESHOLD]
        ur = union_records(sr, lr)
        s, l = {r['topology_hash'] for r in sr}, {r['topology_hash'] for r in lr}
        part = partition(s, l)
        if (part['search'], part['llm'], part['shared'], part['union']) != (
                ref['search_qualified'], ref['llm_qualified'], ref['overlap_qualified'], ref['union_qualified']):
            raise ValueError('Source decomposition does not reconstruct audited endpoint')
        if {r['topology_hash'] for r in ur} != {r['topology_hash'] for r in unions[seed]['qualified']}:
            raise ValueError('Topology union differs from the frozen atlas')
        sf, lf = {families[h] for h in s}, {families[h] for h in l}
        fp = partition(sf, lf)
        sl, ll = controls[seed]['best']['loss'], min(r['loss'] for r in lr)
        if not np.isclose(ll, ref['llm_best_loss'], rtol=1e-12, atol=0):
            raise ValueError('LLM endpoint mismatch')
        search_map, llm_map = {r['topology_hash']: r for r in sr}, {r['topology_hash']: r for r in lr}
        overlap_winners = Counter(winning_source(search_map[h]['loss'], llm_map[h]['loss']) for h in s & l)
        proposed = {r['topology_hash'] for r in raw}
        per_seed.append({'seed': seed, **{'topologies_'+k: v for k,v in part.items()},
            **{'families_'+k: v for k,v in fp.items()}, 'search_best_loss': sl, 'llm_best_loss': ll,
            'llm_search_loss_ratio': ll/sl, 'best_source': winning_source(sl, ll),
            'search_qualified_absent_from_recorded_llm_structures': len(s-proposed),
            'overlap_llm_better': overlap_winners['llm'], 'overlap_search_better': overlap_winners['search'],
            'overlap_tie': overlap_winners['tie']})
        for h in sorted(s | l):
            memberships.append({'seed': seed, 'topology_hash': h, 'family': families[h],
                'qualifying_source': 'shared' if h in s & l else ('search_only' if h in s else 'llm_only'),
                'search_loss': search_map[h]['loss'] if h in s else None,
                'llm_loss': llm_map[h]['loss'] if h in l else None})
        for name, rr in [('search', sr), ('llm', lr), ('union', ur)]:
            labs = [families[r['topology_hash']] for r in rr]
            balance = [integral_balance(reactions_for(r, catalog, fixed)) for r in rr]
            motifs = [aif_like(reactions_for(r, catalog, fixed), b) for r,b in zip(rr,balance)]
            sources.append({'seed': seed, 'source': name, 'topologies': len(rr),
                'families': len(set(labs)), 'rarefied_families_k20': rarefied_richness(labs, 20),
                'aif_like_fraction': float(np.mean(motifs)), 'best_loss': min(r['loss'] for r in rr)})
        search_losses = np.asarray(controls[seed]['losses'])
        llm_losses = np.asarray([r['loss'] for r in union_records(raw)])
        union_losses = np.asarray(unions[seed]['losses'])
        for name, losses in [('search', search_losses), ('llm', llm_losses), ('union', union_losses)]:
            curves.extend({'seed': seed, 'source': name, 'threshold': float(t), 'count': int(n)}
                          for t,n in zip(thresholds, np.searchsorted(losses, thresholds, side='left')))
        p = provenance(control, seed)
        if p['qualified_total'] != len(s) or any(p['first_qualified'][h]['group'].startswith('llm') for h in s):
            raise ValueError('No-LLM provenance does not reconstruct control')
        counts = Counter(p['first_qualified'][h]['group'] for h in s)
        first_rows.append({'seed': seed, **{g: counts[g] for g in ('rl_direct','rl_fit','other_search','other_fit')}})
        provenance_rows.append({'seed': seed, 'first_qualified': p['first_qualified'], 'groups': p['groups']})
        audit_rows.append({'seed': seed, 'control': str(control), 'bank': str(bank),
                           'audited_request_contexts': audited, 'raw_evaluated_members': len(raw)})
        print(f'Source split and first-qualification audit {seed}/20', flush=True)
    write_csv(output/'per_seed.csv', per_seed)
    write_csv(output/'source_metrics.csv', sources)
    write_csv(output/'search_first_qualifications.csv', first_rows)
    write_csv(output/'topology_source_membership.csv', memberships)
    write_csv(output/'threshold_curves.csv', curves)
    dump_json(output/'search_provenance.json', provenance_rows)
    summary = {'n': 20, 'post_hoc': True, 'new_simulations': 0, 'new_model_calls': 0,
        'threshold': THRESHOLD, 'family_cut': .6, 'families_from': str(ATLAS),
        'best_source_counts': dict(Counter(r['best_source'] for r in per_seed)),
        'mean_partition': {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0] if k not in ('seed','best_source')},
        'source_summary': {a: {k: float(np.mean([r[k] for r in sources if r['source']==a]))
                              for k in ('topologies','families','rarefied_families_k20','aif_like_fraction')}
                           for a in ('search','llm','union')},
        'median_best_loss': {a: float(np.median([r['best_loss'] for r in sources if r['source']==a]))
                             for a in ('search','llm','union')},
        'median_paired_llm_search_loss_ratio': float(np.median([r['llm_search_loss_ratio'] for r in per_seed])),
        'mean_search_first_qualifications': {k: float(np.mean([r[k] for r in first_rows]))
                                           for k in first_rows[0] if k != 'seed'}}
    dump_json(output/'summary.json', summary)
    paths = [ATLAS/'inputs.json', ATLAS/'extracted.json.gz', ATLAS/'topology_structures.json',
             RECOVERY/'analysis/summary.json', Path(__file__)]
    dump_json(output/'inputs.json', {'hashes': {str(p): digest(p) for p in paths}, 'audits': audit_rows})
    render(output, per_seed, sources, curves, first_rows)
    report(output, summary)
    print(json.dumps(summary, indent=2), flush=True)


def render(output, rows, sources, curves, first):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    pdf = PdfPages(output/'RPA_INDEPENDENT_UNION_SOURCES.pdf')

    def save(fig, name, caption):
        fig.get_layout_engine().set(rect=(0,.18,1,.77))
        fig.text(.025,.022,textwrap.fill(caption,155),fontsize=8,va='bottom')
        fig.savefig(output/(name+'.png'),dpi=180)
        fig.savefig(output/(name+'.pdf')); pdf.savefig(fig); plt.close(fig)

    fig, axes = plt.subplots(1,3,figsize=(14,5.5),layout='constrained',gridspec_kw={'width_ratios':[.8,1.2,1.1]})
    for r in rows:
        axes[0].plot([0,1],[r['search_best_loss'],r['llm_best_loss']],color='.78',lw=.8,zorder=0)
    for i,k in enumerate(('search_best_loss','llm_best_loss')):
        v=[r[k] for r in rows]; axes[0].scatter(np.full(20,i),v,color=COLORS[i],s=20)
        axes[0].hlines(np.median(v),i-.12,i+.12,color=COLORS[i],lw=3)
    axes[0].set(yscale='log',xticks=[0,1],xticklabels=['AdaGrad/RL','Direct LLM'],ylabel='Best canonical loss',xlim=(-.4,1.4))
    bottom=np.zeros(20)
    for key,color,label in [('search_only',COLORS[0],'Search only'),('shared',COLORS[2],'Shared'),('llm_only',COLORS[1],'LLM only')]:
        values=np.array([r['topologies_'+key] for r in rows])
        axes[1].bar(range(1,21),values,bottom=bottom,color=color,label=label,width=.85)
        bottom+=values
    axes[1].set(xlabel='Matched seed',ylabel='Qualifying topology count',xticks=[1,5,10,15,20])
    axes[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
    for a,color,label in zip(('search','llm','union'),COLORS,LABELS):
        rr=[r for r in curves if r['source']==a]
        t=np.array(sorted({r['threshold'] for r in rr}))
        v=np.array([[r['count'] for r in rr if r['seed']==seed] for seed in range(1,21)])
        mean,sd=v.mean(0),v.std(0,ddof=1)
        axes[2].plot(t,mean,color=color,label=label)
        axes[2].fill_between(t,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.10)
    axes[2].axvline(THRESHOLD,ls='--',color='.4',lw=1)
    axes[2].set(xscale='log',xlabel='Qualification loss threshold',ylabel='Mean qualifying topologies')
    axes[2].legend(frameon=False,fontsize=8,loc='upper left')
    for ax,title in zip(axes,('A  Which source supplies quality?','B  Exclusive and shared contributions','C  Contribution versus quality cutoff')):
        ax.set_title(title,loc='left',fontsize=11)
    save(fig,'01_source_contributions','Twenty matched seeds. Search is the existing no-LLM AdaGrad/RL hybrid, including mutation/random proposals; LLM is the direct supplied-rate insertion-only bank without RL/HOF conditioning. Shared structures are counted once. Bands show sample SD, not confidence intervals. This is terminal aggregation of separately obtained sources, not evidence of interaction or equal consumed budgets.')

    fig,axes=plt.subplots(1,3,figsize=(14,5.5),layout='constrained')
    x=np.arange(20)
    bottom=np.zeros(20)
    for key,color,label in [('search_only',COLORS[0],'Search-only families'),('shared',COLORS[2],'Shared families'),('llm_only',COLORS[1],'LLM-only families')]:
        v=np.array([r['families_'+key] for r in rows]); axes[0].bar(x+1,v,bottom=bottom,color=color,label=label,width=.85);bottom+=v
    axes[0].set(xlabel='Matched seed',ylabel='Structural family count',xticks=[1,5,10,15,20])
    axes[0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2)
    jitter=np.random.default_rng(31).uniform(-.12,.12,20)
    for i,a in enumerate(('search','llm','union')):
        rr=[r for r in sources if r['source']==a]
        v=[r['aif_like_fraction']*100 for r in rr]
        axes[1].scatter(i+jitter,v,color=COLORS[i],s=20,alpha=.8)
        axes[1].hlines(np.mean(v),i-.2,i+.2,color=COLORS[i],lw=3)
    axes[1].set(xticks=[0,1,2],xticklabels=['AdaGrad/RL','Direct LLM','Union'],ylabel='AIF-like qualifying designs (%)',ylim=(-4,104))
    vals=[np.mean([r[k] for r in first]) for k in ('rl_direct','rl_fit','other_search','other_fit')]
    axes[2].barh(range(4),vals,color=['#168577','#68b3a1','#617889','#a5b2bc'])
    axes[2].set_yticks(range(4),['RL, initial rates','RL + AdaGrad','Mutation/random','Mutation/random\n+ AdaGrad'])
    axes[2].invert_yaxis(); axes[2].set_xlabel('Mean first qualifying topologies')
    axes[2].set_xlim(0,max(vals)*1.2)
    for i,v in enumerate(vals): axes[2].text(v+1,i,f'{v:.2f}',va='center',fontsize=9)
    for ax,title in zip(axes,('A  Do both sources add families?','B  Different motif preferences','C  Inside the search component')):
        ax.set_title(title,loc='left',fontsize=11)
    save(fig,'02_families_and_search_provenance','Families reuse the frozen pooled atlas (average-linkage Jaccard cut 0.60); they are not reclustered by source. B: means are bars; dots are seeds. AIF-like denotes a structural motif with a restricted exact integral balance, not proven stability. C assigns each search topology to its first qualifying event; RL + AdaGrad means a policy-emitted topology first qualified after fitting. It does not prove independent invention.')
    pdf.close()


def report(output, summary):
    p=summary['mean_partition']; s=summary['source_summary']
    lines=['# Independent Union: Search and LLM Contributions', '',
        'Twenty matched seeds, frozen qualification threshold 0.016491168958935352. All source sets reconstruct the audited terminal union. No new evaluations or model calls.', '',
        '| Source | Mean qualifying topologies | Mean families | Expected families in 20 designs | AIF-like fraction | Median best loss |',
        '|---|---:|---:|---:|---:|---:|']
    for a,label in zip(('search','llm','union'),LABELS):
        r=s[a]
        lines.append(f"| {label} | {r['topologies']:.2f} | {r['families']:.2f} | {r['rarefied_families_k20']:.2f} | {100*r['aif_like_fraction']:.2f}% | {summary['median_best_loss'][a]:.6g} |")
    lines += ['', '## Complementarity', '',
        f"Per seed, {p['topologies_search_only']:.2f} qualifying topologies are search-only, {p['topologies_llm_only']:.2f} LLM-only, and {p['topologies_shared']:.2f} shared. Removing LLM therefore removes {p['topologies_llm_only']:.2f} qualifying designs; removing search removes {p['topologies_search_only']:.2f}. Source-exclusive means absent from the other source's qualifying set, not proof that it never proposed the structure.", '',
        f"Search contributes {p['families_search_only']:.2f} families absent from the LLM's qualifying set; LLM contributes {p['families_llm_only']:.2f} absent from search. Another {p['families_shared']:.2f} families are shared. Family counts use the previous global clustering without refitting; they do not measure distinct verified biological mechanisms.", '',
        f"Endpoint winners: {summary['best_source_counts']}. Median paired LLM/search loss ratio: {summary['median_paired_llm_search_loss_ratio']:.4f}. Among topologies qualifying in both sets, LLM supplies better rates in {p['overlap_llm_better']:.2f} cases per seed, search in {p['overlap_search_better']:.2f}, and {p['overlap_tie']:.2f} tie.", '',
        '## What Does RL Mean Here?', '',
        'The search source is the no-LLM AdaGrad/RL hybrid, including mutation/random proposals and parameter fitting. Event-time records, not mutable terminal source labels, separate its first qualifications:', '',
        '| First qualifying event | Mean topologies |', '|---|---:|']
    for k,v in summary['mean_search_first_qualifications'].items(): lines.append(f'| {k} | {v:.2f} |')
    lines += ['', 'These categories partition the search-qualified set. They attribute first qualifying emission/fitting, not ultimate policy ancestry or a causal necessity. A later source can rediscover or improve the same structure.', '',
        '## Limits', '',
        'This union reuses the no-LLM run and an insertion-only LLM bank. Saved request contexts were re-audited: RL/HOF, SIL and exclusion state are withheld, while LLM self-feedback is allowed. The bank is restricted to direct supplied-rate proposals; no insertion-only search descendants or AdaGrad-refined LLM candidates enter it.', '',
        'This is terminal complementarity, not synergy: the two components do not interact in the reconstructed union. The bank was produced on a different completion schedule with additional model/numerical costs. It is not an equal-cost standalone-LLM comparison or a newly executed no-communication campaign. Source counts overlap; their sum is not the union. Global uniqueness and within-seed uniqueness are different quantities.', '',
        'AIF-like screens require X1/X2 co-consumption, X3-driven X2 production and a restricted exact integral balance with negative alpha and positive gain. They do not prove stability or parameter robustness. Failed/incomplete evaluations remain nonqualifying. No new significance tests are claimed.', '',
        'Reproduce with a fresh output directory:', '', '```bash',
        f'.venv/bin/python -m comparisons.search_exploration.analyze_rpa_independent_sources --output {output}', '```']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)
