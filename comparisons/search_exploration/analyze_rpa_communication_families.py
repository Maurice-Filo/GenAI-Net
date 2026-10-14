"""Completed RPA communication portfolios: quality, structural families and motifs.

Read-only, post-hoc analysis. No model calls, evaluations or source DB changes.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from comparisons.search_exploration.analyze_rpa_flash_adagrad import FULL, CONTROL
from comparisons.search_exploration.analyze_rpa_llm_provenance import RECOVERY, connect, read
from comparisons.search_exploration.analyze_rpa_mechanisms import (
    dump_json, equation, extra_structure, integral_balance, rarefied_richness,
    reaction_catalog, reactions_for, set_jaccard, stoichiometry, write_csv,
)

ARMS = ('control', 'full', 'llm-to-rl', 'rl-to-llm', 'independent-union')
LABELS = ('AdaGrad/RL', 'Full duplex', 'Insertion only', 'Return only', 'Independent union')
SHORT = ('AdaGrad/RL', 'Full\nduplex', 'Insertion\nonly', 'Return\nonly', 'Independent\nunion*')
COLORS = ('#168577', '#a64e85', '#bc7838', '#376ab3', '#6b6975')
THRESHOLD = .016491168958935352
CUTS = (.3, .45, .6, .75)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_records(*banks, ceiling=.1):
    """One best complete canonical rate vector per topology, irrespective of source."""
    by_hash = {}
    for bank in banks:
        for record in bank:
            loss = record.get('loss')
            if not record.get('simulation_complete') or loss is None:
                continue
            if not np.isfinite(loss) or not 0 <= loss < ceiling:
                continue
            old = by_hash.get(record['topology_hash'])
            if old is None or (loss, record['candidate_hash']) < (old['loss'], old['candidate_hash']):
                by_hash[record['topology_hash']] = record
    return sorted(by_hash.values(), key=lambda r: (r['loss'], r['candidate_hash']))


def search_records(folder):
    with connect(folder) as db:
        db.execute('PRAGMA temp_store=MEMORY')
        rows = db.execute("""SELECT payload FROM (
            SELECT payload, row_number() OVER (
                PARTITION BY json_extract(payload,'$.topology_hash')
                ORDER BY json_extract(payload,'$.loss'),key) AS rank
            FROM designs WHERE json_extract(payload,'$.simulation_complete')=1
            AND json_type(payload,'$.loss') IN ('integer','real')
            AND json_extract(payload,'$.loss') >= 0
            AND json_extract(payload,'$.loss') < 0.1) WHERE rank=1""")
        return union_records([json.loads(row[0]) for row in rows])


def extract(output):
    # Motif/balance screens also run offline without importing the RL trainer.
    from comparisons.search_exploration.analyze_rpa_directional_streams import direct_records
    recovered = read(RECOVERY/'analysis/summary.json')
    baseline = read(FULL/'analysis-vs-adagrad-20260917/summary.json')
    if recovered['audit_errors'] or recovered['matched_seeds'] != list(range(1, 21)):
        raise ValueError('Requires all twenty reconciled, audited directional seeds')
    refs = {('full' if r['arm']=='flash' else r['arm'], r['seed']): (r['qualified_discovered'], r['loss'])
            for r in baseline['endpoints']}
    refs.update({(r['arm'], r['seed']): (r['union_qualified'], r['union_best_loss'])
                 for r in recovered['endpoints']})
    refs.update({('independent-union', r['seed']): (r['union_qualified'], r['union_best_loss'])
                 for r in recovered['independent_unions']})
    control_plan = read(CONTROL/'active_plan.json')
    control = {j['seed']: Path(j.get('output', CONTROL/'runs'/j['name']))
               for j in control_plan['jobs'] if j['arm'] == 'julia'}
    full = {j['seed']: Path(j.get('output', FULL/'runs'/j['name'])) for j in read(FULL/'plan.json')['jobs']}
    directional = {(j['arm'], j['seed']): Path(j['output']) for j in read(RECOVERY/'plan.json')['jobs']}
    inputs = [CONTROL/'active_plan.json', FULL/'plan.json', RECOVERY/'plan.json',
              FULL/'analysis-vs-adagrad-20260917/summary.json', RECOVERY/'analysis/summary.json']
    runs, sources = [], {}
    template = read(control[1]/'manifest.json')['task_config']['rpa']
    for seed in range(1, 21):
        folders = {'control': control[seed], 'full': full[seed],
                   **{a: directional[a, seed] for a in ARMS[2:4]}}
        banks, direct = {}, {}
        for arm, folder in folders.items():
            manifest = read(folder/'manifest.json')
            cfg = manifest['config']
            if (manifest['task_config']['rpa'] != template or cfg['seed'] != seed or
                    cfg['quality_threshold'] != THRESHOLD or cfg['rate_min'] != .001 or
                    cfg['rate_max'] != 100 or cfg['epochs'] != 100 or cfg['batch_size'] != 1023):
                raise ValueError(f'Incompatible task/configuration: {folder}')
            banks[arm] = search_records(folder)
            direct[arm] = direct_records(folder) if arm in ARMS[2:4] else []
            files = [folder/'manifest.json', folder/'completed.json']
            if arm in ARMS[2:4]:
                files.append(folder/'direct_llm_proposals.jsonl')
            sources[str(folder)] = {str(p): digest(p) for p in files}
        for arm in ARMS:
            records = (union_records(banks['control'], direct['llm-to-rl']) if arm == 'independent-union'
                       else union_records(banks[arm], direct[arm]))
            qualified = [r for r in records if r['loss'] < THRESHOLD]
            count, loss = refs[arm, seed]
            if len(qualified) != count or not np.isclose(records[0]['loss'], loss, rtol=1e-12, atol=0):
                raise ValueError(f'Endpoint mismatch: {arm}/{seed}')
            for r in qualified:
                ids, rates = r['reaction_ids'], r['rates']
                if len(ids) != 5 or len(set(ids)) != 5 or set(ids) & {0, 1, 28}:
                    raise ValueError('Invalid reaction contract')
                if len(rates) != 5 or not all(.001 <= k <= 100 for k in rates):
                    raise ValueError('Invalid rate contract')
            runs.append({'arm': arm, 'seed': seed, 'folder': str(folders.get(arm, control[seed])),
                         'losses': [r['loss'] for r in records], 'qualified': qualified, 'best': records[0]})
        print(f'Extracted seed {seed}/20; all five endpoints match audited summaries', flush=True)
    runs.sort(key=lambda r: (ARMS.index(r['arm']), r['seed']))
    with gzip.open(output/'extracted.json.gz', 'wt') as stream:
        json.dump(runs, stream)
    dump_json(output/'inputs.json', {'files': {str(p): digest(p) for p in inputs}, 'runs': sources,
        'script_sha256': digest(Path(__file__)), 'database_mode': 'read-only immutable; completed and checkpointed'})
    return runs


def antithetic_motif(reactions):
    annihilation = any(sorted(r['structure']['reactants']) == ['X_1', 'X_2']
                       and np.all(stoichiometry(r)[:2] < 0) for r in reactions)
    sensor = any(r['structure']['reactants'] == ['X_3'] and stoichiometry(r)[1] > 0 for r in reactions)
    return bool(annihilation and sensor)


def aif_like(reactions, balance):
    return bool(antithetic_motif(reactions) and balance['alpha'] < 0 and balance['restricted_exact_balance'])


def cluster(runs, catalog, fixed, output):
    structures, representative = {}, {}
    for run in runs:
        for r in run['qualified']:
            h = r['topology_hash']
            signature = extra_structure(reactions_for(r, catalog, fixed), fixed)
            if h in structures and signature != structures[h]:
                raise ValueError('Conflicting structures for one topology hash')
            structures[h] = signature
            if h not in representative or r['loss'] < representative[h]['loss']:
                representative[h] = r
    hashes = sorted(structures)
    if len(set(structures.values())) != len(hashes):
        raise ValueError('Distinct topology hashes collapse to the same labelled structure')
    tokens = sorted(set(t for s in structures.values() for t in s))
    token_index = {t: i for i, t in enumerate(tokens)}
    features = np.zeros((len(hashes), len(tokens)), bool)
    for i, h in enumerate(hashes):
        features[i, [token_index[t] for t in structures[h]]] = True
    print(f'Clustering {len(hashes)} unique qualifying topologies', flush=True)
    condensed = pdist(features, metric='jaccard')
    tree = linkage(condensed, method='average')
    distances = squareform(condensed).astype(np.float32)
    index = {h: i for i, h in enumerate(hashes)}
    raw = fcluster(tree, t=.6, criterion='distance')
    counts = Counter(raw)
    renumber = {old: i+1 for i, old in enumerate(sorted(counts, key=lambda k: (-counts[k], k)))}
    labels = np.array([renumber[v] for v in raw])
    # Ten was used in the earlier atlas; fifty adds resolution at a common sample size.
    ks = (10, 50)
    rows, sensitivity, membership = [], [], []
    for run in runs:
        ix = np.array([index[r['topology_hash']] for r in run['qualified']], int)
        run['indices'] = ix.tolist()
        labs = labels[ix]
        frequencies = np.array(list(Counter(labs).values()), float) / len(labs)
        sub = distances[np.ix_(ix, ix)]
        balances, motifs, structural_motifs = [], [], []
        for r in run['qualified']:
            rr = reactions_for(r, catalog, fixed)
            balance = integral_balance(rr)
            balances.append(balance)
            motifs.append(aif_like(rr, balance))
            structural_motifs.append(antithetic_motif(rr))
        rows.append({'arm': run['arm'], 'seed': run['seed'], 'best_loss': run['best']['loss'],
            'qualified_topologies': len(ix), 'families': len(frequencies),
            'effective_families': float(np.exp(-np.sum(frequencies*np.log(frequencies)))),
            'mean_pairwise_distance': float(sub[np.triu_indices(len(ix), 1)].mean()),
            **{f'rarefied_k{k}': rarefied_richness(labs, k) for k in ks},
            'exact_balance_fraction': float(np.mean([b['restricted_exact_balance'] for b in balances])),
            'near_unit_balance_fraction': float(np.mean([b['restricted_near_unit_balance'] for b in balances])),
            'antithetic_motif_fraction': float(np.mean(structural_motifs)),
            'aif_like_fraction': float(np.mean(motifs))})
        for r, lab, balance, motif, structural in zip(run['qualified'], labs, balances, motifs, structural_motifs):
            membership.append({'arm': run['arm'], 'seed': run['seed'], 'topology_hash': r['topology_hash'],
                'family': int(lab), 'loss': r['loss'], 'aif_like_exact': motif,
                'antithetic_structural_motif': structural, **balance})
        for cut in CUTS:
            lab = fcluster(tree, t=cut, criterion='distance')[ix]
            sensitivity.append({'arm': run['arm'], 'seed': run['seed'], 'cut': cut,
                'families': len(set(lab)), **{f'rarefied_k{k}': rarefied_richness(lab, k) for k in ks}})
    exact = np.zeros((len(runs), len(runs)))
    nearest = exact.copy()
    for i, a in enumerate(runs):
        for j in range(i, len(runs)):
            ai, bi = a['indices'], runs[j]['indices']
            exact[i,j] = exact[j,i] = set_jaccard(set(ai), set(bi))
            d = distances[np.ix_(ai, bi)]
            nearest[i,j] = nearest[j,i] = 1 - .5*(d.min(0).mean()+d.min(1).mean())
    families = []
    for lab in sorted(set(labels)):
        ix = np.flatnonzero(labels == lab)
        medoid = int(ix[np.argmin(distances[np.ix_(ix, ix)].mean(1))])
        record = representative[hashes[medoid]]
        rr = reactions_for(record, catalog, fixed)[len(fixed):]
        families.append({'family': int(lab), 'unique_topologies': len(ix),
            'medoid_hash': hashes[medoid], 'medoid_record': record,
            'medoid_reactions': [equation(r) for r in rr],
            'mean_within_distance': float(distances[np.ix_(ix, ix)][np.triu_indices(len(ix), 1)].mean()) if len(ix)>1 else 0.,
            'seed_coverage': {a: sum(bool(set(run['indices']) & set(ix)) for run in runs if run['arm']==a) for a in ARMS}})
    write_csv(output/'per_run.csv', rows)
    write_csv(output/'cluster_cut_sensitivity.csv', sensitivity)
    write_csv(output/'memberships.csv', membership)
    dump_json(output/'families.json', families)
    dump_json(output/'topology_structures.json', [{'topology_hash': h, 'family': int(l),
        'added_structure': structures[h]} for h, l in zip(hashes, labels)])
    np.savez_compressed(output/'clustering.npz', linkage=tree, labels=labels, topology_hashes=hashes,
                        exact_overlap=exact, nearest_similarity=nearest)
    return rows, labels, exact, nearest, sensitivity, families


def contrasts(rows):
    """Paired-seed bootstrap intervals, explicitly exploratory rather than p-values."""
    ix = {(r['arm'], r['seed']): r for r in rows}
    draws = np.random.default_rng(20260918).integers(0, 20, (10000, 20))
    results = []
    for a, b in [('full', 'control'), ('full', 'llm-to-rl'), ('rl-to-llm', 'control'),
                 ('full', 'independent-union')]:
        for metric in ('qualified_topologies', 'best_loss', 'families', 'rarefied_k50', 'mean_pairwise_distance'):
            x = np.array([ix[a, s][metric] for s in range(1, 21)])
            y = np.array([ix[b, s][metric] for s in range(1, 21)])
            if not np.isfinite([x, y]).all():
                raise ValueError('Common-size rarefaction requires all matched seeds')
            d = x-y
            ties = np.isclose(x, y, rtol=1e-9, atol=1e-12)
            wins = (d < 0) if metric == 'best_loss' else (d > 0)
            ci = np.quantile(d[draws].mean(1), [.025, .975])
            results.append({'arm': a, 'reference': b, 'metric': metric,
                'mean_difference': float(d.mean()), 'ci_low': float(ci[0]), 'ci_high': float(ci[1]),
                'wins': int((wins & ~ties).sum()), 'ties': int(ties.sum()),
                'losses': int((~wins & ~ties).sum()), 'n': 20})
    return results


def render(output, runs, rows, labels, exact, nearest, sensitivity, families):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
    rng = np.random.default_rng(19)
    jitter = rng.uniform(-.13, .13, 20)
    pdf = PdfPages(output/'RPA_COMMUNICATION_TOPOLOGY_ATLAS.pdf')
    catalog, fixed = reaction_catalog(CONTROL)

    def save(fig, name, caption):
        fig.get_layout_engine().set(rect=(0, .17, 1, .78))
        fig.text(.025, .022, textwrap.fill(caption, 155), fontsize=8, va='bottom')
        fig.savefig(output/(name+'.png'))
        fig.savefig(output/(name+'.pdf'))
        pdf.savefig(fig)
        plt.close(fig)

    def dots(ax, key, ylabel, log=False):
        for i, arm in enumerate(ARMS):
            values = np.array([r[key] for r in rows if r['arm'] == arm])
            parts = ax.violinplot(values, positions=[i], widths=.72, showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(COLORS[i]); body.set_alpha(.12)
            ax.scatter(i+jitter, values, color=COLORS[i], s=17, alpha=.75)
            ax.hlines(np.median(values), i-.2, i+.2, colors=COLORS[i], lw=2.8)
        ax.set_xticks(range(5), SHORT, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.13)
        if log:
            ax.set_yscale('log')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout='constrained')
    dots(axes[0], 'best_loss', 'Best canonical loss', True)
    dots(axes[1], 'qualified_topologies', 'Qualifying topologies')
    thresholds = np.unique(np.r_[np.geomspace(.0002, .08, 160), THRESHOLD])
    curves = []
    for i, arm in enumerate(ARMS):
        values = np.array([np.searchsorted(r['losses'], thresholds, side='left') for r in runs if r['arm']==arm])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        axes[2].plot(thresholds, mean, color=COLORS[i], label=LABELS[i], lw=1.8)
        axes[2].fill_between(thresholds, np.maximum(0, mean-sd), mean+sd, color=COLORS[i], alpha=.09)
        curves.extend({'arm': arm, 'threshold': t, 'mean_count': m, 'sd': s} for t, m, s in zip(thresholds, mean, sd))
    axes[2].axvline(THRESHOLD, ls='--', color='.3', lw=1)
    axes[2].set(xscale='log', xlabel='Qualification loss threshold', ylabel='Mean qualifying topologies')
    axes[2].legend(frameon=False, fontsize=8, loc='upper left')
    for ax, title in zip(axes, ('A  Endpoint quality', 'B  Portfolio size', 'C  Quality / breadth trade-off')):
        ax.set_title(title, loc='left')
    save(fig, '01_quality', 'Twenty matched seeds per arm, with balance-interrupted attempts replaced. Points are seeds; bars are medians; bands are one sample SD, not significance. *Independent union reuses the no-LLM search and direct insertion-only proposals: it is post-hoc aggregation, not an executed no-communication run.')
    write_csv(output/'threshold_curves.csv', curves)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout='constrained')
    for ax, key, ylabel, title in zip(axes, ('mean_pairwise_distance', 'families', 'rarefied_k50'),
            ('Mean reaction-set Jaccard distance', 'Structural families per seed', 'Expected families in 50 topologies'),
            ('A  Within-run separation', 'B  Family coverage', 'C  Equal-size comparison')):
        dots(ax, key, ylabel)
        ax.set_title(title, loc='left')
    save(fig, '02_family_diversity', 'Average-linkage clustering at Jaccard distance 0.60, using complete added-reaction structures and excluding the common fixed template. Species roles are preserved; rates are ignored. C is exact rarefaction without replacement, controlling portfolio size, not search effort. Families are structural groups, not verified biological mechanisms.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
    for ax, matrix, title in zip(axes, (exact, nearest), ('A  Exact topology overlap', 'B  Closest-structure similarity')):
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
        ax.set_xticks(np.arange(5)*20+9.5, SHORT, fontsize=8)
        ax.set_yticks(np.arange(5)*20+9.5, SHORT, fontsize=8)
        for v in (19.5, 39.5, 59.5, 79.5):
            ax.axvline(v, color='white', lw=.7); ax.axhline(v, color='white', lw=.7)
        ax.set_title(title, loc='left')
        fig.colorbar(im, ax=ax, shrink=.7, label='Similarity')
    save(fig, '03_between_runs', 'One row and column per seed, ordered 1-20 within each arm. A compares entire sets of qualifying topologies. B averages nearest reaction-set similarities symmetrically between portfolios. Diagonal cells are self-comparisons. The union shares the control search and insertion-only proposal bank by construction. Cells are dependent, not statistical replicates.')

    top = [f['family'] for f in families[:12]]
    composition = np.zeros((5, len(top)+1))
    coverage = np.zeros((5, len(top)))
    for i, arm in enumerate(ARMS):
        for run in [r for r in runs if r['arm']==arm]:
            counts = Counter(labels[run['indices']])
            composition[i, :-1] += [counts[f]/len(run['indices'])/20 for f in top]
            composition[i, -1] += sum(n for f, n in counts.items() if f not in top)/len(run['indices'])/20
            coverage[i] += [int(counts[f]>0) for f in top]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    cmap = plt.get_cmap('tab20')
    left = np.zeros(5)
    for j, family in enumerate(top+['Other']):
        axes[0].barh(range(5), composition[:, j]*100, left=left, color=cmap(j) if family!='Other' else '#d5d5d5',
                     label=f'F{family}' if family!='Other' else 'Other', height=.65)
        left += composition[:, j]*100
    axes[0].set_yticks(range(5), LABELS); axes[0].invert_yaxis()
    axes[0].set(xlabel='Mean within-seed portfolio share (%)', xlim=(0, 100))
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17))
    im = axes[1].imshow(coverage, vmin=0, vmax=20, cmap='Blues', aspect='auto')
    axes[1].set_xticks(range(len(top)), [f'F{x}' for x in top], fontsize=8)
    axes[1].set_yticks(range(5), LABELS, fontsize=9)
    for i in range(5):
        for j in range(len(top)):
            axes[1].text(j, i, f'{coverage[i,j]:.0f}', ha='center', va='center', fontsize=8,
                         color='white' if coverage[i,j]>12 else '#222222')
    fig.colorbar(im, ax=axes[1], shrink=.7, label='Seeds containing family')
    axes[0].set_title('A  Family composition', loc='left'); axes[1].set_title('B  Recurrence across seeds', loc='left')
    save(fig, '04_family_composition', 'Family numbers are assigned by global unique-topology count in this atlas; they are not IDs from the earlier atlas. Each seed contributes equal weight to portfolio shares. The twelve largest families are shown separately; all remaining families are grouped as Other. Recurrence measures reproducibility of structural discovery, not functional robustness.')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), layout='constrained')
    dots(axes[0], 'exact_balance_fraction', 'Fraction with restricted exact balance')
    for i, arm in enumerate(ARMS):
        rr = [r for r in rows if r['arm']==arm]
        structural = [r['antithetic_motif_fraction'] for r in rr]
        exact_motif = [r['aif_like_fraction'] for r in rr]
        axes[1].plot(np.c_[i+jitter, i+jitter].T, np.array([exact_motif, structural]), color=COLORS[i], alpha=.15, lw=.7)
        axes[1].scatter(i+jitter, structural, facecolors='none', edgecolors=COLORS[i], s=25, marker='D',
                        label='Structural motif' if i==0 else None)
        axes[1].scatter(i+jitter, exact_motif, color=COLORS[i], s=17,
                        label='Motif + exact balance' if i==0 else None)
    axes[1].set_xticks(range(5), SHORT, fontsize=8)
    axes[1].set_ylabel('Fraction of qualifying topologies')
    for i, arm in enumerate(ARMS):
        mean = [np.mean([r['rarefied_k50'] for r in sensitivity if r['arm']==arm and r['cut']==c]) for c in CUTS]
        axes[2].plot(CUTS, mean, 'o-', color=COLORS[i], label=LABELS[i], ms=4)
    axes[2].set(xlabel='Average-linkage distance cutoff', ylabel='Mean expected families in 50 topologies', xticks=CUTS)
    axes[2].legend(frameon=False, fontsize=8)
    for ax, title in zip(axes, ('A  Balance signatures', 'B  Antithetic-like subset', 'C  Cluster-cut sensitivity')):
        ax.set_title(title, loc='left')
    save(fig, '05_mechanisms_and_sensitivity', 'The restricted test seeks d(X1 + alpha X2)/dt = u1 - gain X3, with positive gain and relative coefficient residual <= 1e-8. In B, diamonds require X1/X2 co-consumption and X3-driven X2 production; circles additionally require exact balance with alpha < 0. These screens do not prove stability or robustness. At cut 0.30 all distinct five-reaction sets remain separate (minimum distance 1/3).')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, family in zip(axes.flat, families[:6]):
        record = family['medoid_record']
        rr = reactions_for(record, catalog, fixed)[len(fixed):]
        lines = [f"F{family['family']}  |  {family['unique_topologies']} structures", '']
        lines += [f'{equation(r)}    [k={r["parameters"][0]:.3g}]' for r in rr]
        coverage = family['seed_coverage']
        lines += ['', f"Seeds: RL {coverage['control']}, Full {coverage['full']}, In {coverage['llm-to-rl']}",
                  f"       Out {coverage['rl-to-llm']}, Union {coverage['independent-union']}",
                  f'Medoid best observed loss: {record["loss"]:.4g}']
        ax.text(.02, .98, '\n'.join(lines), va='top', transform=ax.transAxes, fontsize=9, family='monospace', linespacing=1.5)
        ax.set_axis_off()
    save(fig, '06_representatives', 'Representative added reactions for the six largest families. Each representative minimizes mean Jaccard distance to its family, rather than being selected for lowest loss. Rates shown are that medoid topology\'s best observed rate vector across the cohorts. The two fixed input-modulated reactions are shared and omitted here. Empty denotes the empty complex.')
    pdf.close()


def analyze(output):
    output.mkdir(parents=True, exist_ok=False)
    runs = extract(output)
    catalog, fixed = reaction_catalog(CONTROL)
    rows, labels, exact, nearest, sensitivity, families = cluster(runs, catalog, fixed, output)
    paired = contrasts(rows)
    write_csv(output/'paired_exploratory_intervals.csv', paired)
    summary = []
    for arm in ARMS:
        rr = [r for r in rows if r['arm']==arm]
        row = {'arm': arm, 'n': len(rr), **{'mean_'+k: float(np.mean([r[k] for r in rr]))
               for k in rr[0] if k not in ('arm', 'seed')}}
        row['median_best_loss'] = float(np.median([r['best_loss'] for r in rr]))
        h = {r['topology_hash'] for run in runs if run['arm']==arm for r in run['qualified']}
        row['global_unique_topologies'] = len(h)
        ids = [i for i, run in enumerate(runs) if run['arm']==arm]
        row['between_seed_exact_jaccard'] = float(exact[np.ix_(ids, ids)][np.triu_indices(20, 1)].mean())
        summary.append(row)
    dump_json(output/'summary.json', {'post_hoc': True, 'new_simulations': 0, 'new_model_calls': 0,
        'threshold': THRESHOLD, 'cluster_cut': .6, 'families': len(families), 'unique_topologies': len(labels),
        'arms': summary, 'contrasts': paired, 'confidence_intervals': 'Unadjusted descriptive paired-seed bootstrap, 10000 draws; no new significance tests'})
    render(output, runs, rows, labels, exact, nearest, sensitivity, families)
    lines = ['# RPA Communication: Completed Portfolios and Structural Families', '',
        'All 20 seeds in each arm. Thirteen balance-interrupted directional attempts were replaced; only the reconciled plan is used. Every endpoint was rechecked against the audited summaries. No new simulations or model calls.', '',
        '| Arm | Median best loss | Mean qualifying topologies | Mean families | Families in 50 | Mean Jaccard distance |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['arm']} | {r['median_best_loss']:.6g} | {r['mean_qualified_topologies']:.2f} | {r['mean_families']:.2f} | {r['mean_rarefied_k50']:.2f} | {r['mean_mean_pairwise_distance']:.3f} |")
    lines += ['', '## Interpretation and Scope', '',
        f"The pooled set contains {len(labels)} distinct qualifying topologies in {len(families)} structural families at distance cut 0.60. Family coverage and rarefaction measure breadth of labelled reaction structures, not proven functional diversity.", '',
        'The no-LLM control is an AdaGrad/RL hybrid with mutation/random proposals, not pure RL. Directional endpoints include the union of their search discoveries and valid direct LLM proposals. In particular, return-only LLM candidates are not silently dropped merely because they never enter the search archive.', '',
        'Independent union is post-hoc aggregation of the existing no-LLM search with the insertion-only direct supplied-rate proposal bank. It excludes fitted descendants and HOF-conditioned proposals. It shares data with the other arms and is not an executed no-communication campaign or fixed-proposal causal replay. Actual numerical counts, worker allocations, model request counts and completion schedules differ; one historical control retained early finite timeout settings.', '',
        'Topology identity uses complete reaction records, including input channels and stoichiometric multiplicity. The exact two fixed-template structures are subtracted. Rates and reaction order do not define a new topology; species are not relabelled because their input/output roles are fixed. Distances use added-reaction Jaccard dissimilarity; average linkage uses each globally unique qualifying structure once. Families are conditional on this pooled observed set.', '',
        'Rarefaction reports the exact expected family count after sampling 50 qualifying topologies per seed without replacement. The earlier k=10 metric is retained in CSV. Cluster-cut sensitivity uses 0.30, 0.45, 0.60 and 0.75. Equal portfolio size does not equalize compute or model-call budget.', '',
        'Restricted integral-balance and AIF-like labels are algebraic motif screens at the best observed rates. They do not establish a unique dynamical mechanism, correct setpoint gain, stability or parameter robustness. Perturbation results remain in the separate existing robustness report; they were not rerun or silently extended here.', '',
        'Paired intervals in paired_exploratory_intervals.csv resample the 20 seeds, not individual topologies. They are post-hoc, unadjusted 95% bootstrap intervals, not multiple-testing-controlled significance claims. Shaded threshold bands show sample SD, not confidence intervals.', '',
        '## Reproduce', '', '```bash',
        f'.venv/bin/python -m comparisons.search_exploration.analyze_rpa_communication_families --output {output}',
        '```', '', 'Use a fresh output folder when regenerating; existing outputs are never overwritten. Inputs and hashes are recorded in inputs.json. The combined PDF has six pages; standalone PDF/PNG panels, per-run statistics, memberships, medoid representatives and clustering arrays are alongside it.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    analyze(parser.parse_args().output)
