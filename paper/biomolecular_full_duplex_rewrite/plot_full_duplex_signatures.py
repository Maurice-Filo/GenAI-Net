"""Diagram of the most frequent signatures and their internal topology diversity.

Uses saved qualifying portfolio memberships only; no simulations or refitting.
"""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'category_coverage'
SOURCE = OUTPUT / 'memberships.csv'
PORTFOLIOS = {
    'RPA': HERE.parents[1] / 'comparisons/search_exploration/reports/rpa-communication-portfolio-20260925/diagnostics/selection.json.gz',
    'Logic': Path('/local1/rossin/search-exploration/logic5-portfolio-diagnostics-order-fallback-20260924/selection.json.gz'),
}


def validate_thresholds(records):
    """Match every plotted membership to its saved, strictly qualifying loss."""
    audit = {}
    for task, path in PORTFOLIOS.items():
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        expected, losses, thresholds = set(), [], set()
        for run in data['runs']:
            if run['arm'] != 'full':
                continue
            folder = Path(run.get('folder', run.get('output')))
            manifest = json.loads((folder / 'manifest.json').read_text())
            threshold = manifest['config']['quality_threshold']
            thresholds.add(threshold)
            for member in run['members']:
                loss = member['loss']
                assert np.isfinite(loss) and 0 <= loss < threshold, (task, run['seed'], loss)
                ids = data['candidates'][member['candidate']]['reaction_ids']
                expected.add((int(run['seed']), tuple(sorted(ids))))
                losses.append(loss)
        plotted = [(int(r['seed']), tuple(sorted(map(int, r['reaction_ids'].split(';')))))
                   for r in records if r['task'] == task and r['arm'] == 'full']
        assert len(thresholds) == 1
        assert len(plotted) == len(set(plotted)) == len(losses)
        assert set(plotted) == expected
        audit[task] = dict(threshold=thresholds.pop(), checked_memberships=len(losses),
                           largest_included_loss=max(losses), all_pass=True,
                           source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUTPUT / 'full_duplex_histogram_threshold_audit.json').write_text(json.dumps(audit, indent=2)+'\n')


def summarize(top_k=10):
    with SOURCE.open() as stream:
        records = list(csv.DictReader(stream))
    validate_thresholds(records)
    selected = []
    for task in ('RPA', 'Logic'):
        rows = [r for r in records if r['task'] == task and r['arm'] == 'full']
        totals = Counter(int(r['seed']) for r in rows)
        assert sorted(totals) == list(range(1, 21))
        groups = defaultdict(list)
        for row in rows:
            groups[row['category']].append(row)
        ranked = []
        for category, members in groups.items():
            by_seed = Counter(int(r['seed']) for r in members)
            topologies = {tuple(map(int, r['reaction_ids'].split(';'))) for r in members}
            assert all(len(t) == len(set(t)) == 5 for t in topologies)
            ranked.append(dict(task=task, category=category,
                               pooled_unique_topologies=len(topologies),
                               mean_per_run_share=sum(by_seed[s]/totals[s] for s in totals)/20,
                               runs_represented=len(by_seed),
                               topology_run_memberships=len(members)))
        ranked.sort(key=lambda r: (-r['mean_per_run_share'], r['category']))
        for rank, row in enumerate(ranked[:top_k], 1):
            selected.append(dict(rank=rank, **row))
        other_categories = {r['category'] for r in ranked[top_k:]}
        other_members = [r for r in rows if r['category'] in other_categories]
        other_by_seed = Counter(int(r['seed']) for r in other_members)
        other_topologies = {tuple(map(int, r['reaction_ids'].split(';'))) for r in other_members}
        assert sum(r['pooled_unique_topologies'] for r in ranked) == len({r['reaction_ids'] for r in rows})
        selected.append(dict(rank=top_k+1, task=task, category='Others',
                             pooled_unique_topologies=len(other_topologies),
                             mean_per_run_share=sum(other_by_seed[s]/totals[s] for s in totals)/20,
                             runs_represented=len(other_by_seed), topology_run_memberships=len(other_members)))
    # Reconcile with the independently saved frequency analysis.
    prior = json.loads((OUTPUT / 'full_duplex_signature_frequencies.json').read_text())
    for row in selected:
        if row['category'] == 'Others':
            continue
        expected = prior['tasks'][row['task']]['signatures'][row['rank']-1]
        assert row['category'] == expected['category']
        assert row['pooled_unique_topologies'] == expected['pooled_unique_topologies']
        assert row['runs_represented'] == expected['runs_represented']
        assert np.isclose(row['mean_per_run_share'], expected['mean_per_run_fraction'])
    return selected


def render(rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    fig.text(.05, .96, 'Different topologies within recurring structural signatures',
             fontsize=20, weight='bold', color='#172b3a')
    fig.text(.05, .925, 'Full duplex · top three signatures in each task, ranked by mean share of qualifying topologies per run',
             fontsize=11, color='#52616b')
    colors = {'RPA': '#168577', 'Logic': '#945283'}
    labels = ['Output\nproducing', 'Output\nconsuming', 'Shared\nspecies',
              'Source\nreactions', 'Sink\nreactions', 'Species\ncount']
    for i, task in enumerate(('RPA', 'Logic')):
        ytop = .878 - i*.39
        color = colors[task]
        task_rows = [r for r in rows if r['task'] == task]
        share = sum(r['mean_per_run_share'] for r in task_rows)
        total = sum(r['pooled_unique_topologies'] for r in task_rows)
        fig.text(.05, ytop, task, fontsize=16, weight='bold', color=color)
        fig.text(.145, ytop+.001,
                 f'Top three: {100*share:.1f}% mean portfolio share  ·  {total} distinct topologies across 20 runs',
                 fontsize=11, color='#52616b')
        for j, row in enumerate(task_rows):
            ax = fig.add_axes([.05+j*.307, ytop-.35, .287, .327])
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.axis('off')
            box = FancyBboxPatch((.006, .006), .988, .988,
                                 boxstyle='round,pad=0,rounding_size=.025',
                                 edgecolor='#d8dfe4', facecolor='#fbfcfd', linewidth=1,
                                 transform=ax.transAxes)
            ax.add_patch(box)
            values = list(map(int, row['category'].split(':')))
            ax.text(.06, .915, f"#{row['rank']}   ({', '.join(map(str, values))})",
                    fontsize=13, weight='bold', color=color)
            for x, value, label in zip(np.linspace(.105, .895, 6), values, labels):
                ax.text(x, .805, str(value), ha='center', va='center',
                        fontsize=17, weight='bold', color='#172b3a')
                ax.text(x, .722, label, ha='center', va='center', fontsize=7.7,
                        color='#52616b', linespacing=1.2)
            ax.plot([.06, .94], [.655, .655], color='#d8dfe4', linewidth=.8)
            n = row['pooled_unique_topologies']
            ax.text(.06, .561, str(n), fontsize=27, weight='bold', color=color)
            ax.text(.29 if n >= 100 else .22, .572, 'distinct topologies', fontsize=12, color='#172b3a')
            # Identical dot spacing in all six panels; one dot per distinct topology.
            k = np.arange(n)
            ax.scatter(.072+(k % 20)*.045, .485-(k//20)*.038,
                       s=12, color=color, linewidths=0)
            ax.text(.06, .11, f"{100*row['mean_per_run_share']:.2f}% mean share per run",
                    fontsize=10, weight='bold', color='#172b3a')
            ax.text(.06, .05, f"Present in {row['runs_represented']}/20 runs", fontsize=10, color='#52616b')
    fig.text(.05, .095, 'Each dot is one distinct topology, deduplicated across all 20 runs; rate changes do not create a new topology.',
             fontsize=10, color='#364955')
    fig.text(.05, .070, '“Shared species” counts reactions with a species on both sides. Signature counts exclude fixed input reactions.',
             fontsize=10, color='#364955')
    fig.text(.05, .045, 'Original five-reaction, loss-qualified designs; no pruning, robustness or additional Logic correctness filter. Shares weight runs equally.',
             fontsize=9.5, color='#52616b')
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT / f'full_duplex_top_signatures.{extension}', dpi=200)
    plt.close(fig)


def reaction_frequencies():
    with SOURCE.open() as stream:
        records = list(csv.DictReader(stream))
    ranked = []
    for task, path in PORTFOLIOS.items():
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        topologies = {tuple(sorted(map(int, r['reaction_ids'].split(';'))))
                      for r in records if r['task'] == task and r['arm'] == 'full'}
        starter_ids = {r['reaction_id'] for r in data['fixed']}
        assert all(not (set(topology) & starter_ids) for topology in topologies)
        counts = Counter(reaction for topology in topologies for reaction in topology)
        assert sum(counts.values()) == 5*len(topologies)
        for rank, (rid, count) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])), 1):
            structure = data['catalog'][str(rid)]['structure']
            ranked.append(dict(task=task, rank=rank, reaction_id=rid,
                               reactants=' + '.join(structure['reactants']) or '∅',
                               products=' + '.join(structure['products']) or '∅',
                               qualifying_topologies_containing_reaction=count,
                               qualifying_topologies_total=len(topologies), fraction=count/len(topologies)))
    with (OUTPUT / 'full_duplex_reaction_frequencies.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    return [r for r in ranked if r['rank'] <= 3]


def reaction_math(row):
    def side(text):
        if text == '∅':
            return r'\varnothing'
        terms = []
        for species, count in Counter(text.split(' + ')).items():
            symbol = ('X_{'+species.split('_')[1]+'}') if species.startswith('X_') else r'\mathrm{'+species+'}'
            terms.append((str(count) if count > 1 else '')+symbol)
        return ' + '.join(terms)
    return '$'+side(row['reactants'])+r'\;\rightarrow\;'+side(row['products'])+'$'


def render_compact(rows, reactions):
    """Small companion bar chart; retain the detailed diagram separately."""
    fig = plt.figure(figsize=(11, 6.7))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.65, 1), left=.065, right=.985,
                            bottom=.17, top=.87, wspace=.13, hspace=.92)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    ymax = max(r['pooled_unique_topologies'] for r in rows)*1.16
    for ax, task, color in zip(axes, ('RPA', 'Logic'), ('#168577', '#945283')):
        selected = [r for r in rows if r['task'] == task]
        counts = [r['pooled_unique_topologies'] for r in selected]
        positions = range(len(selected))
        bars = ax.bar(positions, counts, width=.70,
                      color=[('#9aa2a9' if r['category'] == 'Others' else color) for r in selected])
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xticks(positions, [('Others' if r['category'] == 'Others' else '('+r['category'].replace(':', ',')+')') for r in selected],
                      fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title(task, fontsize=10, weight='bold', color=color)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', length=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Distinct topologies', fontsize=9)
    axes[1].tick_params(labelleft=False)
    for i, (task, color) in enumerate(zip(('RPA', 'Logic'), ('#168577', '#945283'))):
        ax = fig.add_subplot(grid[1, i])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.set_title('Most frequent added reactions', fontsize=10, weight='bold',
                     color=color, loc='left', pad=12)
        selected = [r for r in reactions if r['task'] == task]
        for y, row in zip((.80, .49, .18), selected):
            ax.text(.015, y, reaction_math(row), fontsize=12, va='center')
            ax.barh(y, .19, left=.49, height=.14, color='#edf0f2')
            ax.barh(y, .19*row['fraction'], left=.49, height=.14, color=color)
            label = f"{row['qualifying_topologies_containing_reaction']:,} / {row['qualifying_topologies_total']:,} ({100*row['fraction']:.1f}%)"
            ax.text(.985, y, label, fontsize=8.5, ha='right', va='center')
    fig.suptitle('Full duplex: qualifying topologies in the top 10 signatures and all others', fontsize=11, weight='bold', y=.98)
    fig.text(.065, .105, 'Topologies pooled and deduplicated over 20 runs · signatures ranked by mean frequency · Others combines remaining signatures', fontsize=8)
    fig.text(.065, .071, 'Reaction counts use ALL qualifying topologies, including Others; fixed input reactions are excluded. Bars show topology fractions.', fontsize=8)
    fig.text(.065, .037, 'All counted designs pass the saved-loss threshold: RPA < 0.01649117; Logic < 0.04271268 (rounded).', fontsize=8)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT / f'full_duplex_top_signatures_histogram.{extension}', dpi=220)
    plt.close(fig)


def main():
    all_rows = summarize()
    rows = [r for r in all_rows if r['rank'] <= 3]
    render(rows)
    render_compact(all_rows, reaction_frequencies())
    with (OUTPUT / 'full_duplex_top_signatures_histogram.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUTPUT / 'full_duplex_top_signatures.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      ranking='Mean of within-run qualifying-topology fractions; equal weight to 20 runs.',
                      diversity='Union of distinct sorted reaction-ID sets across 20 runs; species labels preserved.',
                      rows=rows)
    (OUTPUT / 'full_duplex_top_signatures.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (OUTPUT / 'full_duplex_top_signatures_caption.md').write_text(
        'The three most frequent structural signatures in full-duplex RPA and Logic output portfolios. '
        'Signatures are ranked by the mean within-run fraction of qualifying topologies across 20 runs. '
        'The six entries count output-producing reactions, output-consuming reactions, reactions with '
        'reactant–product species overlap, sources, sinks, and participating species, respectively. '
        'Counts concern only the five added reactions. Each dot represents one distinct labelled topology '
        'within that signature, pooled and deduplicated across runs. Rates do not distinguish topologies. '
        'The frequency and run-coverage annotations measure recurrence separately from this pooled diversity. '
        'Designs meet the original loss threshold; no pruning, perturbation-success or additional Logic '
        'truth-table correctness filter is applied. Frequency does not establish enrichment relative to '
        'the number of possible topologies in each category.\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
