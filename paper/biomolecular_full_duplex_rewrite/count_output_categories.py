"""Compare topology and descriptor counts in frozen qualifying portfolios.

Read-only analysis of original five-reaction designs; no simulations.
Run from the repository root with .venv/bin/python.
"""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from RL4CRN.search_exploration.archive import descriptor

SOURCES = {
    'RPA': ROOT / 'comparisons/search_exploration/reports/rpa-communication-portfolio-20260925/diagnostics/selection.json.gz',
    'Logic': Path('/local1/rossin/search-exploration/logic5-portfolio-diagnostics-order-fallback-20260924/selection.json.gz'),
}
ARMS = ['control', 'none', 'independent-union', 'llm-to-rl', 'rl-to-llm', 'full']
LABELS = {'control': 'No LLM', 'none': 'Independent',
          'independent-union': 'Independent*', 'llm-to-rl': 'Insertion only',
          'rl-to-llm': 'Return only', 'full': 'Full duplex'}
TOTALS = {'RPA': 1631, 'Logic': 3444}


class Reaction:
    def __init__(self, structure):
        self.reactant_labels = structure['reactants']
        self.product_labels = structure['products']

    def get_stoichiometry_dict(self):
        a, b = Counter(self.reactant_labels), Counter(self.product_labels)
        return {s: b[s] - a[s] for s in a.keys() | b.keys()}


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = HERE / 'category_coverage'
    output.mkdir(exist_ok=True)
    runs, members, groups, hashes = [], [], {}, {}
    for task, path in SOURCES.items():
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, 'rt') as stream:
            data = json.load(stream)
        library = {int(k): Reaction(v['structure']) for k, v in data['catalog'].items()}
        for run in data['runs']:
            arm, seed = run['arm'], run['seed']
            topologies, categories = set(), set()
            pooled_topologies, pooled_categories = groups.setdefault((task, arm), (set(), set()))
            for member in run['members']:
                candidate = data['candidates'][member['candidate']]
                ids = candidate['reaction_ids']
                topology = tuple(sorted(ids))
                assert len(ids) == len(set(ids)) == 5
                assert topology not in topologies
                category = descriptor(ids, library, ['X_3' if task == 'RPA' else 'OUT'],
                                      'reaction_roles_v1', '')
                topologies.add(topology)
                categories.add(category)
                members.append(dict(task=task, arm=arm, seed=seed,
                                    reaction_ids=';'.join(map(str, topology)), category=category))
            pooled_topologies.update(topologies)
            pooled_categories.update(categories)
            runs.append(dict(task=task, arm=arm, seed=seed, topologies=len(topologies),
                             categories=len(categories), topologies_per_category=len(topologies)/len(categories)))
    summaries = []
    for task in SOURCES:
        for arm in ARMS:
            selected = [r for r in runs if r['task'] == task and r['arm'] == arm]
            if not selected:
                continue
            assert sorted(r['seed'] for r in selected) == list(range(1, 21))
            row = dict(task=task, arm=arm, label=LABELS[arm], runs=len(selected))
            for metric in ('topologies', 'categories'):
                values = np.array([r[metric] for r in selected])
                row.update({metric+'_mean': float(values.mean()), metric+'_sd': float(values.std(ddof=1)),
                            metric+'_min': int(values.min()), metric+'_max': int(values.max())})
            topologies, categories = groups[task, arm]
            row.update(pooled_topologies=len(topologies), pooled_categories=len(categories),
                       pooled_category_coverage_percent=100*len(categories)/TOTALS[task])
            summaries.append(row)
    write_csv(output / 'per_run.csv', runs)
    write_csv(output / 'summary.csv', summaries)
    write_csv(output / 'memberships.csv', members)
    (output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    notes = '''# Categories represented in final qualifying portfolios

Original five-reaction designs, at their saved selected rates. Selection is the
frozen best-per-topology, loss-qualified portfolio used by the existing RPA and
Logic diagnostics. It includes numerical and direct-LLM outputs according to
each arm's final union. This is cumulative saved output, not just the bounded
live archive or top-30 Hall of Fame. No pruning, perturbation-success filter or
additional Logic truth-table correctness filter is applied here.

Categories are recomputed with the production six-feature descriptor from
the frozen reaction catalogs. Topology identity is the sorted set of reaction
IDs with species labels preserved. Rates do not distinguish either count.
Per-run means and sample SD use 20 runs; pooled counts are set unions across
those runs, never sums. All counts are descriptive.

RPA Independent* is the reconstructed union of a historical no-LLM search and
the reused insertion-only direct-LLM bank. It is not a fresh independent arm.
Logic Independent is an executed no-communication arm. RPA No LLM is shown
separately and shares its numerical output with the reconstructed union.

Coverage denominators are the exhaustive structural counts previously computed:
1,631 categories for RPA and 3,444 for Logic, with exactly five distinct eligible
order-two reactions. These include categories incapable of satisfying the task.

The figure shows per-run means, sample SD and all individual runs.
Source SHA-256 hashes, per-run counts and every topology/category membership
are saved alongside the summary. No simulations or source changes were made.
'''
    (output / 'README.md').write_text(notes)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'control': '#808080', 'none': '#a98ac5', 'independent-union': '#a98ac5',
              'llm-to-rl': '#4b8bb6', 'rl-to-llm': '#d39a42', 'full': '#168577'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for i, task in enumerate(SOURCES):
        selected = [r for r in summaries if r['task'] == task]
        for j, metric in enumerate(('topologies', 'categories')):
            ax = axes[i, j]
            for k, row in enumerate(selected):
                color = colors[row['arm']]
                values = sorted((r['seed'], r[metric]) for r in runs
                                if r['task'] == task and r['arm'] == row['arm'])
                ax.bar(k, row[metric+'_mean'], color=color, alpha=.55,
                       yerr=row[metric+'_sd'], capsize=4, ecolor='#444444')
                ax.scatter(k+np.linspace(-.16, .16, len(values)), [v for _, v in values],
                           s=13, color=color, edgecolor='white', linewidth=.3, zorder=3)
            ax.set_xticks(range(len(selected)), [r['label'].replace(' ', '\n') for r in selected])
            ax.set_title(f'{task}: qualifying {metric}', loc='left', weight='bold')
            ax.set_ylabel('Count per run')
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
    fig.suptitle('Structural variety in the final qualifying output', fontsize=16, weight='bold')
    fig.tight_layout(rect=(0, .10, 1, .96), h_pad=2)
    fig.text(.06, .065, 'Bars: mean ± sample SD; dots: 20 runs. Original five-reaction designs; no pruning or robustness filter.', fontsize=10)
    fig.text(.06, .032, '*RPA independent output is a reconstructed union; Logic independent output comes from an executed arm.', fontsize=10)
    fig.savefig(output / 'topologies_and_categories.pdf')
    fig.savefig(output / 'topologies_and_categories.png', dpi=180)
    for row in summaries:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
