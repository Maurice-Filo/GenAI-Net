"""RPA timeout sensitivity: frozen convergence and all-discovery topology counts.

Use --refresh to read completed databases; otherwise replot the saved counts.
No simulations, model requests, or writes to campaign outputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proposed_results_figures as figures

THRESHOLD = 0.016491168958935352
COUNTS = HERE / 'rpa_timeout_portfolios.json'
ORDER = ['hybrid', 'full', 'qwen', 'qwen_60min']
COLORS = [figures.BLUE, figures.PURPLE, figures.GREEN, figures.GREEN]


def load_runs():
    reference = figures.read(HERE.parent / 'genainet_comparison/audit.json')['runs']
    qwen = figures.read(HERE / 'qwen_preliminary_convergence_snapshot.json')['qwen_runs']
    followup = figures.read(HERE / 'qwen_rpa_timeout3600_snapshot.json')['runs']
    return [r for r in reference + qwen + followup if r['task'] == 'rpa']


def freeze_counts(runs):
    rows = []
    for arm in ORDER:
        group = sorted([r for r in runs if r['arm'] == arm], key=lambda r: r['seed'])
        assert [r['seed'] for r in group] == list(range(1, 21))
        for run in group:
            folder = Path(run['folder'])
            done = figures.read(folder / 'completed.json')
            assert done['status'] == 'complete' and done['epochs'] == 100
            manifest = figures.read(folder / 'manifest.json')
            assert manifest['config']['quality_threshold'] == THRESHOLD
            path = (folder / 'exploration.sqlite').resolve()
            wal = Path(str(path) + '-wal')
            assert not wal.exists() or wal.stat().st_size == 0
            before = path.stat()
            topologies, hashes, selected = set(), set(), []
            digest = hashlib.sha256()
            records = 0
            with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as db:
                for key, raw in db.execute('SELECT key,payload FROM designs ORDER BY key'):
                    records += 1
                    digest.update(key.encode() + b'\0' + raw.encode() + b'\n')
                    d = json.loads(raw)
                    if d['simulation_complete'] and d['loss'] < THRESHOLD:
                        ids = tuple(sorted(d['reaction_ids']))
                        assert len(ids) == len(set(ids))
                        topologies.add(ids)
                        hashes.add(d['topology_hash'])
                        selected.append(key)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert len(topologies) == len(hashes), 'Stored and canonical topology identities disagree'
            rows.append(dict(arm=arm, seed=run['seed'], folder=str(folder),
                             qualified_topologies=len(topologies), qualifying_parameterizations=len(selected),
                             design_rows=records, design_rows_sha256=digest.hexdigest(),
                             database_bytes=after.st_size, database_mtime_ns=after.st_mtime_ns,
                             final_charged_evaluations=done['counters']['expensive_evaluations']))
            print(arm, run['seed'], len(topologies), flush=True)
    counts = dict(threshold=THRESHOLD, rows=rows, sources=figures.SOURCES.copy(),
                  definition='Distinct sorted added-reaction sets with at least one complete canonical evaluation below threshold, over all recorded designs through terminal drain, independent of archive retention. Fixed starter reactions and species labels are unchanged.')
    COUNTS.write_text(json.dumps(counts, indent=2) + '\n')
    with (HERE / 'rpa_timeout_portfolios.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    figures.style()
    runs = load_runs()
    counts = freeze_counts(runs) if args.refresh else figures.read(COUNTS)
    styles = {'original': (figures.GRAY, '--', 'GenAI-Net'),
              'hybrid': (figures.BLUE, '-', 'No LLM'),
              'full': (figures.PURPLE, '-', 'Full duplex: Flash v4'),
              'qwen': (figures.GREEN, '-', 'Full duplex: Qwen (15 min)'),
              'qwen_60min': (figures.GREEN, '--', 'Full duplex: Qwen (60 min)')}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    fig.subplots_adjust(left=.08, right=.985, top=.72, bottom=.18, wspace=.32)
    fig.legend([Line2D([], [], color=c, ls=ls, lw=1.8) for c, ls, _ in styles.values()],
               [label for _, _, label in styles.values()], loc='upper center',
               ncol=3, fontsize=9, frameon=False, columnspacing=1.4)
    lo = max(r['cost'][0] for r in runs)
    hi = min(102400, min(r['cost'][-1] for r in runs))
    assert hi == 102400
    grid = np.unique([lo, hi] + [c for r in runs for c in r['cost'] if lo <= c <= hi])
    endpoints, terminal_endpoints = {}, {}
    for arm, (color, ls, _) in styles.items():
        group = [r for r in runs if r['arm'] == arm]
        assert len(group) == 20
        values = np.array([figures.sample(r, grid) for r in group])
        low, middle, high = np.quantile(values, [.25, .5, .75], axis=0)
        axes[0].step(grid/1000, middle, where='post', color=color, ls=ls, lw=1.7)
        axes[0].fill_between(grid/1000, low, high, step='post', color=color, alpha=.11, lw=0)
        endpoints[arm] = float(middle[-1])
        terminal_endpoints[arm] = figures.terminal_marker(
            axes[0], group, hi, list(styles).index(arm), len(styles), color,
            hollow=arm == 'qwen_60min')
        assert terminal_endpoints[arm]['median'] <= middle[-1]+1e-12
    figures.axes_style(axes[0], 'A  RPA convergence')
    axes[0].set(yscale='log',
                xlabel=r'Charged allowance ($\times 10^3$)', ylabel='Best loss')
    figures.allowance_axis_with_terminal(axes[0], hi)
    rng = np.random.default_rng(20261008)
    bootstrap = rng.integers(0, 20, (20000, 20))
    jitter = rng.uniform(-.14, .14, 20)
    summaries = {}
    for i, (arm, color) in enumerate(zip(ORDER, COLORS)):
        vals = np.array([r['qualified_topologies'] for r in counts['rows'] if r['arm'] == arm])
        assert len(vals) == 20
        mean = vals.mean()
        low, high = np.quantile(vals[bootstrap].mean(axis=1), [.025, .975])
        axes[1].bar(i, mean, width=.62, color=color, alpha=.45,
                    edgecolor=color, hatch='//' if arm=='qwen_60min' else None)
        axes[1].scatter(i+jitter, vals, s=12, color=color, alpha=.8, zorder=3)
        axes[1].errorbar(i, mean, yerr=[[mean-low],[high-mean]], color='#222222',
                        fmt='none', lw=1.2, capsize=3, zorder=4)
        summaries[arm] = dict(mean=float(mean), median=float(np.median(vals)),
                             mean_ci95=[float(low),float(high)])
    figures.axes_style(axes[1], 'B  Qualifying topologies')
    axes[1].set(xticks=range(4), xticklabels=['No LLM', 'Flash v4', 'Qwen\n15 min', 'Qwen\n60 min'],
                ylabel='Distinct qualifying topologies per run', ylim=(0,None))
    figures.save(fig, 'figureS_qwen_rpa_timeout', None)
    summary = dict(threshold=THRESHOLD, allowance=hi, loss_medians=endpoints,
                   terminal_endpoints=terminal_endpoints,
                   portfolios=summaries, sources=figures.SOURCES)
    (HERE / 'rpa_timeout_comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['portfolios'], indent=2))


if __name__ == '__main__':
    main()
