"""Read-only reconstruction of best-design loss and cumulative topology yield."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1] / 'comparisons/search_exploration/reports/category-negative-loss-pilot-20260925-v4-record-only'
OUT = HERE / 'speciation_ablation'
COLORS = {'baseline': '#168577', 'ablation': '#be6249'}
STYLES = {1: '-', 2: '--', 3: ':'}


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract():
    hashes, trajectories, endpoints = {}, [], []
    def read(path, lines=False):
        blob = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return [json.loads(line) for line in blob.splitlines()] if lines else json.loads(blob)
    campaign = read(CAMPAIGN / 'campaign.json')
    for job in campaign['jobs']:
        folder = (CAMPAIGN / 'runs' / job['name']).resolve()
        manifest = read(folder / 'manifest.json')
        metrics = read(folder / 'metrics.jsonl', lines=True)
        done = read(folder / 'completed.json') if (folder / 'completed.json').exists() else None
        threshold = manifest['config']['quality_threshold']
        assert manifest['config']['seed'] == job['seed']
        assert [m['epoch'] for m in metrics] == list(range(len(metrics)))
        if done:
            assert done['status'] == 'complete' and done['epochs'] == 100
            assert len(metrics) == 101 and done['pending_optimization_jobs'] == 0
        # Read a consistent transaction; original database and reports are unchanged.
        epoch_best = np.full(101, np.inf)
        first_qualified = {}
        event_digest = hashlib.sha256()
        db_path = folder / 'exploration.sqlite'
        with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
            db.execute('BEGIN')
            for eid, payload in db.execute("SELECT id,payload FROM events WHERE kind='pilot_candidate' ORDER BY id"):
                event_digest.update(str(eid).encode() + b'\n' + payload.encode() + b'\n')
                r = json.loads(payload)
                assert r['stream'] in ('search', 'optimization')
                if not r['simulation_complete']:
                    continue
                epoch, loss = r['epoch'], r['loss']
                assert 0 <= epoch <= 100 and np.isfinite(loss) and loss >= 0
                epoch_best[epoch] = min(epoch_best[epoch], loss)
                if loss < threshold:
                    first_qualified[r['topology_hash']] = min(epoch, first_qualified.get(r['topology_hash'], epoch))
        hashes[str(db_path) + '#pilot_candidate_events'] = event_digest.hexdigest()
        best = np.minimum.accumulate(epoch_best)
        curve = []
        for m in metrics:
            epoch = m['epoch']
            assert np.isfinite(best[epoch]) and best[epoch] > 0
            curve.append({'name': job['name'], 'task': job['task'], 'condition': job['condition'],
                'seed': job['seed'], 'epoch': min(epoch+1, 100), 'terminal_drain': epoch == 100,
                'numerical_evaluations': m['Compute/expensive_evaluations'],
                'best_pool_loss': float(best[epoch]),
                'qualifying_topologies': sum(e <= epoch for e in first_qualified.values())})
        assert all(b['best_pool_loss'] <= a['best_pool_loss'] for a, b in zip(curve, curve[1:]))
        assert all(b['qualifying_topologies'] >= a['qualifying_topologies'] for a, b in zip(curve, curve[1:]))
        if done:
            assert np.isclose(curve[-1]['best_pool_loss'], done['metrics']['Archive/best_loss'], rtol=1e-10, atol=1e-12)
            assert curve[-1]['numerical_evaluations'] == done['counters']['expensive_evaluations']
        trajectories += curve
        endpoints.append({**curve[-1], 'status': 'complete' if done else 'incomplete', 'threshold': threshold,
                          'folder': str(folder)})
        print(job['name'], endpoints[-1]['status'], curve[-1]['epoch'], curve[-1]['qualifying_topologies'], flush=True)
    assert len(endpoints) == 12
    payload = {'endpoints': endpoints, 'trajectories': trajectories, 'source_hashes': hashes,
        'pool': 'All complete committed search and canonically reevaluated fitting candidates; no internal optimizer trials.',
        'timing': 'Candidates enter at their recorded commit epoch. T is terminal draining, separated from epoch 100.'}
    (OUT / 'frozen_curves.json').write_text(json.dumps(payload, indent=2) + '\n')
    write_csv(OUT / 'trajectories.csv', trajectories)
    write_csv(OUT / 'endpoints.csv', endpoints)
    return payload


def plot(payload):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for row, task in enumerate(('rpa', 'logic')):
        for condition in ('baseline', 'ablation'):
            for seed in (1, 2, 3):
                run = next(e for e in payload['endpoints'] if (e['task'], e['condition'], e['seed']) == (task, condition, seed))
                points = [r for r in payload['trajectories'] if r['name'] == run['name'] and not r['terminal_drain']]
                terminal = [r for r in payload['trajectories'] if r['name'] == run['name'] and r['terminal_drain']]
                for col, key in enumerate(('best_pool_loss', 'qualifying_topologies')):
                    ax = axes[row, col]
                    ax.step([r['epoch'] for r in points], [r[key] for r in points], where='post',
                        color=COLORS[condition], ls=STYLES[seed], lw=1.9, alpha=.9)
                    if terminal:
                        assert len(terminal) == 1 and run['status'] == 'complete'
                        ax.plot(106, terminal[0][key], marker={1: 'o', 2: 's', 3: '^'}[seed],
                                ms=6, color=COLORS[condition], ls='none')
                    else:
                        ax.plot(points[-1]['epoch'], points[-1][key], 'o', ms=8, mfc='white',
                                mec=COLORS[condition], mew=1.8, zorder=5)
        threshold = next(e['threshold'] for e in payload['endpoints'] if e['task'] == task)
        axes[row, 0].axhline(threshold, ls=(0, (2, 3)), color='#777777', lw=1)
        axes[row, 0].set(yscale='log', ylabel='Best-so-far canonical loss')
        axes[row, 1].set(yscale='symlog', ylim=(0, 300), ylabel='Distinct qualifying topologies')
        axes[row, 1].set_yscale('symlog', linthresh=1)
        axes[row, 1].set_yticks([0, 1, 3, 10, 30, 100, 300])
        axes[row, 1].yaxis.set_major_formatter(ScalarFormatter())
        for col, title in enumerate(('Best-design convergence', 'Cumulative structural yield')):
            ax = axes[row, col]
            ax.set_title(f"{'ABCD'[2*row+col]}   {task.upper() if task == 'rpa' else 'Logic'} — {title}", loc='left', fontsize=12, pad=12)
            ax.set(xlim=(0, 111), xticks=[0, 25, 50, 75, 100, 106], xticklabels=['0', '25', '50', '75', '100', 'T'],
                xlabel='Completed search epochs; T = terminal drain')
            ax.axvline(103, ls=':', lw=.8, color='#aaaaaa')
            ax.grid(axis='y', which='major', alpha=.14)
    for e in payload['endpoints']:
        if e['status'] != 'complete':
            ax = axes[0 if e['task'] == 'rpa' else 1, 1]
            ax.annotate(f"Seed {e['seed']} {e['condition']} incomplete",
                xy=(e['epoch'], e['qualifying_topologies']), xytext=(10, 250),
                textcoords='data', fontsize=9, color=COLORS[e['condition']],
                arrowprops={'arrowstyle': '-', 'color': COLORS[e['condition']], 'lw': .8})
    handles = [Line2D([0], [0], color=COLORS['baseline'], lw=2.5, label='Current hybrid'),
        Line2D([0], [0], color=COLORS['ablation'], lw=2.5, label='Single category + raw-loss reward')]
    handles += [Line2D([0], [0], color='#555555', ls=STYLES[s], marker={1: 'o', 2: 's', 3: '^'}[s],
                       ms=4, lw=1.6, label=f'Seed {s}') for s in (1, 2, 3)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=5, frameon=False, fontsize=10)
    fig.suptitle('No-LLM hybrid: combined category and reward ablation', fontsize=16, y=.985)
    completed = sum(e['status'] == 'complete' for e in payload['endpoints'])
    fig.text(.075, .025,
        f'One curve per seed; {completed}/12 runs complete. Hollow endpoints denote unfinished runs; no extrapolation or pooled endpoint average.\n'
        'Loss includes numerical proposals and canonically evaluated refinements. Yield counts each qualifying topology once, including archive evictions.\n'
        'Count axes are logarithmic above 1 and include zero. Dotted horizontal lines on loss panels mark task-specific qualification thresholds.\n'
        'The intervention changes grouping, reward, retention and sampling together. Epochs do not equalize actual simulation cost.',
        fontsize=9, linespacing=1.6)
    fig.subplots_adjust(left=.08, right=.975, top=.86, bottom=.18, hspace=.4, wspace=.24)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'convergence_and_diversity.{suffix}', dpi=200, facecolor='white')
    plt.close(fig)


def report(payload):
    lines = ['# Combined category/reward ablation: convergence and structural yield', '',
        '![Figure](convergence_and_diversity.png)', '',
        'Every curve is the best observed design or cumulative qualifying topology count within one run. '
        'No median across runs is taken, and the lowest-loss seed is not selected. '
        'The unfinished RPA baseline is shown only through its last checkpoint; its completed ablation counterpart remains visible. '
        'Terminal draining is a separate point T, not assigned to another training epoch.', '',
        '| Task | Seed | Baseline loss | Ablation loss | Baseline structures | Ablation structures | Pair status |',
        '|---|---:|---:|---:|---:|---:|---|']
    for task in ('rpa', 'logic'):
        for seed in (1, 2, 3):
            pair = [next(e for e in payload['endpoints'] if (e['task'], e['seed'], e['condition']) == (task, seed, c))
                    for c in ('baseline', 'ablation')]
            a, b = pair
            status = 'complete' if all(e['status'] == 'complete' for e in pair) else f"incomplete baseline, {a['epoch']} epochs"
            lines.append(f"| {task} | {seed} | {a['best_pool_loss']:.7f} | {b['best_pool_loss']:.7f} | "
                f"{a['qualifying_topologies']} | {b['qualifying_topologies']} | {status} |")
    lines += ['', 'Diversity here means the cumulative number of distinct loss-qualifying labelled topologies. '
        'It is not region count, pairwise structural distance, functional diversity or robustness. '
        'Logic counts use the frozen loss threshold, without an added all-row correctness filter. '
        'Records include all complete committed numerical proposals and canonical reevaluations of fitting results, '
        'excluding internal Julia trials and retaining discoveries later evicted from the archive.', '',
        'The intervention changes region grouping, raw reward, archive retention and parent/imitation sampling together; '
        'it does not isolate speciation alone. Prior reward diagnostics show large failure penalties dominate the '
        'ablated reward magnitude. This small pilot is descriptive; no significance test is added.', '',
        'The plot is reconstructed directly from read-only pilot_candidate event journals and metric boundaries. '
        'Monotonicity and complete-run loss/cost endpoints are checked. All seed trajectories, endpoint statuses and '
        'source hashes are frozen in frozen_curves.json. This refresh does not alter the original campaign report.', '',
        'To replot: `.venv/bin/python paper/biomolecular_full_duplex_rewrite/plot_speciation_ablation.py --frozen`.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    data = json.loads((OUT / 'frozen_curves.json').read_text()) if args.frozen else extract()
    plot(data)
    report(data)
    print(OUT / 'convergence_and_diversity.pdf')


if __name__ == '__main__':
    main()
