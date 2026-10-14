"""Saved-results-only GA / RL--AdaGrad convergence and search success plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'numerical_comparison'
CAMPAIGN = Path('/local1/rossin/search-exploration/rpa-statistical-20seed-20260915')
ARMS = ('julia', 'ga')
LABELS = {'julia': 'RL–AdaGrad hybrid', 'ga': 'Standalone GA'}
COLORS = {'julia': '#168577', 'ga': '#b85848'}


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract():
    hashes = {}
    def read(path, lines=False):
        blob = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return [json.loads(x) for x in blob.splitlines()] if lines else json.loads(blob)
    plan = read(CAMPAIGN / 'active_plan.json')
    audited = read(CAMPAIGN / plan['analysis_dir'] / 'summary.json')
    assert audited['inference_ready']
    endpoints = {(r['arm'], r['seed']): r for r in audited['endpoints']}
    runs, snapshots = [], []
    for job in plan['jobs']:
        arm, seed = job['arm'], job['seed']
        if arm not in ARMS:
            continue
        folder = Path(job.get('output', CAMPAIGN / 'runs' / job['name']))
        completed = read(folder / 'completed.json')
        manifest = read(folder / 'manifest.json')
        rows = read(folder / 'metrics.jsonl', lines=True)
        cfg = manifest['config']
        assert completed['status'] == endpoints[arm, seed]['status'] == 'complete'
        assert manifest['task'] == 'rpa' and manifest['harness'] is False
        assert cfg['quality_threshold'] == plan['threshold'] and cfg['evaluation_budget'] == plan['attempt_ceiling']
        assert cfg['seed'] == seed and completed['pending_optimization_jobs'] == 0
        assert completed['counters'].get('julia/reserved', 0) == 0
        x = np.array([r['Compute/expensive_evaluations'] for r in rows], dtype=int)
        y = np.array([r['Archive/best_loss'] for r in rows], dtype=float)
        assert np.isfinite(y).all() and (y > 0).all()
        assert (np.diff(x) >= 0).all() and (np.diff(y) <= 1e-12).all()
        assert x[-1] == completed['counters']['expensive_evaluations'] == endpoints[arm, seed]['evaluations']
        assert x[-1] <= plan['attempt_ceiling']
        assert y[-1] == completed['metrics']['Archive/best_loss'] == endpoints[arm, seed]['loss']
        assert (completed['epochs'] == 100 if arm == 'julia' else x[-1] == plan['attempt_ceiling'])
        first = np.flatnonzero(y < plan['threshold'])
        runs.append({'arm': arm, 'seed': seed, 'evaluations': x.tolist(), 'loss': y.tolist(),
            'first_success_checkpoint': int(x[first[0]]) if len(first) else None,
            'successful': bool(len(first)), 'joint_evaluations': completed['counters'].get('julia/joint_evaluations', 0),
            'folder': str(folder)})
        snapshots += [{'arm': arm, 'seed': seed, 'checkpoint': i, 'evaluations': int(a), 'best_loss': float(b)}
                      for i, (a, b) in enumerate(zip(x, y))]
    assert {(r['arm'], r['seed']) for r in runs} == {(a, s) for a in ARMS for s in range(1, 21)}
    frozen = {'threshold': plan['threshold'], 'ceiling': plan['attempt_ceiling'], 'runs': runs,
              'source_hashes': hashes}
    (OUT / 'frozen_runs.json').write_text(json.dumps(frozen, indent=2) + '\n')
    write_csv(OUT / 'snapshots.csv', snapshots)
    return frozen


def at_budget(run, budgets):
    x, y = np.asarray(run['evaluations']), np.asarray(run['loss'])
    index = np.searchsorted(x, budgets, side='right') - 1
    assert (index >= 0).all()
    # Once a run stops, retain its final observed design. No further search is
    # implied: the estimand is the best observed design within an allowance.
    return y[index]


def analyze(frozen):
    runs, ceiling, threshold = frozen['runs'], frozen['ceiling'], frozen['threshold']
    start = max(r['evaluations'][0] for r in runs)
    checkpoints = (50000, 100000, 125000, 150000, ceiling)
    grid = np.unique([start, ceiling, *checkpoints,
        *[x for r in runs for x in r['evaluations'] if x >= start]])
    stats, summary, rows, endpoints = {}, {}, [], []
    for arm in ARMS:
        selected = sorted((r for r in runs if r['arm'] == arm), key=lambda r: r['seed'])
        values = np.array([at_budget(r, grid) for r in selected])
        median = np.median(values, axis=0)
        low, high = np.quantile(values, [.25, .75], axis=0)
        successes = (values < threshold).sum(axis=0)
        stops = np.array([r['evaluations'][-1] for r in selected])
        stats[arm] = {'median': median, 'low': low, 'high': high, 'successes': successes,
                      'first_stop': int(stops.min()), 'last_stop': int(stops.max())}
        summary[arm] = {'successful': int(successes[-1]), 'unsuccessful': int(20-successes[-1]),
            'consumed_attempts_min': int(stops.min()), 'consumed_attempts_max': int(stops.max()),
            'median_final_loss': float(median[-1]),
            'last_first_success_checkpoint': max(r['first_success_checkpoint'] for r in selected if r['successful']),
            'checkpoints': []}
        for i, budget in enumerate(grid):
            row = {'arm': arm, 'budget': int(budget), 'n': 20, 'median': float(median[i]),
                'q25': float(low[i]), 'q75': float(high[i]), 'successful': int(successes[i]),
                'unsuccessful': int(20-successes[i]), 'stopped_runs': int((stops < budget).sum())}
            rows.append(row)
            if budget in checkpoints:
                summary[arm]['checkpoints'].append(row)
        for r in selected:
            endpoints.append({'arm': arm, 'seed': r['seed'], 'successful': r['successful'],
                'first_success_checkpoint': r['first_success_checkpoint'], 'final_loss': r['loss'][-1],
                'charged_attempts': r['evaluations'][-1], 'joint_evaluations': r['joint_evaluations'],
                'canonical_evaluations': r['evaluations'][-1]-r['joint_evaluations']})
        assert successes[-1] == sum(r['successful'] for r in selected)
    assert summary['julia']['successful'] == 20 and summary['ga']['successful'] == 13
    write_csv(OUT / 'aggregate_curves.csv', rows)
    write_csv(OUT / 'endpoints.csv', endpoints)
    (OUT / 'summary.json').write_text(json.dumps({'threshold': threshold, 'budget_ceiling': ceiling,
        'methods': summary, 'new_simulations': 0, 'new_hypothesis_tests': 0}, indent=2) + '\n')
    return grid, stats, summary


def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def plot(frozen, grid, stats, summary):
    ceiling, threshold = frozen['ceiling'], frozen['threshold']
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), gridspec_kw={'width_ratios': [1.65, 1.25, 1.]})
    for arm in ARMS:
        s, color = stats[arm], COLORS[arm]
        # The dashed segment explicitly marks the allowance range that includes
        # stopped hybrid runs, without dropping those runs from aggregation.
        cut = np.searchsorted(grid, s['first_stop'], side='right')
        axes[0].step(grid[:cut], s['median'][:cut], where='post', color=color, lw=2.4, label=LABELS[arm])
        if cut < len(grid):
            axes[0].step(grid[cut-1:], s['median'][cut-1:], where='post', color=color, lw=2.4, ls='--')
        axes[0].fill_between(grid, s['low'], s['high'], step='post', color=color, alpha=.14)
        axes[1].step(grid, s['successes'], where='post', color=color, lw=2.4, label=LABELS[arm])
    axes[0].axhline(threshold, color='#555555', ls=':', lw=1.3)
    axes[0].text(ceiling*.98, threshold*1.13, 'Success threshold', color='#555555', ha='right', fontsize=9)
    axes[0].set(yscale='log', ylabel='Best-so-far RPA loss', title='A   Convergence across 20 runs')
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    axes[0].text(.97, .04, 'Median and interquartile range', transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=9, color='#555555')
    axes[1].set(ylabel='Runs with a qualifying design', ylim=(-.5, 21.8), yticks=[0, 5, 10, 15, 20],
                title='B   Success within the budget')
    axes[1].annotate('20 / 20', (ceiling, 20), xytext=(-5, 7), textcoords='offset points',
                     color=COLORS['julia'], ha='right', fontweight='bold')
    axes[1].annotate('13 / 20', (ceiling, 13), xytext=(-5, 7), textcoords='offset points',
                     color=COLORS['ga'], ha='right', fontweight='bold')
    for ax in axes[:2]:
        ax.set_xlim(0, ceiling*1.015)
        ax.set_xticks([0, 50000, 100000, 150000, 200000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k' if x else '0'))
        ax.set_xlabel('Evaluation allowance (charged attempts)')
        ax.grid(axis='y', alpha=.14)
    ax = axes[2]
    success_color, failure_color = '#468a74', '#e4a37e'
    for i, arm in enumerate(ARMS):
        n = summary[arm]['successful']
        ax.bar(i, n, .62, color=success_color, label='Successful' if i == 0 else None)
        ax.bar(i, 20-n, .62, bottom=n, color=failure_color, label='Unsuccessful' if i == 0 else None)
        ax.text(i, n/2, f'{n} successful', color='white', ha='center', va='center', fontweight='bold', fontsize=10)
        if n < 20:
            ax.text(i, n+(20-n)/2, f'{20-n} unsuccessful', ha='center', va='center', fontsize=9)
        ax.text(i, 20.65, f'{n/20:.0%} success', ha='center', fontweight='bold', fontsize=10)
    ax.set(ylim=(0, 23), yticks=[0, 5, 10, 15, 20], xticks=[0, 1],
        xticklabels=['RL–AdaGrad\nhybrid', 'Standalone\nGA'], ylabel='Number of runs',
        title='C   Final search outcomes')
    ax.text(.5, -.2, 'Ceiling: 214,793 attempts per run', transform=ax.transAxes, ha='center', fontsize=9)
    fig.suptitle('RPA without LLM: convergence and reliability under a bounded evaluation budget', fontsize=14, y=.97)
    fig.text(.06, .045,
        'Success: best canonical loss < 0.0164912. All 40 runs completed; “unsuccessful” means no qualifying design.\n'
        'Hybrid stops after 100 epochs (129,229–160,273 attempts); GA uses 214,793. Dashed loss curve includes stopped runs, retaining their last result.\n'
        'An attempt is a canonical score or a joint loss/gradient evaluation; costs differ. The hybrid also includes archive mutation and random proposals.',
        fontsize=8.5, linespacing=1.6)
    fig.subplots_adjust(left=.06, right=.985, bottom=.28, top=.86, wspace=.32)
    save(fig, 'ga_vs_rl_adagrad')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        for run in sorted((r for r in frozen['runs'] if r['arm'] == arm), key=lambda r: r['seed']):
            color = COLORS[arm] if run['successful'] else '#992d32'
            ax.step(run['evaluations'], run['loss'], where='post', color=color,
                    alpha=.38 if run['successful'] else .9, lw=.9 if run['successful'] else 1.4)
            ax.plot(run['evaluations'][-1], run['loss'][-1], 'o', color=color, ms=3)
        ax.axhline(threshold, ls=':', color='#555555', lw=1)
        ax.set(yscale='log', xlim=(0, ceiling*1.02), xlabel='Actual charged evaluation attempts',
               title=f"{LABELS[arm]}: {summary[arm]['successful']}/20 successful")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k' if x else '0'))
        ax.grid(axis='y', alpha=.14)
    axes[0].set_ylabel('Best-so-far RPA loss')
    fig.text(.07, .025, 'One curve per completed run, ending at its actual expenditure. Dark red: searches that never qualify.\n'
             'Dotted line: success threshold. Canonical and joint-gradient evaluation attempts have different costs.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, 'ga_vs_rl_adagrad_individual_runs')


def report(frozen, summary):
    lines = ['# GA versus RL–AdaGrad: RPA convergence and successful searches', '',
        'Saved-results-only analysis of the completed no-LLM cohort: 20 runs per method. No new simulations or tests.', '',
        '![Main plot](ga_vs_rl_adagrad.png)', '',
        '| Evaluation allowance | Hybrid successful | GA successful | Hybrid median loss | GA median loss |',
        '|---:|---:|---:|---:|---:|']
    for a, b in zip(summary['julia']['checkpoints'], summary['ga']['checkpoints']):
        lines.append(f"| {a['budget']:,} | {a['successful']}/20 | {b['successful']}/20 | {a['median']:.7f} | {b['median']:.7f} |")
    lines += ['', '## Interpretation and accounting', '',
        'Success is the first recorded best canonical loss strictly below the frozen RPA threshold '
        f"{frozen['threshold']}. All 40 searches completed. The seven unsuccessful GA searches are not crashed runs, "
        'and failure here does not refer to an individual ODE solver failure.', '',
        'The x-axis is the cumulative charged-evaluation allowance, not epochs, raw scenario solves, proposals or wall time. '
        'A charged attempt can be one canonical evaluation over the task scenarios or one joint Julia loss/gradient evaluation. '
        'Their costs differ. Cache hits do not add a charged evaluation; interrupted historical attempts are excluded from '
        'completed-run trajectories and retained separately in the campaign records.', '',
        'At each allowance, use the last recorded checkpoint at or below it, with no interpolation or backdating. '
        'If a run has finished, retain its last design: the dashed hybrid segment denotes allowances containing at least '
        'one stopped hybrid run. It does not claim that those runs continued searching. Every aggregate includes all 20 seeds. '
        'Shading is the linearly interpolated empirical 25th–75th percentile, not a confidence interval.', '',
        'The hybrid ends after 100 epochs plus draining, consuming 129,229–160,273 attempts. GA consumes 214,793 in every run. '
        'At 100,000 attempts, before either workflow has stopped, observed success is already 20/20 versus 13/20. '
        'All hybrid seeds first qualify by recorded checkpoint 55,359; first-hit counts are checkpoint upper bounds, '
        'not exact individual-evaluation event times.', '',
        'RL–AdaGrad denotes the complete hybrid, including archive mutation and random proposals; '
        'these curves do not isolate RL or the optimizer as the cause of an advantage. '
        'Existing corrected final-loss significance versus GA remains unresolved (Holm p=0.0655); '
        'this figure reports descriptive convergence and search success, without introducing a new test.', '',
        '![Individual runs](ga_vs_rl_adagrad_individual_runs.png)', '',
        'Source: `/local1/rossin/search-exploration/rpa-statistical-20seed-20260915/analysis-concurrent/`.',
        'Frozen trajectories and input hashes: `frozen_runs.json`; observed points: `snapshots.csv`; '
        'budget-aligned summaries: `aggregate_curves.csv`; per-run outcomes: `endpoints.csv`.', '',
        'Replot without the original campaign: `../plot_ga_adagrad.py --frozen` with the repository Python environment.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    frozen = json.loads((OUT / 'frozen_runs.json').read_text()) if args.frozen else extract()
    grid, stats, summary = analyze(frozen)
    plot(frozen, grid, stats, summary)
    report(frozen, summary)
    print(json.dumps(summary, indent=2))
    print(f'Plots written to {OUT}')


if __name__ == '__main__':
    main()
