"""Descriptive historical comparison; no new searches or significance tests.

Legacy runs did not retain all candidates, so qualifying topology counts cannot
be reconstructed. The Logic numerical branch ran alongside an independent LLM;
its archive excludes those proposals, and its plotted costs subtract the
separately logged independent LLM evaluations at each checkpoint.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / 'comparisons/search_exploration/reports'
PAPER = OUT.parent
sys.path.insert(0, str(PAPER))
from generate_dose_overviews import select_runs
BREADTH = {'dose_hill': 'hill', 'dose_ultrasensitive': 'ultrasensitive',
           'dose_biphasic': 'biphasic', 'classifier': 'classifier'}
COLORS = {'original': '#777777', 'hybrid': '#287aa0', 'full': '#ae4d77'}
LABELS = {'original': 'Historical GenAI-Net', 'hybrid': 'Hybrid numerical archive',
          'full': 'Full duplex'}


def read(p):
    return json.loads(p.read_text())


def legacy(task):
    result = []
    for seed in range(20):
        if task in BREADTH:
            folder = ROOT / f'comparisons/rpa_search/data/raw/rl4crn_breadth/{task}_full102400_seed{seed}_cvode_rl_only_breadth'
            done = read(folder / 'completed.json')
            assert done['epochs'] == 100 and done['ode_simulations'] == 102400 and done['task'] == task
        else:
            folder = ROOT / f'comparisons/rpa_search/data/raw/rl4crn/{task}_full102400_seed{seed}_cvode'
        rows = list(csv.DictReader((folder / 'progress.csv').open()))
        assert len(rows) == 100 and int(rows[-1]['step']) == 100
        assert float(rows[-1]['ode_simulations']) == 102400
        if task in BREADTH:
            assert np.isclose(float(rows[-1]['best_so_far_loss']), done['best_loss'])
        result.append(dict(task=task, arm='original', seed=seed, folder=str(folder),
                           epoch=[int(r['step']) for r in rows],
                           cost=[float(r['ode_simulations']) for r in rows],
                           loss=[float(r['best_so_far_loss']) for r in rows]))
    return result


def hybrid(task, arm, seed, folder):
    folder = Path(folder)
    done = read(folder / 'completed.json')
    assert done['status'] == 'complete' and done['epochs'] == 100
    rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    rows = sorted({int(r['epoch']): r for r in rows if 'epoch' in r and int(r['epoch']) < 100}.values(), key=lambda r: r['epoch'])
    assert [r['epoch'] for r in rows] == list(range(100))
    # Never assign post-training stream drain improvements to an earlier cost.
    values = dict(task=task, arm=arm, seed=int(seed), folder=str(folder),
                  epoch=[r['epoch'] + 1 for r in rows],
                  cost=[r['Compute/expensive_evaluations'] for r in rows],
                  loss=[r['Archive/best_loss'] for r in rows])
    values['cost'].append(done['counters']['expensive_evaluations'])
    values['loss'].append(done['metrics']['Archive/best_loss'])
    values['epoch'].append(101)  # distinct terminal/drained point
    if task == 'logic' and arm == 'hybrid':
        manifest = read(folder / 'manifest.json')
        assert manifest['communication_mode'] == 'none'
        assert manifest['llm_fitting'].startswith('Disabled')
        counters = [{k.removeprefix('Compute/'): v for k, v in row.items()
                     if k.startswith('Compute/')} for row in rows] + [done['counters']]
        for counts in counters:
            assert counts['expensive_evaluations'] == sum(
                counts.get('evaluations/' + stream, 0) for stream in ('search', 'optimization', 'llm'))
        values['raw_cost'] = list(values['cost'])
        values['excluded_llm_evaluations'] = [c.get('evaluations/llm', 0) for c in counters]
        values['cost'] = [total - excluded for total, excluded in
                          zip(values['raw_cost'], values['excluded_llm_evaluations'])]
        assert all(c >= 0 for c in values['cost']) and np.all(np.diff(values['cost']) >= 0)
        values['cost_accounting'] = 'numerical_only_excluding_independent_llm'
    values['retained_qualifying'] = done['metrics']['Archive/qualified_topologies']
    return values


def align(run, grid):
    x, y = np.asarray(run['cost']), np.minimum.accumulate(run['loss'])
    ix = np.searchsorted(x, grid, side='right') - 1
    assert (ix >= 0).all()
    return y[ix]


def main():
    runs = legacy('rpa') + legacy('logic')
    frozen = read(PAPER / 'numerical_comparison/frozen_runs.json')
    for r in frozen['runs']:
        if r['arm'] == 'julia':
            runs.append(hybrid('rpa', 'hybrid', r['seed'], r['folder']))
    full = Path('/local1/rossin/search-exploration/rpa-flash-adagrad-20seed-20260917-r1')
    for j in read(full / 'plan.json')['jobs']:
        runs.append(hybrid('rpa', 'full', j['seed'], j.get('output', full / 'runs' / j['name'])))
    for r in csv.DictReader((REPORTS / 'logic5-four-arm-statistics-20260924/endpoints.csv').open()):
        if r['arm'] in ('none', 'full'):
            runs.append(hybrid('logic', 'hybrid' if r['arm'] == 'none' else 'full', r['seed'], r['output']))
    for task, key in BREADTH.items():
        runs.extend(legacy(task))
        selected, excluded = select_runs(key)
        assert len(selected) == 20 and not excluded
        for seed, folder, done, replacement in selected:
            row = hybrid(task, 'full', seed, folder)
            row['replacement'] = replacement
            runs.append(row)
    limitations = [
        'Historical comparison, not a matched algorithm ablation: policy sizes, rate proposal mechanisms, learning updates and execution settings differ.',
        'Legacy task configurations match hybrid RPA/Logic task sections; this does not prove historical source-code identity.',
        'Original GenAI-Net: 256-wide, depth 2, deep block 512. Hybrid: 1024-wide, depth 5, deep block 1024.',
        'One charged call is a task evaluation or hybrid joint loss/gradient call; these do not have equal computational cost.',
        'Logic hybrid numerical archive comes from the independent-operation arm. Its plotted cost subtracts logged independent LLM evaluations at every checkpoint and terminal endpoint; raw counters are retained. This adjusts accounting, not the executed search.',
        'RPA/Logic reference thresholds are historical GenAI-Net median endpoint losses; breadth uses the existing loss<0.05 cutoff. Neither guarantees functional correctness.',
        'Breadth: old and new configurations agree on task solver/tolerances and six added reactions. Task targets live in the shared paper_breadth_tasks builder, not the saved JSON; historical source-code identity is not established.',
        'Breadth full-duplex comparisons include the explicitly selected Ultrasensitive service-failure replacements, irrespective of outcome. Seeds 0–19 vs 1–20 are not treated as paired experiments.',
        'Legacy portfolios are unavailable: no comparison of distinct qualifying topology counts can be recovered from these files.',
        'Epoch panel excludes final stream drain; budget panel includes it only at its recorded cost. Final threshold sweep uses drained endpoints.',
        'No wall-time speedup or statistical significance inferred from these historical runs.'
    ]
    thresholds = {'rpa': frozen['threshold'], 'logic': 0.04271267898491677}
    thresholds.update({task: .05 for task in BREADTH})
    summaries = []
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    with PdfPages(OUT / 'historical_convergence.pdf') as pdf:
        for task in ('rpa', 'logic', *BREADTH):
            selected = [r for r in runs if r['task'] == task]
            # Verify task contracts against original run config.
            baseline = read(Path(selected[0]['folder']) / 'config.json')[task]
            for r in selected:
                assert np.isfinite(r['loss']).all() and np.all(np.diff(r['cost']) >= 0)
                if r['arm'] != 'original':
                    cfg = read(Path(r['folder']) / 'manifest.json')['task_config']
                    assert cfg[task] == baseline
                    original_cfg = read(Path(selected[0]['folder']) / 'config.json')
                    assert cfg['search']['max_added_reactions'] == original_cfg['search']['max_added_reactions']
            ceiling = min(102400, min(r['cost'][-1] for r in selected))
            grid = np.linspace(max(r['cost'][0] for r in selected), ceiling, 301)
            assert all(r['cost'][-1] >= grid[-1] for r in selected)
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), constrained_layout=True)
            for arm in COLORS:
                group = [r for r in selected if r['arm'] == arm]
                if not group:
                    continue
                assert len(group) == 20
                epoch = np.array([np.minimum.accumulate(r['loss'][:100]) for r in group])
                cost = np.array([align(r, grid) for r in group])
                for ax, x, a in ((axes[0, 0], np.arange(1, 101), epoch), (axes[0, 1], grid, cost)):
                    q = np.quantile(a, [.25, .5, .75], axis=0)
                    ax.fill_between(x, q[0], q[2], color=COLORS[arm], alpha=.15)
                    ax.plot(x, q[1], color=COLORS[arm], label=LABELS[arm], lw=1.8)
                    ax.set_yscale('log')
                axes[1, 0].plot(grid, (cost < thresholds[task]).sum(axis=0), color=COLORS[arm], lw=1.8)
                cutoffs = np.geomspace(1e-4, .3, 350)
                final = np.array([r['loss'][-1] for r in group])
                axes[1, 1].plot(cutoffs, (final[:, None] < cutoffs).sum(axis=0), color=COLORS[arm], lw=1.8)
                summaries.append(dict(task=task, arm=arm, n=20, common_allowance=ceiling, median_at_common_allowance=float(np.median(cost[:, -1])),
                                      below_reference_at_common_allowance=int((cost[:, -1] < thresholds[task]).sum()),
                                      median_terminal=float(np.median(final)),
                                      below_reference_terminal=int((final < thresholds[task]).sum()),
                                      reference_threshold=thresholds[task], unique_qualifying_comparison_available=False))
            axes[0, 0].set(title='A  Best loss by epoch (median and IQR)', xlabel='Epoch (different work per epoch)', ylabel='Best loss')
            axes[0, 0].legend(fontsize=8)
            axes[0, 1].set(title='B  Best loss at a common charged allowance', xlabel='Charged evaluation allowance', ylabel='Best loss')
            axes[1, 0].set(title=f'C  Runs below loss cutoff {thresholds[task]:.5g}', xlabel='Charged evaluation allowance', ylabel='Runs (out of 20)', ylim=(-.5, 20.5))
            axes[1, 1].set(title='D  Final-run quality across loss thresholds', xlabel='Loss threshold (strict inequality)', ylabel='Runs below threshold (out of 20)', xscale='log', ylim=(-.5, 20.5))
            axes[1, 1].axvline(thresholds[task], color='.5', ls=':', lw=1)
            fig.suptitle(f'{task.upper()} · historical comparison, 20 runs per method\nDifferent policy sizes; charged calls differ in cost; topology counts unavailable', fontsize=12)
            pdf.savefig(fig)
            fig.savefig(OUT / f'{task}_historical_convergence.png', dpi=170)
            plt.close(fig)
    (OUT / 'audit.json').write_text(json.dumps(dict(limitations=limitations, summaries=summaries, runs=runs), indent=2) + '\n')
    with (OUT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
