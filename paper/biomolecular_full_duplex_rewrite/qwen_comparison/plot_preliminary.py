"""Snapshot partial Qwen trajectories against 20-run reference arithmetic means.

Read-only with respect to campaigns. Partial trajectories are never extrapolated;
the snapshot and aggregate CSV make the figure reproducible after runs advance.
"""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
REFERENCE = OUT.parent / 'genainet_comparison/audit.json'
CAMPAIGN = Path('/local1/rossin/search-exploration/qwen38-matched-rpa-logic-20seed-20261006-r2')
STEM = 'qwen_preliminary_convergence'
STYLES = {
    'original': ('#62666D', '--', 'GenAI-Net'),
    'hybrid': ('#0072B2', '-', 'Hybrid (no LLM insertion)'),
    'full': ('#CF6A18', '-', 'Flash full duplex'),
}
QWEN = '#27834B'


def read_jsonl_snapshot(path):
    content = path.read_bytes()
    # Ignore only a currently incomplete trailing record from an active writer.
    lines = content.splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise
    return rows, hashlib.sha256(content).hexdigest()


def sample(run, grid):
    cost = np.asarray(run['cost'], dtype=float)
    loss = np.minimum.accumulate(np.asarray(run['loss'], dtype=float))
    assert np.isfinite(cost).all() and np.isfinite(loss).all() and (loss > 0).all()
    assert (np.diff(cost) >= 0).all()
    assert grid[0] >= cost[0] and grid[-1] <= cost[-1]
    return loss[np.searchsorted(cost, grid, side='right') - 1]


def snapshot():
    reference_bytes = REFERENCE.read_bytes()
    reference = json.loads(reference_bytes)
    progress = json.loads((CAMPAIGN / 'progress.json').read_text())
    runs, omitted = [], []
    for job in progress['jobs']:
        if job['status'] == 'queued':
            continue
        path = Path(job['output']) / 'metrics.jsonl'
        if not path.exists():
            omitted.append(dict(name=job['name'], reason='No completed epoch yet'))
            continue
        metrics, digest = read_jsonl_snapshot(path)
        checkpoints = sorted({int(m['epoch']): m for m in metrics if 'epoch' in m}.values(), key=lambda m: m['epoch'])
        if not checkpoints:
            omitted.append(dict(name=job['name'], reason='No completed epoch yet'))
            continue
        run = dict(task=job['task'], arm='qwen', seed=job['seed'], folder=job['output'], status=job['status'],
                   metrics_sha256=digest, epoch=[min(m['epoch'] + 1, 100) for m in checkpoints],
                   post_training_drain=[m['epoch'] == 100 for m in checkpoints],
                   cost=[m['Compute/expensive_evaluations'] for m in checkpoints],
                   loss=[m['Archive/best_loss'] for m in checkpoints])
        history = Path(job['output']) / 'llm_history.json'
        histories = json.loads(history.read_text()) if history.exists() else []
        run['llm_errors'] = [h['error'] for h in histories if h.get('error')]
        runs.append(run)
    return dict(snapshot_time=datetime.now(ZoneInfo('Europe/Zurich')).isoformat(), campaign=str(CAMPAIGN),
                reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                reference_runs=[r for r in reference['runs'] if r['task'] in ('rpa', 'logic')],
                qwen_runs=runs, omitted=omitted, reference_limitations=reference['limitations'],
                aggregation='Reference curves: arithmetic mean +/- one standard error across 20 runs. Qwen: completed-run mean when at least two complete; faint individual observed trajectories include partial runs.',
                alignment='Right-continuous best-so-far steps at recorded charged cost; no backward interpolation or extrapolation.',
                protocol_notes=['Qwen: RPA 900 s, Logic 3600 s role timeout; asynchronous single in-flight request, cadence five epochs.',
                    'Qwen concurrency increased from two to eight during the campaign; all observed runs retained, including timeout-affected seeds.',
                    'Logic hybrid control is the numerical archive of the independent-operation arm; its charged allowance subtracts separately logged independent LLM evaluations at each checkpoint.',
                    'Charged evaluation counts do not imply equal wall-clock cost; this is a preliminary descriptive comparison.'])


def render(data):
    all_complete = len(data['qwen_runs']) == 40 and all(r['status'] == 'complete' for r in data['qwen_runs'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
    fig.subplots_adjust(left=.075, right=.985, bottom=.23, top=.72, wspace=.23)
    handles = [Line2D([], [], color=c, ls=s, lw=2, label=l) for c, s, l in STYLES.values()]
    handles.append(Line2D([], [], color=QWEN, lw=2.5, label='Qwen full duplex (preliminary)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .935), ncol=2,
               frameon=False, columnspacing=2.6, handlelength=3)
    fig.suptitle(('Convergence' if all_complete else 'Preliminary convergence') + ': local Qwen and reference approaches', x=.53, y=.98, fontsize=13)
    table, endpoints = [], []
    for ax, task, title in zip(axes, ('rpa', 'logic'), ('Robust perfect adaptation', 'Multi-input logic')):
        references = [r for r in data['reference_runs'] if r['task'] == task]
        lower = max(r['cost'][0] for r in references)
        upper = min(102400, min(r['cost'][-1] for r in references))
        grid = np.unique([lower, upper] + [c for r in references for c in r['cost'] if lower <= c <= upper])
        for arm, (color, style, _) in STYLES.items():
            group = [r for r in references if r['arm'] == arm]
            assert len(group) == 20 and len({r['seed'] for r in group}) == 20
            values = np.array([sample(r, grid) for r in group])
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(len(group))
            assert (mean - se > 0).all()
            ax.fill_between(grid / 1000, mean - se, mean + se, step='post', color=color, alpha=.10, lw=0)
            ax.step(grid / 1000, mean, where='post', color=color, ls=style, lw=2, zorder=3)
            table.extend(dict(task=task, arm=arm, allowance=float(x), mean=float(y), se=float(s), n=20)
                         for x, y, s in zip(grid, mean, se))
        qwen = [r for r in data['qwen_runs'] if r['task'] == task]
        completed_qwen = [r for r in qwen if r['status'] == 'complete']
        for run in qwen:
            lo, hi = max(lower, run['cost'][0]), min(upper, run['cost'][-1])
            if hi < lo:
                continue
            gx = np.unique([lo, hi] + [c for c in run['cost'] if lo <= c <= hi])
            y = sample(run, gx)
            ax.step(gx / 1000, y, where='post', color=QWEN, lw=.85, alpha=.30 if len(completed_qwen)>=2 else .85, zorder=2)
            clipped = run['cost'][-1] > upper
            ax.plot(hi / 1000, y[-1], marker='>' if clipped else 'o', markersize=4.5,
                    color=QWEN, markerfacecolor=QWEN if run['status']=='complete' else 'white',
                    clip_on=False, alpha=.4 if len(completed_qwen)>=2 else 1, zorder=2)
            if hi > 20000 and len(completed_qwen) < 2:
                ax.annotate(f'seed {run["seed"]}', (hi / 1000, y[-1]), xytext=(-5, -12 if task=='logic' else 7),
                            textcoords='offset points', ha='right', fontsize=8, color=QWEN)
            endpoints.append(dict(task=task, seed=run['seed'], status=run['status'], observed_cost=run['cost'][-1],
                                  plotted_cost=hi, plotted_loss=float(y[-1]), last_epoch=run['epoch'][-1]))
        if len(completed_qwen) >= 2:
            lo = max(lower, max(r['cost'][0] for r in completed_qwen))
            hi = min(upper, min(r['cost'][-1] for r in completed_qwen))
            gx = np.unique([lo, hi] + [c for r in completed_qwen for c in r['cost'] if lo <= c <= hi])
            values = np.array([sample(r, gx) for r in completed_qwen])
            mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
            ax.step(gx / 1000, mean, where='post', color=QWEN, lw=2.5, zorder=5)
            ax.fill_between(gx / 1000, mean-se, mean+se, color=QWEN, alpha=.10, step='post', lw=0)
            table.extend(dict(task=task, arm='qwen_completed', allowance=float(x), mean=float(y), se=float(s), n=len(values))
                         for x,y,s in zip(gx,mean,se))
        complete = sum(r['status'] == 'complete' for r in qwen)
        ax.set_title(f'{title}\nQwen: {len(qwen)} observed runs; {complete} completed', fontsize=11, loc='left', pad=10)
        ax.set_yscale('log')
        ax.set_xlim(0, upper / 1000)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(r'Charged-evaluation allowance ($\times 10^3$)', labelpad=8)
        ax.grid(axis='y', which='major', color='#DDE2E6', lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Best loss so far (log scale)')
    stamp = datetime.fromisoformat(data['snapshot_time']).astimezone(ZoneInfo('Europe/Zurich')).strftime('%d %b %Y, %H:%M %Z')
    foot = (('All approaches: mean ± SE, 20 runs each. Faint green traces show individual Qwen runs. No extrapolation.\n' if all_complete else
             'References: mean ± SE, 20 runs each. Green: completed-run mean ± SE; faint traces include partial runs. No extrapolation.\n') +
            'Logic control: numerical archive without LLM insertion; independent LLM evaluation costs are subtracted.\n'
            f'Snapshot: {stamp}. Qwen includes timeout/context-limit failures; results are preliminary and operationally affected.')
    fig.text(.075, .035, foot, ha='left', va='bottom', fontsize=8, color='#444444', linespacing=1.6)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=230, facecolor='white')
    plt.close(fig)
    with (OUT / f'{STEM}_reference_means.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return endpoints


if __name__ == '__main__':
    data = snapshot()
    data['plotted_endpoints'] = render(data)
    (OUT / f'{STEM}_snapshot.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'snapshot': data['snapshot_time'], 'endpoints': data['plotted_endpoints']}, indent=2))
