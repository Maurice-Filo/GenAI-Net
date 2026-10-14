"""Publication figure from the audited historical trajectories; no simulations.

All curves use exact recorded checkpoint costs with right-continuous steps.
Each task stops at the minimum terminal allowance shared by every included run,
capped at the original GenAI-Net budget. No endpoint is extrapolated.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
STEM = 'genainet_hybrid_full_page'
TASKS = [('rpa', 'Robust perfect adaptation'), ('logic', 'Multi-input logic'),
         ('dose_hill', 'Hill dose response'), ('dose_ultrasensitive', 'Ultrasensitive response'),
         ('dose_biphasic', 'Biphasic response'), ('classifier', 'Classification')]
STYLES = {
    'original': ('#545b63', (0, (4, 2)), 'GenAI-Net'),
    'hybrid': ('#0072B2', '-', 'Hybrid numerical search'),
    'full': ('#C55A11', '-', 'Full duplex'),
}


def sample(run, grid):
    cost = np.asarray(run['cost'], dtype=float)
    loss = np.asarray(run['loss'], dtype=float)
    assert np.isfinite(cost).all() and np.isfinite(loss).all() and (loss > 0).all()
    assert (np.diff(cost) >= 0).all()
    index = np.searchsorted(cost, grid, side='right') - 1
    assert (index >= 0).all() and grid[-1] <= cost[-1]
    return np.minimum.accumulate(loss)[index]


def main():
    source = OUT / 'audit.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig = plt.figure(figsize=(11.6, 6.8), facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1],
                          left=.075, right=.985, bottom=.105, top=.89,
                          hspace=.43)
    upper_panels = gs[0].subgridspec(1, 2, wspace=.19)
    lower_panels = gs[1].subgridspec(1, 4, wspace=.35)
    handles = [Line2D([], [], color=color, linestyle=style, lw=1.8, label=label)
               for color, style, label in STYLES.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .985),
               ncol=3, frameon=False, fontsize=9, handlelength=2.7,
               columnspacing=1.8, handletextpad=.6, borderaxespad=0)
    curves, summary = [], []
    for k, (task, title) in enumerate(TASKS):
        ax = fig.add_subplot(upper_panels[0, k] if k < 2 else lower_panels[0, k - 2])
        runs = [r for r in data['runs'] if r['task'] == task]
        lower = max(r['cost'][0] for r in runs)
        upper = min(102400, min(r['cost'][-1] for r in runs))
        # Include every observed change point, not an interpolated regular grid.
        grid = np.unique([lower, upper] + [c for r in runs for c in r['cost'] if lower <= c <= upper])
        for arm, (color, style, label) in STYLES.items():
            group = [r for r in runs if r['arm'] == arm]
            if not group:
                continue
            assert len(group) == 20 and len({r['seed'] for r in group}) == 20
            values = np.array([sample(r, grid) for r in group])
            q25, median, q75 = np.quantile(values, [.25, .5, .75], axis=0)
            assert (np.diff(median) <= 1e-12).all()
            ax.fill_between(grid / 1000, q25, q75, step='post', color=color, alpha=.13, lw=0)
            ax.step(grid / 1000, median, where='post', color=color, ls=style, lw=1.65)
            summary.append(dict(task=task, arm=arm, n=20, allowance=upper,
                                median=float(median[-1]), q25=float(q25[-1]), q75=float(q75[-1])))
            curves.extend(dict(task=task, arm=arm, allowance=float(x), median=float(m),
                               q25=float(lo), q75=float(hi))
                          for x, m, lo, hi in zip(grid, median, q25, q75))
        ax.set_yscale('log')
        ax.set_xlim(0, upper / 1000)
        ax.set_xticks([x for x in [0, 25, 50, 75, 100] if x <= upper / 1000])
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=6))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y', which='major', color='#dce1e5', lw=.5, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(which='minor', length=2)
        ax.set_title(title, loc='left', fontsize=10, pad=11, weight='medium')
        ax.text(-.10 if k < 2 else -.22, 1.065, chr(65 + k), transform=ax.transAxes, fontsize=12, weight='bold')
    fig.text(.018, .50, 'Best loss so far', rotation=90, va='center', fontsize=10)
    fig.text(.53, .035, r'Common charged-evaluation allowance ($\times 10^3$)', ha='center', fontsize=10)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{STEM}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    with (OUT / f'{STEM}_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    (OUT / f'{STEM}_audit.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'alignment': 'Last recorded observation at or below allowance; exact union of checkpoint costs; no extrapolation.',
        'summary': summary, 'limitations': data['limitations'],
    }, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
