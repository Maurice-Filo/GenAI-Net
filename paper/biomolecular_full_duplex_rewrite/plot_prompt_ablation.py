"""Render the RPA prompt comparison from the frozen, completed analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {'reference': '#943C87', 'unspecified': '#26845C'}
LABELS = ['Original prompt', 'Without mechanism suggestions']


def main():
    data = json.loads((HERE / 'prompt_ablation/final_20seed/snapshot.json').read_text())
    assert data['final'] and data['seeds'] == list(range(1, 21))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 10, 'axes.titlesize': 11,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    fig.subplots_adjust(left=.085, right=.98, top=.88, bottom=.14,
                        hspace=.46, wspace=.28)
    jitter = np.random.default_rng(42).uniform(-.16, .16, 20)

    def distribution(ax, values, x, color):
        parts = ax.violinplot(values, positions=[x], widths=.74, showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor('none')
            body.set_alpha(.12)
        ax.scatter(x+jitter, values, s=18, color=color, alpha=.68,
                   linewidths=.4, edgecolors=color, zorder=3)
        ax.hlines(np.median(values), x-.20, x+.20, color=color, lw=2.6, zorder=4)

    for i, (arm, color) in enumerate(COLORS.items()):
        runs = sorted((r for r in data['runs'] if r['arm'] == arm), key=lambda r:r['seed'])
        rows = sorted((r for r in data['per_run'] if r['arm'] == arm), key=lambda r:r['seed'])
        assert [r['seed'] for r in runs] == [r['seed'] for r in rows] == list(range(1, 21))
        losses = []
        for run in runs:
            metrics = sorted((m for m in run['metrics'] if m['epoch'] < 100), key=lambda m:m['epoch'])
            assert [m['epoch'] for m in metrics] == list(range(100))
            losses.append(np.minimum.accumulate([m['Archive/best_loss'] for m in metrics]))
        low, median, high = np.quantile(losses, [.25, .5, .75], axis=0)
        np.testing.assert_allclose(median[-1], data['convergence_endpoints']['epoch'][arm]['median'])
        epochs = np.arange(1, 101)
        axes[0, 0].step(epochs, median, where='post', color=color, lw=2, label=LABELS[i])
        axes[0, 0].fill_between(epochs, low, high, step='post', color=color, alpha=.12, lw=0)
        distribution(axes[0, 1], np.array([r['qualified_topologies'] for r in rows]), i, color)
        distribution(axes[1, 0], np.array([r['balance_fraction'] for r in rows])*100, i, color)
        motif = np.array([r['motif_fraction'] for r in rows])*100
        combined = np.array([r['combined_fraction'] for r in rows])*100
        assert (combined <= motif+1e-9).all()
        x = i+jitter
        axes[1, 1].plot(np.c_[x, x].T, np.array([combined, motif]), color=color, alpha=.22, lw=.7)
        axes[1, 1].scatter(x, motif, facecolors='none', edgecolors=color, s=26,
                           marker='D', linewidths=1, alpha=.85, zorder=3)
        axes[1, 1].scatter(x, combined, color=color, s=17, linewidths=0, alpha=.85, zorder=4)

    axes[0, 0].set(xlabel='Completed epochs', ylabel='Best loss', yscale='log', xlim=(0, 100))
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set(xticks=[0, 1], xticklabels=['Original prompt', 'Without mechanism\nsuggestions'], xlim=(-.5, 1.5))
    axes[0, 1].set(ylabel='Qualifying topologies per run', ylim=(0, None))
    for ax in axes[1]:
        ax.set(ylabel='Qualifying topologies (%)', ylim=(-4, 104), yticks=[0, 25, 50, 75, 100])
    for ax, title in zip(axes.flat, ['A  Convergence', 'B  Qualifying topologies',
                                     'C  Integral balance', 'D  Antithetic-like motifs']):
        ax.set_title(title, loc='left', pad=10)
        ax.grid(axis='y', color='#E8EBEE', lw=.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
        ax.tick_params(width=.7, length=3.5)
    axes[1, 1].legend([Line2D([], [], marker='D', mfc='white', color='#222222', ls='none', ms=5),
                       Line2D([], [], marker='o', color='#222222', ls='none', ms=5)],
                      ['Structural motif', 'Motif + integral balance'], loc='upper center',
                      bbox_to_anchor=(.5, -.26), ncol=2, frameon=False, fontsize=8.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .99), ncol=2, frameon=False)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(HERE / f'results_figure_proposal/figureS_rpa_prompt_ablation.{ext}',
                    dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
