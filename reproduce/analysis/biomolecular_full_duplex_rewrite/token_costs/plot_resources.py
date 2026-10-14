"""Publication figures from frozen token and tool-call aggregates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
TOKEN_COLORS = ['#2F718E', '#B8D7E1', '#B77B42']
COST_COLORS = ['#74628F', '#DAD3E3']
TOOL_NAMES = ['read', 'write', 'glob', 'grep', 'bash', 'edit', 'Other']
TOOL_COLORS = ['#2F718E', '#B77B42', '#8B709C', '#559B8C', '#788CA9', '#C47F78', '#C8C8C3']


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=12)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E6E8EA', linewidth=.6)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.tick_params(axis='x', length=3, color='#777777')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    for edge in [2.5, 6.5, 11.5, 13.5]:
        ax.axhline(edge, color='#D3D6D8', lw=.6)


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=240)
    plt.close(fig)


def plot_tools(tokens, tool_data):
    calls = {r['group']: r for r in tool_data['campaigns']}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 8.3), sharey=True)
    fig.subplots_adjust(left=.30, right=.985, bottom=.09, top=.86, wspace=.14)
    y = np.arange(len(tokens))
    for ax, divisor, title in zip(axes, [1, 20], ['A  Total over 20 runs', 'B  Mean per run']):
        left = np.zeros(len(tokens))
        for name, color in zip(TOOL_NAMES, TOOL_COLORS):
            values = np.array([
                calls[r['group']][name] if name != 'Other' else
                sum(calls[r['group']][t] for t in tool_data['tools'] if t not in TOOL_NAMES)
                for r in tokens
            ])/divisor
            ax.barh(y, values, left=left, height=.64, color=color,
                    edgecolor='white', linewidth=.25)
            left += values
        for i, value in enumerate(left):
            label = f'{value:,.0f}' if divisor == 1 else f'{value:,.1f}'
            ax.text(value + left.max()*.02, i, label, va='center', fontsize=11)
        ax.set_xlim(0, left.max()*1.25)
        ax.set_xlabel('Tool calls')
        style(ax, title)
    axes[0].set_yticks(y, [r['label'] for r in tokens])
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(TOOL_COLORS, TOOL_NAMES)],
               loc='upper center', bbox_to_anchor=(.64, .985), ncol=7,
               frameon=False, fontsize=11, handlelength=1.1, columnspacing=.8)
    save(fig, 'llm_resources_tools')


def plot_tokens_cost(tokens):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 12.5), sharey=True)
    fig.subplots_adjust(left=.30, right=.985, bottom=.06, top=.91, wspace=.15, hspace=.20)
    y = np.arange(len(tokens))
    for row, divisor in enumerate([1, 20]):
        token_ax, cost_ax = axes[row]
        left = np.zeros(len(tokens))
        for field, color in zip(['inputTokens', 'cacheReadTokens', 'outputTokens'], TOKEN_COLORS):
            values = np.array([r[field] for r in tokens])/divisor/1e6
            token_ax.barh(y, values, left=left, height=.65, color=color,
                          edgecolor='white', linewidth=.25)
            left += values
        for i, value in enumerate(left):
            token_ax.text(value + left.max()*.02, i, f'{value:.2f}', va='center', fontsize=11)
        token_ax.set_xlim(0, left.max()*1.25)
        token_ax.set_xlabel('Tokens (millions)')
        maximum = max(r['cost_peak_usd'] or 0 for r in tokens)/divisor
        for i, r in enumerate(tokens):
            if r['cost_offpeak_usd'] is None:
                cost_ax.text(maximum*.02, i, 'Not estimated', color='#707078',
                             va='center', fontsize=10)
                continue
            low, high = r['cost_offpeak_usd']/divisor, r['cost_peak_usd']/divisor
            cost_ax.barh(i, low, height=.65, color=COST_COLORS[0])
            cost_ax.barh(i, high-low, left=low, height=.65, color=COST_COLORS[1])
            cost_ax.plot([high, high], [i-.325, i+.325], color=COST_COLORS[0], lw=.8)
            cost_ax.text(high + maximum*.02, i, f'{low:.2f}–{high:.2f}',
                         va='center', fontsize=11)
        cost_ax.set_xlim(0, maximum*1.52)
        cost_ax.set_xlabel('Estimated API cost (USD)')
        suffix = 'Total over 20 runs' if row == 0 else 'Mean per run'
        style(token_ax, f'{"A" if row == 0 else "C"}  {suffix}')
        style(cost_ax, f'{"B" if row == 0 else "D"}  {suffix}')
        token_ax.set_yticks(y, [r['label'] for r in tokens])
    axes[0, 0].invert_yaxis()
    handles = [Patch(color=c, label=l) for c, l in zip(TOKEN_COLORS,
               ['Other input', 'Cached input', 'Output incl. reasoning'])]
    handles += [Patch(color=c, label=l) for c, l in zip(COST_COLORS,
                ['Off-peak cost', 'Peak increment'])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.64, .992),
               ncol=3, frameon=False, fontsize=11, handlelength=1.3)
    save(fig, 'llm_resources_tokens_cost')


if __name__ == '__main__':
    tokens = json.loads((OUT / 'summary.json').read_text())['campaigns']
    tool_data = json.loads((OUT / 'tool_calls_summary.json').read_text())
    calls = {r['group']: r for r in tool_data['campaigns']}
    assert len(tokens) == len(calls) == 17
    for r in tokens:
        assert r['runs'] == calls[r['group']]['runs'] == 20
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 12,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    plot_tools(tokens, tool_data)
    plot_tokens_cost(tokens)
