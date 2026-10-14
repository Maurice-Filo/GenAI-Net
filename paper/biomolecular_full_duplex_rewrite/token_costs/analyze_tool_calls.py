"""Count authoritative harness tool/call events in the saved study cohorts.

Run from the repository with .venv/bin/python; --cached only redraws figures.
Source session logs and manuscript files are never modified.
"""
import argparse
import csv
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def read_csv(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse(row):
    path = Path(row['file'])
    proc = None
    if path.suffix == '.jsonl':
        stream = path.open('rb')
    else:
        proc = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stream = proc.stdout
    calls = {}
    results = set()
    duplicate_calls = 0
    with stream:
        for line in stream:
            if b'"tool/call"' not in line and b'"tool/result"' not in line:
                continue
            event = json.loads(line)
            data = event.get('data', {})
            if event.get('type') == 'tool/call':
                key = (data.get('turn'), data.get('step'), data['callId'])
                if key in calls:
                    assert calls[key] == data['name'], path
                    duplicate_calls += 1
                calls[key] = data['name']
            elif event.get('type') == 'tool/result':
                call_id = data.get('message', {}).get('source', {}).get('callId')
                if call_id is not None:
                    results.add((data.get('turn'), data.get('step'), call_id))
    if proc is not None:
        err = proc.stderr.read()
        assert proc.wait() == 0, (path, err)
        proc.stderr.close()
    return dict(group=row['group'], run=row['run'], seed=int(row['seed']),
                session=row['session'], counts=dict(Counter(calls.values())),
                calls=len(calls), calls_without_result=len(set(calls) - results),
                results_without_call=len(results - set(calls)),
                duplicate_calls=duplicate_calls)


def extract():
    sources = read_csv('sessions.csv')
    assert len({r['session'] for r in sources}) == len(sources)
    sessions = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = [pool.submit(parse, row) for row in sources]
        for n, future in enumerate(as_completed(pending), 1):
            sessions.append(future.result())
            if n % 500 == 0:
                print(f'{n}/{len(sources)} sessions processed', flush=True)
    sessions.sort(key=lambda r: (r['group'], r['seed'], r['session']))
    tools = sorted({tool for s in sessions for tool in s['counts']},
                   key=lambda tool: -sum(s['counts'].get(tool, 0) for s in sessions))
    runs = {}
    for r in read_csv('runs.csv'):
        runs[r['run']] = dict(group=r['group'], label=r['label'], kind=r['kind'],
                             run=r['run'], seed=int(r['seed']),
                             missing_workspaces=int(r['missing_workspaces']),
                             calls=0, calls_without_result=0, results_without_call=0,
                             **{t: 0 for t in tools})
    for s in sessions:
        r = runs[s['run']]
        for k in ('calls', 'calls_without_result', 'results_without_call'):
            r[k] += s[k]
        for t, count in s['counts'].items():
            r[t] += count
    campaigns = []
    for c in read_csv('campaigns.csv'):
        rr = [r for r in runs.values() if r['group'] == c['group']]
        assert len(rr) == 20 and len({r['seed'] for r in rr}) == 20
        campaigns.append(dict(group=c['group'], label=c['label'], kind=c['kind'], runs=len(rr),
                              **{k: sum(r[k] for r in rr) for k in
                                 ['calls', 'calls_without_result', 'results_without_call',
                                  'missing_workspaces', *tools]}))
    for r in [*runs.values(), *campaigns]:
        assert r['calls'] == sum(r[t] for t in tools)
    assert sum(c['calls'] for c in campaigns) == sum(s['calls'] for s in sessions)
    flat_sessions = [{k: v for k, v in s.items() if k != 'counts'} |
                     {t: s['counts'].get(t, 0) for t in tools} for s in sessions]
    write_csv('tool_calls_sessions.csv', flat_sessions)
    write_csv('tool_calls_runs.csv', list(runs.values()))
    write_csv('tool_calls_campaigns.csv', campaigns)
    data = dict(tools=tools, campaigns=campaigns, sessions=len(sessions),
                notes=[
                    'Same final cohorts and session mappings as the token/cost analysis: 20 runs per campaign.',
                    'Count tool/call events, deduplicated by session, turn, step and callId. Streamed assistant tool-call chunks and tool/result events are not counted as extra calls.',
                    'Calls include unsuccessful or interrupted tool invocations. A matched result does not necessarily indicate success.',
                    'All retained sessions in the selected runs are included, including failed proposal batches. Missing workspace/session logs are not imputed.',
                    'These are agent harness tool invocations, not canonical numerical evaluations or ODE simulations.',
                    'Per-run bars show arithmetic means over all 20 runs, including runs with no recorded calls. Campaign plots show sums.',
                    'Tool arguments and outputs are not copied into these aggregate files.'
                ])
    (OUT / 'tool_calls_summary.json').write_text(json.dumps(data, indent=2) + '\n')
    return data


def render(data, per_run=True):
    campaigns, tools = data['campaigns'], data['tools']
    displayed = [(tool, [tool]) for tool in tools[:6]]
    if len(tools) > 6:
        displayed.append(('Other', tools[6:]))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11.8, 8.5))
    fig.subplots_adjust(left=.30, right=.97, bottom=.10, top=.90)
    colors = ['#287da8', '#ba803b', '#943c87', '#58a48b', '#7e8ca8', '#ca7572',
              '#9aa65a', '#b1d4df', '#877059', '#d2b4d7']
    y = np.arange(len(campaigns))
    left = np.zeros(len(y))
    denominator = np.array([c['runs'] if per_run else 1 for c in campaigns])
    for i, (tool, members) in enumerate(displayed):
        values = np.array([sum(c[t] for t in members) for c in campaigns]) / denominator
        ax.barh(y, values, left=left, color=colors[i % len(colors)],
                label=tool, height=.66, edgecolor='white', linewidth=.35)
        left += values
    for i, total in enumerate(left):
        label = f'{total:,.1f}' if per_run else f'{total:,.0f}'
        ax.text(total + max(left)*.013, i, label, va='center', fontsize=9)
    ax.set_yticks(y, [c['label'] for c in campaigns])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left)*1.14)
    ax.set_xlabel('Mean recorded tool calls per run' if per_run else
                  'Recorded tool calls per 20-run campaign')
    ax.tick_params(axis='y', length=0)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)
    for edge in (2.5, 6.5, 11.5, 13.5):
        ax.axhline(edge, color='.8', linewidth=.6)
    fig.legend(loc='upper center', bbox_to_anchor=(.64, .975),
               ncol=len(displayed), frameon=False, title='Harness tool')
    name = 'tool_calls_per_run' if per_run else 'tool_calls_by_campaign'
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    data = json.loads((OUT / 'tool_calls_summary.json').read_text()) if args.cached else extract()
    render(data)
    render(data, per_run=False)
    print('Total calls:', sum(c['calls'] for c in data['campaigns']))
    print('Tools:', data['tools'])
    for c in data['campaigns']:
        print(c['label'], c['calls'], 'calls;', round(c['calls']/c['runs'], 1), 'per run')
