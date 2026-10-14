"""Reconstruct saved, observed best-so-far losses; no simulation or model calls.

Run from the repository root with .venv/bin/python. Frozen snapshots support
subsequent replotting with --frozen, without access to the original run folders.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / 'jmlr_hybrid_crn'
ARMS = ('none', 'llm-to-rl', 'rl-to-llm', 'full')
LABELS = ('No communication', 'LLM to search', 'Search to LLM', 'Full duplex')
COLORS = ('#737373', '#cf7c24', '#277da8', '#973c85')
TASKS = ('rpa', 'logic')
SOURCES = {}


def read(path, lines=False):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)


def loss(value):
    if value is None:
        return np.inf
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'Invalid saved loss: {value}')
    return value


def same(a, b):
    if not np.isclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError(f'Endpoint mismatch: {a} != {b}')


def history(folder):
    completed = read(folder / 'completed.json')
    assert completed['status'] == 'complete'
    metrics = read(folder / 'metrics.jsonl', lines=True)
    assert set(m['epoch'] for m in metrics) == set(range(101))
    assert all(a['epoch'] <= b['epoch'] for a, b in zip(metrics, metrics[1:]))
    # Index 0 = one completed training epoch; index 100 = terminal draining.
    search, direct = np.full(101, np.inf), np.full(101, np.inf)
    bank = folder / 'direct_llm_proposals.jsonl'
    prefix = None
    if bank.exists():
        rows = read(bank, lines=True)
        assert all(r['direct_supplied_rate_only'] for r in rows)
        records = [r['record'] for r in rows if r['record'] is not None]
        values = [loss(r['loss']) if r['simulation_complete'] else np.inf for r in records]
        prefix = np.r_[np.inf, np.minimum.accumulate(values)]
        counts = [m['LLM/direct_evaluated_records'] for m in metrics]
        assert all(a <= b for a, b in zip(counts, counts[1:]))
        assert counts[-1] == len(records)
    for m in metrics:
        i = m['epoch']
        search[i] = min(search[i], loss(m['Archive/best_loss']))
        if prefix is not None:
            direct[i] = min(direct[i], prefix[m['LLM/direct_evaluated_records']])
    search, direct = np.minimum.accumulate(search), np.minimum.accumulate(direct)
    assert np.all(np.isfinite(search))
    same(search[-1], loss(metrics[-1]['Archive/best_loss']))
    union = np.minimum(search, direct)
    terminal = folder / 'terminal_union.json'
    if terminal.exists():
        end = read(terminal)
        same(search[-1], end['search_best_loss'])
        same(direct[-1], end['llm_best_loss'])
        same(union[-1], end['union_best_loss'])
    else:
        same(union[-1], completed['metrics']['Archive/best_loss'])
    return search, direct, union


def reconstruct():
    channels = read(PAPER / 'generated/channels.json')['rows']
    logic = read(PAPER / 'generated/logic_endpoints.json')
    rpa = read(PAPER / 'generated/rpa_communication.json')
    expected_logic = {(r['arm'], r['seed']): r['loss'] for r in logic['endpoints']}
    cache, curves = {}, {}

    def get(folder):
        folder = Path(folder)
        if folder not in cache:
            cache[folder] = history(folder)
        return cache[folder]

    for task in TASKS:
        for arm in ARMS:
            rows = sorted((r for r in channels if r['task'] == task and r['arm'] == arm),
                          key=lambda r: r['seed'])
            assert [r['seed'] for r in rows] == list(range(1, 21))
            for r in rows:
                search, direct, union = get(r['folder'])
                if task == 'rpa' and arm == 'none':
                    # Direct bank only: never use the donor run's search archive.
                    _, direct, _ = get(Path(r['llm_bank']).parent)
                    union = np.minimum(search, direct)
                if task == 'logic':
                    same(union[-1], expected_logic[arm, r['seed']])
                curves[task, arm, r['seed']] = dict(search=search, direct=direct, union=union)
        if task == 'rpa':
            for arm in ARMS:
                label = 'independent-union' if arm == 'none' else arm
                target = next(r['median_best_loss'] for r in rpa['arms'] if r['arm'] == label)
                same(np.median([curves[task, arm, s]['union'][-1] for s in range(1, 21)]), target)
    return curves


def write_csv(path, rows):
    with path.open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze(curves):
    rows = []
    for (task, arm, seed), streams in curves.items():
        for i in range(101):
            rows.append(dict(task=task, arm=arm, seed=seed,
                             completed_epochs=str(i + 1) if i < 100 else 'T',
                             **{k: float(v[i]) if np.isfinite(v[i]) else '' for k, v in streams.items()}))
    write_csv(HERE / 'convergence/snapshots.csv', rows)
    (HERE / 'convergence/input_hashes.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


def from_frozen():
    curves = {}
    with (HERE / 'convergence/snapshots.csv').open() as src:
        for row in csv.DictReader(src):
            key = row['task'], row['arm'], int(row['seed'])
            if key not in curves:
                curves[key] = {k: np.full(101, np.inf) for k in ('search', 'direct', 'union')}
            i = 100 if row['completed_epochs'] == 'T' else int(row['completed_epochs']) - 1
            for k in curves[key]:
                curves[key][k][i] = float(row[k]) if row[k] else np.inf
    assert len(curves) == 160
    return curves


def summarize(values):
    # Direct-proposal histories can be unavailable early. Retain infinity in
    # the cohort instead of conditioning on the seeds whose LLM returned first.
    ordered = np.sort(values, axis=0)
    median = (ordered[9] + ordered[10]) / 2
    # Empirical quartile order statistics avoid interpolation involving infinity.
    return median, ordered[4], ordered[14]


def plot(curves, stream, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    for ax, task in zip(axes, TASKS):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            values = np.array([curves[task, arm, seed][stream] for seed in range(1, 21)])
            median, lo, hi = summarize(values)
            style = '--' if task == 'rpa' and arm == 'none' else '-'
            shown = label
            ax.step(np.arange(1, 101), np.where(np.isfinite(median[:100]), median[:100], np.nan),
                    where='post', color=color, lw=2, ls=style, label=shown)
            valid = np.isfinite(lo[:100]) & np.isfinite(hi[:100])
            ax.fill_between(np.arange(1, 101), lo[:100], hi[:100], where=valid,
                            step='post', color=color, alpha=.10)
            ax.plot(106, median[-1], 'o', color=color, ms=5)
            ax.vlines(106, lo[-1], hi[-1], color=color, alpha=.55, lw=1.3)
        ax.axvline(103, color='#aaaaaa', lw=.7, ls=':')
        ax.set_yscale('log')
        ax.set(xlim=(0, 110), xticks=[0, 25, 50, 75, 100, 106],
               xticklabels=['0', '25', '50', '75', '100', 'T'],
               xlabel='Completed search epochs; T = terminal drain',
               ylabel='Best-so-far canonical loss (median across 20 runs)')
        ax.set_title('RPA: adaptation across input / disturbance conditions' if task == 'rpa'
                     else 'Logic: four-input Boolean function', fontsize=11, loc='left')
        ax.grid(axis='y', alpha=.16, which='major')
        ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    fig.suptitle(title, fontsize=15, y=.98)
    fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.85, wspace=.26)
    fig.savefig(HERE / f'convergence/{filename}.png', dpi=190)
    fig.savefig(HERE / f'convergence/{filename}.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', action='store_true')
    args = parser.parse_args()
    (HERE / 'convergence').mkdir(parents=True, exist_ok=True)
    curves = from_frozen() if args.frozen else reconstruct()
    if not args.frozen:
        freeze(curves)
    rows, checks = [], []
    for task in TASKS:
        for arm in ARMS:
            for stream in ('union', 'search'):
                values = np.array([curves[task, arm, s][stream] for s in range(1, 21)])
                assert np.all(np.isfinite(values)) and np.all(values > 0)
                assert np.all(np.diff(values, axis=1) <= 1e-12)
                median, lo, hi = summarize(values)
                for i in range(101):
                    row = dict(task=task, arm=arm, stream=stream,
                               completed_epochs=str(i + 1) if i < 100 else 'T',
                               median=float(median[i]), q25=float(lo[i]), q75=float(hi[i]), n=20)
                    rows.append(row)
                    if i in (0, 9, 24, 49, 74, 99, 100):
                        checks.append(row)
    write_csv(HERE / 'convergence/aggregate_curves.csv', rows)
    (HERE / 'convergence/checkpoints.json').write_text(json.dumps(checks, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    plot(curves, 'union', 'Best available design across both search processes', 'communication_convergence')
    plot(curves, 'search', 'Best design in the numerical search archive', 'search_archive_convergence')
    print('Validated 160 seed trajectories; endpoint checks passed. No simulations or model calls.')
    for r in checks:
        if r['completed_epochs'] in ('25', '50', 'T'):
            print(r['task'], r['arm'], r['stream'], r['completed_epochs'], f"{r['median']:.7f}")


if __name__ == '__main__':
    main()
