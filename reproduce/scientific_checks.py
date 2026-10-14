"""Recompute principal published quantities from released scientific records."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np


def main(root, output):
    from reproduce import read, write, rows, resolve, module, csv_numbers, guard_source_access
    p = root/'analysis/biomolecular_full_duplex_rewrite'
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(p))
    guard_source_access()
    report = {}

    numerical = read(p/'numerical_comparison/frozen_runs.json')
    for run in numerical['runs']:
        folder = resolve(run['folder'])
        history = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [m['Compute/expensive_evaluations'] for m in history] == run['evaluations']
        assert [m['Archive/best_loss'] for m in history] == run['loss']
        first = next((x for x, y in zip(run['evaluations'], run['loss']) if y < numerical['threshold']), None)
        assert first == run['first_success_checkpoint']
        assert (first is not None) == run['successful']
    breadth = read(p/'genainet_comparison/audit.json')['runs']
    for run in breadth:
        folder = resolve(run['folder'])
        if run['arm'] == 'original':
            with (folder/'progress.csv').open() as f:
                history = list(csv.DictReader(f))
            cost = [float(m['ode_simulations']) for m in history]
            loss = [float(m['best_so_far_loss']) for m in history]
        else:
            history = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            history = sorted({int(m['epoch']): m for m in history if int(m['epoch']) < 100}.values(), key=lambda m: m['epoch'])
            done = read(folder/'completed.json')
            cost = [m['Compute/expensive_evaluations'] for m in history]+[done['counters']['expensive_evaluations']]
            loss = [m['Archive/best_loss'] for m in history]+[done['metrics']['Archive/best_loss']]
            if run.get('cost_accounting') == 'numerical_only_excluding_independent_llm':
                excluded = [m.get('Compute/evaluations/llm', 0) for m in history]+[done['counters'].get('evaluations/llm', 0)]
                assert cost == run['raw_cost'] and excluded == run['excluded_llm_evaluations']
                cost = [a-b for a, b in zip(cost, excluded)]
        assert cost == run['cost'], (run['task'], run['arm'], run['seed'])
        np.testing.assert_allclose(loss, run['loss'], rtol=1e-12)
    report['numerical_and_breadth'] = dict(numerical_runs=len(numerical['runs']), breadth_curves=len(breadth), matched=True)
    print('Numerical and cross-task convergence: raw histories match', flush=True)

    # Reconstruct all communication curves from original epoch/proposal records.
    c = module('release_convergence', p/'plot_convergence.py')
    history = c.history
    c.history = lambda folder: history(resolve(folder))
    actual, expected = c.reconstruct(), c.from_frozen()
    assert actual.keys() == expected.keys()
    for key in actual:
        for stream in actual[key]:
            np.testing.assert_allclose(actual[key][stream], expected[key][stream], rtol=1e-12)
    report['communication'] = dict(trajectories=len(actual), checkpoints_each=101, matched=True)
    print('Communication: 160 raw trajectories match', flush=True)

    # Recount Logic EIL topologies, descriptors and epoch discoveries.
    from RL4CRN.search_exploration.archive import descriptor
    from count_output_categories import Reaction
    with gzip.open(root/'data/diagnostics/logic/selection.json.gz', 'rt') as f:
        selection = json.load(f)
    library = {int(k): Reaction(v['structure']) for k, v in selection['catalog'].items()}
    audit = read(p/'eil_ablation/audit.json')
    expected_end = {(r['arm'], int(r['seed'])): r for r in csv_numbers(p/'eil_ablation/endpoints.csv')}
    expected_curves = {(r['arm'], int(r['seed']), int(r['completed_epochs'])): r
                       for r in csv_numbers(p/'eil_ablation/curves.csv')}
    for run in audit['runs']:
        folder = resolve(run['path'])
        threshold = read(folder/'manifest.json')['config']['quality_threshold']
        qualifying = {}
        for row in rows(folder, 'designs'):
            d = json.loads(row['payload'])
            if d.get('simulation_complete') and d.get('loss') is not None and d['loss'] < threshold:
                qualifying[d['candidate_hash']] = d
        first = {}
        for row in rows(folder, 'events'):
            if row.get('kind') != 'archive_insertion':
                continue
            d = json.loads(row['payload'])
            if d['candidate_hash'] in qualifying:
                first.setdefault(d['candidate_hash'], int(d['epoch']))
        assert qualifying.keys() <= first.keys()
        topologies = {}
        def add(d, epoch):
            ids = tuple(sorted(d['reaction_ids']))
            category = descriptor(ids, library, ['OUT'], 'reaction_roles_v1', '')
            if 'region' in d:
                assert category == d['region']
            old = topologies.get(ids)
            topologies[ids] = (category, min(epoch, old[1]) if old else epoch)
        for k, d in qualifying.items():
            add(d, first[k])
        for line in (folder/'direct_llm_proposals.jsonl').read_text().splitlines():
            row = json.loads(line)
            d = row.get('record')
            if d and d.get('simulation_complete') and d['loss'] < threshold:
                add(d, int(row['completed_epoch']))
        key = run['arm'], run['seed']
        endpoint = expected_end[key]
        assert len(topologies) == endpoint['qualified']
        assert len({v[0] for v in topologies.values()}) == endpoint['categories']
        terminal = read(folder/'terminal_union.json')
        assert terminal['union_best_loss'] == endpoint['loss']
        assert terminal['union_qualified'] == endpoint['qualified']
        for line in (folder/'metrics.jsonl').read_text().splitlines():
            m = json.loads(line)
            if m['epoch'] >= 100:
                continue
            target = expected_curves[key+(m['epoch']+1,)]
            assert m['Archive/best_loss'] == target['loss']
            assert m['Compute/expensive_evaluations'] == target['evaluations']
            assert sum(v[1] <= m['epoch'] for v in topologies.values()) == target['qualified']
    e = module('release_eil', p/'plot_eil_ablation.py')
    statistics = e.summarize(list(expected_end.values()))
    assert statistics == read(p/'eil_ablation/summary.json')
    report['eil'] = dict(runs=40, endpoint_counts_and_epoch_curves_match=True,
                         statistics_match=True)
    print('EIL: 40 portfolios and their discovery histories match', flush=True)

    # Audit canonical reduced structures and perturbation membership counts.
    from comparisons.search_exploration.plot_logic5_canonical_topology_counts import audit as canonical_audit
    logic = root/'data/diagnostics/logic'
    # Relocation changes a pathname, not the bytes/hash of the frozen selection.
    # The original audit indexes an absolute path. Supply the original key by
    # translating only that dictionary lookup, retaining the original plan bytes.
    original_loads = json.loads
    class HashPaths(dict):
        def __getitem__(self, key):
            if key not in self and str(key).endswith('/selection.json.gz'):
                matches = [v for k, v in self.items() if k.endswith('/selection.json.gz')]
                assert len(matches) == 1
                return matches[0]
            return super().__getitem__(key)
    def relocated_loads(*args, **kwargs):
        d = original_loads(*args, **kwargs)
        if isinstance(d, dict) and 'input_hashes' in d:
            d['input_hashes'] = HashPaths(d['input_hashes'])
        return d
    try:
        json.loads = relocated_loads
        canonical = canonical_audit(logic)
    finally:
        json.loads = original_loads
    reference = read(root/'analysis/jmlr_hybrid_crn/generated/canonical_audit.json')
    for key in ['rows', 'pooled', 'global_counts', 'records_digest', 'discrepancies']:
        assert canonical[key] == reference[key], key
    report['logic_reduction'] = dict(portfolios=len(canonical['rows']),
        unique_originals=canonical['raw_hashes'], unique_reduced=canonical['core_hashes'], matched=True)
    print('Logic: canonical reduction and robustness counts match', flush=True)

    from comparisons.search_exploration.analyze_rpa_mechanisms import reactions_for, integral_balance
    from comparisons.search_exploration.analyze_rpa_communication_families import antithetic_motif, aif_like
    with gzip.open(root/'data/diagnostics/rpa/selection.json.gz', 'rt') as f:
        rpa = json.load(f)
    catalog = {int(k): v for k, v in rpa['catalog'].items()}
    def screen(records):
        flags = []
        for d in records:
            reactions = reactions_for(d, catalog, rpa['fixed'])
            balance = integral_balance(reactions)
            flags.append([antithetic_motif(reactions), balance['restricted_exact_balance'],
                          aif_like(reactions, balance)])
        return np.mean(flags, axis=0)
    with (p/'results_figure_proposal/rpa_integral_screen.csv').open() as f:
        targets = {(v['arm'], int(v['seed'])): v for v in csv.DictReader(f)}
    with gzip.open(root/'data/diagnostics/rpa/extracted.json.gz', 'rt') as f:
        runs = json.load(f)
    for run in runs:
        target = targets[run['arm'], run['seed']]
        assert len(run['qualified']) == int(target['qualified_topologies'])
        np.testing.assert_allclose(screen(run['qualified']), [float(target[k]) for k in
            ['antithetic_motif_fraction', 'exact_balance_fraction', 'aif_like_fraction']], rtol=1e-12)
    prompt = read(p/'prompt_ablation/final_20seed/snapshot.json')
    targets = {(v['arm'], v['seed']): v for v in prompt['per_run']}
    for run in prompt['runs']:
        folder = resolve(run['folder'])
        threshold = read(folder/'manifest.json')['config']['quality_threshold']
        best = {}
        for row in rows(folder, 'designs'):
            d = json.loads(row['payload'])
            if not d.get('simulation_complete') or d.get('loss') is None or not 0 <= d['loss'] < threshold:
                continue
            key = d['topology_hash']
            old = best.get(key)
            if old is None or (d['loss'], d['candidate_hash']) < (old['loss'], old['candidate_hash']):
                best[key] = d
        target = targets[run['arm'], run['seed']]
        assert len(best) == target['qualified_topologies']
        np.testing.assert_allclose(screen(best.values()), [target[k] for k in
            ['motif_fraction', 'balance_fraction', 'combined_fraction']], rtol=1e-12)
    report['rpa_screens'] = dict(communication_portfolios=len(runs), prompt_portfolios=len(prompt['runs']), matched=True)
    print('RPA: structural/balance screens and prompt ablation match', flush=True)

    parity = read(p/'parity_performance/qualification_snapshot.json')
    expected = {(d['seed'], tuple(d['reaction_ids'])): d for d in parity['topologies']}
    checked = 0
    for run in parity['runs']:
        best = {}
        for row in rows(resolve(run['folder']), 'designs'):
            d = json.loads(row['payload'])
            if not d.get('simulation_complete') or d.get('loss') is None or not 0 <= d['loss'] < parity['loss_threshold']:
                continue
            ids = tuple(sorted(d['reaction_ids']))
            old = best.get(ids)
            if old is None or (d['loss'], d['candidate_hash']) < (old['loss'], old['candidate_hash']):
                best[ids] = d
        assert len(best) == run['qualifying_topologies']
        for ids, d in best.items():
            target = expected[run['seed'], ids]
            assert d['candidate_hash'] == target['candidate_hash'] and d['loss'] == target['loss']
            assert d['all_rows_correct'] == target['all_rows_correct']
            if target['diagnostics_available']:
                inputs = np.array(target['inputs'])
                targets = np.array(target['targets'])
                outputs = np.array(target['terminal_outputs'])
                np.testing.assert_array_equal(inputs.sum(axis=1) % 2, targets)
                correct = (2*targets-1)*(outputs-parity['classification_threshold']) > parity['margin']
                assert int(correct.sum()) == target['correct_rows']
                assert bool(correct.all()) == d['all_rows_correct']
            checked += 1
    assert checked == len(expected)
    report['parity'] = dict(runs=len(parity['runs']), qualifying_topologies=checked, matched=True)
    print('Parity: qualifying designs and classification counts match', flush=True)

    totals = defaultdict(Counter)
    tools = defaultdict(Counter)
    sessions = 0
    with gzip.open(root/'data/llm_usage/session_events.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            g = r['group']
            sessions += 1
            for item in r['usage']:
                totals[g].update({k: v or 0 for k, v in item['usage'].items()})
                totals[g]['usage_records'] += 1
            for item in r['tools']:
                tools[g][item['name']] += 1
                tools[g]['calls'] += 1
                tools[g]['calls_without_result'] += not item['result_recorded']
            tools[g]['results_without_call'] += r['orphan_results']
    for target in read(p/'token_costs/summary.json')['campaigns']:
        g = target['group']
        for key in ['inputTokens', 'cacheReadTokens', 'outputTokens', 'reasoningTokens', 'cacheWriteTokens', 'usage_records']:
            assert totals[g][key] == target[key], (g, key)
        if target['kind'] == 'flash':
            cost = sum(totals[g][k]*v/1e6 for k, v in
                       {'inputTokens': .15, 'cacheReadTokens': .003, 'outputTokens': .60}.items())
            np.testing.assert_allclose(cost, target['cost_offpeak_usd'], rtol=1e-12)
    summary = read(p/'token_costs/tool_calls_summary.json')
    for target in summary['campaigns']:
        for key in summary['tools'] + ['calls', 'calls_without_result', 'results_without_call']:
            assert tools[target['group']][key] == target[key], (target['group'], key)
    report['llm_usage'] = dict(sessions=sessions, campaigns=len(totals),
        tool_calls=sum(t['calls'] for t in tools.values()), matched=True)
    print('LLM: token and tool-event totals match', flush=True)
    write(output/'scientific_checks.json', report)


if __name__ == '__main__':
    raise SystemExit('Use python reproduce.py analyze')
