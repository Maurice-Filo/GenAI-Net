"""Frozen RPA communication-portfolio subset and kinetic diagnostics.

Uses the Logic portfolio protocol with RPA's existing task qualification rule.
No search, refitting, model calls, external logging, or source-database writes.
"""

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import fcntl
import gzip
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from RL4CRN.search_exploration.engine import atomic_json
from comparisons.search_exploration.run_rpa_perturbations import directions, perturb, RADII
from comparisons.search_exploration.run_logic5_portfolio_diagnostics import candidate_key
from comparisons.search_exploration.analyze_rpa_mechanisms import (
    cached_function, structural_token, write_csv,
)

ARMS = ('control', 'full', 'llm-to-rl', 'rl-to-llm', 'independent-union')
LABELS = ('No LLM', 'Full duplex', 'Insertion only', 'Return only', 'Independent union*')
ATLAS = Path('/local1/rossin/search-exploration/rpa-llm-interaction-20260918/topology-families-final20-r2')
OLD = Path('/local1/rossin/search-exploration/rpa-rate-perturbations-20260917')
LOGIC = Path('/local1/rossin/search-exploration/logic5-portfolio-diagnostics-order-fallback-20260924')
PLAN = EVALUATOR = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural_identity(records):
    encoded = json.dumps(sorted(structural_token(r) for r in records), separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


def select_core(subsets, epsilon, threshold):
    eligible = [r for r in subsets if r['status'] == 'ok' and r['loss'] < threshold
                and r['max_output_deviation'] <= epsilon]
    if not eligible:
        raise ValueError('The validated original must be an eligible subset')
    return min(eligible, key=lambda r: (r['retained_count'], r['structure_json']))


def prepare(output, workers):
    """Freeze audited best-per-topology portfolios and all exact-vector provenance."""
    from RL4CRN.search_exploration.evaluation import build
    from RL4CRN.utils.results_database import serialize_crn
    from comparisons.search_exploration.analyze_rpa_llm_provenance import connect
    from comparisons.search_exploration.analyze_logic5_robust_provenance import route
    for root in (OLD, LOGIC):
        if sha(root / 'plan.json') != (root / 'plan.sha256').read_text().strip():
            raise ValueError(f'Prior protocol hash mismatch: {root}')
        if not (root / 'completed.json').exists():
            raise ValueError(f'Prior study incomplete: {root}')
    original, logic = read(OLD / 'plan.json'), read(LOGIC / 'plan.json')
    audit = read(ATLAS / 'inputs.json')
    inputs = dict(audit['files'])
    for bank in audit['runs'].values():
        inputs.update(bank)
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f'Audited input changed: {path}')
    with gzip.open(ATLAS / 'extracted.json.gz', 'rt') as stream:
        portfolios = json.load(stream)
    if {(r['arm'], r['seed']) for r in portfolios} != set(itertools.product(ARMS, range(1, 21))):
        raise ValueError('Require all 100 audited arm/seed portfolios')
    evaluator, _, _ = build('rpa', original['task_config'], .001, 100.)
    catalog = {}
    for i in range(len(evaluator.library)):
        reaction = evaluator.library.get_reaction(i)
        catalog[str(i)] = {'reaction_id': i, 'structure': {'type': type(reaction).__name__,
            'reactants': list(reaction.reactant_labels), 'products': list(reaction.product_labels),
            'inputs': list(reaction.input_channels)}}
    if len(catalog) != 91 or not np.array_equal(original['directions'], directions()):
        raise ValueError('RPA reaction library or perturbation directions changed')
    fixed = json.loads(serialize_crn(evaluator.crn_template)['parameters_json'])
    threshold = original['threshold']
    candidates, runs = {}, []
    folders = {(r['arm'], r['seed']): Path(r['folder']) for r in portfolios}
    for run in portfolios:
        arm, seed, folder = run['arm'], run['seed'], Path(run['folder'])
        manifest = read(folder / 'manifest.json')
        if manifest['task_config']['rpa'] != original['task_config']['rpa']:
            raise ValueError('RPA evaluation configuration differs from previous diagnostics')
        members = []
        for rank, c in enumerate(run['qualified'], 1):
            ids, rates = c['reaction_ids'], c['rates']
            if (len(ids) != 5 or len(set(ids)) != 5 or set(ids) & {0, 1, 28}
                    or len(rates) != 5 or not all(.001 <= k <= 100 for k in rates)
                    or not c['simulation_complete'] or not 0 <= c['loss'] < threshold):
                raise ValueError('Invalid selected qualifying design')
            key = candidate_key(ids, rates)
            actions = sorted(zip(ids, rates))
            if key not in candidates:
                candidates[key] = {'id': key, 'reaction_ids': [i for i, _ in actions],
                    'rates': [k for _, k in actions], 'recorded_losses': [], 'saved_orders': [],
                    'winner': False, 'topology_hash': c['topology_hash']}
            selected = candidates[key]
            if selected['topology_hash'] != c['topology_hash']:
                raise ValueError('Exact vector has inconsistent original topology identity')
            selected['recorded_losses'] = sorted(set(selected['recorded_losses'] + [c['loss']]))
            selected['saved_orders'] = sorted(map(list, set(map(tuple, selected['saved_orders'] + [ids]))))
            selected['winner'] |= rank == 1
            members.append({'candidate': key, 'candidate_hash': c['candidate_hash'],
                'loss': c['loss'], 'rank': rank, 'topology_hash': c['topology_hash']})
        if not members or not np.isclose(members[0]['loss'], run['best']['loss'], rtol=1e-12, atol=0):
            raise ValueError('Selected winner does not match audited endpoint')
        # The union's numerical events come only from the historical control;
        # its direct bank comes only from insertion-only, never that arm's search.
        keys = {m['candidate_hash'] for m in members}
        evidence = defaultdict(list)
        with connect(folder) as db:
            params = tuple(sorted(keys))
            marks = ','.join('?' for _ in params)
            for eid, payload in db.execute(
                    f"SELECT id,payload FROM events WHERE kind='archive_insertion' "
                    f"AND json_extract(payload,'$.candidate_hash') IN ({marks}) ORDER BY id", params):
                event = json.loads(payload)
                evidence[event['candidate_hash']].append({'kind': 'search_event', 'event_id': eid,
                    'route': route(event), 'loss': event['loss'], 'parent_hash': event.get('parent_hash')})
        direct_folder = folders['llm-to-rl', seed] if arm == 'independent-union' else folder
        bank = direct_folder / 'direct_llm_proposals.jsonl'
        if bank.exists() and arm != 'control':
            inputs[str(bank)] = sha(bank)
            for line in bank.read_text().splitlines():
                entry = json.loads(line)
                c = entry.get('record')
                if c and c['candidate_hash'] in keys:
                    if not entry['direct_supplied_rate_only'] or not c['simulation_complete']:
                        raise ValueError('Invalid direct-proposal provenance')
                    evidence[c['candidate_hash']].append({'kind': 'direct_bank', 'route': 'llm_direct',
                        'loss': c['loss'], 'completed_epoch': entry['completed_epoch']})
        for member in members:
            ev = evidence[member['candidate_hash']]
            if not ev or any(not np.isclose(e['loss'], member['loss'], rtol=1e-7, atol=1e-7) for e in ev):
                raise ValueError(f'Missing or inconsistent exact-vector provenance: {arm}/{seed}')
            member['evidence'] = ev
            member['routes'] = sorted({e['route'] for e in ev})
        runs.append({'arm': arm, 'seed': seed, 'folder': str(folder), 'members': members})
        print(f'Validated {arm}/{seed}: {len(members)} qualifying vectors', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('records', 'analysis'):
        (output / name).mkdir()
    selection = {'runs': runs, 'candidates': candidates, 'catalog': catalog, 'fixed': fixed}
    with gzip.open(output / 'selection.json.gz', 'wt') as stream:
        json.dump(selection, stream, allow_nan=False)
    for p in (ATLAS / 'inputs.json', ATLAS / 'extracted.json.gz', OLD / 'plan.json',
              LOGIC / 'plan.json', output / 'selection.json.gz'):
        inputs[str(p)] = sha(p)
    source_root = Path(__file__).resolve().parents[2]
    if not (source_root / 'SNAPSHOT.json').exists():
        raise ValueError('Prepare and execute from a frozen source snapshot')
    plan = {'schema': 'rpa-communication-portfolio-diagnostics-v1', 'created': time.time(),
        'workers': workers, 'task_config': original['task_config'], 'threshold': threshold,
        'directions': original['directions'], 'winner_radii': list(RADII), 'portfolio_radius': 1.1,
        'epsilons': logic['epsilons'], 'primary_epsilon': logic['primary_epsilon'],
        'replay_tolerance': logic['replay_tolerance'], 'diagnostic_bounds': original['diagnostic_bounds'],
        'robust_diagnostic_cutoff': logic['robust_diagnostic_cutoff'],
        'original_records': sum(len(r['members']) for r in runs), 'unique_candidates': len(candidates),
        'selection': 'One best recorded rate vector per qualifying topology per arm/seed; no robustness selection.',
        'pruning': 'All 32 subsets; unchanged fixed template and surviving rates; qualify by original RPA loss threshold and maximum output deviation <= epsilon over all 9x1000 output samples. Minimum cardinality then canonical structural serialization. Conditional minimum, not a global mechanism/minimality proof.',
        'perturbations': 'Original and primary reduced portfolios: 32 shared antithetic log-uniform directions at 10%. Original winners: five radii and one-rate-at-a-time tests, exactly as Logic. Additional reduced-winner sweeps at the same radii, labelled separately. No refit/clipping. No new balance-preserving perturbations in this Logic-matched portfolio study.',
        'replay_order': logic['replay_order'],
        'baseline_policy': 'Canonical order then recorded orders. Require saved topology and all saved losses to match, and original qualification. If unresolved, retain unresolved record; never replace candidate. Nonbaseline retries only after numerical failure, never for unfavorable loss.',
        'qualification': 'Complete canonical simulation and loss below the frozen RPA threshold; unlike Logic, there is no binary truth-table criterion. This is not a guarantee of asymptotic RPA.',
        'analysis_unit': 'Seed; count each canonical core once per arm/seed if at least one selected rate witness passes >=29/32 perturbations.',
        'comparison_limit': 'Independent union reuses historical no-LLM search and insertion-only direct banks. RPA channel comparisons remain descriptive. Five arms include that union plus the true no-LLM baseline.',
        'refit': False, 'clip': False, 'new_model_calls': 0, 'scientific_timeout': None,
        'source_root': str(source_root), 'source_hashes': {str(p): sha(p) for p in source_root.rglob('*.py')
            if '__pycache__' not in p.parts}, 'input_hashes': inputs}
    atomic_json(output / 'plan.json', plan)
    (output / 'plan.sha256').write_text(sha(output / 'plan.json') + '\n')
    atomic_json(output / 'status.json', {'phase': 'prepared', 'completed': 0, 'expected': len(candidates)})
    (output / 'PROTOCOL.md').write_text('\n'.join([
        '# RPA communication: matched subset and perturbation diagnostics', '',
        f"100 arm/seed portfolios; {plan['original_records']} memberships; {len(candidates)} exact rate vectors.", '',
        plan['selection'], '', plan['pruning'], '', plan['perturbations'], '',
        'Primary epsilon 0.01; sensitivity epsilons 0.001 and 0.05. Factors lie in [1/(1+p), 1+p]. '
        'Sixteen draws plus their negatives, keyed by reaction ID, copied exactly from the previous RPA study. '
        'Winner radii: 1%, 5%, 10%, 25%, 50%. Robustness cutoff: 29/32 passes at 10%.', '',
        plan['qualification'], '', plan['baseline_policy'], '', plan['comparison_limit'], '',
        'Local execution only. Frozen source snapshot and hashed inputs. Original databases remain read-only. '
        'All failures stay in denominators; partial and unresolved results are explicitly marked. '
        'No contrast inference is emitted before completion; this descriptive protocol performs no significance tests.', '']))
    return plan


def initialize(plan):
    global PLAN, EVALUATOR
    import torch
    torch.set_num_threads(1)
    from RL4CRN.search_exploration.evaluation import build
    PLAN = plan
    EVALUATOR, _, _ = build('rpa', plan['task_config'], *plan['diagnostic_bounds'])
    replay.cache_clear()


@lru_cache(maxsize=192)
def replay(actions):
    from RL4CRN.search_exploration.evaluation import materialize
    from RL4CRN.utils.results_database import serialize_crn
    env = materialize(EVALUATOR, {'reaction_ids': [i for i, _ in actions], 'rates': [k for _, k in actions]})
    loss, info = EVALUATOR._compute_loss(env)
    expected_inputs = np.asarray(list(itertools.product([.5, 1., 1.5], repeat=2)))
    if (not np.array_equal(info['inputs'], expected_inputs)
            or not np.array_equal(np.asarray(info['setpoint']).reshape(-1), expected_inputs[:, 0])
            or not np.allclose(info['time_horizon'], np.linspace(0, 100, 1000), rtol=1e-7, atol=1e-7)):
        raise ValueError('Canonical RPA scenarios, targets or time grid changed')
    outputs = np.asarray(info['outputs'], dtype=float)
    if outputs.shape != (9, 1, 1000):
        raise ValueError(f'Unexpected output shape: {outputs.shape}')
    valid = np.isfinite(loss) and np.isfinite(outputs).all() and not np.any(outputs == 1e4)
    identity = serialize_crn(env.state)
    canonical_hash, canonical_structure = structural_identity(json.loads(identity['structure_json']))
    return {'loss': float(loss) if np.isfinite(loss) else None, 'outputs': outputs,
        'status': 'ok' if valid else 'solver_failure', 'topology_hash': identity['topology_hash'],
        'canonical_hash': canonical_hash, 'canonical_structure': canonical_structure,
        'structure_json': identity['structure_json'],
        'functional': cached_function(info) if valid else {}}


def replay_with_orders(actions, saved_orders, accept=None):
    rates = dict(actions)
    canonical = tuple(sorted(rates))
    orders = [canonical] + sorted({tuple(i for i in order if i in rates) for order in saved_orders} - {canonical})
    attempts = []
    for order in orders:
        if len(order) != len(rates) or set(order) != set(rates):
            raise ValueError('Invalid saved reaction order')
        value = replay(tuple((i, rates[i]) for i in order))
        accepted = value['status'] == 'ok' and (accept is None or accept(value))
        attempts.append({'reaction_ids': list(order), 'status': value['status'],
                         'loss': value['loss'], 'accepted': bool(accepted)})
        if accepted:
            return value, attempts, True
    return value, attempts, False


def row(value):
    return {k: v for k, v in value.items() if k != 'outputs'}


def evaluate_candidate(candidate):
    started = time.monotonic()
    before = replay.cache_info()
    actions = tuple(zip(candidate['reaction_ids'], candidate['rates']))
    saved_orders = candidate['saved_orders']
    def matches(value):
        return (value['topology_hash'] == candidate['topology_hash']
                and all(np.isclose(value['loss'], x, **PLAN['replay_tolerance']) for x in candidate['recorded_losses'])
                and value['loss'] < PLAN['threshold'])
    original, attempts, resolved = replay_with_orders(actions, saved_orders, matches)
    result = {'id': candidate['id'], 'status': 'complete', 'baseline_unresolved': not resolved,
        'baseline': {**row(original), 'order_attempts': attempts, 'stored_losses': candidate['recorded_losses']},
        'subsets': [], 'cores': {}, 'perturbations': []}
    if resolved:
        for count in range(6):
            for retained in itertools.combinations(range(5), count):
                if count == 5:
                    value, orders = original, attempts
                else:
                    value, orders, _ = replay_with_orders(tuple(actions[i] for i in retained), saved_orders)
                result['subsets'].append({**row(value), 'order_attempts': orders,
                    'retained_indices': list(retained), 'retained_count': count,
                    'max_output_deviation': float(np.max(np.abs(value['outputs'] - original['outputs'])))
                        if value['status'] == 'ok' else None})
        for epsilon in PLAN['epsilons']:
            result['cores'][str(epsilon)] = select_core(result['subsets'], epsilon, PLAN['threshold'])

        def sensitivity(current, radius, mode, perturbation_directions):
            ids, rates = [i for i, _ in current], [k for _, k in current]
            for draw, direction in enumerate(perturbation_directions):
                rates_draw = perturb(ids, rates, radius, direction)
                value, orders, _ = replay_with_orders(tuple(zip(ids, rates_draw)), saved_orders)
                passed = value['status'] == 'ok' and value['loss'] < PLAN['threshold']
                result['perturbations'].append({'mode': mode, 'radius': radius, 'draw': draw,
                    'status': value['status'], 'loss': value['loss'], 'qualified': bool(passed),
                    'order_attempts': orders, 'functional': value['functional']})

        radii = PLAN['winner_radii'] if candidate['winner'] else [PLAN['portfolio_radius']]
        for radius in radii:
            sensitivity(actions, radius, 'original', PLAN['directions'])
        if candidate['winner']:
            oat = np.zeros((10, 91))
            for i, (rid, _) in enumerate(actions):
                oat[2*i, rid], oat[2*i+1, rid] = -1, 1
            sensitivity(actions, 1.1, 'oat', oat)
        core = result['cores']['0.01']
        core_actions = tuple(actions[i] for i in core['retained_indices'])
        # Core winner sweeps supplement the matched Logic 10% core endpoint.
        if len(core_actions) == 5:
            result['perturbations'] += [{**r, 'mode': 'core'} for r in result['perturbations'] if r['mode'] == 'original']
        else:
            for radius in radii:
                sensitivity(core_actions, radius, 'core', PLAN['directions'])
    after = replay.cache_info()
    result.update(evaluator_calls=after.misses-before.misses, cache_hits=after.hits-before.hits,
                  elapsed_seconds=time.monotonic()-started)
    return result


def load_plan(output):
    plan = read(output / 'plan.json')
    if sha(output / 'plan.json') != (output / 'plan.sha256').read_text().strip():
        raise ValueError('Plan changed')
    for path, expected in {**plan['source_hashes'], **plan['input_hashes']}.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source/input changed: {path}')
    with gzip.open(output / 'selection.json.gz', 'rt') as stream:
        return plan, json.load(stream)


def analyze(output):
    plan, selection = load_plan(output)
    fingerprint = sha(output / 'plan.json')
    records = {}
    for path in (output / 'records').glob('*.json'):
        r = read(path)
        if r['plan_sha256'] != fingerprint or r['id'] not in selection['candidates'] or r['status'] != 'complete':
            raise ValueError('Invalid diagnostic record')
        records[r['id']] = r
    complete = len(records) == len(selection['candidates'])
    rows, memberships, sweeps = [], [], []
    unresolved = sum(r['baseline_unresolved'] for r in records.values())
    llm_routes = {'llm_direct', 'llm_fit', 'ga_llm'}
    for run in selection['runs']:
        seen = defaultdict(set)
        available = reduced = failed = unknown = 0
        for member in run['members']:
            record = records.get(member['candidate'])
            if record is None:
                continue
            available += 1
            if record['baseline_unresolved']:
                unknown += 1
                continue
            core = record['cores']['0.01']
            original = record['baseline']
            # Independently reconstruct canonical identities from frozen library.
            candidate = selection['candidates'][member['candidate']]
            for expected, indices in ((original, range(5)), (core, core['retained_indices'])):
                structure = selection['fixed'] + [selection['catalog'][str(candidate['reaction_ids'][i])] for i in indices]
                if structural_identity(structure)[0] != expected['canonical_hash']:
                    raise ValueError('Recorded core/baseline identity does not match selected reactions')
            fractions = {}
            for mode in ('original', 'core'):
                draws = [d for d in record['perturbations'] if d['mode'] == mode and d['radius'] == 1.1]
                if len(draws) != 32:
                    raise ValueError('Missing primary perturbation draws')
                fractions[mode] = sum(d['qualified'] for d in draws) / 32
                failed += sum(d['status'] != 'ok' for d in draws)
            seen['originals'].add(original['canonical_hash'])
            seen['cores'].add(core['canonical_hash'])
            if fractions['original'] >= plan['robust_diagnostic_cutoff']:
                seen['robust_originals'].add(original['canonical_hash'])
            if fractions['core'] >= plan['robust_diagnostic_cutoff']:
                seen['robust_cores'].add(core['canonical_hash'])
                if llm_routes.intersection(member['routes']):
                    seen['llm_supported_robust_cores'].add(core['canonical_hash'])
            for epsilon in plan['epsilons']:
                seen[f'cores_epsilon_{epsilon}'].add(record['cores'][str(epsilon)]['canonical_hash'])
            reduced += core['retained_count'] < 5
            memberships.append({'arm': run['arm'], 'seed': run['seed'], 'candidate': member['candidate'],
                'original_hash': original['canonical_hash'], 'core_hash': core['canonical_hash'],
                'retained_count': core['retained_count'], 'original_pass_fraction': fractions['original'],
                'core_pass_fraction': fractions['core'], 'routes': member['routes']})
            if member['rank'] == 1:
                for mode in ('original', 'core'):
                    for radius in plan['winner_radii']:
                        draws = [d for d in record['perturbations'] if d['mode'] == mode and d['radius'] == radius]
                        if len(draws) != 32:
                            raise ValueError('Missing winner sweep')
                        sweeps.append({'arm': run['arm'], 'seed': run['seed'], 'mode': mode, 'radius': radius,
                            'pass_fraction': sum(d['qualified'] for d in draws) / 32,
                            'failed_draws': sum(d['status'] != 'ok' for d in draws)})
        keys = ['originals', 'cores', 'robust_originals', 'robust_cores', 'llm_supported_robust_cores']
        keys += [f'cores_epsilon_{e}' for e in plan['epsilons']]
        rows.append({'arm': run['arm'], 'seed': run['seed'], 'selected': len(run['members']),
            'available': available, 'baseline_unresolved': unknown, 'prunable': reduced,
            'primary_failed_draws_logical': failed, **{key: len(seen[key]) for key in keys}})
    out = output / 'analysis'
    write_csv(out / 'per_run.csv', rows)
    if memberships:
        write_csv(out / 'memberships.csv', memberships)
    if sweeps:
        write_csv(out / 'winner_sweeps.csv', sweeps)
    summary = {'status': 'complete' if complete else 'partial', 'completed': len(records),
        'expected': len(selection['candidates']), 'baseline_unresolved': unresolved,
        'plan_sha256': fingerprint, 'rows': rows, 'memberships': memberships, 'winner_sweeps': sweeps}
    atomic_json(out / 'summary.json', summary)
    lines = ['# RPA communication portfolio diagnostics', '',
        f"Status: {summary['status']}; {len(records)}/{summary['expected']} exact vectors; {unresolved} unresolved baselines.", '',
        'Partial results are monitoring only; no arm comparison is interpreted until all candidates finish and baseline discrepancies are resolved.', '',
        '| Arm | Available memberships | Selected memberships | Unresolved |', '|---|---:|---:|---:|']
    for arm, label in zip(ARMS, LABELS):
        rr = [r for r in rows if r['arm'] == arm]
        lines.append(f"| {label} | {sum(r['available'] for r in rr)} | {sum(r['selected'] for r in rr)} | {sum(r['baseline_unresolved'] for r in rr)} |")
    if complete and not unresolved:
        lines += ['', '| Arm | Mean originals | Mean cores | Mean robust cores | Mean LLM-supported robust cores |',
                  '|---|---:|---:|---:|---:|']
        for arm, label in zip(ARMS, LABELS):
            rr = [r for r in rows if r['arm'] == arm]
            means = [np.mean([r[k] for r in rr]) for k in ('originals', 'cores', 'robust_cores', 'llm_supported_robust_cores')]
            lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.2f}' for v in means) + ' |')
        plot(out, rows, sweeps)
        lines += ['', '![Portfolio counts](portfolio_counts.png)', '', '![Winner sensitivity](winner_sweeps.png)']
    lines += ['', plan['pruning'], '', plan['perturbations'], '', plan['comparison_limit'], '',
        'Robust means at least 29/32 loss-qualified draws at 10%; this is a sampled diagnostic, not an adaptation guarantee. '
        'Canonical identities include the fixed template, reaction multiplicity and input channels. '
        'A structure is counted once per seed when at least one selected parameterization passes. '
        'LLM support uses exact-vector archive events/direct-bank evidence; multiple routes are retained. '
        'All subset, perturbation and numerical-order attempts remain in records/. No significance tests are claimed.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return summary


def plot(output, rows, sweeps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#168577', '#a64e85', '#bc7838', '#376ab3', '#6b6975')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ('cores', 'robust_cores', 'llm_supported_robust_cores'),
                                ('Distinct reduced structures', 'Robust reduced structures', 'LLM-supported robust structures')):
        for i, (arm, color) in enumerate(zip(ARMS, colors)):
            values = [r[metric] for r in rows if r['arm'] == arm]
            ax.scatter(np.full(len(values), i), values, color=color, alpha=.4, s=16)
            ax.plot([i-.2, i+.2], [np.mean(values)]*2, color='black', lw=2)
        ax.set(title=title, xticks=range(5), xticklabels=LABELS, ylabel='Count per seed')
        ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    fig.savefig(output / 'portfolio_counts.png', dpi=180)
    fig.savefig(output / 'portfolio_counts.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, ax in zip(('original', 'core'), axes):
        for arm, label, color in zip(ARMS, LABELS, colors):
            values = [np.mean([r['pass_fraction'] for r in sweeps if r['arm'] == arm
                              and r['mode'] == mode and r['radius'] == radius]) for radius in RADII]
            ax.plot([100*(r-1) for r in RADII], values, 'o-', label=label, color=color)
        ax.set(title=f'Winning designs: {mode}', xlabel='Multiplicative radius (%)',
               ylabel='Mean fraction of qualifying draws', ylim=(-.02, 1.02))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'winner_sweeps.png', dpi=180)
    fig.savefig(output / 'winner_sweeps.pdf')
    plt.close(fig)


def execute(output, smoke=False):
    plan, selection = load_plan(output)
    fingerprint = sha(output / 'plan.json')
    candidates = selection['candidates']
    done, calls, unresolved = set(), 0, 0
    for path in (output / 'records').glob('*.json'):
        r = read(path)
        if r['plan_sha256'] != fingerprint or r['status'] != 'complete' or r['id'] not in candidates:
            raise ValueError('Invalid cached diagnostic')
        done.add(r['id'])
        calls += r['evaluator_calls']
        unresolved += r['baseline_unresolved']
    todo = sorted((c for key, c in candidates.items() if key not in done), key=lambda c: (not c['winner'], c['id']))
    if smoke:
        todo = todo[:2]
    workers = min(2, plan['workers']) if smoke else plan['workers']
    iterator, pending = iter(todo), {}
    started = time.time()
    def progress(phase):
        atomic_json(output / 'status.json', {'phase': phase, 'completed': len(done), 'expected': len(candidates),
            'evaluator_calls': calls, 'baselines_unresolved': unresolved, 'workers': workers,
            'pid': os.getpid(), 'started': started, 'updated': time.time(), 'in_flight': len(pending)})
    progress('smoke' if smoke else 'running')
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(plan,)) as pool:
            exhausted, last = False, 0.
            while pending or not exhausted:
                while not exhausted and len(pending) < 2*workers:
                    candidate = next(iterator, None)
                    if candidate is None:
                        exhausted = True
                    else:
                        pending[pool.submit(evaluate_candidate, candidate)] = candidate['id']
                ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    key = pending.pop(future)
                    try:
                        record = future.result()
                    except BaseException as error:
                        atomic_json(output / 'controller_failed.json', {'candidate': key,
                            'type': type(error).__name__, 'error': str(error), 'time': time.time()})
                        raise
                    record['plan_sha256'] = fingerprint
                    atomic_json(output / 'records' / f'{key}.json', record)
                    done.add(key)
                    calls += record['evaluator_calls']
                    unresolved += record['baseline_unresolved']
                if time.monotonic()-last > 20:
                    progress('smoke' if smoke else 'running')
                    print(f'Persisted {len(done)}/{len(candidates)} candidates; {calls} calls; {unresolved} unresolved', flush=True)
                    last = time.monotonic()
        phase = 'complete' if len(done) == len(candidates) else 'smoke_passed'
        progress(phase)
        analyze(output)
        if phase == 'complete':
            atomic_json(output / 'completed.json', read(output / 'status.json'))
    except BaseException:
        progress('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output, args.workers)
    if args.execute or args.smoke:
        with (output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(output, smoke=args.smoke)
    if args.analyze:
        analyze(output)
