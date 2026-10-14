"""Apply the existing Logic reduction/perturbation protocol to a fresh campaign."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys


def diagnose(root, workers):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from search_analysis import records
    from RL4CRN.search_exploration.evaluation import build
    from comparisons.search_exploration.run_logic5_portfolio_diagnostics import (
        initialize, evaluate_candidate, candidate_key, EPSILONS, RADII)
    c = json.loads((root / 'campaign.json').read_text())
    candidates, members, reference = {}, [], None
    for entry in c['jobs']:
        output = root / entry['path'] / 'output'
        if not (output / 'completed.json').exists():
            continue
        if json.loads((output / 'completed.json').read_text()).get('status') != 'complete':
            continue
        m = json.loads((output / 'manifest.json').read_text())
        if m['task'] != 'logic':
            raise ValueError('This diagnostic is for five-reaction Logic; use analyze --rpa-screen for RPA')
        if reference and (m['task_config'] != reference['task_config'] or
                          m['config']['quality_threshold'] != reference['config']['quality_threshold']):
            raise ValueError('Diagnostic requires shared task settings and threshold')
        reference = m
        selected = {}
        for row in records(output):
            if row['loss'] >= m['config']['quality_threshold']:
                continue
            key = tuple(sorted(row['reaction_ids']))
            if key not in selected or row['loss'] < selected[key]['loss']:
                selected[key] = row
        winner = min((row['loss'] for row in selected.values()), default=None)
        keys = []
        for row in selected.values():
            if len(row['reaction_ids']) != 5 or len(set(row['reaction_ids'])) != 5:
                raise ValueError('The paper\'s Logic diagnostic requires five distinct added reactions')
            key = candidate_key(row['reaction_ids'], row['rates'])
            item = candidates.setdefault(key, dict(id=key, reaction_ids=row['reaction_ids'], rates=row['rates'],
                loss=row['loss'], topology_hash=row['topology_hash'], recorded_losses=[], saved_orders=[],
                recorded_correctness=[], winner=False))
            item['recorded_losses'].append(row['loss']); item['saved_orders'].append(row['reaction_ids'])
            if row.get('all_rows_correct') is not None:
                item['recorded_correctness'].append(row['all_rows_correct'])
            item['winner'] |= row['loss'] == winner
            keys.append(key)
        members.append(dict(seed=entry['seed'], candidates=keys))
    if reference is None:
        raise ValueError('No completed Logic runs to diagnose')
    evaluator, _, _ = build('logic', reference['task_config'], .001, 100.)
    z = np.random.default_rng(20260917).uniform(-1, 1, (16, len(evaluator.library)))
    plan = dict(task_config=reference['task_config'], threshold=reference['config']['quality_threshold'],
                replay_tolerance={'rtol': 1e-7, 'atol': 1e-7}, epsilons=list(EPSILONS),
                directions=np.concatenate((z, -z)).tolist(), winner_radii=list(RADII), portfolio_radius=1.1,
                workers=workers, members=members, candidates=candidates)
    destination = root / 'diagnostics'
    destination.mkdir(exist_ok=False)
    (destination / 'records').mkdir()
    (destination / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    source_files = [Path(__file__), Path(__file__).parent/'diagnostics/logic_pruning.py',
                    Path(__file__).parents[1]/'comparisons/search_exploration/run_logic5_portfolio_diagnostics.py',
                    Path(__file__).parents[1]/'comparisons/search_exploration/run_rpa_perturbations.py']
    (destination / 'source_hashes.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files}, indent=2)+'\n')
    count = 0
    results = {}
    # Bounded submission avoids queuing an entire large portfolio in memory.
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(plan,)) as pool:
        iterator = iter(candidates.values()); pending = {}
        while True:
            while len(pending) < 2*workers:
                candidate = next(iterator, None)
                if candidate is None:
                    break
                pending[pool.submit(evaluate_candidate, candidate)] = candidate['id']
            if not pending:
                break
            future = next(as_completed(pending)); key = pending.pop(future)
            result = future.result(); results[key] = result
            (destination/'records'/f'{key}.json').write_text(json.dumps(result, allow_nan=False)+'\n')
            count += 1
            print(f'Diagnosed {count}/{len(candidates)} candidates', flush=True)
    summaries = []
    for member in members:
        reduced, robust, unresolved = set(), set(), 0
        for key in member['candidates']:
            result = results[key]
            if result['baseline_unresolved']:
                unresolved += 1; continue
            core = result['cores'].get('0.01')
            if core:
                identity = core['structure_json']
                reduced.add(identity)
                passed = sum(r['valid'] for r in result['perturbations'] if r['mode']=='core' and r['radius']==1.1)
                if passed >= 29:
                    robust.add(identity)
        summaries.append(dict(seed=member['seed'], qualifying_representatives=len(member['candidates']),
            reduced_structures=len(reduced), robust_reduced_structures=len(robust), unresolved_baselines=unresolved))
    (destination/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print(f'Diagnostics: {destination}; same reduction and 32-draw perturbation functions as the paper')
