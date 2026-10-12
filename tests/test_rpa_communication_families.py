import numpy as np

from comparisons.search_exploration.analyze_rpa_communication_families import (
    ARMS, aif_like, contrasts, union_records,
)
from comparisons.search_exploration.analyze_rpa_mechanisms import (
    extra_structure, integral_balance, structural_token,
)


def record(topology, candidate, loss, complete=True):
    return {'topology_hash': topology, 'candidate_hash': candidate,
            'loss': loss, 'simulation_complete': complete}


def test_terminal_union_keeps_isolated_llm_and_best_rates_only():
    search = [record('a', 'search', .01), record('b', 'failed', .001, False)]
    direct = [record('a', 'llm', .002), record('c', 'isolated', .005),
              record('d', 'nan', float('nan')), record('e', 'negative', -.01)]
    rows = union_records(search, direct)
    assert [r['candidate_hash'] for r in rows] == ['llm', 'isolated']
    assert union_records(direct, search) == rows


def test_complete_structures_preserve_inputs_and_subtract_exact_template():
    def reaction(inputs):
        return {'reaction_id': 1, 'structure': {'type': 'MassAction',
            'reactants': [], 'products': ['X_1'], 'inputs': inputs}}
    fixed, unmodulated = reaction(['u_1']), reaction([None])
    assert extra_structure([fixed, unmodulated], [fixed]) == (structural_token(unmodulated),)
    assert structural_token(fixed) != structural_token(unmodulated)


def test_aif_screen_is_order_invariant_but_requires_exact_balance():
    def reaction(reactants, products, inputs=None):
        return {'parameters': [1.], 'structure': {'type': 'MassAction',
            'reactants': reactants, 'products': products, 'inputs': inputs or [None]}}
    rr = [reaction([], ['X_1'], ['u_1']), reaction(['X_3'], ['X_3', 'X_2']),
          reaction(['X_2', 'X_1'], [])]
    assert aif_like(rr, integral_balance(rr))
    rr.append(reaction(['X_1'], []))
    assert not aif_like(rr, integral_balance(rr))


def test_paired_intervals_use_seed_differences_and_correct_loss_direction():
    rows = []
    for arm in ARMS:
        for seed in range(1, 21):
            delta = 1 if arm == 'full' else 0
            rows.append({'arm': arm, 'seed': seed, 'qualified_topologies': 100+seed+delta,
                'families': 20+seed+delta, 'rarefied_k50': 10+seed+delta,
                'mean_pairwise_distance': .5+delta*.01, 'best_loss': .01-delta*.001})
    cc = [r for r in contrasts(rows) if r['arm']=='full' and r['reference']=='control']
    assert all(r['wins']==20 and r['ties']==0 and r['n']==20 for r in cc)
    count = next(r for r in cc if r['metric']=='qualified_topologies')
    assert count['ci_low'] == count['ci_high'] == count['mean_difference'] == 1
    loss = next(r for r in cc if r['metric']=='best_loss')
    assert np.isclose(loss['mean_difference'], -.001)
