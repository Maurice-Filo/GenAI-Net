"""Scientific-contract checks for the RPA adaptation of the Logic protocol."""

import numpy as np
import pytest

from comparisons.search_exploration import run_rpa_portfolio_diagnostics as protocol
from comparisons.search_exploration.run_rpa_perturbations import directions, perturb


def subset(n, loss, deviation, structure, status='ok'):
    return {'retained_count': n, 'loss': loss, 'max_output_deviation': deviation,
            'structure_json': structure, 'status': status}


def test_minimum_is_conditional_on_both_loss_and_entire_output_tolerance():
    rows = [subset(0, .1, 0., 'a'), subset(1, .001, .011, 'b'),
            subset(2, .001, .01, 'c'), subset(5, .001, 0., 'd'),
            subset(1, .0001, 0., 'e', 'solver_failure')]
    assert protocol.select_core(rows, .01, .02) is rows[2]
    assert protocol.select_core(rows, .001, .02) is rows[3]
    assert protocol.select_core(rows, .05, .02) is rows[1]


def test_ties_use_structure_not_best_loss_and_threshold_is_strict():
    rows = [subset(1, .02, 0., 'a'), subset(2, .019, 0., 'b'), subset(2, .001, 0., 'c')]
    assert protocol.select_core(rows, .01, .02) is rows[1]
    with pytest.raises(ValueError):
        protocol.select_core(rows[:1], .01, .02)


def test_successful_but_unfavorable_perturbation_is_never_retried(monkeypatch):
    calls = []
    def replay(actions):
        calls.append(actions)
        return {'status': 'ok', 'loss': 100.}
    monkeypatch.setattr(protocol, 'replay', replay)
    value, attempts, resolved = protocol.replay_with_orders(((2, .5), (1, 2.)), [[2, 1]])
    assert resolved and value['loss'] == 100.
    assert len(attempts) == 1 and calls == [((1, 2.), (2, .5))]


def test_failed_order_recovery_preserves_surviving_rates(monkeypatch):
    calls = []
    def replay(actions):
        calls.append(actions)
        return {'status': 'solver_failure' if len(calls) == 1 else 'ok', 'loss': 10.}
    monkeypatch.setattr(protocol, 'replay', replay)
    _, attempts, resolved = protocol.replay_with_orders(((1, 2.), (2, .5)), [[3, 2, 1]])
    assert resolved and len(attempts) == 2
    assert calls == [((1, 2.), (2, .5)), ((2, .5), (1, 2.))]


def test_baseline_disagreement_remains_unresolved(monkeypatch):
    monkeypatch.setattr(protocol, 'replay', lambda _: {'status': 'ok', 'loss': 10.})
    _, attempts, resolved = protocol.replay_with_orders(((1, 2.),), [[1]], lambda v: v['loss'] < .02)
    assert not resolved and len(attempts) == 1


def test_perturbations_shared_by_reaction_and_not_clipped():
    z = directions()
    assert z.shape == (32, 91)
    assert np.array_equal(z[:16], -z[16:])
    full = perturb([1, 20, 70], [1., 2., 3.], 1.1, z[0])
    reduced = perturb([70, 1], [3., 1.], 1.1, z[0])
    assert reduced == [full[2], full[0]]
    assert perturb([1], [100.], 1.5, np.ones(91)) == [150.]


def test_structural_identity_ignores_order_and_rates_but_keeps_multiplicity():
    a = {'reaction_id': 1, 'parameters': [1.], 'structure': {
        'type': 'Reaction', 'reactants': ['X1'], 'products': ['X2'], 'inputs': []}}
    b = {'reaction_id': 2, 'parameters': [2.], 'structure': {
        'type': 'Reaction', 'reactants': ['X2'], 'products': ['X1'], 'inputs': []}}
    assert protocol.structural_identity([a, b]) == protocol.structural_identity([b, {**a, 'parameters': [3.]}])
    assert protocol.structural_identity([a, b]) != protocol.structural_identity([a, a, b])
