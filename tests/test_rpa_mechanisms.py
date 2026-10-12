import itertools

import numpy as np
import pytest

from comparisons.search_exploration.analyze_rpa_mechanisms import (
    cached_function, extra_structure, integral_balance, rarefied_richness,
    set_jaccard, structural_token,
)


def reaction(reactants, products, rate=1., inputs=None, rid=1):
    return {"reaction_id":rid, "parameters":[rate], "structure":{
        "type":"MassAction", "reactants":reactants,"products":products,
        "inputs":inputs or [None]}}


def test_exact_template_subtraction_preserves_same_id_unmodulated_reaction():
    fixed=reaction([], ["X_1"], inputs=["u_1"])
    added=reaction([], ["X_1"])
    assert extra_structure([fixed,added],[fixed]) == (structural_token(added),)
    assert structural_token(fixed) != structural_token(added)


def test_order_invariance_but_not_stoichiometric_multiplicity():
    a=reaction(["X_1","X_2"],["X_3"])
    b=reaction(["X_2","X_1"],["X_3"])
    assert structural_token(a)==structural_token(b)
    assert structural_token(a)!=structural_token(reaction(["X_1","X_1"],["X_3"]))
    assert extra_structure([a,b],[])==(structural_token(a),)*2


def test_empty_portfolio_is_not_perfect_agreement():
    assert np.isnan(set_jaccard(set(),set()))
    assert set_jaccard({1,2},{2,3})==pytest.approx(1/3)


def test_rarefaction_matches_exhaustive_enumeration():
    labels=[1,1,1,2,3]
    expected=np.mean([len({labels[i] for i in selection}) for selection in itertools.combinations(range(5),3)])
    assert rarefied_richness(labels,3)==pytest.approx(expected)
    assert rarefied_richness(labels,5)==3
    assert np.isnan(rarefied_richness(labels,6))
    assert np.isnan(rarefied_richness([],10))


def test_integral_balance_cancels_antithetic_reactions():
    r=[reaction([], ["X_1"],inputs=["u_1"]),
       reaction(["X_3"],[],inputs=["u_2"]),
       reaction(["X_3"],["X_3","X_2"]),
       reaction(["X_1","X_2"],[],rate=3.)]
    result=integral_balance(r)
    assert result["alpha"]==pytest.approx(-1.)
    assert result["gain"]==pytest.approx(1.)
    assert result["restricted_exact_balance"]
    assert not integral_balance(r+[reaction(["X_1"],[],rate=.2)])["restricted_exact_balance"]


def test_cached_function_detects_hidden_drift_despite_perfect_output():
    u=np.array(list(itertools.product([.5,1.,1.5],repeat=2)))
    y=np.repeat(u[:,0,None,None],1000,axis=2)
    x=np.repeat(y,3,axis=1)
    x[:,0,:]+=np.linspace(0,10,1000)
    result=cached_function({"outputs":y,"trajectories":x,"inputs":u,"setpoint":u[:,0,None]})
    assert result["max_final_relative_error"]==0
    assert result["max_disturbance_spread"]==0
    assert result["max_state_tail_change"]>0
