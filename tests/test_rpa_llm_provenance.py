import numpy as np
import pytest
from comparisons.search_exploration.analyze_rpa_llm_provenance import group, winner_weights, classify_exposure
from comparisons.search_exploration.run_rpa_llm_robustness import rate_key

def test_fit_is_not_neural_rate_learning():
    assert group({'source':'llm','origin':'llm','stream':'optimization'})=='llm_fit'
    assert group({'source':'rl','origin':'rl','stream':'optimization'})=='rl_fit'
    assert group({'source':'ga','origin':'llm','stream':'optimization'})=='llm_mutation'
    assert group({'source':'llm','origin':'llm','stream':None})=='llm_direct'

def test_ties_are_shared_not_given_to_first_source():
    assert np.allclose(winner_weights([1.,1.,2.]),[.5,.5,0])
    assert np.allclose(winner_weights([np.inf,np.inf]),[0,0])

def test_unexpected_parentage_is_not_silently_relabelled():
    with pytest.raises(ValueError):
        group({'source':'rl','origin':'llm','stream':'search'})

def test_robustness_cache_preserves_alignment_and_exact_rates():
    a={'reaction_ids':[3,2],'rates':[.1,.2]}
    b={'reaction_ids':[2,3],'rates':[.2,.1]}
    assert rate_key(a)==rate_key(b)
    assert rate_key(a)!=rate_key({'reaction_ids':[2,3],'rates':[.1,.2]})
    assert rate_key(a)!=rate_key({'reaction_ids':[3,2],'rates':[np.nextafter(.1,1),.2]})

def test_return_exposure_excludes_prior_llm_matches():
    hof=[{'topology_hash':'a','source':'RL','loss':.01}]
    new=classify_exposure('a',hof,set(),.005)
    assert new['rl_hof_match_new_to_llm'] and new['improves_matching_hof']
    old=classify_exposure('a',hof,{'a'},.005)
    assert old['rl_labelled_hof_match'] and not old['rl_hof_match_new_to_llm']
    assert not classify_exposure('a',[],set(),.005)['hof_match']
