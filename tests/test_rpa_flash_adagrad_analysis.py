import pytest

from comparisons.search_exploration.analyze_rpa_flash_adagrad import topology_losses, matched_settings


def test_topology_counts_use_best_complete_rate_setting():
    records = [dict(topology_hash=k, loss=v, simulation_complete=c) for k,v,c in
               [('a', .02, True), ('a', .01, True), ('b', .001, False), ('c', .03, True)]]
    assert topology_losses(records) == [.01, .03]


def test_nonfinite_complete_loss_is_not_hidden():
    with pytest.raises(ValueError):
        topology_losses([dict(topology_hash='a', loss=float('nan'), simulation_complete=True)])


def test_settings_allow_disclosed_scheduling_not_scientific_changes():
    a = dict(config=dict(workers=32, seed=1), julia_recipe={'evaluations':24},
             task_config={'rpa': {}}, policy={'deep_layer_size':1024}, post_prune=False,
             parameter_operator='julia_adagrad')
    b = {**a, 'config': {'workers':6, 'seed':1}, 'julia_recipe': {'evaluations':24, 'loss_disagreement_policy':'warn'}}
    assert matched_settings(a,b) == {'workers':[32,6]}
    b['config']['seed'] = 2
    with pytest.raises(ValueError):
        matched_settings(a,b)
