import numpy as np

from comparisons.search_exploration.plot_rpa_independent_loss_time import (
    previous_values, proposal_prefix, upper_summary,
)


def test_prefix_excludes_failed_scores_and_does_not_backdate_future_results():
    rows=[{'record':None},{'record':{'simulation_complete':False,'loss':.0001}},
          {'record':{'simulation_complete':True,'loss':.02}},
          {'record':{'simulation_complete':True,'loss':.001}}]
    p=proposal_prefix(rows)
    assert np.isinf(p[0]) and np.isinf(p[1])
    assert p[2]==.02 and p[3]==.001


def test_snapshot_lookup_has_no_future_leak_and_retains_final_result():
    result=previous_values([3,5,5,9],[.1,.05,.04,.01],[0,3,4,5,20])
    np.testing.assert_equal(result,[np.inf,.1,.1,.04,.01])


def test_upper_median_does_not_drop_missing_sources():
    values=np.r_[np.full(10,.001),np.full(10,np.inf)][:,None]
    assert np.isinf(upper_summary(values)[0][0])
    values[10]=.002
    assert upper_summary(values)[0][0]==.002
