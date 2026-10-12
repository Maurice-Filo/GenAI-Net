from comparisons.search_exploration.analyze_rpa_independent_sources import partition, winning_source


def test_source_partition_does_not_double_count_overlap():
    p=partition({'a','b','c'}, {'c','d'})
    assert p=={'search_only':2,'shared':1,'llm_only':1,'search':3,'llm':2,'union':4}
    assert p['search']+p['llm']-p['shared']==p['union']
    assert p['search_only']+p['llm_only']+p['shared']==p['union']


def test_shared_families_need_not_mean_shared_topologies():
    families={'a':1,'b':1}
    assert partition({'a'}, {'b'})['shared']==0
    assert partition({families['a']}, {families['b']})['shared']==1


def test_best_loss_credit_handles_ties():
    assert winning_source(.01,.001)=='llm'
    assert winning_source(.001,.01)=='search'
    assert winning_source(.01,.01+1e-13)=='tie'
