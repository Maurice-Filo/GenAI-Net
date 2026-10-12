import pytest

from comparisons.search_exploration.analyze_rpa_perturbation_counts import summarize_counts


def fixture():
    plan={"threshold":1.,"directions":[0]*4}
    data=[{"design":"a","mode":"baseline","status":"ok","loss":.5},
          {"design":"b","mode":"baseline","status":"ok","loss":2.}]
    for name in ("a","b"):
        for draw in range(4):
            data.append({"arm":"ga","design":name,"mode":"all_rates","radius":1.1,
                "status":"incomplete" if name=="a" and draw==3 else "ok",
                "loss":.5 if name=="a" or draw==0 else 2.,"draw":draw})
    return plan,data


def test_absolute_counts_separate_retention_from_new_qualification():
    plan,data=fixture()
    r=summarize_counts(plan,data)[0]
    assert r["tested_winners"]==2
    assert r["baseline_qualified_winners"]==1
    assert r["qualifying_draws_from_qualified_baselines"]==3
    assert r["tested_draws_from_qualified_baselines"]==4
    assert r["qualifying_draws_all_baselines"]==4
    assert r["tested_draws_all_baselines"]==8
    assert r["new_qualifying_draws_from_nonqualifying_baselines"]==1
    assert r["mean_remaining_qualified_winners_per_direction"]==.75
    assert r["min_remaining_qualified_winners_per_direction"]==0
    assert r["max_remaining_qualified_winners_per_direction"]==1
    assert r["baseline_qualified_winners_passing_any"]==1
    assert r["baseline_qualified_winners_passing_at_least_half"]==1
    assert r["baseline_qualified_winners_passing_all"]==0


def test_incomplete_baseline_does_not_count_as_retention():
    plan,data=fixture()
    data[0]["status"]="incomplete"
    r=summarize_counts(plan,data)[0]
    assert r["baseline_qualified_winners"]==0
    assert r["qualifying_draws_from_qualified_baselines"]==0
    assert r["qualifying_draws_all_baselines"]==4
    assert r["mean_remaining_qualified_winners_per_direction"]==0


def test_duplicate_or_missing_draws_are_rejected():
    plan,data=fixture()
    with pytest.raises(ValueError,match="directions"):
        summarize_counts(plan,data[:-1])
    data[-1]["draw"]=0
    with pytest.raises(ValueError,match="directions"):
        summarize_counts(plan,data)
