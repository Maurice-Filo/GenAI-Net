import numpy as np

from comparisons.search_exploration.analyze_rpa_perturbations import aggregate, bootstrap_mean
from comparisons.search_exploration.run_rpa_perturbations import (
    balance_preserving_ids, directions, jobs, perturb,
)


def test_antithetic_directions_and_no_clipping():
    z=directions()
    assert z.shape==(32,91)
    assert np.array_equal(z[:16],-z[16:])
    rates=perturb([2,3],[100.,.001],1.5,np.ones(91))
    assert rates[0]==150.
    assert np.isclose(rates[1],.0015)
    a=perturb([2,3],[2.,3.],1.5,z[0])
    b=perturb([2,3],[2.,3.],1.5,-z[0])
    assert np.allclose(np.array(a)*b,np.array([2.,3.])**2)


def test_rate_directions_are_keyed_by_reaction_not_order():
    a=perturb([2,3],[2.,3.],1.1,directions()[0])
    b=perturb([3,2],[3.,2.],1.1,directions()[0])
    assert a==b[::-1]
    assert perturb([2,3],[2.,3.],1.1,np.ones(91),[2])[1]==3.


def test_balance_preserving_mask_excludes_sensor_gain():
    def r(reactants,products):
        return {"structure":{"reactants":reactants,"products":products}}
    catalog={1:r(["X_1","X_2"],[]),2:r(["X_3"],["X_3","X_2"]),3:r(["X_1"],["X_1","X_3"])}
    b={"restricted_exact_balance":True,"alpha":-1.}
    assert balance_preserving_ids({"reaction_ids":[1,2,3]},catalog,b)==[1,3]


def test_jobs_are_unique_and_preserve_topology_and_baseline():
    d={"id":"test","arm":"julia","seed":1,"candidate":{"reaction_ids":[2,3,4,5,6],"rates":[1.]*5},"balance_preserving_ids":[2,3]}
    plan={"designs":[d],"directions":directions().tolist(),"radii":[1.01,1.1],"oat_radius":1.1}
    jj=list(jobs(plan))
    assert len(jj)==1+2*32*2+10
    assert len({j["id"] for j in jj})==len(jj)
    assert jj[0]["rates"]==[1.]*5
    assert all(j["reaction_ids"]==[2,3,4,5,6] for j in jj)
    for j in jj:
        if j["mode"]=="balance_preserving":
            assert j["rates"][2:]==[1.]*3


def aggregation_fixture():
    design={"id":"test","arm":"julia","seed":1,"candidate":{"loss":.5},
        "aif_like_exact":True,"balance":{"restricted_exact_balance":True}}
    plan={"designs":[design],"threshold":.9}
    baseline={"design":"test","mode":"baseline","radius":1.,"status":"ok","loss":1.}
    sample={"design":"test","mode":"all_rates","radius":1.1,"status":"ok","loss":.8,
        "changed_ids":[2,3,4,5,6],"max_final_relative_error":.01,"max_disturbance_spread":.02}
    return plan,baseline,sample


def test_aggregation_uses_fresh_baseline_and_keeps_failed_draws():
    plan,b,s=aggregation_fixture()
    failed={**s,"status":"incomplete","loss":.01}
    rows,audit,_=aggregate(plan,[b,s,s,failed])
    r=rows[0]
    assert r["draws"]==3 and r["failures"]==1
    assert r["qualified_fraction"]==2/3
    assert r["final_error_below_5pct_fraction"]==2/3
    assert r["disturbance_spread_below_5pct_fraction"]==2/3
    assert r["median_loss_ratio"]==.8
    assert r["fraction_within_2x_baseline"]==2/3
    assert not r["baseline_qualified"]  # Stored loss passes, reevaluated loss does not.
    assert np.isinf(r["p90_loss"]) and np.isinf(r["max_loss"])
    assert audit[0]["relative_drift"]==1.
    assert audit[0]["drift_above_1pct"]


def test_all_failures_are_infinite_not_dropped_or_given_penalty_loss():
    plan,b,s=aggregation_fixture()
    rows,_,_=aggregate(plan,[b,{**s,"status":"error","loss":None}])
    r=rows[0]
    assert r["qualified_fraction"]==0
    assert np.isinf(r["median_loss_ratio"])
    assert np.isnan(r["median_final_error_finite"])


def test_bad_baseline_is_disclosed_not_silently_replaced_by_stored_loss():
    plan,b,s=aggregation_fixture()
    rows,audit,_=aggregate(plan,[{**b,"status":"error","loss":None},s])
    assert not rows[0]["baseline_valid"]
    assert not rows[0]["baseline_qualified"]
    assert np.isnan(rows[0]["median_loss_ratio"])
    assert audit[0]["status"]=="error"


def test_seed_bootstrap_zero_and_one_are_exact():
    assert bootstrap_mean([1.,1.,1.],np.random.default_rng(0))==(1.,1.,1.)
    assert bootstrap_mean([0.,0.,0.],np.random.default_rng(0))==(0.,0.,0.)
    assert all(np.isnan(x) for x in bootstrap_mean([],np.random.default_rng(0)))
