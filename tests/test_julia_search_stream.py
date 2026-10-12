from concurrent.futures import Future
from collections import defaultdict
from copy import deepcopy
import json
import os
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from RL4CRN.parameter_optimization import JuliaOptimizer, ReactionLibrary, Reaction, Experiment
from RL4CRN.search_exploration.evaluation import build, evaluate_once
from RL4CRN.search_exploration.store import BudgetExhausted
from RL4CRN.search_streams.julia_adapter import task_experiment
from RL4CRN.search_streams.julia_fitting import TopologyFitter
from RL4CRN.search_streams.julia_stream import JuliaRecipe, JuliaFitPool, JuliaStreamTrainer, reserve_joint, settle_joint, loss_agreement
from RL4CRN.search_streams.pruning import acceptable, prune_archive
from RL4CRN.search_streams.store import CompactStore
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS


@pytest.fixture(scope="module")
def worker():
    if not os.environ.get("SYNTH_EVO_PROJECT"):
        pytest.skip("Set SYNTH_EVO_PROJECT for real Julia tests")
    with JuliaOptimizer(project=os.environ["SYNTH_EVO_PROJECT"], startup_timeout=240) as value:
        yield value


def decay():
    library = ReactionLibrary(("X",), (), (), {7: Reaction((1,), (0,))})
    t = np.linspace(0, 3, 61)
    exp = Experiment(t, [[1.]], [[]], [[1.]], np.exp(-.7*t)[None, None, :], loss="l2")
    return library, exp


def test_runtime_budget_does_not_modify_optimizer_configuration(worker):
    library, exp = decay()
    model = worker.prepare(library.select([7]), exp)
    first = worker.optimize(model, [[.2]], steps=0, step_policy="feasible", evaluation_limit=1)["results"][0]
    second = worker.optimize(model, states=[first["state"]], steps=30,
        step_policy="feasible", evaluation_limit=3)["results"][0]
    assert first["evaluations"] == 1
    assert second["evaluations"] == 3
    assert second["state"]["config"] == first["state"]["config"]
    assert second["status"] == "evaluation_budget_exhausted"
    assert first["state"]["step"] == 0
    with pytest.raises(ValueError, match="evaluation_limit"):
        worker.optimize(model, [[1.]], evaluation_limit=0, step_policy="feasible")


def test_multistart_joint_cap_continuation_and_exact_screen_cache(worker):
    library, exp = decay()
    fitter = TopologyFitter(worker, library, exp)
    candidate = {"reaction_ids": [7], "rates": [.2]}
    result = fitter.fit(candidate, 17, 24)
    assert len(result["states"]) == 2
    assert result["evaluations"] == 24
    assert sum(r.get("evaluations", 0) for r in result["traces"]) == 24
    assert sum(r["evaluations"] for r in result["traces"][:4]) == 4
    repeat = fitter.fit(candidate, 17, 4)
    assert repeat["screen_cache_hits"] == 4
    assert repeat["evaluations"] <= 4
    resumed = fitter.fit(candidate, 17, 3, prior=result)
    assert resumed["evaluations"] == 3
    assert resumed["best_loss"] <= result["best_loss"]
    assert result["states"][0]["step"] < max(s["step"] for s in resumed["states"])


def test_real_julia_no_timeout_preserves_evaluation_budget(worker):
    library, exp = decay()
    fitter = TopologyFitter(worker, library, exp)
    result = fitter.fit({"reaction_ids": [7], "rates": [.2]}, 17, 12, timeout=None)
    assert result["evaluations"] == 12
    assert result["best_loss"] is not None


def test_failed_screen_does_not_discard_other_starts():
    class Worker:
        def prepare(self, *args):
            return "m"

        def optimize(self, model, starts, **kw):
            if starts and starts[0][0] == 1.:
                return {"results": [{"status": "evaluation_failed", "evaluations": 1, "scenario_solves": 1}]}
            state = {"best_loss": .1, "best_rates": [.5]} if starts else kw["states"][0]
            return {"results": [{"state": state, "status": "ok", "best_rates": [.5],
                "evaluations": 1 if starts else kw["evaluation_limit"], "scenario_solves": 1}]}
    library, exp = decay()
    result = TopologyFitter(Worker(), library, exp).fit({"reaction_ids": [7], "rates": [1.]}, 1)
    assert result["evaluations"] == 24
    assert result["best_rates"] == [.5]
    assert len(result["states"]) == 2
    assert result["traces"][0]["status"] == "evaluation_failed"


def test_joint_reservations_protect_search_and_refund_only_unused(tmp_path):
    store = CompactStore(tmp_path / "s.sqlite", 30, "julia-test", 10)
    try:
        token, credit = reserve_joint(store, 24)
        assert credit == 20
        with pytest.raises(BudgetExhausted):
            reserve_joint(store, 1)
        for i in range(10):
            _, key = store.reserve({"reaction_ids": [7], "rates": [1.+i]}, "search")
            store.complete(key, {"loss": .1})
        settle_joint(store, token, credit, 7)
        assert store.count("expensive_evaluations") == 17
        assert store.count("julia/joint_evaluations") == 7
        token, credit = reserve_joint(store, 24)
        assert credit == 13
        settle_joint(store, token, credit)  # a killed worker's unknown work is not silently refunded
        assert store.count("expensive_evaluations") == 30
        assert store.count("julia/unknown_evaluations_upper_bound") == 13
        assert store.count("julia/reserved") == 0
    finally:
        store.close()


def test_scheduler_is_distinct_nonblocking_and_failed_topologies_reenter(tmp_path):
    trainer = JuliaStreamTrainer.__new__(JuliaStreamTrainer)
    trainer.optimization_pool = SimpleNamespace(recipe=JuliaRecipe(admissions=4),
        admission_limit=lambda pending: 4, run=lambda job: job)
    trainer.optimize_per_epoch, trainer.queue_capacity = 4, 4
    trainer.optimization_jobs, trainer.scheduled_keys = {}, set()
    trainer.optimizer_rng = np.random.default_rng(1)
    trainer.config, trainer.state = SimpleNamespace(seed=1), SimpleNamespace(epoch=0)
    submitted = []
    def submit(fn, job):
        submitted.append(job)
        return Future()  # never completes: scheduling must not wait
    trainer.optimizer_executor = SimpleNamespace(submit=submit)
    trainer.store = CompactStore(tmp_path / "s.sqlite", 100, "test", 50)
    records = [{"topology_hash": str(i), "candidate_hash": str(i), "loss": float(i),
        "simulation_complete": i < 4, "reaction_ids": [i+1], "rates": [1.]} for i in range(8)]
    try:
        assert trainer._schedule(records + records) == 4
        assert len({j["candidate"]["topology_hash"] for j in submitted}) == 4
        assert [j["candidate"]["topology_hash"] for j in submitted[:2]] == ["0", "1"]
        assert trainer._schedule(records) == 0
        trainer.optimization_jobs.clear()
        trainer.state.epoch += 1
        assert trainer._schedule([records[-1]]) == 1
        trainer.optimization_jobs.clear()
        trainer.state.epoch += 1
        assert trainer._schedule([records[-1]]) == 1
        assert submitted[-1]["seed"] != submitted[-2]["seed"]
    finally:
        trainer.store.close()


def test_compute_controller_calibrates_before_parallel_wave():
    pool = JuliaFitPool.__new__(JuliaFitPool)
    pool.lock = threading.Lock()
    pool.recipe, pool.search_pool = JuliaRecipe(), SimpleNamespace(compute_seconds=100.)
    pool.compute_seconds, pool.job_estimate, pool.completed_jobs = 0., 1., 0
    assert pool.admission_limit(0) == 1
    assert pool.admission_limit(1) == 0
    pool.completed_jobs, pool.compute_seconds, pool.job_estimate = 1, 10., 5.
    assert pool.admission_limit(0) == 3
    assert pool.admission_limit(1) == 2
    pool.compute_seconds = 30.
    assert pool.admission_limit(0) == 0


def test_observed_loss_difference_is_warning_not_fatal():
    value = loss_agreement(1.0405492305529764, 1.0414303766810842)
    assert value["status"] == "warning"
    assert value["relative_difference"] < .001
    assert loss_agreement(1., 1.)["status"] == "agree"
    assert loss_agreement(0., 0.)["status"] == "agree"
    assert loss_agreement(.5, 1.)["status"] == "error"
    assert loss_agreement(.001, .002)["status"] == "error"


def test_warning_policy_preserves_large_disagreement_without_aborting():
    value = loss_agreement(.5543738581707589, .5718524765711763, mismatch_policy="warn")
    assert value["status"] == "warning"
    assert value["large_disagreement"] is True
    assert value["relative_difference"] == pytest.approx(.030564908112699105)
    assert value["policy"]["authoritative_loss"] == "canonical"
    assert loss_agreement(.001, 100., mismatch_policy="warn")["status"] == "warning"
    with pytest.raises(ValueError):
        loss_agreement(float("nan"), 1., mismatch_policy="warn")
    with pytest.raises(ValueError):
        JuliaRecipe(loss_disagreement_policy="ignore")


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -.001])
def test_invalid_loss_remains_fatal(invalid):
    with pytest.raises(ValueError, match="finite nonnegative"):
        loss_agreement(invalid, 1.)
    with pytest.raises(ValueError, match="finite nonnegative"):
        loss_agreement(1., invalid)


@pytest.mark.parametrize("kind", ["warning", "error", "large-warning", "incomplete"])
def test_pool_uses_canonical_score_and_preserves_agreement_diagnostics(tmp_path, kind):
    pool = JuliaFitPool.__new__(JuliaFitPool)
    pool.available = queue.Queue()
    pool.available.put(object())
    pool.output = tmp_path
    pool.recipe = JuliaRecipe(extensions=0, loss_disagreement_policy="warn" if kind == "large-warning" else "abort")
    pool.timeout = 60.
    pool.lock = threading.Lock()
    pool.extended = defaultdict(int)
    pool.compute_seconds, pool.completed_jobs, pool.job_estimate = 0., 0, 1.
    pool.search_pool = SimpleNamespace(compute_seconds=1.)
    pool.store = CompactStore(tmp_path / "s.sqlite", 100, "agreement-test", 50)
    canonical_loss = 1.0414303766810842
    julia_loss = .5 if kind in ("error", "large-warning") else 1.0405492305529764
    pool.canonical_pool = SimpleNamespace(evaluate=lambda c: {"loss": canonical_loss,
        "simulation_complete": kind != "incomplete", "task_info": {}, "candidate_hash": "fit"})
    phase = {"best_rates": [.1], "reaction_ids": [7], "best_loss": julia_loss,
        "evaluations": 1, "scenario_solves": 9, "traces": []}
    pool._fit = lambda *args: phase
    job = {"candidate": {"rates": [1.], "reaction_ids": [7], "loss": 5.},
        "seed": 0, "epoch": 0, "job_key": "test", "novel": False}
    try:
        if kind == "error":
            with pytest.raises(RuntimeError, match="Substantial"):
                pool.run(job)
        else:
            out = pool.run(job)
            if kind in ("warning", "large-warning"):
                assert out["best"]["loss"] == canonical_loss
                assert out["loss_agreement"]["status"] == "warning"
                assert pool.store.count("julia/loss_agreement_warning") == 1
                assert pool.store.count("julia/large_loss_disagreements") == int(kind == "large-warning")
            else:
                assert out["best"] is None
                assert out["failures"][0]["status"] == "canonical_incomplete"
        saved = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert saved["outcome"]["canonical_loss"] == canonical_loss
        assert saved["outcome"]["julia_loss"] == julia_loss
        assert ("fatal_error" in saved) == (kind == "error")
    finally:
        pool.store.close()


def test_pruning_loss_and_logical_acceptance():
    p = {"loss": .01, "simulation_complete": True, "all_rows_correct": True}
    assert acceptable(p, {**p, "loss": .0102}, .02, True)
    assert not acceptable(p, {**p, "loss": .01021}, .02)
    assert not acceptable(p, {**p, "loss": .009, "all_rows_correct": False}, .02, True)
    assert not acceptable(p, {**p, "loss": .001, "simulation_complete": False}, .02)
    assert not acceptable(p, {**p, "loss": .009}, .008)
    assert not acceptable(p, {**p, "loss": float("nan")}, .02)


def pruning_fixture():
    library = ReactionLibrary(("X",), ("u",), (Reaction((0,), (1,), 0, 1.),),
        {1: Reaction((0,), (1,)), 7: Reaction((1,), (0,)), 9: Reaction((2,), (0,))})
    record = {"reaction_ids": [7, 9], "rates": [.1, .2], "loss": .01, "qualified": True,
        "topology_hash": "p", "candidate_hash": "p", "simulation_complete": True, "all_rows_correct": True}
    return library, record


def test_pruning_separate_provenance_budget_and_single_deletion():
    library, original = pruning_fixture()
    saved = deepcopy(original)
    def fit(c, budget):
        return {"reaction_ids": c["reaction_ids"], "best_rates": c["rates"], "evaluations": budget}
    def evaluate(c):
        return {"loss": .01, "simulation_complete": True, "all_rows_correct": True}
    result = prune_archive([original, original], library=library, fit=fit, evaluate=evaluate,
        threshold=.02, require_all_rows=True)
    assert original == saved
    assert len(result["parents"]) == 1
    item = result["parents"][0]
    assert len(item["trials"]) == 1
    assert item["evaluations"] == 13  # parent + (11 joint + one verification)
    assert item["derived"]["reaction_ids"] == [9]
    assert item["derived"]["parent_hash"] == "p"
    assert item["derived"]["not_inserted_into_search_archive"]
    assert library.select([9]).reactions[0].fixed_rate == 1.
    assert library.select([9]).reactions[0].input_channel == 0


def test_pruning_failure_budget_and_callback_bugs_are_visible():
    library, original = pruning_fixture()
    def failfit(c, budget):
        return {"best_rates": None, "evaluations": budget}
    result = prune_archive([original], library=library, fit=failfit,
        evaluate=lambda c: original, threshold=.02, parent_budget=14)
    assert result["evaluations"] <= 14
    assert result["parents"][0]["derived"] is None
    def bug(c, budget):
        raise RuntimeError("implementation bug")
    with pytest.raises(RuntimeError, match="implementation bug"):
        prune_archive([original], library=library, fit=bug, evaluate=lambda c: original, threshold=.02)


@pytest.mark.parametrize("task", ["rpa", "logic"])
def test_task_adapter_canonical_loss(worker, task):
    cfg = json.loads(CONFIGS[task].read_text())
    evaluator, spec, _ = build(task, cfg, .001, 100.)
    library, exp = task_experiment(evaluator, spec)
    if task == "rpa":
        ids = [18, 51, 7, 13, 80]
        rates = [5.79576795, 11.87837895, 47.32750789, .7406736, 2.34991266]
    else:
        ids = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids][:evaluator.max_added_reactions]
        rates = [.1] * len(ids)
    canonical = evaluate_once(evaluator, {"reaction_ids": ids, "rates": rates})
    assert canonical["simulation_complete"]
    value = worker.evaluate(worker.prepare(library.select(ids), exp), rates, trajectories=True)
    assert value["loss"] == pytest.approx(canonical["loss"], rel=2e-4, abs=2e-6)
    assert np.asarray(value["outputs"]) == pytest.approx(np.asarray(canonical["task_info"]["outputs"]), rel=2e-4, abs=2e-6)
    assert exp.max_species == 1e4


def test_adapter_rejects_other_objectives():
    with pytest.raises(ValueError, match="L1 only"):
        task_experiment(None, SimpleNamespace(kind="oscillator_mean", norm=1))


def test_real_pruning_accepts_disposable_unobserved_reaction(worker):
    from RL4CRN.iocrns.iocrn import IOCRN
    from RL4CRN.iocrns.reactions import MassAction

    fixed = Reaction((1, 0), (0, 0), fixed_rate=.7)
    library = ReactionLibrary(("X", "Y"), (), (fixed,),
        {7: Reaction((0, 1), (0, 0)), 9: Reaction((0, 0), (0, 1))})
    times = np.linspace(0, 3, 61)
    targets = np.exp(-.7*times)[None, None, :]
    exp = Experiment(times, [[1., 0.]], [[]], [[1., 0.]], targets, max_species=1e4)
    fitter = TopologyFitter(worker, library, exp)

    def canonical(c):
        reactions = [MassAction(["X"], [], params=[.7])]
        for rid, rate in zip(c["reaction_ids"], c["rates"]):
            reactions.append(MassAction(["Y"] if rid == 7 else [], [] if rid == 7 else ["Y"], params=[rate]))
        crn = IOCRN(reactions, ["X"], solver="CVODE", rtol=1e-8, atol=1e-8)
        crn.compile()
        initial = np.array([1. if s == "X" else 0. for s in crn.species_labels])
        _, _, y, _ = crn.transient_response([np.array([])], [initial], times)
        values = np.asarray(y)
        return {"loss": float(np.mean(np.abs(values - targets))),
            "simulation_complete": bool(np.isfinite(values).all() and not (values == 1e4).any())}

    parent = {"reaction_ids": [7, 9], "rates": [.1, .2], "topology_hash": "synthetic",
        "candidate_hash": "original", "qualified": True}
    parent.update(canonical(parent))
    result = prune_archive([parent], library=library,
        fit=lambda c, budget: fitter.fit(c, 0, budget, starts=1), evaluate=canonical, threshold=.001)
    item = result["parents"][0]
    assert item["status"] == "accepted"
    assert len(item["derived"]["reaction_ids"]) == 1
    assert item["derived"]["loss"] < 1e-6
    assert item["evaluations"] <= 13
    assert parent["reaction_ids"] == [7, 9]
