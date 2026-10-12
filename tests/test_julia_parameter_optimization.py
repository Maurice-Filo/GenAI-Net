"""Opt in to real Julia tests with SYNTH_EVO_PROJECT=/path/to/SynthEvo."""

import os
import copy

import numpy as np
import pytest

from RL4CRN.parameter_optimization import (
    CRNSpec, Experiment, JuliaOptimizer, Reaction, ReactionLibrary,
)


def decay_problem():
    crn = CRNSpec(("X",), (), (Reaction((1,), (0,)),))
    times = np.linspace(0, 3, 61)
    experiment = Experiment(times, [[1.]], [[]], [[1.]],
                            np.exp(-.7 * times)[None, None, :], loss="l2")
    return crn, experiment


def test_spec_validation():
    crn, exp = decay_problem()
    assert crn.parameter_count == 1
    assert exp.payload(crn)["targets"][0][0][0] == 1
    with pytest.raises(ValueError):
        CRNSpec(("X",), (), (Reaction((.5,), (0,)),))
    with pytest.raises(ValueError):
        CRNSpec(("X",), (), (Reaction((1,), (0,), input_channel=0),))
    with pytest.raises(ValueError):
        Experiment([0, 0], [[1]], [[]], [[1]], [[[0, 0]]]).payload(crn)
    with pytest.raises(ValueError):
        Experiment([0, 1], [[1]], [[]], [[1]], [[[0, np.nan]]]).payload(crn)


def test_library_selection_preserves_order_and_masks_template():
    fixed = Reaction((0,), (1,), input_channel=0, fixed_rate=1.)
    library = ReactionLibrary(("X",), ("u",), (fixed,), {
        7: Reaction((1,), (0,)), 9: Reaction((2,), (0,)),
        1: Reaction((0,), (1,)), 0: Reaction((0,), (0,)),
    })
    assert library.select([9, 7]).reactions[-2:] == (library.reactions[9], library.reactions[7])
    assert library.select([7]).parameter_count == 1
    for ids in ([1], [0], [7, 7], [99]):
        with pytest.raises(ValueError):
            library.select(ids)


@pytest.fixture(scope="module")
def worker():
    project = os.environ.get("SYNTH_EVO_PROJECT")
    if not project:
        pytest.skip("Set SYNTH_EVO_PROJECT to run real Julia integration tests")
    with JuliaOptimizer(project=project, startup_timeout=240) as optimizer:
        yield optimizer


def test_analytic_decay_and_cache(worker):
    crn, exp = decay_problem()
    model = worker.prepare(crn, exp)
    again = worker.prepare(crn, exp)
    assert model == again
    out = worker.evaluate(model, [.3], trajectories=True)
    t = exp.times
    y = np.exp(-.3 * t)
    err = y - np.exp(-.7 * t)
    assert np.asarray(out["outputs"])[0, 0] == pytest.approx(y, abs=2e-7)
    assert out["loss"] == pytest.approx(np.mean(err**2), abs=2e-8)
    assert out["gradient"][0] == pytest.approx(np.mean(2 * err * -t * y), abs=2e-7)
    assert worker.info["pid"] == worker.stats()["pid"]
    assert worker.stats()["prepares"] == 1


@pytest.mark.parametrize("method", ["adagrad", "adam"])
def test_optimization_resume_best_and_async(worker, method):
    model = worker.prepare(*decay_problem())
    whole = worker.optimize(model, [[.15], [1.5]], steps=30, method=method, learning_rate=.2)
    first = worker.submit(model, [[.15]], steps=12, method=method, learning_rate=.2).result()
    second = worker.optimize(model, steps=18, states=[first["results"][0]["state"]],
                             method=method, learning_rate=.2)
    a, b = whole["results"][0], second["results"][0]
    assert a["status"] == b["status"] == "ok"
    assert a["state"]["step"] == b["state"]["step"] == 30
    assert a["final_rates"] == pytest.approx(b["final_rates"], abs=1e-12)
    assert a["best_loss"] == pytest.approx(b["best_loss"], abs=1e-12)
    assert a["evaluations"] == first["results"][0]["evaluations"] + b["evaluations"] == 31
    for result in whole["results"]:
        assert result["best_loss"] < result["history"][0]["loss"] / 10
        assert result["best_loss"] <= result["final_loss"]
        score = worker.evaluate(model, result["best_rates"])
        assert score["loss"] == pytest.approx(result["best_loss"], abs=1e-12)
    with pytest.raises(ValueError, match="configuration"):
        worker.optimize(model, states=[a["state"]], method=method, learning_rate=.4)
    changed = copy.deepcopy(a["state"])
    changed["theta"][0] += .1
    with pytest.raises(ValueError, match="modified"):
        worker.optimize(model, states=[changed], method=method, learning_rate=.2)


def test_bounds_and_bad_start_isolation(worker):
    model = worker.prepare(*decay_problem())
    out = worker.optimize(model, [[200.], [.2], [-1.]], steps=2, bounds=(.1, 1.), learning_rate=20.)
    assert [r["status"] for r in out["results"]] == ["ok", "ok", "invalid_start"]
    assert out["results"][0]["initial_clipped"]
    for r in out["results"][:2]:
        assert .1 <= r["best_rates"][0] <= 1.
        assert .1 <= r["final_rates"][0] <= 1.
    with pytest.raises(ValueError):
        worker.optimize(model, [[1]], method="typo")


def test_repeated_species_zero_state_modulation_and_finite_difference(worker):
    # Modulated and unmodulated production are deliberately distinct here.
    crn = CRNSpec(("X", "Y"), ("u",), (
        Reaction((0, 0), (1, 0), input_channel=0, fixed_rate=.4),
        Reaction((0, 0), (1, 0)), Reaction((2, 0), (0, 1)),
        Reaction((0, 1), (0, 0)),
    ))
    t = np.linspace(0, 2, 51)
    exp = Experiment(t, [[0, 0], [.3, .1]], [[0], [1.5]], np.eye(2),
                     np.full((2, 2, len(t)), 2.), weights=np.linspace(.25, 2, len(t)), loss="l1")
    model = worker.prepare(crn, exp)
    p = np.array([.2, .7, .3])
    result = worker.evaluate(model, p, trajectories=True)
    fd = []
    for j in range(3):
        delta = np.eye(3)[j] * 1e-5
        fd.append((worker.evaluate(model, p + delta)["loss"] -
                   worker.evaluate(model, p - delta)["loss"]) / 2e-5)
    assert result["gradient"] == pytest.approx(fd, rel=2e-5, abs=1e-7)
    # Independent Python CVODE and the real reward aggregation.
    from RL4CRN.iocrns.iocrn import IOCRN
    from RL4CRN.iocrns.reactions import MassAction
    from RL4CRN.utils.utils import performance_metric
    reactions = [MassAction([], ["X"], input_channels=["u"], params=[.4]),
                 MassAction([], ["X"], params=[.2]),
                 MassAction(["X", "X"], ["Y"], params=[.7]),
                 MassAction(["Y"], [], params=[.3])]
    network = IOCRN(reactions, ["X", "Y"], solver="CVODE", atol=1e-8, rtol=1e-8)
    network.compile()
    ys = []
    for x0, u in zip(exp.initial_states, exp.inputs):
        network.reset()  # The legacy IOCRN caches its last response on the instance.
        _, _, outputs, _ = network.transient_response([np.asarray(u)], [np.asarray(x0)], t)
        ys.append(outputs[0])
    assert np.asarray(result["outputs"]) == pytest.approx(np.asarray(ys), abs=2e-6)
    expected = performance_metric([np.array([2., 2.])] * 2, ys,
                                  np.tile(exp.weights, (2, 1)))
    assert result["loss"] == pytest.approx(expected, abs=2e-7)


def test_explicit_solver_failure_and_recovery(worker):
    crn, exp = decay_problem()
    exp.max_steps = 1
    model = worker.prepare(crn, exp)
    result = worker.optimize(model, [[1.], [.5]], steps=2)
    assert all(r["status"] == "evaluation_failed" for r in result["results"])
    assert all(r["best_loss"] is None for r in result["results"])
    assert all(r["evaluations"] == 1 and r["scenario_solves"] == 1 for r in result["results"])
    assert all(r["completed_evaluations"] == 0 for r in result["results"])
    good = worker.prepare(*decay_problem())
    assert np.isfinite(worker.evaluate(good, [.3])["loss"])


def test_worker_deadline_kills_process():
    project = os.environ.get("SYNTH_EVO_PROJECT")
    if not project:
        pytest.skip("Set SYNTH_EVO_PROJECT")
    with JuliaOptimizer(project=project, startup_timeout=240) as worker:
        model = worker.prepare(*decay_problem())
        with pytest.raises(TimeoutError):
            worker.evaluate(model, [.3], timeout=1e-9)
        assert worker.process.poll() is not None


def test_species_reaction_permutation_and_tsit5(worker):
    crn = CRNSpec(("X", "Y"), (), (Reaction((1, 0), (0, 1)), Reaction((0, 1), (0, 0))))
    exp = Experiment(np.linspace(0, 2, 31), [[1., 0.]], [[]], [[1., 2.]],
                     np.full((1, 1, 31), 3.), relative=True)
    first = worker.evaluate(worker.prepare(crn, exp), [.4, .2], trajectories=True)
    reordered = CRNSpec(("Y", "X"), (), (Reaction((1, 0), (0, 0)), Reaction((0, 1), (1, 0))))
    exp.initial_states, exp.output_matrix, exp.solver = [[0., 1.]], [[2., 1.]], "tsit5"
    second = worker.evaluate(worker.prepare(reordered, exp), [.2, .4], trajectories=True)
    assert first["loss"] == pytest.approx(second["loss"], abs=1e-7)
    assert first["gradient"] == pytest.approx(second["gradient"][::-1], abs=2e-7)
    assert np.asarray(first["outputs"]) == pytest.approx(np.asarray(second["outputs"]), abs=2e-7)


def test_real_rpa_library_and_canonical_cvode(worker):
    from comparisons.search_exploration.benchmark_julia_rates import rpa_fixture
    crn, exp, rates, canonical = rpa_fixture()
    model = worker.prepare(crn, exp)
    result = worker.evaluate(model, rates, trajectories=True)
    reference = canonical(rates)
    assert reference["simulation_complete"]
    assert result["loss"] == pytest.approx(reference["loss"], abs=2e-6)
    assert np.asarray(result["outputs"]) == pytest.approx(reference["outputs"], abs=2e-5)
    # Check the derivative away from the good design's many absolute-loss kinks.
    # A 1e-3 finite difference crosses those kinks at the original near-optimum.
    rates = rates * [.7, 1.2, .8, 1.3, 1.1]
    result = worker.evaluate(model, rates)
    fd = []
    for j in range(len(rates)):
        delta = np.eye(len(rates))[j] * 1e-3
        fd.append((worker.evaluate(model, rates + delta)["loss"] -
                   worker.evaluate(model, rates - delta)["loss"]) / 2e-3)
    assert result["gradient"] == pytest.approx(fd, rel=3e-3, abs=2e-6)
    # Regression from the first fitting probe: OUT tracked well while X_2
    # exceeded the canonical 10,000 concentration ceiling.
    inadmissible = [4.781990621178806, 12.100315756896052, 50.91168895987312,
                    .8143936818104359, 3.0450384350528217]
    assert not canonical(inadmissible)["simulation_complete"]
    with pytest.raises(RuntimeError, match="ceiling"):
        worker.evaluate(model, inadmissible)


def test_constant_fixed_network(worker):
    crn = CRNSpec(("X",), (), (Reaction((0,), (1,), fixed_rate=.3),))
    times = np.linspace(0, 2, 31)
    exp = Experiment(times, [[0.]], [[]], [[1.]], np.zeros((1, 1, 31)))
    model = worker.prepare(crn, exp)
    result = worker.evaluate(model, [], trajectories=True)
    assert result["gradient"] == []
    assert result["outputs"][0][0] == pytest.approx(.3 * times, abs=1e-8)
    with pytest.raises(ValueError, match="No free parameters"):
        worker.optimize(model, [[]])


def test_species_ceiling_is_individual_failure_not_a_low_loss(worker):
    crn = CRNSpec(("X",), (), (Reaction((0,), (1,)),))
    exp = Experiment(np.linspace(0, 2, 31), [[0.]], [[]], [[1.]],
                     np.ones((1, 1, 31)), max_species=.2)
    model = worker.prepare(crn, exp)
    result = worker.optimize(model, [[1.], [.05]], steps=0)["results"]
    assert result[0]["status"] == "evaluation_failed"
    assert "ceiling" in result[0]["error"]
    assert result[0]["best_loss"] is None
    assert result[1]["status"] == "ok"
    assert result[1]["best_loss"] == pytest.approx(.95, abs=1e-8)


def test_cache_eviction_and_checkpoint_across_workers(worker):
    project = os.environ["SYNTH_EVO_PROJECT"]
    crn, exp = decay_problem()
    key = worker.prepare(crn, exp)
    checkpoint = worker.optimize(key, [[.2]], steps=3)["results"][0]["state"]
    expected = worker.optimize(key, states=[checkpoint], steps=4)["results"][0]
    with JuliaOptimizer(project=project, startup_timeout=240, cache_size=1) as other:
        assert other.prepare(crn, exp) == key
        actual = other.optimize(key, states=[checkpoint], steps=4)["results"][0]
        assert actual["final_rates"] == pytest.approx(expected["final_rates"], abs=1e-12)
        assert actual["best_loss"] == pytest.approx(expected["best_loss"], abs=1e-12)
        exp.loss = "l1"
        new_key = other.prepare(crn, exp)
        assert new_key != key
        assert other.stats()["models"] == 1
        with pytest.raises(ValueError, match="evicted"):
            other.evaluate(key, [.2])
        assert np.isfinite(other.evaluate(new_key, [.2])["loss"])


def bounded_birth_problem():
    return (CRNSpec(("X",), (), (Reaction((0,), (1,)),)),
            Experiment(np.linspace(0, 2, 31), [[0.]], [[]], [[1.]],
                       np.ones((1, 1, 31)), max_species=.2))


@pytest.mark.parametrize("method", ["adagrad", "adam"])
def test_feasible_step_retries_without_reaccumulating(worker, method):
    key = worker.prepare(*bounded_birth_problem())
    initial = worker.evaluate(key, [.08])
    result = worker.optimize(key, [[.08]], steps=1, method=method, learning_rate=1.,
                             step_policy="feasible", max_backtracks=4)["results"][0]
    assert result["status"] == "ok"
    assert result["state"]["step"] == 1
    assert result["evaluations"] == 5
    assert result["completed_evaluations"] == 2
    assert [t["status"] for t in result["trials"]] == ["rejected"] * 3 + ["accepted"]
    assert result["trials"][-1]["scale"] == .125
    g = .08 * initial["gradient"][0]
    if method == "adagrad":
        assert result["state"]["accumulator"] == pytest.approx([g*g])
    else:
        assert result["state"]["moment"] == pytest.approx([.1*g])
    assert result["best_loss"] < initial["loss"]
    assert worker.evaluate(key, result["final_rates"])["loss"] == pytest.approx(result["final_loss"])


def test_feasible_budget_exhaustion_preserves_valid_state(worker):
    key = worker.prepare(*bounded_birth_problem())
    result = worker.optimize(key, [[.08]], steps=10, learning_rate=1., step_policy="feasible",
                             max_evaluations=2)["results"][0]
    assert result["status"] == "evaluation_budget_exhausted"
    assert result["evaluations"] == 2
    assert result["state"]["step"] == 0
    assert result["state"]["accumulator"] == [0.]
    assert result["final_rates"] == pytest.approx([.08])
    assert result["final_loss"] == pytest.approx(.92)
    result = worker.optimize(key, [[.08], [1.]], steps=1, learning_rate=1.,
                             step_policy="feasible", max_backtracks=0)["results"]
    assert [r["status"] for r in result] == ["backtracking_exhausted", "evaluation_failed"]
    assert result[0]["state"]["step"] == 0
    assert result[1]["best_loss"] is None


def test_feasible_exact_continuation(worker):
    key = worker.prepare(*bounded_birth_problem())
    options = dict(step_policy="feasible", learning_rate=1., max_backtracks=12)
    whole = worker.optimize(key, [[.08]], steps=3, **options)["results"][0]
    first = worker.optimize(key, [[.08]], steps=1, **options)["results"][0]
    tail = worker.optimize(key, states=[first["state"]], steps=2, **options)["results"][0]
    assert whole["status"] == tail["status"] == "ok"
    assert whole["final_rates"] == pytest.approx(tail["final_rates"], abs=1e-12)
    assert whole["state"]["accumulator"] == pytest.approx(tail["state"]["accumulator"], abs=1e-12)
    assert whole["evaluations"] == first["evaluations"] + tail["evaluations"]


@pytest.mark.parametrize("step_policy", ["stop", "feasible"])
def test_task_l1_has_no_rate_penalty_or_pruning(worker, step_policy):
    crn = CRNSpec(("X", "Y"), (), (
        Reaction((0, 0), (1, 0)), Reaction((0, 0), (0, 1)),
    ))
    exp = Experiment(np.linspace(0, 2, 31), [[0., 0.]], [[]], [[1., 0.]],
                     np.zeros((1, 1, 31)), loss="l1")
    key = worker.prepare(crn, exp)
    # Y is unobserved and uncoupled: its rate must not contribute a sparsity loss.
    for rate in (.001, 30.):
        evaluated = worker.evaluate(key, [.2, rate])
        assert evaluated["loss"] == pytest.approx(.2, abs=1e-10)
        assert evaluated["gradient"] == pytest.approx([1., 0.], abs=1e-10)
        result = worker.optimize(key, [[.2, rate]], steps=3,
                                 step_policy=step_policy)["results"][0]
        assert result["status"] == "ok"
        assert len(result["final_rates"]) == len(result["best_rates"]) == 2
        assert result["final_rates"][1] == pytest.approx(rate, rel=1e-12)
        assert result["best_rates"][1] == pytest.approx(rate, rel=1e-12)
