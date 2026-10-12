from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.evaluation import build, EvaluationService, evaluate_once
from RL4CRN.search_exploration.engine import ExplorationTrainer
from RL4CRN.search_exploration.harness import CachedHarnessEvaluator
from RL4CRN.search_exploration.store import Store
from RL4CRN.utils.input_interface import Session
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS


def make_trainer(path, config=None, pool_type=None):
    cfg = config or ExplorationConfig(epochs=2, batch_size=3, workers=1, fit_evaluations=2,
        evaluation_budget=50, plot_every=10, sil_batch_size=2, sil_top_k=4, archive_capacity=30)
    evaluator, task, session_cfg = build("logic", json.loads(CONFIGS["logic"].read_text()), cfg.rate_min, cfg.rate_max)
    session_cfg.train.seed = cfg.seed
    session_cfg.train.n_cpus = 1
    session_cfg.train.batch_size = cfg.batch_size
    session_cfg.policy.width = 16
    session_cfg.policy.depth = 1
    session_cfg.policy.deep_layer_size = 16
    session = Session.from_config(session_cfg, task, device="cpu")
    path.mkdir(parents=True, exist_ok=True)
    store = Store(path / "exploration.sqlite", cfg.evaluation_budget, "test-v1")
    class FakePool:
        def evaluate(self, c, timeout=None):
            from RL4CRN.search_exploration.evaluation import materialize
            from RL4CRN.utils.results_database import serialize_crn
            identity = serialize_crn(materialize(evaluator, c).state)
            loss = .01 + sum(c["reaction_ids"]) / 1000000
            return {"loss": loss, "topology_hash": identity["topology_hash"],
                    "candidate_hash": identity["candidate_hash"], "simulation_complete": True,
                    "all_rows_correct": True, "task_info": {}}
    service = EvaluationService(store, (pool_type or FakePool)(),
        [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids],
        evaluator.max_added_reactions, (cfg.rate_min, cfg.rate_max))
    trainer = ExplorationTrainer(session, evaluator, service, cfg, path)
    return trainer


def close(trainer):
    trainer.clear_llm_graph(wait=True)
    trainer.close_results_database()
    trainer.store.close()


def test_structural_policy_no_rate_gradients_and_no_invalid_actions(tmp_path):
    trainer = make_trainer(tmp_path)
    try:
        context = trainer.context()
        selected = trainer.learner.sample(6, context)
        assert all(len(set(ids)) == 5 and not set(ids) & trainer.evaluator.forbidden_reaction_ids for ids in selected)
        proposals = [{"reaction_ids": ids, "policy_version": 0} for ids in selected]
        metrics = trainer.learner.update(proposals, [1, 0, 1, 0, 2, 0], context, [])
        assert metrics["Policy/gradient_norm"] > 0
        assert all(p.grad is None for p in trainer.s.policy.continuous_parameter_generator.parameters())
        with pytest.raises(ValueError, match="Stale"):
            trainer.learner.update(proposals, [0]*6, context, [])
    finally:
        close(trainer)


def test_cached_harness_members_and_canonical_loss(tmp_path):
    from RL4CRN.llm.schemas import LLMCandidate
    trainer = make_trainer(tmp_path)
    try:
        proxy = CachedHarnessEvaluator(trainer.evaluator, trainer.service)
        ids = trainer.allowed[:5]
        good = LLMCandidate(ids, [[1.]]*5)
        bad = LLMCandidate([0, *ids[:4]], [[1.]]*5)
        results = proxy.evaluate_many([good, bad, good])
        assert [r.valid for r in results] == [True, False, True]
        assert results[0].loss == results[2].loss
        assert results[0].exploration_record["reaction_ids"] == ids
        assert trainer.store.count("expensive_evaluations") == 1
        assert trainer.store.count("evaluations/llm") == 1
    finally:
        close(trainer)


def test_checkpoint_matches_uninterrupted_policy_and_archive(tmp_path):
    uninterrupted = make_trainer(tmp_path / "a")
    interrupted = None
    resumed = None
    try:
        uninterrupted.step_epoch()
        uninterrupted.step_epoch()
        expected = uninterrupted.learner.state_dict()
        expected_archive = uninterrupted.discovery.ranked()
        interrupted = make_trainer(tmp_path / "b")
        interrupted.step_epoch()
        close(interrupted)
        interrupted = None
        resumed = make_trainer(tmp_path / "b")
        resumed.load(tmp_path / "b/checkpoint.pt")
        resumed.step_epoch()
        for key, value in expected.items():
            torch.testing.assert_close(value, resumed.learner.state_dict()[key], rtol=0, atol=0)
        assert expected_archive == resumed.discovery.ranked()
        assert uninterrupted.store.counters() == resumed.store.counters()
    finally:
        close(uninterrupted)
        if interrupted:
            close(interrupted)
        if resumed:
            close(resumed)


def test_pending_cohort_saved_before_dispatch(tmp_path):
    trainer = make_trainer(tmp_path)
    try:
        trainer.pending = trainer._new_proposals()
        trainer.save(tmp_path / "checkpoint.pt")
        saved = json.loads(json.dumps(trainer.pending))
        trainer.pending = None
        trainer.load(tmp_path / "checkpoint.pt")
        assert trainer.pending == saved
        trainer.step_epoch()
        assert trainer.state.epoch == 1
    finally:
        close(trainer)


@pytest.mark.parametrize("outward", [True, False])
def test_request_zero_withheld_and_later_context_isolated(tmp_path, outward):
    from RL4CRN.llm.graphs import LLMGraphRunResult, default_decider_writer_spec
    calls = []
    class Graph:
        client = SimpleNamespace(exploration_context=None)
        def run_round(self, **kwargs):
            calls.append((kwargs, self.client.exploration_context))
            return LLMGraphRunResult(default_decider_writer_spec(), "", "", {}, [], [])
    trainer = make_trainer(tmp_path)
    try:
        # Populate the HOF before request zero to prove withholding is not merely an empty archive.
        trainer.step_epoch()
        trainer.configure_llm_graph(Graph(), every=1, start_epoch=0, task_description="test",
            cross_communication=outward, withhold_initial_hof=True)
        trainer._maybe_run_llm_graph(0)
        trainer.wait_for_llm_graph(timeout=5)
        assert calls[0][0]["hall_of_fame_iter"] == ()
        assert calls[0][1] is None
        trainer._maybe_run_llm_graph(1)
        trainer.wait_for_llm_graph(timeout=5)
        assert bool(calls[1][0]["hall_of_fame_iter"]) is outward
        assert (calls[1][1] is not None) is outward
    finally:
        close(trainer)


def test_inward_disabled_keeps_results_outside_training_archive(tmp_path):
    from RL4CRN.llm.schemas import LLMCandidate
    from RL4CRN.llm.graphs import LLMGraphRunResult, default_decider_writer_spec
    trainer = make_trainer(tmp_path)
    proxy = CachedHarnessEvaluator(trainer.evaluator, trainer.service)
    evaluation = proxy.evaluate(LLMCandidate(trainer.allowed[:5], [[1.]]*5))
    class Graph:
        def run_round(self, **kwargs):
            return LLMGraphRunResult(default_decider_writer_spec(), "", "", {}, [evaluation.candidate], [evaluation])
    try:
        trainer.configure_llm_graph(Graph(), every=1, task_description="test", add_to_hall_of_fame=False,
                                    cross_communication=False, withhold_initial_hof=True)
        trainer._maybe_run_llm_graph(0)
        trainer.wait_for_llm_graph(timeout=5)
        assert not trainer.discovery.entries
        assert not trainer.s.mult_env.hall_of_fame
        assert len(trainer._llm_loop["isolated_candidates"]) == 1
        trainer.merge_isolated_llm_candidates()
        assert len(trainer.s.mult_env.hall_of_fame) == 1
    finally:
        close(trainer)


def test_fake_comet_receives_metrics_and_figures(tmp_path):
    from RL4CRN.search_exploration.telemetry import Telemetry
    class Logger:
        metrics, figures = [], []
        def log_metrics(self, values, step):
            self.metrics.append((values, step))
        def log_figure(self, **kwargs):
            self.figures.append(kwargs["figure_name"])
    logger = Logger()
    telemetry = Telemetry(tmp_path, logger)
    telemetry.metrics(0, {"Archive/best_loss": .1, "Archive/qualified_regions": 2,
                         "Archive/qualified_topologies": 3, "Compute/expensive_evaluations": 5})
    telemetry.plot()
    assert len(logger.metrics) == 1 and len(logger.figures) == 2
    assert (tmp_path / "progress.pdf").stat().st_size > 1000
    assert (tmp_path / "sources.png").stat().st_size > 1000


def test_compatibility_sample_and_snapshot_provenance(tmp_path):
    trainer = make_trainer(tmp_path)
    try:
        hof = trainer.sample(2, 2)
        assert len(hof) == 2
        assert trainer.learner.version == 0
        assert not trainer.discovery.entries
        trainer.step_epoch()
        trainer.flush_results_database()
        import sqlite3
        with sqlite3.connect(tmp_path / "results.sqlite") as conn:
            row = conn.execute("SELECT cumulative_total_evaluations FROM hof_snapshots").fetchone()
            assert row[0] == trainer.store.count("expensive_evaluations")
            labels = conn.execute("SELECT emitter,provenance_class FROM hof_snapshot_entries").fetchall()
            assert labels and all(provenance.endswith("_seed_refined") for _, provenance in labels)
    finally:
        close(trainer)


def test_structural_refinement_attribution_and_budget(tmp_path):
    cfg = ExplorationConfig(epochs=1, batch_size=2, workers=1, fit_evaluations=2,
        structural_refinement_steps=1, evaluation_budget=20, plot_every=10,
        sil_batch_size=2, sil_top_k=4, archive_capacity=30)
    trainer = make_trainer(tmp_path, cfg)
    try:
        trainer.step_epoch()
        assert trainer.store.count("expensive_evaluations") <= 8
        events = trainer.store.db.execute("SELECT payload FROM events WHERE kind='archive_insertion'").fetchall()
        records = [json.loads(row[0]) for row in events]
        assert any(r["source"] == "ga" and r["parent_hash"] for r in records)
    finally:
        close(trainer)


def _slow_evaluation_worker(connection, delay):
    import time
    connection.send(("ready", None))
    try:
        while True:
            connection.recv()
            time.sleep(delay)
            connection.send(("ok", {"loss": 1.}))
    except (EOFError, BrokenPipeError):
        pass


def test_worker_timeout_kills_and_replaces_process():
    import multiprocessing
    import queue
    from RL4CRN.search_exploration.evaluation import EvaluatorPool
    pool = object.__new__(EvaluatorPool)
    pool.timeout = .05
    pool.available = queue.Queue()
    pool.slots = []
    ctx = multiprocessing.get_context("spawn")
    def start(delay=0):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_slow_evaluation_worker, args=(child, delay))
        p.start()
        child.close()
        assert parent.poll(30)
        assert parent.recv()[0] == "ready"
        return p, parent
    original = start(2)
    pool.slots = [original]
    pool.available.put(original)
    pool._start = start
    try:
        with pytest.raises(TimeoutError):
            pool.evaluate({})
        assert not original[0].is_alive()
        assert pool.evaluate({})["loss"] == 1
    finally:
        pool.close()


def reference_trajectories(learner, count, context, sequences=None):
    from RL4CRN.search_exploration.evaluation import materialize
    envs = [materialize(learner.evaluator, {"reaction_ids": [], "rates": []}) for _ in range(count)]
    selected, logps, entropies, states = [[] for _ in envs], [], [], []
    for step in range(learner.evaluator.max_added_reactions):
        state = torch.stack([learner.session.tensorizer.tensorize(learner.session.observer.observe(e.state)) for e in envs])
        states.append(state)
        dist = learner.distribution_from_state(state, context)
        actions = dist.sample() if sequences is None else torch.tensor([s[step] for s in sequences])
        logps.append(dist.log_prob(actions))
        entropies.append(dist.entropy())
        for i, rid in enumerate(actions.tolist()):
            selected[i].append(rid)
            raw = {"reaction index": rid, "parameters": [learner.cfg.initial_rate],
                   "continuous parameters": [learner.cfg.initial_rate], "discrete parameters": None}
            envs[i].step(learner.evaluator.actuator.actuate(raw), learner.evaluator.stepper, raw_action=raw)
    return selected, torch.stack(logps).sum(0), torch.stack(entropies).mean(0), states


@pytest.mark.parametrize("task_name", ["logic", "rpa"])
def test_tensor_rollout_matches_observations_logits_rng_and_gradients(task_name):
    from RL4CRN.search_exploration.policy import StructuralLearner
    cfg = ExplorationConfig()
    evaluator, task, session_cfg = build(task_name, json.loads(CONFIGS[task_name].read_text()), cfg.rate_min, cfg.rate_max)
    session_cfg.train.n_cpus, session_cfg.train.batch_size = 1, 3
    session_cfg.policy.width, session_cfg.policy.depth, session_cfg.policy.deep_layer_size = 16, 1, 16
    session = Session.from_config(session_cfg, task, device="cpu")
    learner = StructuralLearner(session, evaluator, cfg)
    context = [.2, .1, .3, .4]
    rng = torch.get_rng_state()
    actual = learner.trajectories(3, context)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    expected = reference_trajectories(learner, 3, context)
    assert actual[0] == expected[0]
    assert torch.equal(after, torch.get_rng_state())
    for a, b in zip(learner._sample_cache[2], expected[3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(actual[1:], expected[1:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    parameters = list(learner.parameters())
    a = torch.autograd.grad(actual[1].mean() + .1*actual[2].mean(), parameters)
    b = torch.autograd.grad(expected[1].mean() + .1*expected[2].mean(), parameters)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    replay = learner.trajectories(3, context, actual[0])
    torch.testing.assert_close(replay[1], actual[1], rtol=0, atol=0)
    learner._sample_cache = None
    rebuilt = learner.trajectories(3, context, actual[0])
    torch.testing.assert_close(rebuilt[1], replay[1], rtol=0, atol=0)


def test_structure_only_session_preserves_weights_and_rng_without_rate_head():
    from copy import deepcopy
    from RL4CRN.search_exploration.session import build_session
    cfg = ExplorationConfig()
    evaluator, task, sc = build("logic", json.loads(CONFIGS["logic"].read_text()), cfg.rate_min, cfg.rate_max)
    sc.train.n_cpus, sc.train.batch_size = 1, 2
    sc.policy.width, sc.policy.depth, sc.policy.deep_layer_size = 256, 1, 512
    original = Session.from_config(deepcopy(sc), task, device="cpu")
    rng = torch.get_rng_state()
    optimized = build_session(deepcopy(sc), task, device="cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert optimized.policy.continuous_parameter_generator is None
    for name in ("encoder", "reaction_structure_head"):
        for a, b in zip(getattr(original.policy, name).parameters(), getattr(optimized.policy, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_pending_journal_recovers_without_rewriting_model(tmp_path, monkeypatch):
    import RL4CRN.search_exploration.engine as engine
    expected = make_trainer(tmp_path / "expected")
    interrupted = None
    resumed = None
    try:
        expected.step_epoch()
        expected.step_epoch()
        interrupted = make_trainer(tmp_path / "interrupted")
        writes = []
        original_write = engine.atomic_torch
        def write(path, value):
            writes.append((Path(path).name, "optimizer" in value))
            return original_write(path, value)
        monkeypatch.setattr(engine, "atomic_torch", write)
        def fail(proposal):
            raise RuntimeError("simulated power interruption before fitting")
        monkeypatch.setattr(interrupted, "_fit_burst", fail)
        with pytest.raises(RuntimeError, match="interruption"):
            interrupted.step_epoch()
        assert writes == [("checkpoint.pt", True), ("pending.pt", False)]
        journal = torch.load(tmp_path / "interrupted/pending.pt", weights_only=False)
        assert journal["pending"] is not None and "policy" not in journal
        close(interrupted)
        interrupted = None
        resumed = make_trainer(tmp_path / "interrupted")
        resumed.load(tmp_path / "interrupted/checkpoint.pt")
        resumed.step_epoch()
        resumed.step_epoch()
        for name, value in expected.learner.state_dict().items():
            torch.testing.assert_close(value, resumed.learner.state_dict()[name], rtol=0, atol=0)
        assert expected.discovery.ranked() == resumed.discovery.ranked()
        assert expected.store.counters() == resumed.store.counters()
        assert sum(name == "checkpoint.pt" for name, _ in writes) == 3
        assert sum(name == "pending.pt" for name, _ in writes) == 2
        # The old pending journal must not override a newer completed checkpoint.
        resumed.load(tmp_path / "interrupted/checkpoint.pt")
        assert resumed.pending is None and resumed.state.epoch == 2
    finally:
        close(expected)
        if interrupted:
            close(interrupted)
        if resumed:
            close(resumed)


def test_dispatch_intent_fails_closed_on_resume(tmp_path):
    import RL4CRN.search_exploration.engine as engine
    trainer = make_trainer(tmp_path)
    try:
        trainer.save(tmp_path / "checkpoint.pt")
        engine.atomic_torch(tmp_path / "pending.pt", {**trainer._runtime_state(),
            "base_checkpoint_id": trainer._checkpoint_id, "llm_dispatch_in_progress": True})
        with pytest.raises(ValueError, match="reconciliation"):
            trainer.load(tmp_path / "checkpoint.pt")
    finally:
        close(trainer)
