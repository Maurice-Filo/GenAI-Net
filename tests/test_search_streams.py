from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace
import zlib

import numpy as np
import pytest

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.evaluation import build, EvaluationService, evaluate_once, materialize
from RL4CRN.search_exploration.refinement import fit
from RL4CRN.search_exploration.session import build_session
from RL4CRN.search_exploration.store import Store, BudgetExhausted, canonical
from RL4CRN.search_streams.store import CompactStore
from RL4CRN.search_streams.workers import StreamPool
from RL4CRN.search_streams.engine import StreamTrainer
from RL4CRN.search_streams.archive import StreamDiscoveryArchive
from RL4CRN.utils.results_database import serialize_crn
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS


def candidate(ids=(2, 3), rates=(1., 2.)):
    return {"reaction_ids": list(ids), "rates": list(rates)}


def test_source_interaction_uses_event_attribution_and_deduplicates_topologies():
    from comparisons.search_exploration.analyze_source_interaction import summarize
    designs = {
        "a": {"loss": .01, "simulation_complete": True, "topology_hash": "A", "source": "ga"},
        "b": {"loss": .005, "simulation_complete": True, "topology_hash": "A"},
        "c": {"loss": .002, "simulation_complete": True, "topology_hash": "B"},
    }
    events = [
        {"candidate_hash": "a", "source": "rl", "stream": "search"},
        {"candidate_hash": "b", "source": "rl", "stream": "optimization"},
        {"candidate_hash": "c", "source": "ga", "stream": "search", "parent_hash": "b", "origin": "rl"},
        {"candidate_hash": "c", "source": "rl", "stream": "search"},
    ]
    result = summarize(designs, events, .02)
    groups = {row["group"]: row for row in result["groups"]}
    assert result["distinct_qualifying_topologies"] == 2
    assert groups["rl/search"]["first_qualifications"] == 1
    assert groups["rl/optimization"]["first_qualifications"] == 0
    assert groups["ga/search"]["first_qualifications"] == 1
    assert groups["rl/search"]["qualifying_topologies_seen"] == 2
    assert result["direct_archive_mutation"]["qualified_origin/rl"] == 1
    assert result["direct_archive_mutation"]["improved_parent_loss"] == 1


def test_archive_retains_high_utility_nonrepresentative_before_poor_design():
    counts = {}
    def count(key, n=0):
        counts[key] = counts.get(key, 0) + n
        return counts[key]
    store = SimpleNamespace(count=count, save_design=lambda record: None)
    archive = StreamDiscoveryArchive(ExplorationConfig(archive_capacity=2, sil_top_k=2))
    for name, loss, qualified in (("best", .001, True), ("poor", 1., False), ("good", .002, True)):
        archive.add({"topology_hash": name, "candidate_hash": name, "region": "same",
            "loss": loss, "qualified": qualified, "simulation_complete": True,
            "reaction_ids": [2], "rates": [1.]}, store)
    assert set(archive.entries) == {"best", "good"}
    restored = StreamDiscoveryArchive(archive.cfg)
    restored.load_state_dict(archive.state_dict())
    assert restored.ranked() == archive.ranked()


def test_compact_cache_exact_keys_diagnostics_and_reopen(tmp_path):
    path = tmp_path / "cache.sqlite"
    store = CompactStore(path, 8, "test", 4, memory_entries=1)
    c = candidate()
    try:
        action, key = store.reserve(c, "search")
        assert action == "evaluate"
        store.complete(key, {"loss": .1})
        swapped = candidate((3, 2), (2., 1.))
        assert store.reserve(swapped, "search") == ("cached", {"loss": .1})
        assert store.db.execute("SELECT count(*) FROM diagnostics").fetchone()[0] == 0
        assert store.cached(c)["task_info"] == {}
        assert store.cached(c, include_diagnostics=False) == {"loss": .1}
        blob = zlib.compress(canonical({"outputs": [1., 2.]}).encode())
        store.retain_diagnostics(c, blob)
        assert store.cached(swapped)["task_info"]["outputs"] == [1., 2.]
        changed = candidate(rates=(1., np.nextafter(2., 3.)))
        assert store.reserve(changed, "search")[0] == "evaluate"
        assert store.count("expensive_evaluations") == 2
        store.fail(store.evaluation_key(changed), "simulated failure")
    finally:
        store.close()
    reopened = CompactStore(path, 8, "test", 4)
    try:
        assert reopened.cached(c, include_diagnostics=False) == {"loss": .1}
        assert reopened.count("expensive_evaluations") == 2
    finally:
        reopened.close()


def test_shared_budget_protects_search_and_charges_failures(tmp_path):
    store = CompactStore(tmp_path / "cache.sqlite", 3, "test", 2)
    try:
        _, key = store.reserve(candidate(), "optimization")
        store.fail(key, "failure remains charged")
        with pytest.raises(BudgetExhausted, match="search budget protected"):
            store.reserve(candidate(rates=(2., 3.)), "optimization")
        for rate in (3., 4.):
            _, key = store.reserve(candidate(rates=(rate, 1.)), "search")
            store.complete(key, {"loss": rate})
        with pytest.raises(BudgetExhausted):
            store.reserve(candidate(rates=(5., 1.)), "search")
        assert store.count("expensive_evaluations") == 3
    finally:
        store.close()


def test_duplicate_inflight_coalesces_without_holding_lock(tmp_path):
    store = CompactStore(tmp_path / "cache.sqlite", 3, "test", 2)
    try:
        _, key = store.reserve(candidate(), "search")
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(store.reserve, candidate(), "optimization")
            time.sleep(.02)
            store.complete(key, {"loss": .2})
            assert future.result(timeout=1) == ("cached", {"loss": .2})
        assert store.count("expensive_evaluations") == 1
    finally:
        store.close()


class FakeStreamPool:
    def __init__(self, evaluator, store, gate=None):
        self.evaluator, self.store, self.gate = evaluator, store, gate
        self.slots = [None]

    def run(self, job):
        c = dict(job["candidate"])
        if self.gate:
            assert self.gate.wait(10), "test gate not released"
            c["rates"] = [2.] * len(c["rates"])
        action, value = self.store.reserve(c, job["category"])
        if action == "evaluate":
            identity = serialize_crn(materialize(self.evaluator, c).state)
            value_result = {"loss": .001 if self.gate else .2,
                "topology_hash": identity["topology_hash"], "candidate_hash": identity["candidate_hash"],
                "simulation_complete": True, "all_rows_correct": True}
            self.store.complete(value, value_result)
        else:
            value_result = value
        return {"best": {**c, **value_result}, "calls": 1, "failures": [], "termination": "completed",
                "packed_diagnostics": zlib.compress(b"{}")}


@pytest.mark.parametrize("rl_enabled", [True, False])
@pytest.mark.parametrize("mutation_enabled", [True, False])
def test_delayed_optimizer_does_not_block_search_or_supply_onpolicy_rewards(tmp_path, rl_enabled, mutation_enabled):
    cfg = ExplorationConfig(epochs=2, batch_size=4, evaluation_budget=50, workers=1,
        fit_evaluations=2, archive_capacity=20, sil_top_k=4, sil_batch_size=2, plot_every=10)
    if not rl_enabled:
        cfg = replace(cfg, rl_fraction=0.)
    if not mutation_enabled:
        cfg = replace(cfg, mutation_fraction=0.)
    evaluator, task, sc = build("rpa", json.loads(CONFIGS["rpa"].read_text()), cfg.rate_min, cfg.rate_max)
    sc.train.seed, sc.train.batch_size, sc.train.n_cpus = 0, 4, 1
    sc.policy.width, sc.policy.depth, sc.policy.deep_layer_size = 16, 1, 16
    session = build_session(sc, task)
    store = CompactStore(tmp_path / "exploration.sqlite", 50, "test", 8)
    gate = threading.Event()
    search = FakeStreamPool(evaluator, store)
    optimizer = FakeStreamPool(evaluator, store, gate)
    allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
    service = EvaluationService(store, search, allowed, 5, (.001, 100.))
    trainer = StreamTrainer(session, evaluator, service, cfg, tmp_path, search_pool=search,
        optimization_pool=optimizer, optimization_workers=1, queue_capacity=2, optimize_per_epoch=2)
    if not rl_enabled:
        def forbidden_update(*args, **kwargs):
            pytest.fail("No-RL mode must not update the policy or sample imitation elites")
        trainer.learner.update = forbidden_update
        trainer.discovery.imitation = forbidden_update
    try:
        trainer.step_epoch()
        trainer.step_epoch()
        assert trainer.state.epoch == 2
        assert trainer.learner.version == (2 if rl_enabled else 0)
        if not mutation_enabled:
            assert store.count("proposals/ga") == 0
        if not rl_enabled:
            assert store.count("proposals/rl") == 0
            assert trainer.last_metrics["Policy/enabled"] == 0
        assert trainer.optimization_merged == 0
        assert len(trainer.optimization_jobs) == 2
        assert trainer.discovery.ranked(1)[0]["loss"] == .2
        trainer.flush_results_database()
        import sqlite3
        with sqlite3.connect(tmp_path / "results.sqlite") as database:
            classes = [r[0] for r in database.execute("SELECT provenance_class FROM hof_snapshot_entries")]
            assert classes and all(c.startswith("direct_") for c in classes)
        assert store.count("optimization/queue_full_epochs") == 1
        version = trainer.learner.version
        gate.set()
        trainer.finish()
        assert trainer.learner.version == version
        assert not trainer.optimization_jobs
        assert trainer.optimization_improved == 2
        assert trainer.discovery.ranked(1)[0]["loss"] == .001
        with sqlite3.connect(tmp_path / "results.sqlite") as database:
            classes = [r[0] for r in database.execute("SELECT provenance_class FROM hof_snapshot_entries")]
            assert any(c.endswith("_seed_refined") for c in classes)
        records = [json.loads(p[0]) for p in store.db.execute("SELECT payload FROM events WHERE kind='archive_insertion'")]
        late = [r for r in records if r["stream"] == "optimization"]
        assert late and all(r["epoch"] == 2 and r["policy_version"] == 0 and r["parent_hash"] for r in late)
        with pytest.raises(ValueError, match="reconciliation"):
            trainer.load(tmp_path / "checkpoint.pt")
    finally:
        gate.set()
        trainer.close_streams()
        trainer.close_results_database()
        store.close()


def test_worker_fitting_matches_reference_cvode_and_reuses_initial_loss(tmp_path):
    task_config = json.loads(CONFIGS["rpa"].read_text())
    cfg = ExplorationConfig(fit_evaluations=4, evaluation_budget=30, fit_seconds=60, workers=1)
    evaluator, _, _ = build("rpa", task_config, cfg.rate_min, cfg.rate_max)
    ids = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids][:5]
    c = {"reaction_ids": ids, "rates": [1.] * 5, "proposal_id": "test", "source": "rl", "origin": "rl"}
    reference = Store(tmp_path / "reference.sqlite", 30, "test")
    compact = CompactStore(tmp_path / "compact.sqlite", 30, "test", 10)
    class DirectPool:
        def evaluate(self, proposal, timeout=None):
            return evaluate_once(evaluator, proposal)
    service = EvaluationService(reference, DirectPool(), ids, 5, (.001, 100.))
    pool = StreamPool(("rpa", task_config, .001, 100.), 1, 30, compact, 60)
    try:
        expected = fit(c, service, cfg, 123, include_diagnostics=False)
        first = pool.run({"category": "search", "candidate": c})
        assert first["best"]["loss"] == evaluate_once(evaluator, c)["loss"]
        actual = pool.run({"category": "optimization", "candidate": c, "config": cfg.to_dict(), "seed": 123})
        assert actual["failures"] == []
        assert actual["best"]["loss"] == expected["best"]["loss"]
        assert actual["best"]["rates"] == expected["best"]["rates"]
        assert actual["cache_hits"] >= 1
        assert compact.count("expensive_evaluations") == reference.count("expensive_evaluations")
        assert compact.db.execute("SELECT count(*) FROM diagnostics").fetchone()[0] == 0
        compatibility = EvaluationService(compact, pool, ids, 5, (.001, 100.))
        changed = {**c, "rates": [3.] * 5}
        before = compact.count("expensive_evaluations")
        result, hit = compatibility.evaluate(changed, category="llm")
        assert not hit and "task_info" in result
        assert compact.count("expensive_evaluations") == before + 1
        assert compact.count("evaluations/llm") == 1
        again, hit = compatibility.evaluate(changed, category="llm")
        assert hit and again["loss"] == result["loss"]
    finally:
        pool.close()
        compact.close()
        reference.close()


def test_incomplete_starting_trajectory_can_enter_random_refinement_slots(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import RL4CRN.search_streams.engine as engine
    trainer = object.__new__(StreamTrainer)
    trainer.store = CompactStore(tmp_path / "cache.sqlite", 30, "test", 10)
    trainer.optimization_jobs, trainer.scheduled_keys = {}, set()
    trainer.queue_capacity, trainer.optimize_per_epoch = 4, 4
    trainer.config = ExplorationConfig()
    trainer.optimizer_rng = np.random.default_rng(0)
    trainer.state = SimpleNamespace(epoch=0)
    trainer.library = {}
    trainer.evaluator = SimpleNamespace(crn_template=SimpleNamespace(output_labels=[]))
    trainer.optimization_pool = SimpleNamespace(run=lambda job: None)
    trainer.optimizer_executor = SimpleNamespace(submit=lambda *args: Future())
    monkeypatch.setattr(engine, "descriptor", lambda *args: "test-region")
    record = {**candidate(), "simulation_complete": False, "loss": 10000., "topology_hash": "topology",
              "candidate_hash": "candidate", "origin": "rl", "source": "rl", "proposal_id": "test"}
    try:
        assert trainer._schedule([record]) == 1
        assert len(trainer.optimization_jobs) == 1
        assert trainer._schedule([record]) == 0
    finally:
        trainer.store.close()
