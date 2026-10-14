from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import time

import numpy as np
import pytest

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.store import Store, BudgetExhausted
from RL4CRN.search_exploration.archive import DiscoveryArchive
from RL4CRN.search_exploration.refinement import fit


def candidate(rate=1.):
    return {"reaction_ids": [5, 8], "rates": [rate, 1.], "proposal_id": "p", "source": "rl"}


def test_cache_exact_order_independent_budget_and_failure(tmp_path):
    store = Store(tmp_path / "store.sqlite", 2, "evaluator-v1")
    def compute(c):
        return {"loss": sum(c["rates"])}
    assert not store.evaluate(candidate(), compute)[1]
    reverse = {**candidate(), "reaction_ids": [8, 5]}
    assert store.evaluate(reverse, compute)[1]
    assert not store.evaluate(candidate(np.nextafter(1., 2.)), compute)[1]
    with pytest.raises(BudgetExhausted):
        store.evaluate(candidate(2.), compute)
    assert store.count("expensive_evaluations") == 2
    store.close()
    with pytest.raises(ValueError, match="identity"):
        Store(tmp_path / "store.sqlite", 2, "changed-evaluator")


def test_coalesces_concurrent_identical_evaluations(tmp_path):
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    started, finish = threading.Event(), threading.Event()
    def compute(c):
        started.set()
        assert finish.wait(5)
        return {"loss": 1.}
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(store.evaluate, candidate(), compute)
        assert started.wait(5)
        b = pool.submit(store.evaluate, candidate(), compute)
        time.sleep(.05)
        finish.set()
        assert a.result()[0] == b.result()[0]
    assert store.count("expensive_evaluations") == 1
    store.close()


def test_failures_not_cached_and_budget_not_refunded(tmp_path):
    store = Store(tmp_path / "store.sqlite", 3, "v1")
    def broken(c):
        raise TimeoutError("numerical worker")
    for _ in range(2):
        with pytest.raises(TimeoutError):
            store.evaluate(candidate(), broken)
    assert store.count("expensive_evaluations") == 2
    assert store.count("cache_hits") == 0
    store.close()


def record(i, loss, region="a"):
    return {"candidate_hash": f"c{i}", "topology_hash": f"t{i}", "loss": loss,
            "region": region, "qualified": loss < .05, "simulation_complete": True,
            "reaction_ids": [5, 8], "rates": [1., 1.], "origin": "rl"}


def test_archive_repeat_novelty_and_small_top_imitation(tmp_path):
    cfg = ExplorationConfig(archive_capacity=100, sil_top_k=3, sil_batch_size=2)
    archive = DiscoveryArchive(cfg)
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    gain, novelty = archive.add(record(0, .01), store)
    again, next_novelty = archive.add(record(0, .01), store)
    assert gain > 0 and again == 0 and next_novelty < novelty
    for i in range(1, 20):
        archive.add(record(i, .01 + i / 1000, str(i % 4)), store)
    elites = archive.imitation(np.random.default_rng(0))
    assert len(archive.entries) == 20 and len(elites) == 2
    assert {r["topology_hash"] for r in elites} <= {r["topology_hash"] for r in archive.ranked(3)}
    state = archive.state_dict()
    other = DiscoveryArchive(cfg)
    other.load_state_dict(state)
    assert other.ranked() == archive.ranked()
    store.close()


def test_bounded_archive_and_full_disk_history(tmp_path):
    cfg = ExplorationConfig(archive_capacity=3, sil_top_k=2)
    archive = DiscoveryArchive(cfg)
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    for i in range(10):
        archive.add(record(i, .01 + i / 100), store)
    assert len(archive.entries) == 3
    assert store.db.execute("SELECT COUNT(*) FROM designs").fetchone()[0] == 10
    assert len(archive.heap) <= 3 * cfg.archive_capacity
    store.close()


@pytest.mark.parametrize("backend", ["evolution", "powell"])
def test_fitting_budget_bounds_and_exact_replay(tmp_path, backend):
    cfg = ExplorationConfig(optimizer=backend, fit_evaluations=8)
    store = Store(tmp_path / "store.sqlite", 50, "v1")
    class Service:
        def __init__(self):
            self.store = store
        def evaluate(self, c, **kwargs):
            assert all(cfg.rate_min <= v <= cfg.rate_max for v in c["rates"])
            return store.evaluate(c, lambda c: {"loss": sum((v - 2)**2 for v in c["rates"]),
                                                "simulation_complete": True})
    result = fit(candidate(), Service(), cfg, 12)
    assert result["calls"] <= 8
    assert result["best"]["loss"] <= 2
    count = store.count("expensive_evaluations")
    assert fit(candidate(), Service(), cfg, 12) == result
    assert store.count("expensive_evaluations") == count
    store.close()


def test_failed_fit_retryable_not_excluded(tmp_path):
    cfg = ExplorationConfig(fit_evaluations=2)
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    class Service:
        def __init__(self):
            self.store = store
        def evaluate(self, c, **kwargs):
            raise TimeoutError("worker killed")
    result = fit(candidate(), Service(), cfg, 0)
    assert result["best"] is None
    assert store.processing_state(result["fit_key"])[0] == "retry-eligible"
    store.close()


def test_no_random_share_rejected():
    with pytest.raises(ValueError):
        ExplorationConfig(rl_fraction=.75, mutation_fraction=.25)


def test_frozen_reference_sources_match_manifest():
    import hashlib
    import tarfile
    from comparisons.search_exploration.freeze_legacy import DEFAULT
    manifest = json.loads((DEFAULT / "manifest.json").read_text())
    archive = DEFAULT / "scientific_source.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with tarfile.open(archive) as stream:
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(stream.extractfile(name).read()).hexdigest() == expected
    assert len(manifest["files"]) > 100


def test_lower_loss_but_incorrect_does_not_remove_success_credit(tmp_path):
    archive = DiscoveryArchive(ExplorationConfig())
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    winner = record(0, .03)
    assert archive.add(winner, store)[0] > 0
    incorrect = {**record(1, .01), "qualified": False}
    assert archive.add(incorrect, store)[0] == 0
    assert archive.add(winner, store)[0] == 0
    assert archive.region_best["a"]["qualified"]
    store.close()


def test_archive_prefers_region_representatives_over_duplicates(tmp_path):
    cfg = ExplorationConfig(archive_capacity=2, sil_top_k=1)
    archive = DiscoveryArchive(cfg)
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    archive.add(record(0, .01, "a"), store)
    archive.add(record(1, .03, "b"), store)
    archive.add(record(2, .009, "a"), store)
    assert {r["region"] for r in archive.ranked()} == {"a", "b"}
    store.close()


def test_fit_is_invariant_to_reaction_order(tmp_path):
    cfg = ExplorationConfig(fit_evaluations=5)
    store = Store(tmp_path / "store.sqlite", 30, "v1")
    class Service:
        def __init__(self):
            self.store = store
        def evaluate(self, c, **kwargs):
            return store.evaluate(c, lambda c: {"loss": sum(k*k for k in c["rates"]), "simulation_complete": True})
    original = candidate(2.)
    reverse = {**original, "reaction_ids": list(reversed(original["reaction_ids"])), "rates": list(reversed(original["rates"]))}
    a = fit(original, Service(), cfg, 9)
    b = fit(reverse, Service(), cfg, 9)
    assert a == b and store.count("fit_cache_hits") == 1
    store.close()


def test_compact_fit_replay_restores_diagnostics_and_current_provenance(tmp_path):
    cfg = ExplorationConfig(fit_evaluations=1)
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    diagnostics = {"outputs": [[0., .2, .5]], "solver": "CVODE"}
    class Service:
        def __init__(self):
            self.store = store
        def evaluate(self, c, **kwargs):
            return store.evaluate(c, lambda c: {
                "loss": .02, "simulation_complete": True, "task_info": diagnostics,
                "candidate_hash": "exact-design", "topology_hash": "topology"})
    first = fit(candidate(), Service(), cfg, 9)
    stored = store.processing_state(first["fit_key"])[1]
    assert "task_info" not in stored["best"]
    store.save_design(first["best"])
    design = json.loads(store.db.execute("SELECT payload FROM designs").fetchone()[0])
    assert "task_info" not in design
    assert design["evaluation_cache_key"] == store.evaluation_key(first["best"])
    second = fit({**candidate(), "source": "random", "proposal_id": "second"}, Service(), cfg, 9)
    assert second["best"]["task_info"] == diagnostics
    assert second["best"]["source"] == "random"
    assert second["best"]["proposal_id"] == "second"
    assert store.count("expensive_evaluations") == 1
    store.close()


def test_lazy_cache_does_not_decode_diagnostics_under_lock(tmp_path, monkeypatch):
    import zlib
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    payload = {"loss": .01, "task_info": {"outputs": list(range(10000))}}
    summary, _ = store.evaluate(candidate(), lambda c: payload, include_diagnostics=False)
    assert summary == {"loss": .01}
    original = zlib.decompress
    def decompress(blob):
        completed = threading.Event()
        def read():
            store.count("expensive_evaluations")
            completed.set()
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        assert completed.wait(2), "Diagnostics decoded while holding the store lock"
        thread.join()
        return original(blob)
    monkeypatch.setattr(zlib, "decompress", lambda b: pytest.fail("Unexpected trajectory load"))
    assert store.cached(candidate(), include_diagnostics=False) == summary
    assert store.evaluate(candidate(), lambda c: pytest.fail("Cache miss"), include_diagnostics=False)[0] == summary
    monkeypatch.setattr(zlib, "decompress", decompress)
    assert store.cached(candidate()) == payload
    assert store.evaluate(candidate(), lambda c: pytest.fail("Cache miss"))[0] == payload
    store.close()


def test_store_transaction_rollback_and_read_only_counters(tmp_path):
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    statements = []
    store.db.set_trace_callback(statements.append)
    assert store.count("missing") == 0
    assert all(not q.startswith("INSERT") for q in statements)
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.count("x", 1)
            store.event("test", {})
            raise RuntimeError("abort")
    assert store.count("x") == 0
    assert store.db.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    with store.transaction():
        store.count("x", 2)
        with store.transaction():
            store.count("y", 3)
    assert store.counts(["x", "y", "z"]) == {"x": 2, "y": 3, "z": 0}
    store.close()


def test_cache_reads_legacy_payload_without_rewriting(tmp_path):
    import zlib
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    payload = {"loss": .2, "task_info": {"outputs": [1., 2.]}}
    packed = zlib.compress(json.dumps(payload).encode())
    store.db.execute("INSERT INTO evaluations VALUES (?,?)", (store.evaluation_key(candidate()), packed))
    assert store.cached(candidate()) == payload
    assert store.cached(candidate(), include_diagnostics=False) == {"loss": .2}
    assert store.db.execute("SELECT payload FROM evaluations").fetchone()[0] == packed
    store.close()


def test_parent_cohort_sampler_matches_original_draws(tmp_path):
    import math
    archive = DiscoveryArchive(ExplorationConfig())
    store = Store(tmp_path / "store.sqlite", 10, "v1")
    for i in range(30):
        archive.add(record(i, .01 + i/10000, str(i % 7)), store)
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    sample = archive.parent_sampler(store)
    for _ in range(50):
        grouped = {}
        for entry in archive.entries.values():
            grouped.setdefault(entry["region"], []).append(entry)
        keys = sorted(grouped)
        weights = np.array([1 / math.sqrt(1 + store.count("region:" + k)) for k in keys])
        region = keys[int(a.choice(len(keys), p=weights/weights.sum()))]
        expected = grouped[region][int(a.integers(len(grouped[region])))]
        assert sample(b) == expected
    store.close()


def test_failed_completion_rolls_back_summary_and_diagnostics(tmp_path, monkeypatch):
    store = Store(tmp_path / "store.sqlite", 3, "v1")
    original = store.event
    def event(kind, value):
        if kind == "evaluation_completed":
            raise OSError("simulated disk failure")
        return original(kind, value)
    monkeypatch.setattr(store, "event", event)
    payload = {"loss": .2, "task_info": {"outputs": [1.]}}
    with pytest.raises(OSError, match="disk failure"):
        store.evaluate(candidate(), lambda c: payload)
    assert not store.inflight
    assert store.count("expensive_evaluations") == 1
    for table in ("evaluation_summaries", "diagnostics"):
        assert store.db.execute("SELECT count(*) FROM " + table).fetchone()[0] == 0
    monkeypatch.setattr(store, "event", original)
    assert store.evaluate(candidate(), lambda c: payload)[0] == payload
    assert store.count("expensive_evaluations") == 2
    store.close()


def test_coalesced_callers_can_request_different_payload_sizes(tmp_path):
    store = Store(tmp_path / "store.sqlite", 3, "v1")
    started, finish = threading.Event(), threading.Event()
    payload = {"loss": .2, "task_info": {"outputs": [1.]}}
    def compute(c):
        started.set()
        assert finish.wait(5)
        return payload
    with ThreadPoolExecutor(2) as executor:
        a = executor.submit(store.evaluate, candidate(), compute, include_diagnostics=False)
        assert started.wait(5)
        b = executor.submit(store.evaluate, candidate(), compute)
        deadline = time.monotonic() + 3
        while store.count("coalesced_hits") == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        finish.set()
        assert a.result(timeout=3)[0] == {"loss": .2}
        assert b.result(timeout=3)[0] == payload
    assert store.count("expensive_evaluations") == 1
    assert store.count("coalesced_hits") == 1
    store.close()
