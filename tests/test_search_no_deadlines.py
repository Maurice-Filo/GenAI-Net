import json
import queue
from types import SimpleNamespace

import pytest

from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.refinement import fit
from RL4CRN.search_streams.workers import StreamPool
from RL4CRN.search_streams.store import CompactStore
from comparisons.search_exploration.resume_rpa_statistical import amended_job


def test_none_deadlines_are_explicit_and_old_defaults_survive():
    cfg = ExplorationConfig(evaluation_seconds=None, fit_seconds=None)
    assert cfg.to_dict()["evaluation_seconds"] is None
    assert ExplorationConfig().evaluation_seconds == 30
    assert ExplorationConfig().fit_seconds == 60
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ExplorationConfig(evaluation_seconds=value)


def test_evolution_fitting_without_clock_cutoff_still_obeys_call_budget(monkeypatch):
    import RL4CRN.search_exploration.refinement as module
    clock = iter(range(0, 100000, 1000))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    calls = []
    class Store:
        def processing_state(self, key):
            return None
        def processing(self, *args):
            pass
    class Service:
        store = Store()
        def evaluate(self, candidate, *, timeout, **kwargs):
            assert timeout is None
            calls.append(candidate)
            return {"loss": .1, "simulation_complete": True}, False
    cfg = ExplorationConfig(fit_seconds=None, evaluation_seconds=None, fit_evaluations=5)
    result = fit({"reaction_ids": [2], "rates": [1.], "proposal_id": "p"}, Service(), cfg, 1,
        include_diagnostics=False)
    assert len(calls) == result["calls"] == 5
    assert result["termination"] == "evaluation_limit"


def fake_pool(messages, timeout=None):
    class Connection:
        waits = []
        def send(self, message):
            pass
        def poll(self, wait):
            self.waits.append(wait)
            return True
        def recv(self):
            value = messages.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
    conn = Connection()
    pool = StreamPool.__new__(StreamPool)
    pool.available = queue.Queue()
    pool.available.put((None, conn))
    pool.timeout = timeout
    pool.fit_seconds = None
    pool.store = SimpleNamespace(reserve=lambda *a: ("evaluate", "key"), complete=lambda *a: None,
        fail=lambda *a: None)
    pool._stop = lambda slot: None
    return pool, conn


def test_stream_poll_and_per_evaluation_wait_are_unbounded():
    messages = [("rpc", ("reserve", ({}, "search"))),
        ("rpc", ("complete", ("key", {"loss": .1}))), ("ok", {"loss": .1})]
    pool, conn = fake_pool(messages)
    assert pool.run({"category": "search", "candidate": {}}) == {"loss": .1}
    assert conn.waits == [None, None, None]
    assert pool.available.qsize() == 1


def test_existing_timed_mode_still_uses_finite_waits():
    pool, conn = fake_pool([("ok", {})], timeout=30)
    pool.run({"category": "search", "candidate": {}})
    assert 0 < conn.waits[0] <= 40


def test_dead_worker_errors_are_not_hidden_and_do_not_starve_queued_jobs():
    pool, _ = fake_pool([EOFError("worker died")])
    with pytest.raises(EOFError):
        pool.run({"category": "search", "candidate": {}})
    assert pool.available.qsize() == 1


def test_coalesced_evaluation_has_no_hidden_120_second_cutoff(tmp_path):
    store = CompactStore(tmp_path / "store.sqlite", 10, "test", 10, wait_timeout=None)
    candidate = {"reaction_ids": [2], "rates": [1.]}
    class Waiting:
        def result(self, timeout):
            assert timeout is None
            return {"loss": .1}
    try:
        store.inflight[store.evaluation_key(candidate)] = Waiting()
        assert store.reserve(candidate, "search") == ("cached", {"loss": .1})
    finally:
        store.close()


def test_amendment_preserves_queued_configs_and_uses_distinct_outputs(tmp_path):
    folder = tmp_path / "amendment"
    folder.mkdir()
    cfg = tmp_path / "original.json"
    cfg.write_text(json.dumps(ExplorationConfig().to_dict()))
    before = cfg.read_bytes()
    old = {"name": "julia-seed02", "arm": "julia", "seed": 2, "status": "queued",
        "command": ["runner", "--output", "old", "--exploration-config", str(cfg)]}
    job = amended_job(tmp_path, old, {"status": "queued"}, folder)
    assert old["command"][2] == "old" and cfg.read_bytes() == before
    assert job["output"].endswith("julia-seed02-no-timeouts")
    new = json.loads(open(job["config_path"]).read())
    assert new["evaluation_seconds"] is None and new["fit_seconds"] is None
    assert new["evaluation_budget"] == json.loads(before)["evaluation_budget"]


def test_julia_transport_accepts_none_but_rejects_infinity():
    import threading
    from RL4CRN.parameter_optimization.client import JuliaOptimizer
    worker = JuliaOptimizer.__new__(JuliaOptimizer)
    worker._lock = threading.RLock()
    worker._closed = False
    worker.process = SimpleNamespace(poll=lambda: None,
        stdin=SimpleNamespace(write=lambda text: None, flush=lambda: None))
    received = []
    worker._receive = lambda timeout: received.append(timeout) or {"ok": True}
    assert worker._request("evaluate", timeout=None) == {"ok": True}
    assert received == [None]
    with pytest.raises(ValueError):
        worker._request("evaluate", timeout=float("inf"))
    worker._closed = True
