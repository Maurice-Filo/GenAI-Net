from copy import deepcopy
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from RL4CRN.llm.graphs import LLMGraphRunResult, default_decider_writer_spec
from RL4CRN.llm.schemas import LLMCandidate
from RL4CRN.search_exploration.config import ExplorationConfig
from RL4CRN.search_exploration.evaluation import build, EvaluationService, evaluate_once
from RL4CRN.search_exploration.session import build_session
from RL4CRN.search_streams.harness import HarnessJuliaStreamTrainer, StreamHarnessEvaluator
from RL4CRN.search_streams.julia_stream import JuliaRecipe, JuliaStreamTrainer
from RL4CRN.search_streams.store import CompactStore
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS


@pytest.fixture
def trainer(tmp_path):
    cfg = ExplorationConfig(epochs=3, batch_size=4, evaluation_budget=80, workers=1,
        archive_capacity=20, sil_top_k=4, sil_batch_size=2, plot_every=10)
    evaluator, task, sc = build("rpa", json.loads(CONFIGS["rpa"].read_text()), .001, 100.)
    sc.train.seed, sc.train.batch_size, sc.train.n_cpus = 0, 4, 1
    sc.policy.width, sc.policy.depth, sc.policy.deep_layer_size = 16, 1, 16
    session = build_session(sc, task)
    store = CompactStore(tmp_path / "exploration.sqlite", 80, "test", 12)
    class Pool:
        slots = [None]
        compute_seconds = 10.
        def evaluate(self, c, timeout=None):
            return evaluate_once(evaluator, c)
        def run(self, job):
            c = job["candidate"]
            result, hit = store.evaluate(c, self.evaluate, category="search")
            return {"best": {**c, **result}, "calls": 1, "failures": [], "termination": "completed"}
    search = Pool()
    optimizer = SimpleNamespace(recipe=JuliaRecipe(), admission_limit=lambda pending: 0,
        compute_seconds=0., search_pool=search, job_estimate=1.)
    allowed = [r.ID for r in evaluator.library.reactions if r.ID not in evaluator.forbidden_reaction_ids]
    service = EvaluationService(store, search, allowed, 5, (.001, 100.))
    obj = HarnessJuliaStreamTrainer(session, evaluator, service, cfg, tmp_path,
        search_pool=search, optimization_pool=optimizer, optimization_workers=1,
        queue_capacity=2, optimize_per_epoch=2)
    yield obj
    obj.close_streams()
    obj.close_results_database()
    store.close()


def test_real_cache_evaluator_reuses_budget_and_does_not_hide_infrastructure(trainer):
    proxy = StreamHarnessEvaluator(trainer.evaluator, trainer.service)
    c = LLMCandidate(trainer.allowed[:5], [[1.]] * 5)
    first = proxy.evaluate(c)
    assert first.valid
    before = trainer.store.count("expensive_evaluations")
    assert proxy.evaluate(c).loss == first.loss
    assert trainer.store.count("expensive_evaluations") == before
    with patch.object(trainer.service, "evaluate", side_effect=OSError("worker died")):
        with pytest.raises(RuntimeError, match="infrastructure"):
            proxy.evaluate(c)
    assert isinstance(proxy.fatal_error, OSError)


def test_incomplete_canonical_result_is_not_insertable(trainer):
    proxy = StreamHarnessEvaluator(trainer.evaluator, trainer.service)
    c = LLMCandidate(trainer.allowed[:5], [[1.]] * 5)
    record = {"task_info": {}, "simulation_complete": False, "loss": 10000.}
    with patch.object(trainer.service, "evaluate", return_value=(record, False)):
        result = proxy.evaluate(c)
    assert not result.valid and result.message == "canonical_incomplete"
    assert proxy.fatal_error is None


def test_withheld_then_outward_context_and_nonblocking_insertion(trainer):
    gate, entered = threading.Event(), threading.Event()
    snapshots = []
    proxy = StreamHarnessEvaluator(trainer.evaluator, trainer.service)
    evaluation = proxy.evaluate(LLMCandidate(trainer.allowed[:5], [[1.]] * 5))
    class Graph:
        client = SimpleNamespace(exploration_context=None)
        def run_round(self, **kwargs):
            snapshots.append((kwargs, deepcopy(self.client.exploration_context)))
            entered.set()
            assert gate.wait(10)
            return LLMGraphRunResult(default_decider_writer_spec(), "", "", {}, [evaluation.candidate], [evaluation])
    trainer.step_epoch()  # populate HOF before testing request-zero withholding
    trainer.configure_llm_graph(Graph(), every=1, task_description="test", num_candidates=1,
        withhold_initial_hof=True, cross_communication=True, add_to_hall_of_fame=True)
    try:
        trainer._maybe_run_llm_graph(0)
        assert entered.wait(2)
        assert snapshots[0][0]["hall_of_fame_iter"] == ()
        assert snapshots[0][1] is None
        assert "withheld" in snapshots[0][0]["sil_feedback_text"]
        trainer.step_epoch()  # pending model does not block policy/search
        assert trainer.state.epoch == 2 and trainer.learner.version == 2
        assert trainer.llm_rounds_merged == 0
        gate.set()
        trainer.wait_for_llm_graph(timeout=5)
        assert trainer.llm_rounds_merged == 1 and trainer.llm_candidates_merged == 1
        assert trainer.llm_fit_candidates
        assert any(r["origin"] == "llm" for r in trainer.discovery.ranked())
        assert any(e.state.last_task_info.get("source") == "LLM" for e in trainer.s.mult_env.hall_of_fame)
        trainer._maybe_run_llm_graph(2)
        trainer.wait_for_llm_graph(timeout=5)
        assert snapshots[-1][0]["hall_of_fame_iter"]
        assert snapshots[-1][1]["archive_size"] > 0
    finally:
        gate.set()


def test_llm_candidates_share_scheduler_and_terminal_drain(trainer):
    record = {"topology_hash": "llm", "source": "llm"}
    trainer.llm_fit_candidates = [record]
    with patch.object(JuliaStreamTrainer, "_schedule", return_value=1) as schedule:
        assert trainer._schedule([{"topology_hash": "rl"}]) == 1
    assert schedule.call_args.args[0] == [{"topology_hash": "rl"}, record]
    assert trainer.llm_fit_candidates == []
    events = []
    with patch.object(trainer, "wait_for_llm_graph", side_effect=lambda **kw: events.append("llm")), \
         patch.object(trainer, "_merge_ready", side_effect=lambda **kw: events.append("fits")), \
         patch.object(trainer, "_schedule", side_effect=lambda rs: events.append("late_fit_selection")), \
         patch.object(JuliaStreamTrainer, "finish", side_effect=lambda: events.append("finish")):
        trainer.finish()
    assert events == ["llm", "fits", "late_fit_selection", "finish"]


def test_terminal_database_counters_include_llm(trainer):
    trainer.store.count("evaluations/llm", 3)
    trainer.store.count("expensive_evaluations", 3)
    from RL4CRN.utils.input_interface import Trainer
    with patch.object(Trainer, "persist_hof_boundary") as persist:
        trainer.persist_hof_boundary(epoch=3, boundary="terminal_boundary", rl_evaluations=0,
                                    llm_final_evaluations=0, llm_tool_evaluations=0, total_evaluations=0)
    assert persist.call_args.kwargs["llm_final_evaluations"] == 3
    assert persist.call_args.kwargs["total_evaluations"] == 3
