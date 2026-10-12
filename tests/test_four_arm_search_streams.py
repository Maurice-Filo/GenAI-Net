from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_harness_search_streams import trainer
from RL4CRN.llm.graphs import LLMGraphRunResult, default_decider_writer_spec
from RL4CRN.llm.schemas import LLMCandidate
from RL4CRN.search_streams.harness import StreamHarnessEvaluator
from RL4CRN.search_streams.directional_harness import topology_union
from RL4CRN.search_streams.four_arm_harness import FourArmHarnessTrainer, channels


@pytest.fixture
def directional(trainer):
    trainer.__class__ = FourArmHarnessTrainer
    trainer.communication_mode = None
    trainer.direct_llm_records = []
    trainer.terminal_union = None
    return trainer


@pytest.mark.parametrize("mode", ["full", "llm-to-rl", "rl-to-llm", "none"])
def test_attach_selects_only_requested_channel(directional, mode):
    directional.attach_harness(communication_mode=mode, dsh_home=directional.output / "home", prompt="test")
    settings = directional._llm_loop
    assert settings["add_to_hall_of_fame"] is channels(mode)[0]
    assert settings["cross_communication"] is channels(mode)[1]
    assert settings["withhold_initial_hof"]
    assert settings["graph"].client.model == "deepseek-v4-flash"
    assert settings["graph"].client.harness_protocol == "workspace-loss-v1"


@pytest.mark.parametrize("mode", ["full", "llm-to-rl", "rl-to-llm", "none"])
@pytest.mark.parametrize("complete", [True, False])
def test_directional_information_and_insertion_boundaries(directional, mode, complete):
    t = directional
    t.communication_mode = mode
    t.step_epoch()
    proxy = StreamHarnessEvaluator(t.evaluator, t.service)
    candidate = LLMCandidate(t.allowed[:5], [[.7]] * 5)
    evaluation = proxy.evaluate(candidate)
    if not complete:
        object.__setattr__(evaluation, "valid", False)
        object.__setattr__(evaluation, "message", "canonical_incomplete")
        evaluation.exploration_record["simulation_complete"] = False
    snapshots = []
    class Graph:
        client = SimpleNamespace(exploration_context=None)
        def run_round(self, **kwargs):
            snapshots.append((kwargs, deepcopy(self.client.exploration_context)))
            return LLMGraphRunResult(default_decider_writer_spec(), "", "", {}, [candidate], [evaluation])
    t.configure_llm_graph(Graph(), every=1, task_description="test", num_candidates=1,
        withhold_initial_hof=True, cross_communication=channels(mode)[1], add_to_hall_of_fame=channels(mode)[0])
    archive_before = deepcopy(t.discovery.state_dict())
    visits_before = t.store.counts(["region:" + k for k in t.discovery.region_best])
    hof_before = [str(e.state) for e in t.s.mult_env.hall_of_fame]
    for epoch in (0, 1):
        t._maybe_run_llm_graph(epoch)
        t.wait_for_llm_graph(timeout=5)
    assert not snapshots[0][0]["hall_of_fame_iter"]
    assert snapshots[0][1] is None
    assert bool(snapshots[1][0]["hall_of_fame_iter"]) is channels(mode)[1]
    assert (snapshots[1][1] is not None) is channels(mode)[1]
    assert len(t.direct_llm_records) == 2
    assert len((t.output / "direct_llm_proposals.jsonl").read_text().splitlines()) == 2
    if not channels(mode)[0]:
        assert not t.llm_fit_candidates
        assert t.discovery.state_dict() == archive_before
        assert t.store.counts(["region:" + k for k in t.discovery.region_best]) == visits_before
        assert [str(e.state) for e in t.s.mult_env.hall_of_fame] == hof_before
        assert not t._llm_loop["isolated_candidates"]
        assert t.llm_candidates_merged == 0
        with pytest.raises(RuntimeError, match="fit an LLM"):
            t._schedule([{"source": "llm"}])
        with pytest.raises(RuntimeError, match="LLM lineage"):
            t._record({"origin": "llm"}, "ga", "bad")
    else:
        assert t.llm_fit_candidates
        if complete:
            assert any(r["origin"] == "llm" for r in t.discovery.ranked())


def test_union_deduplicates_structures_and_excludes_incomplete():
    def r(topology, loss, complete=True):
        return {"topology_hash": topology, "loss": loss, "simulation_complete": complete}
    original = [r("a", .01), r("b", .1)]
    proposals = [r("a", .002), r("a", .003), r("b", .001), r("c", .004), r("d", 0, False)]
    before = deepcopy(original)
    out = topology_union(original, proposals, .02)
    assert out["search_qualified"] == 1
    assert out["llm_qualified"] == 3
    assert out["union_qualified"] == 3
    assert out["additional_qualified"] == 2
    assert out["overlap_qualified"] == 1
    assert original == before

