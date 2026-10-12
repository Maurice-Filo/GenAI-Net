import pytest

from comparisons.llm_seeded_evolution.analysis.live_hof_diversity import structural_summary, reconstruct
from comparisons.llm_seeded_evolution.search import Candidate, ScoredCandidate


def test_distance_excludes_rate_and_order_and_tracks_ancestry():
    a = ScoredCandidate(Candidate((5, 6, 7, 8, 9), (1,) * 5, "llm", proposal_id="p1"), .01, "a")
    b = ScoredCandidate(Candidate((10, 8, 7, 6, 5), (2,) * 5, "reaction_mutation", "a", "p1"), .02, "b")
    result = structural_summary([a, b])
    assert result["mean_pairwise_jaccard_distance"] == pytest.approx(1 / 3)
    assert result["one_replacement_pair_fraction"] == 1
    assert result["universal_added_reactions"] == [5, 6, 7, 8]
    assert result["distinct_added_reactions"] == 6
    assert result["direct_llm"] == result["ga_with_llm_ancestry"] == 1
    assert result["no_recorded_llm_ancestry"] == 0


def test_singleton_has_no_pairwise_estimate():
    result = structural_summary([ScoredCandidate(Candidate((5, 6), (1, 1)), .1, "a")])
    assert result["mean_pairwise_jaccard_distance"] is None


def test_reconstruction_retains_best_parameterization_and_checks_checkpoint():
    rows = [{"candidate_evaluations": str(i), "search_evaluations": str(i), "llm_evaluations": "0",
             "reaction_ids": "[5,6,7,8,9]", "rate_constants": "[1,1,1,1,1]", "loss": str(loss),
             "topology_hash": "same", "source": "random", "proposal_id": "", "parent_hash": ""}
            for i, loss in enumerate([.03, .01, .02], 1)]
    checkpoint = {"candidate_evaluations": "3", "search_evaluations": "3", "llm_evaluations": "0",
                  "best_so_far_loss": ".01", "qualified_topologies": "1"}
    result = reconstruct(rows, [checkpoint], threshold=.04, forbidden={0, 1})[3]
    assert result["all_hof"]["topologies"] == 1
    assert result["best_loss"] == .01
    checkpoint["best_so_far_loss"] = ".02"
    with pytest.raises(ValueError, match="checkpoint"):
        reconstruct(rows, [checkpoint], threshold=.04, forbidden={0, 1})
