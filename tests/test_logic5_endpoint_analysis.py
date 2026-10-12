import numpy as np
import pytest
from scipy.stats import wilcoxon, PermutationMethod

from comparisons.search_exploration import analyze_logic5_four_arm as analysis


def test_values_match_seeds_not_file_order():
    rows = [{"arm": "full", "seed": 2, "loss": 2}, {"arm": "none", "seed": 1, "loss": 99},
            {"arm": "full", "seed": 1, "loss": 1}]
    assert analysis.values(rows, "full", "loss").tolist() == [1, 2]


def test_all_six_pairs_two_endpoints_share_one_correction(monkeypatch):
    rows = [{"arm": a, "seed": s, "loss": i+s, "qualified": 100+i+s}
            for i, a in enumerate(analysis.ARMS) for s in (2, 1)]
    calls = []

    def paired(a, b, **kwargs):
        assert np.all(a >= b)
        calls.append(kwargs)
        return {"p": .001}

    monkeypatch.setattr(analysis, "paired_endpoint", paired)
    results = analysis.statistics(rows)
    assert len(results) == 12
    assert len({(r["a"], r["b"], r["metric"]) for r in results}) == 12
    assert all(r["holm_p_12"] == pytest.approx(.012) for r in results)
    assert sum(c["smaller"] for c in calls) == 6
    assert all(c["resamples"] == 20000 and c["seed"] == 20260924 for c in calls)


@pytest.mark.parametrize("d", [[0, 0, 0], [1, 1, -1, 2, -3], [0, 1, -1, 2, -3],
                               [1, 2, 3, 4], [-1, -1, -1, 0], [0, 0, 1, -2, 2, 4]])
def test_exact_rank_distribution_matches_exhaustive_permutations(d):
    expected = (wilcoxon(d, zero_method="pratt", method=PermutationMethod(n_resamples=np.inf)).pvalue
                if np.any(d) else 1.)
    assert analysis.exact_pratt_p(d) == pytest.approx(expected)


def test_twenty_untied_matches_standard_exact_wilcoxon():
    d = np.arange(1, 21) * np.where(np.arange(20) % 3 == 0, -1, 1)
    assert analysis.exact_pratt_p(d) == pytest.approx(wilcoxon(d, method="exact").pvalue)
