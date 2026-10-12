import copy

import numpy as np
import pytest

from comparisons.search_exploration.plot_logic5_portfolio_robustness import ARMS, summarize


def fixture_data():
    rows = [{"arm": arm, "seed": seed, "valid": 50, "cores": 20+seed,
             "robust_cores": 2 + (arm == "full"), "baseline_unresolved": 0}
            for arm in ARMS for seed in range(1, 21)]
    tests = [{"a": "full", "b": arm, "metric": "robust_cores", "mean_difference": 1.,
              "mean_difference_ci95": [1., 1.], "holm_p_18": .123}
             for arm in ARMS[:-1]]
    return {"status": "complete", "rows": rows, "tests": tests, "baseline_recovered": 2,
            "baseline_unresolved": 0, "plan_sha256": "example"}


def test_fractions_are_computed_per_seed_and_original_tests_preserved():
    data = fixture_data()
    result = summarize(data)
    expected = np.mean([100*3/(20+seed) for seed in range(1, 21)])
    assert result["groups"]["full"]["robust_percent"]["mean"] == pytest.approx(expected)
    assert expected != pytest.approx(100*3/30.5)
    assert result["ratio_of_mean_robust_counts_full_vs_none"] == 1.5
    assert len(result["contrasts"]) == 3
    assert all(t["holm_p_18"] == .123 for t in result["contrasts"])
    assert data == fixture_data()


@pytest.mark.parametrize("change", ["missing", "duplicate", "unresolved", "counts", "test"])
def test_inconsistent_inputs_are_rejected(change):
    data = copy.deepcopy(fixture_data())
    if change == "missing":
        data["rows"].pop()
    elif change == "duplicate":
        data["rows"][-1] = data["rows"][0]
    elif change == "unresolved":
        data["rows"][0]["baseline_unresolved"] = 1
    elif change == "counts":
        data["rows"][0]["robust_cores"] = 100
    else:
        data["tests"][0]["mean_difference"] = 2.
    with pytest.raises(ValueError):
        summarize(data)
