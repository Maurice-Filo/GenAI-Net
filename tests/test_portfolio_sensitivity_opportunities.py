import numpy as np
from paper.iclr2027_genai_net_llm import generate_portfolio_sensitivity_opportunities as analysis

from paper.iclr2027_genai_net_llm.generate_portfolio_sensitivity_opportunities import (
    COHORTS, SENSITIVITY_FIGURE_COHORTS, classify_request, count_support, qualified_counts,
)


def test_threshold_is_strict_and_topologies_not_rates_are_counted():
    entries = [{"topology_hash":"a", "loss":1.0}, {"topology_hash":"a", "loss":.5},
               {"topology_hash":"b", "loss":1.0}, {"topology_hash":"c", "loss":float("nan")}]
    np.testing.assert_array_equal(qualified_counts(entries, 1, [.5,1,2]), [0,1,2])


def test_threshold_sweep_retains_capacity():
    entries = [{"topology_hash":str(i), "loss":.1} for i in range(40)]
    np.testing.assert_array_equal(qualified_counts(entries, 1, [1,2]), [30,30])


def test_last_training_epoch_has_one_update_but_terminal_wait_has_none():
    row = {"finite_valid":10, "completed_epoch":99, "created_at":10}
    assert classify_request(row, 11) == ("available", 1)
    assert classify_request({**row, "completed_epoch":100}, 11) == ("late_valid", 0)
    assert classify_request(row, 9) == ("late_valid", 0)


def test_empty_and_failed_requests_are_not_usable_batches():
    row = {"finite_valid":0, "completed_epoch":10, "created_at":10}
    assert classify_request(row, 11) == ("empty", 90)
    assert classify_request({**row, "completed_epoch":100}, 11) == ("late_empty", 0)


def test_count_support_retains_zero_yield_seeds():
    assert count_support([0, 0, 1, 3]) == {
        "n_seeds": 4, "median_qualified": .5, "min_qualified": 0,
        "max_qualified": 3, "zero_fraction": .5, "std_qualified": np.sqrt(2)}


def test_figure_selection_does_not_exclude_classifier_from_analysis():
    assert SENSITIVITY_FIGURE_COHORTS == ("local_logic", "flash_rpa")
    assert "flash_classifier" in COHORTS


def test_threshold_lower_panels_show_counts_not_cap_fraction(monkeypatch):
    sensitivity, endpoints, summaries = [], [], []
    for cohort, (_, _, arms) in COHORTS.items():
        for arm in arms:
            sensitivity.append({"cohort": cohort, "arm": arm, "multiplier": 1,
                                "mean_paired_gain": 2, "ci_low": 1, "ci_high": 3,
                                "mean_qualified": 12, "std_qualified": 3, "cap_fraction": .25})
            endpoints.append({"cohort": cohort, "arm": arm, "qualified": 12})
            summaries.append({"cohort": cohort, "arm": arm,
                              **{f"mean_{state}": 0 for state in analysis.STATES}})
    checked = []

    def inspect(fig, stem):
        if stem == "portfolio_threshold_sensitivity":
            for ax in fig.axes[2:]:
                np.testing.assert_array_equal(ax.lines[0].get_ydata(), [12])
                band_y = ax.collections[0].get_paths()[0].vertices[:, 1]
                assert band_y.min() == 9
                assert band_y.max() == 15
                assert ax.get_ylim()[1] > 30
            assert fig.axes[2].get_ylabel() == "Qualified topologies\n(mean $\\pm$ 1 SD)"
            checked.append(stem)
        analysis.plt.close(fig)

    monkeypatch.setattr(analysis, "save", inspect)
    analysis.figures(sensitivity, endpoints, summaries)
    assert checked == ["portfolio_threshold_sensitivity"]
