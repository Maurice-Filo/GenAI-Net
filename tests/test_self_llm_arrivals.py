import numpy as np

from comparisons.search_exploration.plot_self_llm_arrivals import arrival_curve, bands


def test_no_backdating_and_reordered_arrivals():
    events = [
        {"request": 0, "arrival_minutes": 10., "batch_best_loss": .1},
        {"request": 1, "arrival_minutes": 5., "batch_best_loss": .5},
        {"request": 2, "arrival_minutes": 15., "batch_best_loss": None},
    ]
    values = arrival_curve(events, [0., 4., 5., 9., 10., 15., 20.])
    np.testing.assert_equal(values, [np.inf, np.inf, .5, .5, .1, .1, .1])


def test_no_conditioning_on_early_responders():
    values = np.array([[.1]] * 10 + [[np.inf]] * 10)
    lo, median, hi = bands(values)
    assert lo[0] == .1
    assert np.isinf(median[0]) and np.isinf(hi[0])
    values[10] = .2
    assert bands(values)[1][0] == .2
