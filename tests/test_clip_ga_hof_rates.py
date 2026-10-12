import itertools

import numpy as np
import pytest

from comparisons.llm_seeded_evolution.analysis.clip_hof_rates import clipped_rates, functional_accuracy


def test_clipping_changes_only_outside_bounds():
    assert clipped_rates([.0001, .001, .1, 100., 4.4e6]) == [.001, .001, .1, 100., 100.]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_rates_not_hidden(bad):
    with pytest.raises(ValueError):
        clipped_rates([bad])


def test_accuracy_and_failure_handling():
    inputs = np.array(list(itertools.product([0, 1], repeat=4)))
    targets = ((inputs[:, 0] * inputs[:, 1]) + (inputs[:, 1] * inputs[:, 2]) + (inputs[:, 2] * inputs[:, 3])) > 0
    info = {"inputs": inputs, "setpoint": targets[:, None],
            "outputs": [np.full((1, 1000), float(t)) for t in targets]}
    assert functional_accuracy(info)["correct_rows"] == 16
    info["outputs"][0][0, -1] = .5
    assert functional_accuracy(info)["correct_rows"] == 15
    info["outputs"][0] = np.full((1, 1000), 1e4)
    assert functional_accuracy(info)["correct_rows"] is None
