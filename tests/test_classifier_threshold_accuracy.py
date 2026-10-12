import numpy as np
import pytest

from paper.iclr2027_genai_net_llm.analyze_classifier_threshold_accuracy import endpoint_accuracy


def test_correct_winner_need_not_reach_binary_target():
    targets = np.tile([1., 0.], (32, 1))
    result = endpoint_accuracy(np.tile([.3, .1], (32, 1)), targets)
    assert result["winner_correct"] == 32
    assert result["binary_correct"] == 0


def test_ties_are_not_assigned_to_first_class():
    targets = np.tile([1., 0.], (32, 1))
    result = endpoint_accuracy(np.tile([.5, .5], (32, 1)), targets)
    assert result["winner_correct"] == 0
    assert result["ambiguous"] == 32
    assert result["binary_correct"] == 0


def test_perfect_one_hot_targets_and_one_wrong_case():
    targets = np.tile([[1., 0.], [0., 1.]], (16, 1))
    outputs = targets.copy()
    outputs[3] = outputs[3, ::-1]
    result = endpoint_accuracy(outputs, targets)
    assert result["winner_correct"] == 31
    assert result["binary_correct"] == 31


def test_invalid_scenarios_are_not_silently_classified():
    targets = np.tile([1., 0.], (32, 1))
    for outputs in (np.ones((31, 2)), np.full((32, 2), np.nan)):
        with pytest.raises(ValueError):
            endpoint_accuracy(outputs, targets)
