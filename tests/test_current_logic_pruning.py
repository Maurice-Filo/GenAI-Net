"""Self-contained checks for the current Logic reduction procedure."""
from copy import deepcopy
import numpy as np
import pytest
from reproduce.diagnostics.logic_pruning import added_records, logic_accuracy, seed_summary, select_core


def truth_table():
    targets = np.tile([[0.], [1.]], (8, 1))
    return np.repeat(targets[:, :, None], 1000, axis=2), targets


def test_all_rows_must_pass_and_threshold_ties_fail():
    outputs, targets = truth_table()
    assert logic_accuracy(outputs, targets)["all16_correct"]
    outputs[0, 0, -1] = .5
    result = logic_accuracy(outputs, targets)
    assert result["correct_rows"] == 15
    assert not result["all16_correct"]
    outputs[0, 0, -1] = .5 - .5e-6
    assert not logic_accuracy(outputs, targets)["all16_correct"]
    outputs[0, 0, -1] = .5 - 2e-6
    assert logic_accuracy(outputs, targets)["all16_correct"]


def test_endpoint_correctness_does_not_imply_late_window_correctness():
    outputs, targets = truth_table()
    outputs[1, 0, 900] = .1
    result = logic_accuracy(outputs, targets)
    assert result["all16_correct"]
    assert not result["late_all16_correct"]


def test_invalid_arrays_abort_instead_of_counting_as_correct():
    outputs, targets = truth_table()
    with pytest.raises(ValueError):
        logic_accuracy(outputs[:15], targets)
    outputs[0, 0, 2] = np.nan
    with pytest.raises(ValueError):
        logic_accuracy(outputs, targets)


def test_template_subtraction_checks_channels_and_rates_not_numeric_id():
    template = [reaction(0, "u_1")]
    additions = [reaction(i + 1, rate=i + 2.) for i in range(5)]
    stored = [reaction(9, "u_1"), *additions]
    assert added_records(stored, template) == additions
    stored[0]["parameters"] = [2.]
    with pytest.raises(ValueError, match="exact fixed template"):
        added_records(stored, template)


def test_exhaustive_selection_allows_joint_removal_and_deterministic_ties():
    # Single removals fail, but a joint removal is valid; a greedy pass misses it.
    rows = [core(5, "original", deviation=0), core(4, "single", correct=False),
            core(3, "b"), core(3, "a"), core(2, "too_different", deviation=.02),
            core(1, "above_loss_threshold", loss=.1), {"status": "solver_failure"}]
    assert select_core(rows, .01, .04)["topology_hash"] == "a"
    assert select_core(rows, .001, .04)["topology_hash"] == "original"
    assert select_core(rows, .05, .04)["topology_hash"] == "too_different"


def test_core_counts_deduplicate_structures_within_eligible_population():
    record = {"qualified": True, "all16_correct": True, "late_all16_correct": True,
              "selected_cores": {"0.01": core(3, "same_core")}}
    records = [deepcopy(record), deepcopy(record), {"qualified": False}]
    result = seed_summary(records, .01)
    assert result["qualified_all16_correct"] == 2
    assert result["distinct_pruned_cores"] == 1
    assert result["merged_candidates"] == 1
    assert result["prunable_candidates"] == 2
    assert result["removed_reactions"] == 4


def test_loss_cutoff_is_strict_and_output_tolerance_is_inclusive():
    original = core(5, "original", deviation=0)
    rows = [original, core(1, "at_loss_cutoff", loss=.04),
            core(3, "at_deviation_tolerance", deviation=.01),
            core(2, "nonfinite_loss", loss=float("nan"))]
    assert select_core(rows, .01, .04)["topology_hash"] == "at_deviation_tolerance"


def reaction(index, input_channel=None, rate=1.):
    return {"index": index, "reaction_id": 1, "parameters": [rate],
            "structure": {"type": "MassAction", "inputs": [input_channel],
                          "reactants": [], "products": ["X_1"]}}


def core(count, name, loss=.02, deviation=.005, correct=True):
    return {"status": "ok", "retained_count": count, "structure_json": name,
            "topology_hash": name, "all16_correct": correct, "loss": loss,
            "max_output_deviation": deviation}
