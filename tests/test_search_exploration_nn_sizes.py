from copy import deepcopy

import pytest

from comparisons.search_exploration.benchmark_nn_sizes import comparison_rows, describe_times


def cases():
    shared = {"task": "logic", "samples": 16, "elites": 16, "seed": 0, "dtype": "torch.float32",
              "hidden_width": 1024, "hidden_depth": 5, "reaction_steps": 5,
              "imitation_bank_hash": "same-bank", "cpu_threads": 1, "warmup": 2,
              "repeats": 5, "torch": "test", "library_size": 421}
    return [{**shared, "embedding": size, "timings": {"total_seconds": {"median": seconds}},
             "parameters": size * 10, "memory": {"training_tensor_bytes": size * 160}}
            for size, seconds in ((1024, .5), (10240, 1.))]


def test_size_comparison_direction_and_memory():
    result = comparison_rows(cases())[0]
    assert result["speedup_large_over_small"] == 2
    assert result["time_reduction_percent"] == 50
    assert result["small_training_mib"] < result["large_training_mib"]


@pytest.mark.parametrize("field", ["imitation_bank_hash", "hidden_width", "cpu_threads", "seed", "repeats"])
def test_size_comparison_rejects_unmatched_settings(field):
    values = cases()
    values[1][field] = "different"
    with pytest.raises(ValueError, match="Unmatched"):
        comparison_rows(values)


def test_size_comparison_rejects_duplicates_and_missing_arms():
    values = cases()
    for invalid in ([], values[:1], values + [deepcopy(values[0])]):
        with pytest.raises(ValueError):
            comparison_rows(invalid)


def test_timing_summary_uses_median_and_extremes():
    result = describe_times([{k: value for k in ("sampling_seconds", "update_seconds", "total_seconds")}
                             for value in (1., 2., 9.)])
    assert result["total_seconds"] == {"min": 1., "median": 2., "max": 9.}
