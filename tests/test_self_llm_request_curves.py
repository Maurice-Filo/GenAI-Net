import json
import math

import pytest

from comparisons.search_exploration.plot_self_llm_request_curves import (
    TASKS, aggregate, request_prefix, self_context,
)


def test_failed_or_invalid_batches_never_improve_prefix():
    values = request_prefix([
        [], [{"loss": .2, "valid": True}],
        [{"loss": .001, "valid": False}, {"loss": float("nan"), "valid": True},
         {"loss": -.1, "valid": True}],
        [{"loss": .1, "valid": True}], [],
    ])
    assert math.isinf(values[0])
    assert values[1:] == [.2, .2, .1, .1]


def test_context_rejects_outward_state_and_records_missing_self_feedback(tmp_path):
    context = tmp_path / "CONTEXT"
    context.mkdir()
    state = {"hall_of_fame_size": 0, "rl_epoch_at_snapshot": 20,
             "llm_feedback": "No previous LLM feedback is available.",
             "llm_best": "No LLM-generated candidate has been accepted yet."}
    (context / "SEARCH_STATE.json").write_text(json.dumps(state))
    (context / "HALL_OF_FAME.json").write_text('{"entries": []}')
    for name in ("SIL_STATUS.md", "EXCLUDED_TOPOLOGIES.md"):
        (context / name).write_text("Cross-source state withheld.")
    assert not self_context(tmp_path)["has_self_feedback"]
    state["llm_feedback"] = "valid; loss=0.1"
    (context / "SEARCH_STATE.json").write_text(json.dumps(state))
    assert self_context(tmp_path)["has_self_feedback"]
    (context / "HALL_OF_FAME.json").write_text('{"entries": [{"loss": 0.01}]}')
    with pytest.raises(ValueError, match="Outward HOF"):
        self_context(tmp_path)


def test_summary_keeps_all_twenty_seeds_and_iqr():
    rows = [{"task": task, "request": request, "seed": seed, "best_so_far_loss": seed + 1.,
             "reaches_rl_reference": seed < 10, "has_self_feedback": request > 0,
             "status": "completed", "rl_median_reference": 10.}
            for task in TASKS for request in range(5) for seed in range(20)]
    result = aggregate(rows)
    assert len(result) == 15
    assert result[0]["median"] == 10.5
    assert result[0]["q25"] == 5.75
    assert result[0]["q75"] == 15.25
    assert result[0]["reaches_rl_reference"] == 10
    rows.pop()
    with pytest.raises(ValueError, match="twenty seeds"):
        aggregate(rows)


def test_missing_prefix_not_silently_dropped():
    rows = [{"task": "rpa", "request": 0, "seed": seed,
             "best_so_far_loss": None if seed == 0 else .1} for seed in range(20)]
    with pytest.raises(ValueError, match="censored"):
        aggregate(rows)
