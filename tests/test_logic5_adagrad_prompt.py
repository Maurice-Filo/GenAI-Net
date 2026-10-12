import json
from itertools import product
from pathlib import Path

import numpy as np

from comparisons.rpa_search.src.common.logic_task import build_logic_fn
from RL4CRN.search_exploration.evaluation import build
from RL4CRN.utils.input_interface import build_weights
from RL4CRN.utils.utils import performance_metric

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "comparisons/search_exploration/prompts/logic5-evidence-exploitation-v2.md"
CONFIG = ROOT / "comparisons/rpa_search/configs/logic_100k.json"


def test_task_prompt_keeps_new_operational_standard():
    text = PROMPT.read_text()
    for clause in ("exactly five distinct allowed added reactions", "no support species",
                   "four fixed template reactions unchanged", "template-reserved IDs",
                   "rates above 100 are clamped to 100", "Request 0 intentionally has no initial HOF",
                   "six evidence-guided refinements", "not a validity quota",
                   "Distinguish supplied-rate results from fitted results",
                   "Do not request new evaluations", "independent rejection of invalid members"):
        assert clause in text
    assert "S_1" not in text and "XOR" not in text
    assert "Writer encodes them without\nchanging scientific choices" in text


def test_truth_table_and_prompt_examples():
    config = json.loads(CONFIG.read_text())
    fn = build_logic_fn(config["logic"]["formula"])
    for u in product((0, 1), repeat=4):
        assert fn(u) == any(u[i] and u[i + 1] for i in range(3))
    assert fn([1, 1, 0, 0]) and not fn([1, 0, 1, 0])


def test_live_evaluator_matches_new_prompt_with_explicit_production_bounds():
    config = json.loads(CONFIG.read_text())
    # New runs need a copied production config, not an edit to the legacy benchmark.
    config["search"]["rate_constant_range"] = [.001, 100.]
    evaluator, task, _ = build("logic", config, .001, 100.)
    assert evaluator.max_added_reactions == 5
    assert evaluator.forbidden_reaction_ids == frozenset({0, 1, 2, 3, 4})
    assert len(evaluator.initial_reaction_ids) == 4
    assert set(evaluator.crn_template.species_labels) == {"X_1", "X_2", "X_3", "X_4", "OUT"}
    assert task.kind == "logic" and task.norm == 1
    assert str(evaluator.crn_template.solver).upper() == "CVODE"
    assert evaluator.crn_template.rtol == evaluator.crn_template.atol == 1e-8
    assert (evaluator.min_parameter_value, evaluator.max_parameter_value) == (.001, 100.)


def test_loss_normalization_matches_prompt():
    weights = build_weights(1, 1000, "transient")
    np.testing.assert_array_equal(weights[0, :200], .25)
    np.testing.assert_array_equal(weights[0, 200:800], 1.)
    np.testing.assert_array_equal(weights[0, 800:], 2.)
    loss = performance_metric([np.array([0.])] * 16, [np.ones((1, 1000))] * 16,
                              weights, norm=1)
    assert np.isclose(loss, 1.05)
    assert "16*1000, not by the sum of weights" in PROMPT.read_text()
