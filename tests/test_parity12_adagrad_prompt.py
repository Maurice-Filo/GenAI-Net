from itertools import product
from pathlib import Path

import numpy as np

from comparisons.rpa_search.src.common.logic_task import build_logic_fn
from RL4CRN.utils.input_interface import build_weights
from RL4CRN.utils.utils import performance_metric

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / 'comparisons/search_exploration/prompts/parity12-evidence-exploitation-v2.md'


def test_prompt_preserves_task_and_rpa_operational_guidance():
    text = PROMPT.read_text()
    for clause in ('exactly twelve distinct allowed added reactions', '[0.001, 100]',
                   'all 16 combinations', 'X_1, X_2, X_3, X_4, S_1, S_2, OUT',
                   'Request 0 intentionally has no initial HOF',
                   'six evidence-guided refinements', 'not a validity quota',
                   'neither selection for fitting nor improvement is guaranteed',
                   'Distinguish supplied-rate results from fitted results',
                   'Writer encodes them without changing scientific choices',
                   'independent rejection of invalid members'):
        assert clause in text
    assert 'disturbance u_2' not in text
    assert 'X_3 = u_1' not in text


def test_parity_targets_match_all_sixteen_rows():
    fn = build_logic_fn('parity_4')
    rows = list(product((0, 1), repeat=4))
    assert sum(fn(row) for row in rows) == 8
    for row in rows:
        assert fn(row) == (sum(row) in (1, 3))


def test_weighted_loss_is_mean_not_weight_normalized():
    weights = build_weights(1, 1000, 'transient')
    np.testing.assert_array_equal(weights[0, :200], .25)
    np.testing.assert_array_equal(weights[0, 200:800], 1.)
    np.testing.assert_array_equal(weights[0, 800:], 2.)
    score = performance_metric([np.array([0.])]*16, [np.ones((1, 1000))]*16,
                               weights, norm=1)
    assert np.isclose(score, 1.05)
    assert 'divided by the sum of weights' in PROMPT.read_text()
