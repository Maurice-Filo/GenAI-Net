import json
import sqlite3

import numpy as np

from comparisons.search_exploration.plot_rpa_thresholds import complete_pairs, qualifying_counts, topology_losses


def test_threshold_counts_deduplicate_best_valid_design_and_ignore_frozen_flag(tmp_path):
    rows = [
        {"topology_hash": "A", "loss": .03, "simulation_complete": True, "qualified": False},
        {"topology_hash": "A", "loss": .01, "simulation_complete": True, "qualified": True},
        {"topology_hash": "B", "loss": .02, "simulation_complete": True, "qualified": False},
        {"topology_hash": "C", "loss": .001, "simulation_complete": False, "qualified": True},
        {"topology_hash": "D", "loss": None, "simulation_complete": True},
    ]
    with sqlite3.connect(tmp_path / "exploration.sqlite") as db:
        db.execute("CREATE TABLE designs(payload TEXT)")
        db.executemany("INSERT INTO designs VALUES (?)", [(json.dumps(r),) for r in rows])
    losses = topology_losses(tmp_path)
    np.testing.assert_array_equal(losses, [.01, .02])
    np.testing.assert_array_equal(qualifying_counts(losses, [.005, .01, .015, .02, .03]), [0, 0, 1, 1, 2])


def test_incomplete_and_unmatched_seeds_are_not_pooled():
    jobs = [{"name": f"{a}{s}", "arm": a, "seed": s} for s in (1, 2) for a in ("julia", "evolution", "ga")]
    plan = {"arms": ["julia", "evolution", "ga"], "seeds": [1, 2], "jobs": jobs}
    progress = {"jobs": [{**j, "status": "running" if j["name"] == "ga2" else "complete"} for j in jobs]}
    seeds, matched = complete_pairs(plan, progress)
    assert seeds == [1] and len(matched) == 3


def test_count_curve_is_monotone_and_empty_archive_stays_zero():
    thresholds = np.geomspace(.0001, 1., 100)
    counts = qualifying_counts([.1, .001, .001, .2], thresholds)
    assert np.all(np.diff(counts) >= 0)
    np.testing.assert_array_equal(qualifying_counts([], thresholds), np.zeros(100))
