from copy import deepcopy
import json
import sqlite3

import pytest

from comparisons.search_exploration.run_julia_rpa_evaluation import prepare


def reference():
    return {"schema": "search-streams-v2", "task": "rpa", "config": {
        "seed": 0, "epochs": 100, "batch_size": 1023, "evaluation_budget": 1636800,
        "fit_evaluations": 16, "quality_threshold": .016491168958935352,
        "rl_fraction": .5, "mutation_fraction": .25}}


def test_real_campaign_preserves_control_and_matching_settings():
    original = reference()
    saved = deepcopy(original)
    result = prepare(original, {"status": "complete", "counters": {"expensive_evaluations": 214793}})
    assert original == saved
    assert result == {**saved["config"], "evaluation_budget": 214793, "fit_evaluations": 24}


@pytest.mark.parametrize("field,value", [("schema", "search-streams-v1"), ("task", "logic")])
def test_rejects_incompatible_controls(field, value):
    original = reference()
    original[field] = value
    with pytest.raises(ValueError):
        prepare(original, {"status": "complete"})


def test_no_automatic_replacement_of_unfinished_reference():
    with pytest.raises(ValueError, match="already be complete"):
        prepare(reference(), {"status": "running"})


def test_comparison_counts_discoveries_not_just_retained_archive(tmp_path, monkeypatch):
    from comparisons.search_exploration import run_julia_rpa_evaluation as campaign
    paths = [tmp_path / "run", tmp_path / "old-hybrid", tmp_path / "old-ga"]
    for path in paths:
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(reference()))
        (path / "completed.json").write_text(json.dumps({"elapsed_seconds": 120.,
            "counters": {"expensive_evaluations": 100},
            "metrics": {"Archive/best_loss": .001, "Archive/qualified_topologies": 1}}))
        with sqlite3.connect(path / "exploration.sqlite") as db:
            db.execute("CREATE TABLE designs (payload TEXT)")
            db.executemany("INSERT INTO designs VALUES (?)", [(json.dumps({"qualified": True,
                "topology_hash": key}),) for key in ("A", "A", "B")])
    monkeypatch.setattr(campaign, "REFERENCES", {"Evolution-fit hybrid": paths[1], "Standalone GA": paths[2]})
    campaign.comparison(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert all(r["qualified_discovered"] == 2 and r["qualified_retained"] == 1 for r in result)
    assert "not equal compute" in (tmp_path / "REPORT.md").read_text()


def test_report_handles_job_without_feasible_candidate(tmp_path):
    from comparisons.search_exploration.report_julia_stream import render
    metrics = {"Archive/best_loss": .1, "Archive/topologies": 1, "Archive/qualified_topologies": 0,
        "Optimization/improved": 0, "Compute/expensive_evaluations": 1, "Julia/measured_fit_share": .2}
    files = {"manifest.json": {"config": {"batch_size": 1}, "search_workers": 1,
        "optimization_workers": 1, "julia_recipe": {"compute_share": .2}},
        "completed.json": {"status": "complete", "epochs": 1, "elapsed_seconds": 1.,
            "counters": {"expensive_evaluations": 1}, "metrics": metrics, "pending_optimization_jobs": 0},
        "optimization_results.json": [{"best": None}],
        "julia_jobs/no-feasible.json": {"job": {"candidate": {"simulation_complete": False}},
            "outcome": {"best": None}, "phases": []}}
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"epoch": 0, **metrics}) + "\n")
    render(tmp_path)
    assert "rescued to canonical completion: 0" in (tmp_path / "REPORT.md").read_text()
    assert (tmp_path / "julia_summary.png").exists()
