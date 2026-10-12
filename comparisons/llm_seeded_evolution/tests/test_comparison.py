import json

import pytest

from comparisons.llm_seeded_evolution import compare
from comparisons.llm_seeded_evolution.replay import sha256


def finished_run(tmp_path, monkeypatch, *, pilot=False, task="logic"):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "analysis_plan_v2.json").write_text(json.dumps({
        "quality_thresholds": {task: 0.1}, "portfolio_top_k": 30}))
    monkeypatch.setattr(compare, "GENERATED", generated)
    run = tmp_path / "raw" / "llm_seeded_evolution" / "seed0"
    run.mkdir(parents=True)
    replay = {"communication_mode": "none", "expose_state_to_llm": False,
              "insert_llm_into_rl": False, "search_budget": 100, "llm_evaluations": 2,
              "source_terminal_archive": [{"loss": 0.08, "topology_hash": "source"}]}
    (run / "replay.json").write_text(json.dumps(replay))
    (run / "config.json").write_text(json.dumps({"evolution": {"batch_size": 32, "workers": 2}}))
    (run / "terminal_archive.json").write_text(json.dumps([
        {"loss": 0.05, "topology_hash": "first"}, {"loss": 0.07, "topology_hash": "second"}]))
    summary = {"task": task, "seed": 0, "status": "complete", "pilot": pilot,
               "search_evaluations": 100, "llm_evaluations": 2, "candidate_evaluations": 102,
               "quality_threshold": 0.1, "portfolio_top_k": 30, "best_loss": 0.05,
               "qualified_topologies": 2, "replay_sha256": sha256(run / "replay.json"),
               "config_sha256": sha256(run / "config.json"),
               "terminal_archive_sha256": sha256(run / "terminal_archive.json")}
    (run / "completed.json").write_text(json.dumps(summary))
    return run


def test_paper_comparison_excludes_pilots(tmp_path, monkeypatch):
    finished_run(tmp_path, monkeypatch, pilot=True)
    with pytest.raises(ValueError, match="pilots are excluded"):
        compare.collect_runs(tmp_path / "raw", "logic", [0])


def test_comparison_exports_same_stream_pair_and_correct_direction(tmp_path, monkeypatch):
    finished_run(tmp_path, monkeypatch)
    out = tmp_path / "output"
    compare.main(["--task", "logic", "--seeds", "0", "--raw-root", str(tmp_path / "raw"),
                  "--output-dir", str(out), "--methods", "no_communication", "llm_seeded_evolution"])
    summary = json.loads((out / "logic_summary.json").read_text())
    paired = summary["paired_evolution_minus_independent_union"]
    assert paired["best_loss"]["wins"] == 1
    assert paired["qualified_topologies"]["median_delta"] == 1
    assert "No communication (terminal union)" in (out / "logic_table.tex").read_text()


def test_comparison_rejects_changed_archive(tmp_path, monkeypatch):
    run = finished_run(tmp_path, monkeypatch)
    (run / "terminal_archive.json").write_text("[]")
    with pytest.raises(ValueError, match="unverified terminal archive"):
        compare.collect_runs(tmp_path / "raw", "logic", [0])


def test_classifier_defaults_to_same_stream_comparison(tmp_path, monkeypatch):
    finished_run(tmp_path, monkeypatch, task="classifier")
    out = tmp_path / "output"
    compare.main(["--task", "classifier", "--seeds", "0", "--raw-root", str(tmp_path / "raw"),
                  "--output-dir", str(out)])
    summary = json.loads((out / "classifier_summary.json").read_text())
    assert set(summary["methods"]) == {"no_communication", "llm_seeded_evolution"}
    with pytest.raises(ValueError, match="do not cover Classifier"):
        compare.main(["--task", "classifier", "--methods", "full"])
