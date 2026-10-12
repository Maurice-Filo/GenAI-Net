import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comparisons.search_exploration import rpa_directional_stream_campaign as campaign
from comparisons.search_exploration import analyze_rpa_directional_streams as analysis


def test_plan_preserves_science_and_balances_arms(tmp_path):
    plan = campaign.prepare(tmp_path / "campaign", concurrency=10, comet_project="test")
    assert len(plan["jobs"]) == 40
    assert plan["communication"] == ["llm-to-rl", "rl-to-llm"]
    assert plan["concurrency"] == 10
    for seed in range(1, 21):
        jobs = [j for j in plan["jobs"] if j["seed"] == seed]
        assert {j["arm"] for j in jobs} == set(campaign.MODES)
        for j in jobs:
            command = j["command"]
            assert command[command.index("--communication-mode") + 1] == j["arm"]
            assert command[command.index("--search-workers") + 1] == "6"
            assert command[command.index("--llm-every") + 1] == "5"
            cfg = json.loads(Path(command[command.index("--exploration-config")+1]).read_text())
            assert (cfg["seed"], cfg["epochs"], cfg["batch_size"], cfg["evaluation_budget"]) == (seed,100,1023,214793)
            assert cfg["fit_seconds"] is None and cfg["evaluation_seconds"] is None
    assert plan["jobs"][0]["arm"] != plan["jobs"][2]["arm"]


def test_existing_campaign_occupancy_is_counted(tmp_path):
    p = tmp_path / "progress.json"
    p.write_text(json.dumps({"status": "running", "jobs": [
        {"status": "running", "pid": 123}, {"status": "running", "pid": 456}, {"status": "complete"}]}))
    def alive(pid, _):
        if pid == 456:
            raise ProcessLookupError()
    with patch.object(campaign.os, "kill", side_effect=alive):
        assert campaign.external_active(tmp_path) == 1


def test_union_rejects_hof_conditioned_bank(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"communication_mode": "rl-to-llm", "withhold_initial_hof": True}))
    with pytest.raises(ValueError, match="insertion-only"):
        analysis.audit_self_feedback_bank(tmp_path)


def test_union_requires_withheld_saved_contexts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"communication_mode": "llm-to-rl", "withhold_initial_hof": True}))
    w = tmp_path / "harness-workspaces" / "request0"
    (w / "CONTEXT").mkdir(parents=True)
    (w / "run_manifest.json").write_text("{}")
    (w / "CONTEXT/HALL_OF_FAME.json").write_text('{"entries": []}')
    for name in ("SIL_STATUS.md", "EXCLUDED_TOPOLOGIES.md"):
        (w / "CONTEXT" / name).write_text("Search state withheld")
    assert analysis.audit_self_feedback_bank(tmp_path) == 1
    (w / "CONTEXT/EXPLORATION_STATE.json").write_text("{}")
    with pytest.raises(ValueError, match="leaked"):
        analysis.audit_self_feedback_bank(tmp_path)


def test_union_cost_guard_and_no_fit_descendants(tmp_path):
    cfg = {"seed": 1, "epochs": 100, "batch_size": 1023, "rate_min": .001,
           "rate_max": 100., "quality_threshold": .02, "evaluation_budget": 100}
    ctrl, bank = tmp_path / "control", tmp_path / "bank"
    ctrl.mkdir(); bank.mkdir()
    for root in (ctrl, bank):
        (root / "manifest.json").write_text(json.dumps({"config": cfg, "task_config": {"rpa": {}}}))
    (ctrl / "completed.json").write_text(json.dumps({"counters": {"expensive_evaluations": 99}}))
    (bank / "llm_history.json").write_text("[]")
    direct = [{"topology_hash": "llm", "loss": .01, "simulation_complete": True}]
    with patch.object(analysis,"audit_self_feedback_bank",return_value=1), \
         patch.object(analysis,"designs",return_value=[]), \
         patch.object(analysis,"direct_records",return_value=direct):
        result = analysis.pooled_control(ctrl, bank)
        assert result["canonical_joint_charge_upper_bound"] == 100
        assert result["union_qualified"] == 1
        assert not result["matched_completion_schedule"]
        direct.append(dict(direct[0]))
        with pytest.raises(ValueError, match="exceeds"):
            analysis.pooled_control(ctrl, bank)
