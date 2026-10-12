import json

import pytest

from comparisons.llm_seeded_evolution.campaign import command
from comparisons.llm_seeded_evolution.fixed_stream_report import collect
from comparisons.llm_seeded_evolution.replay import sha256
from comparisons.rpa_search.src.common.io import write_json


def test_local_only_command_has_no_external_logging_and_uses_1023(tmp_path):
    plan = {"workers": 4, "comet_project": None, "campaign_id": "test"}
    job = {"seed": 3, "replay": "/frozen/replay.json", "run_id": "unique"}
    cmd = command(tmp_path, plan, job)
    assert "--comet-project" not in cmd
    assert cmd[cmd.index("--batch-size") + 1] == "1023"
    assert cmd[cmd.index("--source-mode") + 1] == "llm-to-rl"
    assert cmd[cmd.index("--replay") + 1] == job["replay"]


def completed(tmp_path):
    replay = {"communication_mode": "llm-to-rl", "search_budget": 102300,
              "llm_evaluations": 40,
              "source_terminal_archive": [{"topology_hash": "rl", "loss": .08}]}
    write_json(tmp_path / "replay.json", replay)
    write_json(tmp_path / "config.json", {"evolution": {"batch_size": 1023}})
    write_json(tmp_path / "terminal_archive.json", [
        {"topology_hash": "ga1", "loss": .05}, {"topology_hash": "ga2", "loss": .06}])
    result = {"status": "complete", "pilot": False, "search_evaluations": 102300,
        "llm_evaluations": 40, "candidate_evaluations": 102340,
        "quality_threshold": .1, "portfolio_top_k": 30, "best_loss": .05,
        "qualified_topologies": 2, "replay_sha256": sha256(tmp_path / "replay.json"),
        "config_sha256": sha256(tmp_path / "config.json"),
        "terminal_archive_sha256": sha256(tmp_path / "terminal_archive.json")}
    write_json(tmp_path / "completed.json", result)
    return {"jobs": [{"output": str(tmp_path), "cohort": "local_logic", "seed": 0,
                       "replay_sha256": result["replay_sha256"]}]}


def test_fixed_stream_endpoint_compares_actual_source_archive(tmp_path):
    plan = completed(tmp_path)
    row, = collect(plan)
    assert (row["ga_qualified"], row["rl_qualified"]) == (2, 1)
    assert (row["ga_loss"], row["rl_loss"]) == (.05, .08)
    assert row["replayed_proposals"] == 40


def test_fixed_stream_report_rejects_different_proposals(tmp_path):
    plan = completed(tmp_path)
    plan["jobs"][0]["replay_sha256"] = "other"
    with pytest.raises(ValueError, match="planned proposal stream"):
        collect(plan)


def test_replay_serialization_is_identical_in_freeze_and_runner(tmp_path):
    replay = {"z": 1, "events": [{"b": 2, "a": 3}]}
    write_json(tmp_path / "frozen.json", replay)
    write_json(tmp_path / "runtime.json", json.loads((tmp_path / "frozen.json").read_text()))
    assert sha256(tmp_path / "frozen.json") == sha256(tmp_path / "runtime.json")
