import json
from pathlib import Path

import pytest

from comparisons.search_exploration import logic5_four_arm_campaign as campaign


def test_frozen_plan_has_all_80_jobs_and_matched_budgets(tmp_path):
    ready = tmp_path / "ready.json"
    ready.write_text(json.dumps({"status": "passed", "prompt_sha256": campaign.digest(campaign.PROMPT)}))
    root = tmp_path / "campaign"
    plan = campaign.prepare(root, ready)
    assert len(plan["jobs"]) == 80
    assert plan["withhold_initial_hof"] is True
    assert plan["harness_role_timeout_seconds"] == 3600
    for seed in range(1, 21):
        jobs = [j for j in plan["jobs"] if j["seed"] == seed]
        assert {j["arm"] for j in jobs} == set(campaign.MODES)
        config = json.loads((root / "configs" / f"search-seed{seed:02d}.json").read_text())
        assert (config["epochs"], config["batch_size"], config["evaluation_budget"]) == (100, 1023, 214793)
        assert config["fit_seconds"] is None and config["evaluation_seconds"] is None
        task = json.loads((root / "configs" / f"task-seed{seed:02d}.json").read_text())
        assert task["logic"]["formula"] == "chain_or_pairs_4"
        assert task["search"]["max_added_reactions"] == 5
        assert task["search"]["rate_constant_range"] == [.001, 100.]
        for j in jobs:
            command = j["command"]
            assert command[command.index("--communication-mode") + 1] == j["arm"]
            assert command[command.index("--llm-timeout") + 1] == "3600"
            assert "--comet-project" in command
    campaign.verify_inputs(plan)
    with pytest.raises(FileExistsError):
        campaign.prepare(root, ready)


def test_prompt_change_cannot_bypass_preflight(tmp_path):
    ready = tmp_path / "ready.json"
    ready.write_text('{"status":"passed","prompt_sha256":"wrong"}')
    with pytest.raises(ValueError, match="preflight"):
        campaign.prepare(tmp_path / "campaign", ready)
