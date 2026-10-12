import json
from types import SimpleNamespace
from unittest.mock import patch

from RL4CRN.llm import workspace_loss_protocol as protocol
from comparisons.llm_crn_generation import run_parity12_communication_queue as queue
from comparisons.llm_crn_generation import run_mmc2_harness_hybrid as hybrid


def test_twelve_reactions_and_no_other_scientific_changes():
    check = queue.preflight()
    assert check["added_reactions"] == 12
    assert check["total_reactions"] == 16
    assert check["masked_ids"] == [0, 1, 2, 3, 4]
    assert check["rate_bounds"] == [.001, 100.]
    assert protocol.config_path_for_variant("logic_parity", protocol.TASK_VARIANT) != protocol.config_path_for_variant("logic_parity", protocol.TWELVE_VARIANT)
    assert protocol.review_paths(protocol.TASK_VARIANT) != protocol.review_paths(protocol.TWELVE_VARIANT)
    protocol.validate_protocol_release(protocol.TASK_VARIANT)
    protocol.validate_protocol_release(protocol.TWELVE_VARIANT)


def test_commands_keep_budget_schedule_model_and_isolation():
    for mode in queue.old.MODES:
        new = queue.command(mode)
        original = queue.old.campaign_command(mode, protocol.PROTOCOL)
        for flag in ["--model", "--epochs", "--seeds", "--rl-batch-size", "--total-candidate-budget",
                     "--communication-mode", "--llm-every", "--llm-timeout", "--comet-project", "--max-parallel"]:
            assert new[new.index(flag)+1] == original[original.index(flag)+1]
        assert "--withhold-initial-hof" in new
        assert new[new.index("--task-prompt-variant")+1] == protocol.TWELVE_VARIANT
        assert queue.campaign_id(mode) != queue.old.campaign_id(mode, protocol.PROTOCOL)


def test_production_trainer_receives_twelve_reaction_budget():
    captured = {}
    def trainer(cfg, task, logger):
        captured["budget"] = cfg.train.max_added_reactions
        return SimpleNamespace(s=object())
    with patch.object(hybrid, "make_session_and_trainer", side_effect=trainer), patch.object(
        hybrid.LLMCandidateEvaluator, "from_session", return_value=object()
    ):
        config, _, _, _ = hybrid.build_hybrid_run(task_name="logic_parity", seed=0, n_cpus=4,
            rl_batch_size=1023, logger=None, task_prompt_variant=protocol.TWELVE_VARIANT)
    assert captured["budget"] == config["search"]["max_added_reactions"] == 12
