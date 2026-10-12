import unittest

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_logic_no_communication_campaign import (
    CAMPAIGN_ID,
    METHOD,
    RUN_SUFFIX,
    audit_command,
    campaign_command,
)


class LocalQwen38LogicNoCommunicationCampaignTests(unittest.TestCase):
    def test_logic_campaign_matches_the_rpa_ablation_settings(self):
        command = campaign_command("Qwen3.8-test")
        joined = " ".join(command)
        expected = (
            "--tasks logic",
            "--max-parallel 8",
            "--cpus-per-run 4",
            "--seeds 20",
            "--seed-start 0",
            "--epochs 100",
            "--rl-batch-size 1023",
            "--total-candidate-budget 102400",
            "--llm-candidates 10",
            "--llm-every 20",
            "--max-agent-evaluations 0",
            "--max-literature-searches 0",
            "--global-llm-concurrency 8",
            "--generation-backend harness",
            "--communication-mode none",
            "--withhold-initial-hof",
            "--candidate-validation-policy independent-members",
        )
        for fragment in expected:
            self.assertIn(fragment, joined)
        self.assertNotIn("--tasks rpa", joined)

    def test_namespaces_and_audit_are_logic_specific(self):
        command = campaign_command("Qwen3.8-test")
        audit = audit_command("Qwen3.8-test")
        self.assertIn("logic-no-communication", CAMPAIGN_ID)
        self.assertIn("logic_no_communication", METHOD)
        self.assertIn("logic_no_communication", RUN_SUFFIX)
        self.assertEqual(command[command.index("--method-name") + 1], METHOD)
        self.assertEqual(command[command.index("--run-suffix") + 1], RUN_SUFFIX)
        self.assertEqual(audit[audit.index("--tasks") + 1], "logic")
        self.assertEqual(audit[audit.index("--communication-mode") + 1], "none")


if __name__ == "__main__":
    unittest.main()
