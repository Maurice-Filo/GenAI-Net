import unittest

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa_logic_campaign import (
    CAMPAIGN_ID,
    ENDPOINT,
    METHOD,
    RUN_SUFFIX,
    audit_command,
    campaign_command,
)


class LocalQwen38RpaLogicCampaignTests(unittest.TestCase):
    def test_campaign_is_matched_and_withholds_request_zero_hof(self):
        command = campaign_command("Qwen3.8-test")
        joined = " ".join(command)
        expected = (
            "--tasks rpa logic",
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
            "--communication-mode full",
            "--withhold-initial-hof",
            "--candidate-validation-policy independent-members",
        )
        for fragment in expected:
            self.assertIn(fragment, joined)

    def test_local_identity_and_audit_are_frozen(self):
        command = campaign_command("Qwen3.8-test")
        audit = audit_command("Qwen3.8-test")
        self.assertEqual(ENDPOINT, "http://127.0.0.1:18082/v1")
        self.assertIn("20seed", CAMPAIGN_ID)
        self.assertEqual(command[command.index("--method-name") + 1], METHOD)
        self.assertEqual(command[command.index("--run-suffix") + 1], RUN_SUFFIX)
        self.assertEqual(command[command.index("--model") + 1], "Qwen3.8-test")
        self.assertEqual(audit[audit.index("--model") + 1], "Qwen3.8-test")


if __name__ == "__main__":
    unittest.main()
