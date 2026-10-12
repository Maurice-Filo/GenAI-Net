import unittest

from comparisons.llm_crn_generation.run_flash_half_duplex_queue import (
    MAIN_TASK_ORDER,
    MODEL,
    MODES,
    audit_command,
    campaign_command,
    campaign_id,
    method_name,
    run_suffix,
)


class FlashHalfDuplexQueueTests(unittest.TestCase):
    def test_logic_main_campaign_precedes_rpa(self):
        self.assertEqual(MAIN_TASK_ORDER, ("logic", "rpa"))

    def test_both_directional_arms_are_distinct(self):
        self.assertEqual(MODES, ("llm-to-rl", "rl-to-llm"))
        for mode in MODES:
            self.assertIn(mode, campaign_id("rpa", mode, "sentinel"))
            self.assertIn(mode.replace("-", "_"), method_name(mode))
            self.assertIn(mode.replace("-", "_"), run_suffix(mode))

    def test_sentinel_uses_frozen_flash_contract(self):
        for mode in MODES:
            command = campaign_command("logic", mode, "sentinel")
            joined = " ".join(command)
            self.assertIn("--tasks logic", joined)
            self.assertEqual(command[command.index("--seed-start") + 1], "0")
            self.assertEqual(command[command.index("--seeds") + 1], "1")
            self.assertIn("--max-parallel 1 --cpus-per-run 4", joined)
            self.assertIn(f"--model {MODEL}", joined)
            self.assertIn("--llm-provider deepseek-official", joined)
            self.assertIn(f"--communication-mode {mode}", joined)
            self.assertIn("--withhold-initial-hof", joined)
            self.assertIn("--candidate-validation-policy independent-members", joined)
            self.assertIn("--forbidden-topology-m 0", joined)

    def test_main_arms_offer_twenty_concurrent_seed_jobs_for_one_task(self):
        for mode in MODES:
            command = campaign_command("rpa", mode, "main")
            joined = " ".join(command)
            self.assertEqual(command[command.index("--seed-start") + 1], "1")
            self.assertEqual(command[command.index("--seeds") + 1], "19")
            self.assertIn("--max-parallel 10 --cpus-per-run 4", joined)
            self.assertIn("--epochs 100", joined)
            self.assertIn("--rl-batch-size 1023", joined)
            self.assertIn("--total-candidate-budget 102400", joined)
            self.assertIn("--llm-candidates 10 --llm-every 20", joined)
            self.assertIn("--global-llm-concurrency 8", joined)

    def test_audit_matches_each_phase(self):
        for mode in MODES:
            sentinel = " ".join(audit_command("rpa", mode, "sentinel"))
            main = " ".join(audit_command("rpa", mode, "main"))
            self.assertIn("--seed-start 0 --seeds 1", sentinel)
            self.assertIn("--seed-start 1 --seeds 19", main)
            self.assertIn(f"--communication-mode {mode}", sentinel)
            self.assertIn(f"--model {MODEL}", main)


if __name__ == "__main__":
    unittest.main()
