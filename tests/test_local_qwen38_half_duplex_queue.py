import unittest

from comparisons.llm_crn_generation.run_local_qwen38_half_duplex_queue import (
    MODES,
    audit_command,
    campaign_command,
    campaign_id,
    method_name,
    run_suffix,
)


class LocalQwen38HalfDuplexQueueTests(unittest.TestCase):
    def test_both_directional_arms_are_distinct(self):
        self.assertEqual(MODES, ("llm-to-rl", "rl-to-llm"))
        for mode in MODES:
            self.assertIn(mode, campaign_id("rpa", mode, "sentinel"))
            self.assertIn(mode.replace("-", "_"), method_name(mode))
            self.assertIn(mode.replace("-", "_"), run_suffix(mode))

    def test_sentinel_is_small_and_hof_free(self):
        for mode in MODES:
            command = campaign_command("Qwen-test", "rpa", mode, "sentinel")
            joined = " ".join(command)
            self.assertIn("--tasks rpa", joined)
            self.assertEqual(command[command.index("--seed-start") + 1], "0")
            self.assertEqual(command[command.index("--seeds") + 1], "1")
            self.assertIn("--max-parallel 1 --cpus-per-run 4", joined)
            self.assertIn(f"--communication-mode {mode}", joined)
            self.assertIn("--withhold-initial-hof", joined)
            self.assertIn("--candidate-validation-policy independent-members", joined)
            self.assertIn("--generation-backend harness", joined)
            self.assertIn("--forbidden-topology-m 0", joined)

    def test_main_arms_fill_64_cpu_slots_together(self):
        for mode in MODES:
            command = campaign_command("Qwen-test", "logic", mode, "main")
            joined = " ".join(command)
            self.assertEqual(command[command.index("--seed-start") + 1], "1")
            self.assertEqual(command[command.index("--seeds") + 1], "19")
            self.assertIn("--max-parallel 4 --cpus-per-run 4", joined)
            self.assertIn("--epochs 100", joined)
            self.assertIn("--rl-batch-size 1023", joined)
            self.assertIn("--total-candidate-budget 102400", joined)
            self.assertIn("--llm-candidates 10 --llm-every 20", joined)
            self.assertIn("--global-llm-concurrency 8", joined)

    def test_audit_matches_each_phase(self):
        for mode in MODES:
            sentinel = " ".join(
                audit_command("Qwen-test", "rpa", mode, "sentinel")
            )
            main = " ".join(audit_command("Qwen-test", "rpa", mode, "main"))
            self.assertIn("--seed-start 0 --seeds 1", sentinel)
            self.assertIn("--seed-start 1 --seeds 19", main)
            self.assertIn(f"--communication-mode {mode}", sentinel)
            self.assertIn("--model Qwen-test", main)


if __name__ == "__main__":
    unittest.main()
