import ast
import unittest
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "comparisons/llm_crn_generation/run_mmc2_harness_hybrid.py"
)


class HybridRunnerConfigurationTests(unittest.TestCase):
    def test_cli_timeout_is_passed_to_harness_client(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HarnessLLMClient"
        ]

        self.assertEqual(len(calls), 1)
        timeout = next(
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "timeout_seconds"
        )
        self.assertEqual(ast.unparse(timeout), "args.llm_timeout")

    def test_no_communication_is_terminally_pooled_after_llm_drain(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("trainer.wait_for_llm_graph", calls)
        self.assertIn("trainer.merge_isolated_llm_candidates", calls)
        source = RUNNER.read_text(encoding="utf-8")
        self.assertLess(
            source.index("trainer.wait_for_llm_graph"),
            source.index("trainer.merge_isolated_llm_candidates"),
        )
        self.assertIn('choices=("full", "none")', source)

    def test_initial_hof_can_be_withheld_without_disabling_communication(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("--withhold-initial-hof", source)
        self.assertIn("withhold_initial_hof=args.withhold_initial_hof", source)

    def test_contract_v2_uses_two_stage_graph_member_validation_and_new_bounds(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("HarnessDeciderWriterCRNGraph", source)
        self.assertIn('default="independent-members"', source)
        self.assertIn("LLM_RATE_MIN = 0.001", source)
        self.assertIn("LLM_RATE_MAX = 100.0", source)
        self.assertIn('"llm_model_requests_per_round": 2 if', source)
        self.assertIn("two-stage-decider-writer/independent-members", source)
        self.assertIn("validate_experiment_release(", source)


if __name__ == "__main__":
    unittest.main()
