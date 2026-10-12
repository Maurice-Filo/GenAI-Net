import os
import subprocess
import sys
import unittest
from pathlib import Path

from comparisons.llm_crn_generation.run_local_qwen38_contract_v2_rpa_no_communication_campaign import (
    CAMPAIGN_ID,
    METHOD,
    RUN_SUFFIX,
    audit_command,
    campaign_command,
)


class LocalQwen38RpaNoCommunicationCampaignTests(unittest.TestCase):
    def test_entrypoint_bootstraps_repository_imports(self):
        root = Path(__file__).resolve().parents[1]
        scripts = (
            root
            / "comparisons/llm_crn_generation/"
            "run_local_qwen38_contract_v2_rpa_no_communication_campaign.py",
            root
            / "comparisons/llm_crn_generation/"
            "run_local_qwen38_contract_v2_logic_no_communication_campaign.py",
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ""
        for script in scripts:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import runpy; runpy.run_path({str(script)!r}, run_name='entrypoint_test')",
                ],
                cwd="/tmp",
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_campaign_differs_from_full_duplex_only_where_intended(self):
        command = campaign_command("Qwen3.8-test")
        joined = " ".join(command)
        expected = (
            "--tasks rpa",
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
        self.assertNotIn("--tasks rpa logic", joined)

    def test_namespaces_and_audit_are_isolated(self):
        command = campaign_command("Qwen3.8-test")
        audit = audit_command("Qwen3.8-test")
        self.assertIn("no-communication", CAMPAIGN_ID)
        self.assertIn("no_communication", METHOD)
        self.assertIn("no_communication", RUN_SUFFIX)
        self.assertEqual(command[command.index("--method-name") + 1], METHOD)
        self.assertEqual(command[command.index("--run-suffix") + 1], RUN_SUFFIX)
        self.assertEqual(audit[audit.index("--communication-mode") + 1], "none")
        self.assertEqual(audit[audit.index("--model") + 1], "Qwen3.8-test")


if __name__ == "__main__":
    unittest.main()
