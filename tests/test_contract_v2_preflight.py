import json
import unittest
from pathlib import Path

from paper.iclr2027_genai_net_llm.audit_contract_v2_readiness import audit_readiness


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/contract_v2_preflight/preflight_report.json"
)
REGISTRY = (
    ROOT
    / "paper/iclr2027_genai_net_llm/generated/historical_request0_hof_registry.json"
)


class ContractV2PreflightTests(unittest.TestCase):
    def test_manuscript_readiness_fails_closed_before_approval_and_results(self):
        readiness = audit_readiness()

        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(readiness["static_preflight_status"], "pass")
        self.assertTrue(any("prompt approval" in error for error in readiness["errors"]))
        self.assertTrue(any("analysis plan" in error.lower() for error in readiness["errors"]))
        self.assertTrue(any("primary registry" in error for error in readiness["errors"]))

    def test_all_deterministic_task_contracts_pass_static_mask_audit(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["model_calls_made"], 0)
        self.assertEqual(report["author_prompt_approval"], "pending")
        self.assertEqual(len(report["tasks"]), 8)
        for task in report["tasks"]:
            self.assertEqual(task["status"], "pass", task)
            self.assertFalse(task["negative_test"]["valid"])
            self.assertIn("forbidden", task["negative_test"]["message"].lower())
            self.assertEqual(task["rate_bounds"], [0.001, 100.0])
            self.assertEqual(task["candidate_validation_policy"], "independent-members")

    def test_historical_request_zero_registry_is_protocol_selection_only(self):
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

        self.assertEqual(registry["evidence_scope"], "protocol-selection-only")
        self.assertIn("quarantine", Path(registry["quarantine_root"]).parts)
        self.assertIn("fixed input template", registry["mandatory_disclosure"])
        self.assertIn("endpoint performance", registry["forbidden_uses"])


if __name__ == "__main__":
    unittest.main()
