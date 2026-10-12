import unittest

from comparisons.llm_crn_generation.audit_contract_v2_sentinel import (
    classify_llm_failure,
)


class ContractV2AuditFailurePolicyTests(unittest.TestCase):
    def test_model_output_failures_are_observed_outcomes(self):
        messages = (
            "DeepSeek Harness returned an empty response.",
            "Harness returned invalid JSON: Expecting value: line 1 column 1 (char 0)",
            "Harness returned no independently valid CRN candidates.",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(classify_llm_failure(message), "model_outcome")

    def test_infrastructure_failures_remain_blocking(self):
        messages = (
            "[Errno 28] No space left on device",
            "DeepSeek Harness exceeded the 3600s timeout.",
            "DeepSeek Harness failed; inspect stderr.txt (exit code 1).",
            "Connection refused",
        )
        for message in messages:
            with self.subTest(message=message):
                self.assertEqual(classify_llm_failure(message), "infrastructure")


if __name__ == "__main__":
    unittest.main()
