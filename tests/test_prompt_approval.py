import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from comparisons.llm_crn_generation.generate_contract_v2_prompt_review import (
    current_review_records,
)
from comparisons.llm_crn_generation.prompt_approval import validate_prompt_approval


class PromptApprovalTests(unittest.TestCase):
    def test_generated_review_matches_runtime_prompt_sources(self):
        review = Path(
            "paper/iclr2027_genai_net_llm/generated/CONTRACT_V2_PROMPT_REVIEW.json"
        )
        self.assertEqual(
            json.loads(review.read_text(encoding="utf-8")),
            current_review_records(),
        )

    def test_missing_approval_fails_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review.json"
            review.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "pending author prompt approval"):
                validate_prompt_approval(root / "approval.json", review_path=review)

    def test_approval_must_match_current_review_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review.json"
            approval = root / "approval.json"
            review.write_text("{}\n", encoding="utf-8")
            approval.write_text(
                json.dumps(
                    {
                        "approval_status": "approved",
                        "prompt_review_sha256": "stale",
                        "approved_by": "author",
                        "approved_at": "2026-09-02T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_prompt_approval(approval, review_path=review)

            approval_payload = json.loads(approval.read_text(encoding="utf-8"))
            approval_payload["prompt_review_sha256"] = hashlib.sha256(
                review.read_bytes()
            ).hexdigest()
            approval.write_text(json.dumps(approval_payload), encoding="utf-8")

            result = validate_prompt_approval(approval, review_path=review)
            self.assertEqual(result["approval_status"], "approved")
            self.assertTrue(result["approval_file_sha256"])


if __name__ == "__main__":
    unittest.main()
