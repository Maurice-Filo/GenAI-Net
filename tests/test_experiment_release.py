import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from comparisons.llm_crn_generation.experiment_release import (
    DEFAULT_PROMPT_REVIEW,
    DETERMINISTIC_TASKS,
    validate_analysis_plan,
    validate_experiment_release,
)
from comparisons.llm_crn_generation.prompt_approval import file_sha256


class ExperimentReleaseTests(unittest.TestCase):
    def test_draft_analysis_plan_cannot_release_experiments(self):
        draft = (
            Path(__file__).parents[1]
            / "paper/iclr2027_genai_net_llm/generated/analysis_plan_v2.DRAFT.json"
        )
        with self.assertRaisesRegex(RuntimeError, "exactly 'frozen'"):
            validate_analysis_plan(draft)

    def test_sentinel_and_paper_release_require_distinct_gates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            approval = root / "approval.json"
            analysis = root / "analysis.json"
            preflight = root / "preflight.json"
            sentinel = root / "sentinel.json"
            approval.write_text(
                json.dumps(
                    {
                        "approval_status": "approved",
                        "prompt_review_sha256": file_sha256(DEFAULT_PROMPT_REVIEW),
                        "approved_by": "author",
                        "approved_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            analysis.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "quality_thresholds": {
                            task: float(index + 1)
                            for index, task in enumerate(sorted(DETERMINISTIC_TASKS))
                        },
                        "frozen_by": "author",
                        "frozen_at": datetime.now(timezone.utc).isoformat(),
                        "failed_run_policy": "report all failures; no silent replacement",
                        "tie_policy": "absolute difference <= 1e-12",
                    }
                ),
                encoding="utf-8",
            )
            preflight.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "tasks": [{} for _ in range(8)],
                        "model_calls_made": 0,
                    }
                ),
                encoding="utf-8",
            )

            released = validate_experiment_release(
                stage="sentinel",
                prompt_approval_path=approval,
                analysis_plan_path=analysis,
                static_preflight_path=preflight,
                sentinel_report_path=sentinel,
            )
            self.assertEqual(released["stage"], "sentinel")

            with self.assertRaisesRegex(RuntimeError, "sentinel report is missing"):
                validate_experiment_release(
                    stage="paper",
                    prompt_approval_path=approval,
                    analysis_plan_path=analysis,
                    static_preflight_path=preflight,
                    sentinel_report_path=sentinel,
                )

            sentinel.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "prompt_review_sha256": file_sha256(DEFAULT_PROMPT_REVIEW),
                        "analysis_plan_sha256": file_sha256(analysis),
                    }
                ),
                encoding="utf-8",
            )
            released = validate_experiment_release(
                stage="paper",
                prompt_approval_path=approval,
                analysis_plan_path=analysis,
                static_preflight_path=preflight,
                sentinel_report_path=sentinel,
            )
            self.assertEqual(released["stage"], "paper")


if __name__ == "__main__":
    unittest.main()
