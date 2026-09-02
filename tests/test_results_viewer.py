import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from RL4CRN.utils.results_viewer import (
    CampaignProgressReader,
    _parse_crn_presentation,
    _reasoning_sections,
)


class ResultsViewerPresentationTests(unittest.TestCase):
    def test_campaign_progress_combines_completed_and_active_epochs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / "status.json"
            status.write_text(
                json.dumps(
                    {
                        "completed": [{"task": "logic", "seed": 0}],
                        "active": [{"task": "logic", "seed": 1, "pid": 12}],
                        "pending": [],
                        "failed": [],
                    }
                ),
                encoding="utf-8",
            )
            progress = root / "runs" / "trial" / "logic_full_seed1_test" / "progress.csv"
            progress.parent.mkdir(parents=True)
            progress.write_text(
                "step,elapsed_seconds,best_so_far_loss\n25,10.0,0.5\n",
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "campaigns": [
                            {
                                "label": "test",
                                "model": "model",
                                "tasks": ["logic"],
                                "seeds": 2,
                                "epochs": 100,
                                "status_path": str(status),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            snapshot = CampaignProgressReader(plan).snapshot()

            self.assertEqual(snapshot["totals"]["completed"], 1)
            self.assertEqual(snapshot["totals"]["active"], 1)
            self.assertAlmostEqual(snapshot["totals"]["progress_percent"], 62.5)

    def test_unactivated_conditional_campaign_is_not_counted_as_pending(self):
        with TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "campaigns": [
                            {
                                "label": "conditional",
                                "model": "model",
                                "tasks": ["logic", "rpa"],
                                "seeds": 5,
                                "epochs": 100,
                                "conditional": True,
                                "phase": "conditional",
                                "status_path": str(Path(directory) / "future/status.json"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            snapshot = CampaignProgressReader(plan).snapshot()

            self.assertEqual(snapshot["totals"]["runs"], 0)
            self.assertEqual(snapshot["totals"]["pending"], 0)
            self.assertEqual(snapshot["campaigns"][0]["potential_runs"], 10)

    def test_declared_task_subset_filters_a_mixed_legacy_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "campaign_manifest.json").write_text(
                json.dumps({"tasks": ["logic", "rpa"], "seeds": list(range(20))}),
                encoding="utf-8",
            )
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "completed": [{"task": "rpa", "seed": seed} for seed in range(20)],
                        "failed": [{"task": "logic", "seed": seed} for seed in range(20)],
                        "active": [],
                        "pending": [],
                    }
                ),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "campaigns": [
                            {
                                "label": "RPA only",
                                "model": "model",
                                "tasks": ["rpa"],
                                "seeds": 20,
                                "epochs": 100,
                                "status_path": str(root / "status.json"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            snapshot = CampaignProgressReader(plan).snapshot()

            self.assertEqual(snapshot["totals"]["runs"], 20)
            self.assertEqual(snapshot["totals"]["completed"], 20)
            self.assertEqual(snapshot["totals"]["failed"], 0)

    def test_reasoning_notes_are_split_by_model_request(self):
        sections = _reasoning_sections(
            "# Notes\n\n## Search approach\nTemplate\n\n"
            "## Call 0001 (DECIDER)\n- inspected HoF\n\n"
            "## Call 0002 (WRITER)\nSelected ten candidates\n\n"
            "## External evaluator outcome\n10 valid"
        )

        self.assertEqual([section["title"] for section in sections], [
            "Call 0001 (DECIDER)",
            "Call 0002 (WRITER)",
        ])
        self.assertIn("inspected HoF", sections[0]["content"])
        self.assertNotIn("External evaluator", sections[1]["content"])

    def test_crn_text_becomes_structured_equations_and_latex(self):
        presentation = _parse_crn_presentation(
            "\n".join(
                [
                    "Inputs: ['u_1']",
                    "Species: ['X_1', 'OUT']",
                    "Output Species: ['OUT']",
                    "∅ ----> X_1; [MAK(1.0, u_1)]",
                    "X_1 + X_1 ----> OUT; [MAK(0.25)]",
                ]
            ),
            [1, 7],
        )

        self.assertEqual(presentation["outputs"], ["OUT"])
        self.assertEqual(presentation["reactions"][0]["kind"], "template")
        self.assertEqual(presentation["reactions"][1]["kind"], "designed")
        self.assertEqual(
            presentation["reactions"][1]["reactants"],
            [{"species": "X_1", "coefficient": 2}],
        )
        self.assertIn(r"2\,\mathrm{X}_{1}", presentation["latex"])
        self.assertIn(r"\xrightarrow{k=0.25}", presentation["latex"])

    def test_malformed_headers_do_not_break_reaction_rendering(self):
        presentation = _parse_crn_presentation(
            "Species: not-python\nA ----> B; [unknown]", [9]
        )

        self.assertEqual(presentation["species"], [])
        self.assertEqual(presentation["reactions"][0]["reaction_id"], 9)
        self.assertIsNone(presentation["reactions"][0]["rate"])


if __name__ == "__main__":
    unittest.main()
