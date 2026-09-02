import json
import tempfile
import unittest
from pathlib import Path

from comparisons.llm_crn_generation.assess_breadth_consistency import assess_task
from comparisons.llm_crn_generation.paper_breadth_tasks import (
    DETERMINISTIC_TASKS,
    biphasic_target,
    hill_target,
    ultrasensitive_target,
)
from comparisons.rpa_search.src.common.config import load_config


class BreadthCampaignTests(unittest.TestCase):
    def test_frozen_dose_targets_have_expected_shapes(self):
        self.assertEqual(hill_target(0.0), 0.0)
        self.assertLess(ultrasensitive_target(0.25), ultrasensitive_target(0.75))
        self.assertGreater(biphasic_target(0.4), biphasic_target(1.0))

    def test_rl_only_baseline_uses_frozen_breadth_configuration(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "comparisons/llm_crn_generation/configs/paper_breadth_100epoch.json"
        )

        self.assertEqual(len(DETERMINISTIC_TASKS), 6)
        self.assertEqual(config["search"]["max_added_reactions"], 6)
        self.assertEqual(config["classifier"]["rtol"], 1e-6)
        self.assertEqual(config["rl4crn"]["policy_width"], 256)

    def test_dispersion_rule_extends_inconsistent_five_seed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method = "method"
            suffix = "suffix"
            for seed, loss in enumerate([1.0, 1.1, 0.9, 1.05, 100.0]):
                path = root / method / f"dose_hill_full102400_seed{seed}_{suffix}"
                path.mkdir(parents=True)
                (path / "completed.json").write_text(
                    json.dumps({"best_loss": loss}), encoding="utf-8"
                )

            result = assess_task(root, "dose_hill", method, suffix, 102400)

            self.assertFalse(result["consistent"])
            self.assertTrue(result["extend_to_ten_seeds"])


if __name__ == "__main__":
    unittest.main()
