import json
import tempfile
import unittest
from pathlib import Path

from comparisons.llm_crn_generation.quarantine_pre_mask_runs import (
    apply_inventory,
    build_inventory,
    task_from_name,
)


class QuarantinePreMaskRunsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.campaigns = root / "campaigns"
        self.raw = root / "raw"
        self.quarantine = root / "quarantine"
        self.campaigns.mkdir()
        self.raw.mkdir()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _trial(self, campaign: Path, task: str, seed: int) -> Path:
        trial = campaign / "runs" / f"campaign-{task}-seed{seed}"
        run = trial / f"{task}_full102400_seed{seed}_cvode_llm"
        run.mkdir(parents=True)
        (run / "results.sqlite").write_bytes(b"database")
        return trial

    def test_task_names_handle_stochastic_rpa_before_rpa(self):
        self.assertEqual(task_from_name("stochastic_rpa_full1_seed0"), "stochastic_rpa")
        self.assertEqual(task_from_name("campaign-rpa-seed0"), "rpa")

    def test_inventory_splits_no_template_tasks_and_retains_controls(self):
        mixed = self.campaigns / "mixed"
        self._trial(mixed, "classifier", 0)
        self._trial(mixed, "rpa", 0)
        self._write_json(mixed / "campaign_manifest.json", {"tasks": ["classifier", "rpa"]})
        self._write_json(
            mixed / "status.json",
            {"active": [], "completed": [{"task": "classifier"}, {"task": "rpa"}]},
        )
        control = self.campaigns / "rl-only-control"
        control.mkdir()
        raw_method = self.raw / "genai_net_llm_mixed"
        (raw_method / "classifier_full102400_seed0").mkdir(parents=True)
        (raw_method / "rpa_full102400_seed0").mkdir()

        inventory, split_campaigns, split_methods = build_inventory(
            campaign_root=self.campaigns,
            raw_root=self.raw,
            quarantine_root=self.quarantine,
            include_derived=False,
        )

        sources = {Path(record["source"]).name for record in inventory["moves"]}
        self.assertIn("campaign-rpa-seed0", sources)
        self.assertIn("rpa_full102400_seed0", sources)
        self.assertNotIn("campaign-classifier-seed0", sources)
        self.assertNotIn("rl-only-control", sources)

        apply_inventory(
            inventory,
            quarantine_root=self.quarantine,
            split_campaigns=split_campaigns,
            split_methods=split_methods,
        )
        self.assertTrue((mixed / "runs/campaign-classifier-seed0").is_dir())
        self.assertFalse((mixed / "runs/campaign-rpa-seed0").exists())
        self.assertEqual(
            json.loads((mixed / "campaign_manifest.json").read_text())["tasks"],
            ["classifier"],
        )
        self.assertTrue((self.quarantine / "manifest.json").is_file())
        self.assertEqual(
            json.loads((self.quarantine / "manifest.json").read_text())["status"],
            "complete",
        )


if __name__ == "__main__":
    unittest.main()
