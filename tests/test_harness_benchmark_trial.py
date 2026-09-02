import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from RL4CRN.llm.benchmark_prompts import (
    get_mmc2_task_prompt,
    get_mmc2_task_prompt_variant,
    get_reported_mmc2_task_prompt_2026,
)
from comparisons.llm_crn_generation.run_mmc2_harness_trial import (
    append_csv,
    completed_steps,
    parse_args,
)


class HarnessBenchmarkTrialTests(unittest.TestCase):
    def test_prompt_uses_requested_cvode_solver(self):
        prompt = get_mmc2_task_prompt("logic", solver="CVODE")

        self.assertIn("Simulation: CVODE", prompt)
        self.assertNotIn("Simulation: LSODA", prompt)
        self.assertIn("transient-weighted mean absolute error", prompt)
        self.assertNotIn("binary cross-entropy", prompt)
        self.assertIn("[0.001, 100]", prompt)

    def test_reported_logic_prompt_is_frozen_for_reproduction(self):
        prompt = get_reported_mmc2_task_prompt_2026("logic", solver="CVODE")

        self.assertIn("binary cross-entropy", prompt)
        self.assertNotIn("transient-weighted mean absolute error", prompt)
        self.assertEqual(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "b4bf9c2ca99d04da7393e760ac279409adc37358caf530cb629b5c9a89b26719",
        )

    def test_logic_trajectory_prompt_matches_executable_contract(self):
        prompt = get_mmc2_task_prompt_variant(
            "logic", variant="logic-trajectory", solver="CVODE"
        )

        required = (
            "four binary signals u_1, u_2, u_3, u_4",
            "X_1, X_2, X_3, X_4 and output species OUT",
            "(u_1 AND u_2) OR (u_2 AND u_3) OR (u_3 AND u_4)",
            "all 16 vectors in {0, 1}^4",
            "CVODE over t = 0 to 100 with 1000 time points",
            "rtol = atol = 1e-8",
            "output-trajectory L1 error over all 16 rows",
            "0.25 for the first 20%",
            "1.0 for the middle 60%",
            "2.0 for the final 20%",
            "exactly five distinct reactions",
            "null emptyset-to-emptyset reaction is inadmissible",
            "one scalar rate constant",
            "[0.001, 100]",
            "Transient deviations are also charged",
            "later time points weighted most strongly",
        )
        for text in required:
            self.assertIn(text, prompt)
        self.assertNotIn("binary cross-entropy", prompt)
        self.assertEqual(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "6974c971ca0265e8e93bd694748bb9bad5b87ab536118b18b098f3b1a446fd39",
        )

    def test_logic_trajectory_variant_rejects_other_tasks(self):
        with self.assertRaisesRegex(ValueError, "valid only for task='logic'"):
            get_mmc2_task_prompt_variant("rpa", variant="logic-trajectory")

    def test_trial_defaults_match_pilot_protocol(self):
        args = parse_args([])

        self.assertEqual(args.runs, 3)
        self.assertEqual(args.proposals, 10)
        self.assertEqual(args.solver, "CVODE")

    def test_progress_rows_are_restart_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "progress.csv"
            append_csv(
                path,
                {
                    "trial_id": "trial",
                    "task": "rpa",
                    "run_index": 2,
                    "proposal_step": 4,
                    "valid": True,
                    "loss": 0.1,
                    "best_so_far_loss": 0.1,
                    "message": "valid",
                    "workspace": "/tmp/run",
                    "duration_seconds": 1.0,
                },
            )

            self.assertEqual(completed_steps(path), {("rpa", 2, 4)})
            with path.open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
