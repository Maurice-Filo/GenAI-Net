import json
import unittest

from RL4CRN.llm.benchmark_prompts import get_mmc2_task_prompt
from comparisons.llm_crn_generation.run_mmc2_harness_smoke import CONFIGS
from comparisons.llm_crn_generation.run_parity4_communication_300epoch_queue import (
    BUDGET,
    EPOCHS,
    MODES,
    SEEDS,
    audit_command,
    campaign_command,
    campaign_id,
    validate_task_definition,
)
from comparisons.rpa_search.src.common.logic_task import build_logic_fn


class ParityLogicCampaignTests(unittest.TestCase):
    def test_parity_truth_table_is_balanced_and_every_input_is_essential(self):
        parity = build_logic_fn("parity_4")
        outputs = []
        for value in range(16):
            inputs = [(value >> index) & 1 for index in range(4)]
            output = parity(inputs)
            outputs.append(output)
            self.assertEqual(output, bool(sum(inputs) % 2))
            for index in range(4):
                flipped = list(inputs)
                flipped[index] ^= 1
                self.assertNotEqual(output, parity(flipped))
        self.assertEqual(sum(outputs), 8)

    def test_configuration_preserves_logic_numerics_with_new_structure(self):
        config = json.loads(CONFIGS["logic_parity"].read_text(encoding="utf-8"))
        logic = config["logic_parity"]
        self.assertEqual(logic["n_inputs"], 4)
        self.assertEqual(logic["n_support_species"], 2)
        self.assertFalse(logic["include_dilution"])
        self.assertEqual(logic["solver"], "CVODE")
        self.assertEqual(logic["t_f"], 100.0)
        self.assertEqual(logic["n_t"], 1000)
        self.assertEqual(logic["ic"], ["constant", 0.01])
        self.assertEqual(logic["weights"], "transient")
        self.assertEqual(config["search"]["max_added_reactions"], 8)
        self.assertEqual(config["rl4crn"]["epochs"], 300)

    def test_prompt_is_standalone_and_balances_exploitation_with_exploration(self):
        prompt = get_mmc2_task_prompt("logic_parity", solver="CVODE")
        self.assertIn("u_1 XOR u_2 XOR u_3 XOR u_4", prompt)
        self.assertIn("exactly eight distinct reactions", prompt)
        self.assertIn("transient-weighted mean absolute error", prompt)
        self.assertIn("Use the received Hall of Fame actively", prompt)
        self.assertIn("Do not restrict proposals to Hall-of-Fame derivatives", prompt)
        self.assertNotIn("existing Logic", prompt)

    def test_four_arm_campaign_grid_is_frozen(self):
        self.assertEqual(MODES, ("full", "llm-to-rl", "rl-to-llm", "none"))
        self.assertEqual(SEEDS, 20)
        self.assertEqual(EPOCHS, 300)
        self.assertEqual(BUDGET, 307200)
        self.assertEqual(len({campaign_id(mode) for mode in MODES}), 4)
        for mode in MODES:
            command = campaign_command(mode)
            joined = " ".join(command)
            self.assertIn("--tasks logic_parity", joined)
            self.assertIn("--seeds 20 --seed-start 0", joined)
            self.assertIn("--epochs 300", joined)
            self.assertIn("--communication-mode " + mode, joined)
            self.assertIn("--withhold-initial-hof", command)
            audit = audit_command(mode)
            self.assertEqual(audit[audit.index("--epochs") + 1], "300")
            self.assertEqual(audit[audit.index("--communication-mode") + 1], mode)

    def test_definition_preflight_builds_the_eight_reaction_evaluator(self):
        report = validate_task_definition()
        self.assertEqual(report["reaction_budget"], 8)
        self.assertEqual(
            set(report["species"]),
            {"X_1", "X_2", "X_3", "X_4", "S_1", "S_2", "OUT"},
        )
        self.assertEqual(len(report["truth_table"]), 16)
        self.assertEqual(sum(row["target"] for row in report["truth_table"]), 8)


if __name__ == "__main__":
    unittest.main()
