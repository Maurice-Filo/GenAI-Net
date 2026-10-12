import unittest

from paper.iclr2027_genai_net_llm.analyze_contract_v2_primary import control_loss


class ContractV2PrimaryAnalysisTests(unittest.TestCase):
    def test_control_loss_uses_the_value_frozen_by_the_audit(self):
        plan = {
            "control_audit": {
                "rpa": {
                    "controls": [
                        {
                            "seed": 7,
                            "endpoint_loss": 0.0125,
                            "artifact_path": "intentionally/absent",
                        }
                    ]
                }
            }
        }

        self.assertEqual(control_loss(plan, "rpa", 7), 0.0125)


if __name__ == "__main__":
    unittest.main()
