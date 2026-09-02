import json
import unittest

from paper.iclr2027_genai_net_llm.generate_communication_mechanism_analysis import (
    canonical_parameterization_identifier,
    jaccard_distance,
    mean_pairwise_distance,
    qualified_topology_structures,
    structural_record_token,
)


class CommunicationMechanismAnalysisTests(unittest.TestCase):
    def test_parameterization_identifier_ignores_serialization_order(self):
        first = (
            "topology",
            json.dumps(
                [
                    {"index": 0, "reaction_id": 7, "parameters": [2.0]},
                    {"index": 1, "reaction_id": 3, "parameters": [1.0]},
                ]
            ),
        )
        permuted = (
            "topology",
            json.dumps(
                [
                    {"index": 0, "reaction_id": 3, "parameters": [1.0]},
                    {"index": 1, "reaction_id": 7, "parameters": [2.0]},
                ]
            ),
        )

        self.assertEqual(
            canonical_parameterization_identifier(first),
            canonical_parameterization_identifier(permuted),
        )

    @staticmethod
    def _record(reaction_id, reactants, products, inputs):
        return structural_record_token(
            {
                "reaction_id": reaction_id,
                "structure": {
                    "type": "MassAction",
                    "reactants": reactants,
                    "products": products,
                    "inputs": inputs,
                },
            }
        )

    def test_qualified_structures_remove_exact_fixed_records(self):
        fixed_production = self._record(1, [], ["X_1"], ["u_1"])
        fixed_degradation = self._record(28, ["X_3"], [], ["u_2"])
        common = self._record(9, ["X_1"], ["X_2"], [None])
        first = self._record(3, ["X_2"], ["X_3"], [None])
        second = self._record(7, ["X_3"], ["X_1"], [None])
        slow = self._record(5, ["X_1"], [], [None])
        fixed = (fixed_production, fixed_degradation)
        candidates = [
            {"topology": "a", "loss": 0.1},
            {"topology": "b", "loss": 0.2},
            {"topology": "too-slow", "loss": 0.5},
        ]
        topology_records = {
            "a": (*fixed, first, common),
            "b": (*fixed, second, common),
            "too-slow": (*fixed, slow, common),
        }

        structures = qualified_topology_structures(
            candidates,
            threshold=0.3,
            topology_records=topology_records,
            fixed_records=fixed,
        )

        self.assertEqual(
            structures,
            {"a": frozenset({first, common}), "b": frozenset({second, common})},
        )
        self.assertAlmostEqual(mean_pairwise_distance(structures.values()), 2 / 3)

    def test_fixed_modulated_and_added_unmodulated_reactions_can_share_id(self):
        fixed_modulated = self._record(28, ["X_3"], [], ["u_2"])
        added_unmodulated = self._record(28, ["X_3"], [], [None])
        candidates = [{"topology": "collision", "loss": 0.1}]

        structures = qualified_topology_structures(
            candidates,
            threshold=0.2,
            topology_records={
                "collision": (fixed_modulated, added_unmodulated)
            },
            fixed_records=(fixed_modulated,),
        )

        self.assertEqual(structures["collision"], frozenset({added_unmodulated}))

    def test_exact_duplicate_of_fixed_record_survives_multiset_subtraction(self):
        fixed = self._record(1, [], ["X_1"], [None])

        structures = qualified_topology_structures(
            [{"topology": "duplicate", "loss": 0.1}],
            threshold=0.2,
            topology_records={"duplicate": (fixed, fixed)},
            fixed_records=(fixed,),
        )

        self.assertEqual(structures["duplicate"], frozenset({fixed}))

    def test_jaccard_distance_handles_identical_and_empty_sets(self):
        self.assertEqual(jaccard_distance(frozenset(), frozenset()), 0.0)
        self.assertEqual(jaccard_distance(frozenset({1}), frozenset({1})), 0.0)

if __name__ == "__main__":
    unittest.main()
