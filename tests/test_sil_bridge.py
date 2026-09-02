import unittest

from RL4CRN.llm.sil_bridge import _deep_plain_copy, _make_audit_candidate


class _Reaction:
    def __init__(self, num_parameters):
        self.num_parameters = num_parameters


class _Library:
    def __init__(self):
        self.reactions = [_Reaction(1), _Reaction(2), _Reaction(1), _Reaction(3)]

    def __len__(self):
        return len(self.reactions)

    def get_reaction(self, index):
        return self.reactions[index]


class SILBridgeTests(unittest.TestCase):
    def test_audit_candidate_respects_forbidden_ids_and_parameter_arities(self):
        evaluator = type(
            "Evaluator",
            (),
            {
                "library": _Library(),
                "forbidden_reaction_ids": frozenset({1}),
                "max_added_reactions": 2,
            },
        )()

        candidate = _make_audit_candidate(
            evaluator,
            initially_masked=[False, False, True, False],
        )

        self.assertEqual(candidate.reaction_ids, [0, 3])
        self.assertEqual(candidate.parameter_values, [[1.0], [1.0, 1.0, 1.0]])

    def test_plain_copy_does_not_mutate_nested_config(self):
        original = {"logic": {"solver": "LSODA"}, "values": [1, {"x": 2}]}
        copied = _deep_plain_copy(original)
        copied["logic"]["solver"] = "CVODE"

        self.assertEqual(original["logic"]["solver"], "LSODA")


if __name__ == "__main__":
    unittest.main()
