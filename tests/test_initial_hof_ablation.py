import unittest

from comparisons.rpa_search.scripts.analyze_initial_hof_ablation import _source_counts


class InitialHallOfFameAblationTests(unittest.TestCase):
    def test_source_counts_use_exact_topology_and_parameters(self):
        snapshot = [
            {"identifier": ("a", "p1")},
            {"identifier": ("a", "p2")},
            {"identifier": ("b", "p3")},
        ]

        exact_llm, rl_origin = _source_counts(snapshot, {("a", "p1")})

        self.assertEqual(exact_llm, 1)
        self.assertEqual(rl_origin, 2)


if __name__ == "__main__":
    unittest.main()
