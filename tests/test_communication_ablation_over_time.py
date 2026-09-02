import unittest

from comparisons.rpa_search.scripts.plot_communication_ablation_over_time import (
    _deduplicate,
    _through,
    assert_no_preterminal_llm_leakage,
    elite_metrics,
)


def candidate(identifier, topology, loss, reactions):
    return {
        "identifier": (topology, identifier),
        "topology": topology,
        "loss": loss,
        "reaction_ids": frozenset(reactions),
    }


class CommunicationAblationTests(unittest.TestCase):
    def test_deduplicate_keeps_best_parameterization_record(self):
        rows = [
            candidate("p", "a", 2.0, [1]),
            candidate("p", "a", 1.0, [1]),
        ]
        result = _deduplicate(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["loss"], 1.0)

    def test_elite_metrics_distinguish_parameter_and_topology_diversity(self):
        rows = [
            candidate("p1", "a", 1.0, [1, 2]),
            candidate("p2", "a", 1.1, [1, 2]),
            candidate("p3", "b", 1.2, [1, 3]),
        ]
        result = elite_metrics(rows, elite_size=3)

        self.assertEqual(result["best_loss"], 1.0)
        self.assertEqual(result["unique_topologies"], 2.0)
        self.assertGreater(result["effective_topologies"], 1.0)
        self.assertAlmostEqual(result["mean_pairwise_jaccard"], 2.0 / 3.0)

    def test_through_accumulates_source_candidates(self):
        first = candidate("p1", "a", 2.0, [1])
        second = candidate("p2", "b", 1.0, [2])

        self.assertEqual(_through({0: [first], 2: [second]}, 1), [first])
        self.assertEqual(_through({0: [first], 2: [second]}, 2), [first, second])

    def test_preterminal_llm_leakage_is_rejected(self):
        llm = candidate("p1", "a", 1.0, [1])

        with self.assertRaisesRegex(RuntimeError, "exact LLM candidates"):
            assert_no_preterminal_llm_leakage(
                {0: [llm], 300: [llm]},
                [(10, llm)],
            )

        assert_no_preterminal_llm_leakage(
            {0: [candidate("p2", "b", 2.0, [2])], 300: [llm]},
            [(10, llm)],
        )


if __name__ == "__main__":
    unittest.main()
