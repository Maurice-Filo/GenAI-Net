import unittest

from paper.iclr2027_genai_net_llm.analyze_llm_proposal_diversity import proposal_metrics


def proposal(topology, loss, valid=True, structure="[]"):
    return {"topology_hash": topology, "loss": loss, "valid": valid,
            "structure_json": structure}


class ProposalDiversityTests(unittest.TestCase):
    def test_rate_variants_count_once_using_best_loss(self):
        metrics, unique = proposal_metrics([
            proposal("a", .2), proposal("a", .01), proposal("b", .3)], .1)
        self.assertEqual(metrics["valid_proposals"], 3)
        self.assertEqual(metrics["unique_topologies"], 2)
        self.assertEqual(metrics["qualified_topologies"], 1)
        self.assertAlmostEqual(metrics["repeat_fraction"], 1 / 3)
        self.assertEqual(unique[0]["loss"], .01)

    def test_invalid_and_nonfinite_results_do_not_invent_topologies(self):
        metrics, unique = proposal_metrics([
            proposal("a", .01, valid=False), proposal("b", None),
            proposal("c", float("nan")), proposal("d", float("inf"))], .1)
        self.assertEqual(metrics["unique_topologies"], 0)
        self.assertEqual(metrics["qualified_topologies"], 0)
        self.assertIsNone(metrics["repeat_fraction"])
        self.assertEqual(unique, [])

    def test_threshold_is_strict(self):
        metrics, _ = proposal_metrics([proposal("a", .1)], .1)
        self.assertEqual(metrics["qualified_topologies"], 0)

    def test_missing_structure_or_hash_fails_loudly(self):
        for row in (proposal("a", .01, structure=None), proposal(None, .01)):
            with self.assertRaises(RuntimeError):
                proposal_metrics([row], .1)

    def test_no_archive_capacity_is_applied(self):
        metrics, _ = proposal_metrics([proposal(str(i), .01) for i in range(50)], .1)
        self.assertEqual(metrics["qualified_topologies"], 50)


if __name__ == "__main__":
    unittest.main()
