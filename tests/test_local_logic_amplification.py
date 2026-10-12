import unittest

from paper.iclr2027_genai_net_llm.generate_local_logic_amplification import (
    available_proposals, contributions, pooled_portfolio, step_values,
)


def candidate(topology, loss, provenance="direct_llm", **extra):
    return {"topology_hash": topology, "loss": loss, "provenance_class": provenance, **extra}


class LocalLogicAmplificationTests(unittest.TestCase):
    def test_late_llm_reply_is_not_counted_at_launch(self):
        rows = [candidate("a", .01, launched_epoch=0, created_at=40)]
        self.assertEqual(available_proposals(rows, 39), [])
        self.assertEqual(available_proposals(rows, 40), rows)

    def test_pool_counts_rate_variants_once_and_respects_capacity(self):
        hof = [candidate("a", .08), candidate("b", .03)]
        llm = [candidate("a", .01), candidate("c", .02)]
        self.assertEqual([row["topology_hash"] for row in pooled_portfolio(hof, llm, 2)], ["a", "c"])

    def test_refinement_credit_requires_new_success(self):
        refined = "rl_parameter_refinement_of_llm_topology"
        llm = [candidate("already_good", .02), candidate("rescued", .08)]
        pool = [candidate("already_good", .01, refined), candidate("rescued", .01, refined),
                candidate("novel", .01, "rl_native_topology")]
        counts = contributions(pool, llm, .04)
        self.assertEqual(counts, {"llm_sufficient": 1, "rl_rescued": 1, "rl_additional": 1, "rl_shared": 0})

    def test_shared_topology_without_exposure_is_not_called_refinement(self):
        llm = [candidate("shared", .08)]
        pool = [candidate("shared", .02, "rl_native_topology")]
        counts = contributions(pool, llm, .04)
        self.assertEqual(counts["rl_rescued"], 0)
        self.assertEqual(counts["rl_shared"], 1)

    def test_parameter_variant_that_still_fails_is_not_a_success(self):
        pool = [candidate("a", .05, "rl_parameter_refinement_of_llm_topology")]
        self.assertEqual(sum(contributions(pool, [candidate("a", .08)], .04).values()), 0)

    def test_budget_interpolation_does_not_use_future_results(self):
        rows = [{"evaluations": 10, "completed_epochs": 1, "qualified": 0},
                {"evaluations": 20, "completed_epochs": 2, "qualified": 3}]
        self.assertEqual(step_values(rows, [10, 19, 20]).tolist(), [0, 0, 3])
        with self.assertRaises(ValueError):
            step_values(rows, [9, 20])
        with self.assertRaises(ValueError):
            step_values(rows, [10, 21])


if __name__ == "__main__":
    unittest.main()
