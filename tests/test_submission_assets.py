import unittest

from paper.iclr2027_genai_net_llm.generate_submission_assets import (
    summarize_provenance,
)


class SubmissionAssetTests(unittest.TestCase):
    def test_rank_one_provenance_uses_zero_indexed_hof_rank(self):
        rows = [
            {
                "model": "model",
                "task": "rpa",
                "arm": "full_duplex",
                "seed": 0,
                "rank": 0,
                "qualified": True,
                "provenance": "direct_llm",
            },
            {
                "model": "model",
                "task": "rpa",
                "arm": "full_duplex",
                "seed": 0,
                "rank": 1,
                "qualified": True,
                "provenance": "rl_native",
            },
        ]

        summary = summarize_provenance(rows)
        rank_one = next(row for row in summary if row["scope"] == "rank_one")

        self.assertEqual(rank_one["n"], 1)
        self.assertEqual(rank_one["direct_llm"], 1)
        self.assertEqual(rank_one["rl_native"], 0)


if __name__ == "__main__":
    unittest.main()
