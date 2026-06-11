"""Compute the brute-force workload needed to make m<=5 exact."""

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "maxrpa_ns_sweep.csv"
OUT_PATH = ROOT / "exact_up_to_5_cost.csv"


def n_complexes(n: int) -> int:
    return 1 + 2 * n + math.comb(n, 2)


def complexes(n: int):
    out = []

    def rec(pos, remaining, cur):
        if pos == n:
            out.append(tuple(cur))
            return
        for value in range(remaining + 1):
            cur[pos] = value
            rec(pos + 1, remaining - value, cur)

    rec(0, 2, [0] * n)
    return sorted(out)


def reactant_pattern_candidate_count(n: int, m: int) -> int:
    """Count m-reaction CRNs containing at least one special reactant pair.

    This is an exact combinatorial prefilter count.  It ignores q-feasibility,
    but it counts only CRNs where some two reactions differ in the X1 reactant
    count and share all other reactant counts.
    """

    comps = complexes(n)
    c = len(comps)
    by_nonoutput = defaultdict(lambda: defaultdict(int))
    for alpha in comps:
        by_nonoutput[alpha[1:]][alpha[0]] += c - 1

    bad = [0] * (m + 1)
    good = [0] * (m + 1)
    bad[0] = 1

    for bins in by_nonoutput.values():
        sizes = list(bins.values())
        total = sum(sizes)
        ways_any = [math.comb(total, k) for k in range(m + 1)]
        ways_bad = [0] * (m + 1)
        ways_bad[0] = 1
        for k in range(1, m + 1):
            ways_bad[k] = sum(math.comb(size, k) for size in sizes if size >= k)
        ways_good = [ways_any[k] - ways_bad[k] for k in range(m + 1)]

        next_bad = [0] * (m + 1)
        next_good = [0] * (m + 1)
        for old in range(m + 1):
            for k in range(m - old + 1):
                next_bad[old + k] += bad[old] * ways_bad[k]
                next_good[old + k] += good[old] * ways_any[k] + bad[old] * ways_good[k]
        bad, good = next_bad, next_good

    return good[m]


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype={"total_crns": object})
    rows = []
    for _, row in df[df["m_reactions"] <= 5].iterrows():
        total = int(row["total_crns"])
        sampled = row["mode"] == "sample"
        rows.append(
            {
                "n_species": int(row["n_species"]),
                "m_reactions": int(row["m_reactions"]),
                "total_crns": total,
                "reactant_pattern_candidate_crns": reactant_pattern_candidate_count(
                    int(row["n_species"]),
                    int(row["m_reactions"]),
                ),
                "current_mode": row["mode"],
                "additional_crns_for_exact": total if sampled else 0,
                "current_samples": int(row["samples"]) if sampled else 0,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    missing = int(out["additional_crns_for_exact"].sum())
    missing_prefilter = int(
        out.loc[out["current_mode"] == "sample", "reactant_pattern_candidate_crns"].sum()
    )
    sampled_now = int(out["current_samples"].sum())
    exact_now = int(out.loc[out["current_mode"] == "exact", "total_crns"].sum())
    print("additional_crns_for_exact_m_le_5:", missing)
    print("additional_reactant_pattern_candidates_m_le_5:", missing_prefilter)
    print("current_exact_or_known_crns_m_le_5:", exact_now)
    print("current_samples_in_missing_cells:", sampled_now)
    if sampled_now:
        print("additional_vs_current_samples_ratio:", missing / sampled_now)


if __name__ == "__main__":
    main()
