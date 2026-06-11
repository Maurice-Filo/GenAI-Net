"""List 5-species, 2-reaction CRNs excluded only by rank(S)=n.

For n=5 and m=2, full row rank is impossible because S is 5 x 2.  This script
therefore lists the CRNs that pass the reactant-pattern and q-sign conditions
before the rank filter, and writes them to a CSV for inspection.
"""

from __future__ import annotations

import csv
from itertools import combinations, product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rank_excluded_5s2r.csv"
N_SPECIES = 5


def build_complexes(n: int) -> list[tuple[int, ...]]:
    complexes = [
        alpha
        for alpha in product(range(3), repeat=n)
        if sum(alpha) <= 2
    ]
    return sorted(complexes)


def fmt_complex(v: tuple[int, ...]) -> str:
    pieces = []
    for i, count in enumerate(v, start=1):
        if count == 1:
            pieces.append(f"X{i}")
        elif count > 1:
            pieces.append(f"{count}X{i}")
    return "0" if not pieces else "+".join(pieces)


def sub(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x - y for x, y in zip(a, b))


def dot(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x * y for x, y in zip(a, b))


def build_reactions(complexes: list[tuple[int, ...]]) -> list[dict[str, tuple[int, ...]]]:
    reactions = []
    for alpha in complexes:
        for beta in complexes:
            if alpha != beta:
                reactions.append({
                    "alpha": alpha,
                    "beta": beta,
                    "zeta": sub(beta, alpha),
                })
    return reactions


def reactant_pattern_ok(a: dict[str, tuple[int, ...]],
                        b: dict[str, tuple[int, ...]]) -> bool:
    return a["alpha"][0] != b["alpha"][0] and a["alpha"][1:] == b["alpha"][1:]


def forms_allow_opposite_signs_full_space(za: tuple[int, ...],
                                          zb: tuple[int, ...]) -> bool:
    """m=2 has no 'other' reactions, so q can be any vector in R^n.

    We need some q with q.za > 0 and q.zb < 0.  This fails only when one form is
    identically zero, or when za and zb are nonnegative multiples of each other.
    """

    if all(x == 0 for x in za) or all(x == 0 for x in zb):
        return False

    collinear = True
    for i in range(len(za)):
        for j in range(i + 1, len(za)):
            if za[i] * zb[j] != za[j] * zb[i]:
                collinear = False
                break
        if not collinear:
            break
    if not collinear:
        return True

    for ai, bi in zip(za, zb):
        if ai != 0:
            return ai * bi < 0
    return False


def reaction_str(reaction: dict[str, tuple[int, ...]]) -> str:
    return f"{fmt_complex(reaction['alpha'])} -> {fmt_complex(reaction['beta'])}"


def main() -> None:
    reactions = build_reactions(build_complexes(N_SPECIES))
    rows = []
    for i, j in combinations(range(len(reactions)), 2):
        ri = reactions[i]
        rj = reactions[j]
        ordered_pairs = []
        if reactant_pattern_ok(ri, rj) and forms_allow_opposite_signs_full_space(ri["zeta"], rj["zeta"]):
            ordered_pairs.append(f"{i}->{j}")
        if reactant_pattern_ok(rj, ri) and forms_allow_opposite_signs_full_space(rj["zeta"], ri["zeta"]):
            ordered_pairs.append(f"{j}->{i}")
        if not ordered_pairs:
            continue

        rows.append({
            "crn_rank_excluded_index": len(rows),
            "reaction_i": i,
            "reaction_j": j,
            "reaction_i_text": reaction_str(ri),
            "reaction_j_text": reaction_str(rj),
            "zeta_i": ri["zeta"],
            "zeta_j": rj["zeta"],
            "passing_ordered_pairs": ";".join(ordered_pairs),
            "rank_S_upper_bound": 2,
            "required_rank": N_SPECIES,
            "exclusion_reason": "rank(S) <= m = 2 < n = 5",
        })

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"written: {OUT}")
    print(f"excluded CRNs: {len(rows)}")
    print(f"total 5s2r CRNs: {len(reactions) * (len(reactions) - 1) // 2}")


if __name__ == "__main__":
    main()
