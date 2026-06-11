# Exact maxRPA enumeration for 3 species and 6 reactions

This standalone folder computes the exact fraction of labelled 3-species,
6-reaction CRNs satisfying deterministic maximal robust perfect adaptation
under the Gupta-Khammash stoichiometric characterization.

The solver:

- builds the 10 complexes with total molecularity at most 2,
- builds the 90 nontrivial directed reactions,
- enumerates all unordered no-duplicate 6-reaction subsets,
- tests every ordered special reaction pair,
- uses exact integer rank/nullspace geometry for the stoichiometric q-test.

No floating point arithmetic is used for the pass/fail decision.

Run:

```bash
g++ -O3 -std=c++17 -Wall -Wextra -pedantic maxrpa_3s6r_exact.cpp -o maxrpa_3s6r_exact
./maxrpa_3s6r_exact
```

Expected sanity checks:

- complexes: 10
- directed reactions: 90
- total CRNs: `C(90, 6) = 622,614,630`
