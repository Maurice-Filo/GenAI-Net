# maxRPA screen over species and reaction counts

This folder screens deterministic maxRPA fractions for:

- species counts `n = 2, 3, 4, 5`,
- reaction counts `m = 2, ..., 10`,
- output species fixed as `X1`,
- complexes of total molecularity at most 2,
- directed nontrivial reactions with no duplicates.

The generated table is:

```text
maxrpa_ns_sweep.csv
```

The CSV includes the Wilson interval fields directly:

```text
ci_method, ci_low, ci_high, ci_low_percent, ci_high_percent,
ci_half_width, ci_half_width_percent
```

The generated figures are:

```text
maxrpa_ns_sweep.png
maxrpa_ns_sweep.pdf
maxrpa_ns_sweep_sampled_ci.png
maxrpa_ns_sweep_sampled_ci.pdf
maxrpa_ns_sweep_ci_width.png
maxrpa_ns_sweep_ci_width.pdf
```

Important distinction:

- `mode = exact`: exhaustive enumeration of all `C(R, m)` CRNs, or a previously
  computed exact value from the dedicated 3-species integer solvers.
- `mode = sample`: fixed-seed Monte Carlo estimate with `200000` samples.

The high-dimensional cells cannot be exhaustively enumerated naively.  For
example, `n=5, m=10` has `42,251,630,707,215,665,736` possible CRNs.

The plot marks sampled cells with open markers and a `*` annotation in the
heatmap.  Error bars are 95% Wilson binomial confidence intervals for the
sampled fraction.

The separate CI-width plot shows the Wilson half-widths directly, because the
intervals are too small to be visually prominent on the main fraction scale.
The sampled-CI plot shows only sampled cells with their Wilson ribbons.

The brute-force cost of making all `m <= 5` cells exact is summarized in:

```text
exact_up_to_5_cost.csv
```

There is also a pair-first exact enumerator:

```text
maxrpa_ns_m_pairfirst.cpp
```

It enumerates a candidate special pair first, then chooses the remaining
reactions.  To avoid double-counting CRNs with multiple special pairs, it counts
a CRN only for the lexicographically first pair that also passes the q-test.  It
has been checked against the exact 3-species counts through `m=5`.

Run:

```bash
g++ -O3 -std=c++17 -Wall -Wextra maxrpa_ns_m_screen.cpp -o maxrpa_ns_m_screen
python run_ns_sweep.py
python plot_ns_sweep.py
```
