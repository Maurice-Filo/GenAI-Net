# Exact maxRPA sweep for 3-species CRNs

This folder contains a generic exact integer solver for 3-species CRNs with
`m` reactions, plus the sweep results for `m = 2, 3, 4, 5, 6`.

The `m = 2, 3, 4` cases were enumerated with:

```bash
g++ -O3 -std=c++17 -Wall -Wextra -pedantic maxrpa_3s_m_exact.cpp -o maxrpa_3s_m_exact
./maxrpa_3s_m_exact 2
./maxrpa_3s_m_exact 3
./maxrpa_3s_m_exact 4
```

The `m = 5, 6` values were copied from the already-computed exact standalone
enumerations in the sibling folders to avoid rerunning the expensive cases.

The combined table is:

```text
maxrpa_3s_sweep.csv
```

The summary figures are:

```text
maxrpa_3s_sweep.png
maxrpa_3s_sweep.pdf
```
