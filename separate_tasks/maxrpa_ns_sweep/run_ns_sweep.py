"""Run/assemble the n=2..5, m=2..10 maxRPA sweep.

Exact enumeration is used only where the total number of CRNs is modest.
Larger cells are estimated by fixed-seed Monte Carlo sampling and marked as
such in the CSV and plot.
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXE = ROOT / "maxrpa_ns_m_screen"
CSV_PATH = ROOT / "maxrpa_ns_sweep.csv"
SAMPLES = 200_000
EXACT_TOTAL_LIMIT = 20_000_000
WILSON_Z = 1.96


def n_complexes(n: int) -> int:
    return 1 + 2 * n + math.comb(n, 2)


def n_reactions(n: int) -> int:
    c = n_complexes(n)
    return c * (c - 1)


def parse_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().split()[0]
    return out


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def add_interval_fields(row: dict[str, object]) -> dict[str, object]:
    """Add confidence interval fields in fraction and percent units."""

    fraction = float(row["fraction"])
    if row["mode"] == "sample":
        denominator = int(row["denominator"])
        count = int(row["maxrpa_count"])
        ci_low, ci_high = wilson_interval(count, denominator)
    else:
        ci_low = fraction
        ci_high = fraction

    row["ci_method"] = "wilson_95" if row["mode"] == "sample" else "exact"
    row["ci_low"] = ci_low
    row["ci_high"] = ci_high
    row["ci_low_percent"] = 100.0 * ci_low
    row["ci_high_percent"] = 100.0 * ci_high
    row["ci_half_width"] = 0.5 * (ci_high - ci_low)
    row["ci_half_width_percent"] = 50.0 * (ci_high - ci_low)
    return row


def run_cell(n: int, m: int, mode: str, seed: int = 0) -> dict[str, object]:
    cmd = [str(EXE), mode, str(n), str(m)]
    if mode == "sample":
        cmd += [str(SAMPLES), str(seed)]
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    parsed = parse_output(proc.stdout)
    count = int(parsed["maxRPA count"])
    total = int(parsed["total CRNs"])
    if mode == "exact":
        denominator = total
    else:
        denominator = int(parsed["samples"])
    decimal = float(parsed["decimal portion"])
    return add_interval_fields({
        "n_species": n,
        "m_reactions": m,
        "n_complexes": int(parsed["number of complexes"]),
        "n_directed_reactions": int(parsed["number of possible directed reactions"]),
        "total_crns": total,
        "mode": mode,
        "samples": denominator if mode == "sample" else "",
        "seed": seed if mode == "sample" else "",
        "maxrpa_count": count,
        "denominator": denominator,
        "fraction": decimal,
        "standard_error": math.sqrt(decimal * (1.0 - decimal) / denominator)
        if mode == "sample"
        else 0.0,
    })


def main() -> None:
    # Always recompute from the executable.  This prevents stale cached rows
    # when the pass/fail definition changes, e.g. after changing the rank test.
    rows_by_key = {}

    for n in range(2, 6):
        r = n_reactions(n)
        for m in range(2, 11):
            if (n, m) in rows_by_key:
                continue

            total = math.comb(r, m)
            if total <= EXACT_TOTAL_LIMIT:
                rows_by_key[(n, m)] = run_cell(n, m, "exact")
            else:
                seed = 1000 * n + m
                rows_by_key[(n, m)] = run_cell(n, m, "sample", seed=seed)

    fieldnames = [
        "n_species",
        "m_reactions",
        "n_complexes",
        "n_directed_reactions",
        "total_crns",
        "mode",
        "samples",
        "seed",
        "maxrpa_count",
        "denominator",
        "fraction",
        "standard_error",
        "ci_method",
        "ci_low",
        "ci_high",
        "ci_low_percent",
        "ci_high_percent",
        "ci_half_width",
        "ci_half_width_percent",
    ]
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"n={row['n_species']} m={row['m_reactions']} "
            f"{row['mode']} fraction={float(row['fraction']):.6g}"
        )


if __name__ == "__main__":
    main()
