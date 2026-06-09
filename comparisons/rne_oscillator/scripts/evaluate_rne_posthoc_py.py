#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import roadrunner
import tellurium as te


def parse_fitness(astr: str) -> float | None:
    for line in astr.splitlines():
        if line.startswith("#fitness"):
            parts = line.split()
            if len(parts) < 2:
                return None
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def has_oscillator_eigens(eigens) -> bool:
    arr = np.asarray(eigens, dtype=complex).reshape(-1)
    return bool(np.any((arr.real >= 0.0) & (np.abs(arr.imag) > 0.0)))


def concentrations_nonnegative(rr) -> bool:
    conc = np.asarray(rr.getFloatingSpeciesConcentrations(), dtype=float)
    return bool(np.all(conc >= 0.0))


def load_antimony(astr: str):
    return te.loada(astr)


def is_oscillator(astr: str) -> bool:
    rr = load_antimony(astr)
    try:
        rr.steadyState()
        eigens = rr.getFullEigenValues()
        if has_oscillator_eigens(eigens) and concentrations_nonnegative(rr):
            return True
    except Exception:
        pass

    for tight in (False, True):
        try:
            rr.resetToOrigin()
            rr.timeCourseSelections = ["time", *rr.getFloatingSpeciesIds()]
            if tight:
                try:
                    rr.getIntegrator().relative_tolerance = 1e-10
                except Exception:
                    pass
            rr.simulate(0, 50, 100000)
            break
        except Exception:
            if tight:
                return False

    try:
        eigens = rr.getFullEigenValues()
        if has_oscillator_eigens(eigens):
            return True
    except Exception:
        pass

    try:
        rr.steadyState()
        eigens = rr.getFullEigenValues()
        return has_oscillator_eigens(eigens) and concentrations_nonnegative(rr)
    except Exception:
        return False


def is_broken_oscillator(astr: str) -> bool:
    rr = load_antimony(astr)
    try:
        rr.steadyState()
        eigens = rr.getFullEigenValues()
        return has_oscillator_eigens(eigens) and not concentrations_nonnegative(rr)
    except Exception:
        return False


def fix_broken_oscillator(astr: str) -> str | None:
    lines = astr.splitlines()
    for idx, line in enumerate(lines):
        if "->" not in line:
            continue
        old = lines[idx]
        lines[idx] = "#" + old
        fixed = "\n".join(lines)
        if is_oscillator(fixed):
            return fixed
        lines[idx] = old
    return None


def evaluate_file(path: Path) -> dict[str, object]:
    astr = path.read_text(encoding="utf-8")
    fitness = parse_fitness(astr)
    loss = None if fitness in (None, 0.0) else 1.0 / fitness
    row: dict[str, object] = {
        "seed": path.stem,
        "file": str(path),
        "fitness": "" if fitness is None else fitness,
        "loss": "" if loss is None else loss,
        "parse_ok": True,
        "rne_is_oscillator": False,
        "rne_is_broken_oscillator": False,
        "rne_fixed_by_reaction_removal": False,
        "rne_posthoc_success": False,
        "error": "",
    }
    try:
        good = is_oscillator(astr)
        row["rne_is_oscillator"] = good
        if good:
            row["rne_posthoc_success"] = True
            return row
        broken = is_broken_oscillator(astr)
        row["rne_is_broken_oscillator"] = broken
        if broken:
            fixed = fix_broken_oscillator(astr)
            row["rne_fixed_by_reaction_removal"] = fixed is not None
            row["rne_posthoc_success"] = fixed is not None
    except Exception as exc:
        row["parse_ok"] = False
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ant_dir")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    input_dir = Path(args.input_ant_dir)
    output_csv = Path(args.output_csv)
    files = sorted(input_dir.glob("*.ant"))
    rows = [evaluate_file(path) for path in files]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "file",
        "fitness",
        "loss",
        "parse_ok",
        "rne_is_oscillator",
        "rne_is_broken_oscillator",
        "rne_fixed_by_reaction_removal",
        "rne_posthoc_success",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(bool(row["rne_posthoc_success"]) for row in rows)
    print(f"Evaluated {len(rows)} networks")
    print(f"RNE posthoc successes: {successes}/{len(rows)}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
