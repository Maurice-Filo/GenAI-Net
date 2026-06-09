#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHOD = "rl4crn_rne_oscillator_trace_risk09_lr1e4_entropy001_s4_c03"
DEFAULT_LOSSES_CSV = (
    ROOT
    / "comparisons/rne_oscillator/figures/"
    / "rne_oscillator_latest_losses_risk09_lr1e4_entropy001_s4_c03.csv"
)
DEFAULT_RAW_ROOT = ROOT / "comparisons/rne_oscillator/data/raw"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "comparisons/rne_oscillator/data/posthoc/"
    / f"{DEFAULT_METHOD}_antimony"
)


REACTION_RE = re.compile(
    r"^(?P<lhs>.*?)\s*-+>\s*(?P<rhs>.*?);\s*\[MAK\((?P<rate>[-+0-9.eE]+)\)\]$"
)


def _species_side(text: str) -> list[str]:
    text = text.strip()
    if not text or text == "∅":
        return []
    return [part.strip() for part in text.split("+")]


def _rate_law(reactants: list[str], k_name: str) -> str:
    if not reactants:
        return k_name
    return "*".join([k_name, *reactants])


def _reaction_to_antimony(idx: int, reaction: str) -> tuple[str, str]:
    match = REACTION_RE.match(reaction.strip())
    if match is None:
        raise ValueError(f"Could not parse reaction: {reaction!r}")

    reactants = _species_side(match.group("lhs"))
    products = _species_side(match.group("rhs"))
    if not reactants and not products:
        return "", ""
    rate = float(match.group("rate"))
    k_name = f"k{idx}"

    lhs = " + ".join(reactants)
    rhs = " + ".join(products)
    line = f"_J{idx}: {lhs} -> {rhs}; {_rate_law(reactants, k_name)};"
    return line, f"{k_name} = {rate:.17g};"


def network_to_antimony(network: dict, fitness: float | None) -> str:
    species = list(network["species"])
    reactions = list(network["reactions"])

    reaction_lines: list[str] = []
    rate_lines: list[str] = []
    for idx, reaction in enumerate(reactions, start=1):
        reaction_line, rate_line = _reaction_to_antimony(idx, reaction)
        if not reaction_line:
            continue
        reaction_lines.append(reaction_line)
        rate_lines.append(rate_line)

    initial = {
        "X_1": 1.0,
        "X_2": 5.0,
        "X_3": 9.0,
    }

    lines = [
        f"#fitness: {fitness:.17g}" if fitness is not None else "#fitness:",
        "species " + ", ".join(species) + ";",
        "",
        "// Reactions:",
        *reaction_lines,
        "",
        "// Species initializations:",
        *[f"{label} = {initial.get(label, 0.0):.17g};" for label in species],
        "",
        "// Variable initializations:",
        *rate_lines,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--losses-csv", default=str(DEFAULT_LOSSES_CSV))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    losses_csv = Path(args.losses_csv)
    raw_dir = Path(args.raw_root) / args.method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(losses_csv.open(newline="", encoding="utf-8")))
    manifest_rows: list[dict[str, str]] = []

    for row in rows:
        seed_value = row["seed"]
        seed = seed_value if str(seed_value).startswith("seed") else f"seed{int(seed_value):03d}"
        network_path = raw_dir / seed / "best_network.json"
        if not network_path.exists():
            manifest_rows.append(
                {
                    "seed": seed,
                    "antimony_file": "",
                    "loss": row.get("saved_best_loss", ""),
                    "exported": "False",
                    "error": f"missing {network_path}",
                }
            )
            continue

        network = json.loads(network_path.read_text(encoding="utf-8"))
        loss_text = row.get("saved_best_loss", "") or row.get("best_loss", "") or row.get("loss", "")
        loss = float(loss_text)
        fitness = 1.0 / loss if loss > 0 else None
        antimony = network_to_antimony(network, fitness)
        ant_path = output_dir / f"{seed}.ant"
        ant_path.write_text(antimony, encoding="utf-8")
        manifest_rows.append(
                {
                    "seed": seed,
                    "antimony_file": str(ant_path),
                "loss": f"{loss:.17g}",
                "exported": "True",
                "error": "",
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "antimony_file", "loss", "exported", "error"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    exported = sum(row["exported"] == "True" for row in manifest_rows)
    print(f"Exported {exported}/{len(manifest_rows)} networks to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
