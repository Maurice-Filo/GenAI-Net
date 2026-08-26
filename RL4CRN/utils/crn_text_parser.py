"""Parse printed CRN text into reaction objects.

This module handles the human-readable CRN format emitted by ``IOCRN.__str__``.
It intentionally returns reactions rather than constructing an IOCRN, so callers
can choose output labels, solver settings, or add/remove reactions before
building the network.
"""

from __future__ import annotations

import re
from typing import List, Optional

from RL4CRN.iocrns.reactions import CatalyticMichaelisMenten, MassAction, Reaction


_MAK_RE = re.compile(
    r"^\s*(?P<lhs>.*?)\s*-+>\s*(?P<rhs>.*?);\s*"
    r"\[\s*MAK\s*\(\s*(?P<body>.*?)\s*\)\s*\]\s*$"
)

_CATALYTIC_MM_RE = re.compile(
    r"^\s*(?P<lhs>.*?)\s*-+>\s*(?P<rhs>.*?);\s*"
    r"\[\s*CatalyticMM\s*\(\s*(?P<body>.*?)\s*\)\s*\]\s*$"
)


def _parse_species_side(text: str) -> List[str]:
    """Parse one side of a reaction into a list of species labels."""
    text = text.strip()
    if text in {"", "∅", "emptyset", "None"}:
        return []
    return [part.strip() for part in text.split("+") if part.strip()]


def _parse_mak_body(text: str) -> tuple[float, Optional[str]]:
    """Parse ``MAK(k)`` or ``MAK(k, u_i)`` contents."""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) == 1:
        return float(parts[0]), None
    if len(parts) == 2:
        return float(parts[0]), parts[1]
    raise ValueError(f"Could not parse MAK body {text!r}; expected 'k' or 'k, input'.")


def _parse_key_value_body(text: str) -> dict[str, str]:
    """Parse comma-separated ``key=value`` fields."""
    out = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"Could not parse key-value field {part!r}.")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _single_species(side: List[str], *, field_name: str, line: str) -> str:
    """Return the only species on a reaction side, or raise a helpful error."""
    if len(side) != 1:
        raise ValueError(f"CatalyticMM requires one {field_name} species in line {line!r}.")
    return side[0]


def reaction_from_line(line: str, *, params_controllability: bool = True) -> Reaction:
    """Parse one printed reaction line.

    Args:
        line: A line such as ``"X_1 + X_6 ----> X_1; [MAK(0.4)]"`` or
            ``"S ----> P; [CatalyticMM(E=E, k=1.0, K=0.5)]"``.
        params_controllability: Value assigned to the reaction parameter
            controllability flag(s).

    Returns:
        A parsed reaction.

    Raises:
        ValueError: If the line is not a supported reaction line.
    """
    match = _MAK_RE.match(line)
    if match is not None:
        reactants = _parse_species_side(match.group("lhs"))
        products = _parse_species_side(match.group("rhs"))
        rate, input_channel = _parse_mak_body(match.group("body"))

        return MassAction(
            reactant_labels=reactants,
            product_labels=products,
            input_channels=[input_channel],
            params=[rate],
            params_controllability=[params_controllability],
        )

    match = _CATALYTIC_MM_RE.match(line)
    if match is not None:
        substrate = _single_species(
            _parse_species_side(match.group("lhs")),
            field_name="substrate",
            line=line,
        )
        product = _single_species(
            _parse_species_side(match.group("rhs")),
            field_name="product",
            line=line,
        )
        body = _parse_key_value_body(match.group("body"))
        try:
            catalyst = body["E"]
            maximal_rate = float(body["k"])
            michaelis_constant = float(body["K"])
        except KeyError as exc:
            raise ValueError(
                f"Could not parse CatalyticMM body {match.group('body')!r}; "
                "expected E=..., k=..., K=...."
            ) from exc

        return CatalyticMichaelisMenten(
            substrate_label=[substrate],
            product_label=[product],
            catalyst_label=[catalyst],
            input_channels=[None, None],
            params=[maximal_rate, michaelis_constant],
            params_controllability=[params_controllability, params_controllability],
        )

    raise ValueError(f"Not a supported reaction line: {line!r}")


def reactions_from_text(text: str, *, params_controllability: bool = True) -> List[Reaction]:
    """Parse all supported printed reactions from CRN text.

    Header lines such as ``CRN ...``, ``Inputs:``, ``Species:``, and
    ``Output Species:`` are ignored. Any non-empty line containing a reaction
    arrow is parsed strictly.

    Args:
        text: Printed CRN text.
        params_controllability: Value assigned to every reaction parameter's
            controllability flag.

    Returns:
        List of parsed reactions in the order they appear.

    Raises:
        ValueError: If no reactions are found or if a reaction-like line cannot
            be parsed.
    """
    reactions: List[Reaction] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "---->" not in stripped and "-->" not in stripped:
            continue
        try:
            reactions.append(
                reaction_from_line(stripped, params_controllability=params_controllability)
            )
        except ValueError as exc:
            raise ValueError(f"Failed to parse reaction on line {line_no}: {line!r}") from exc

    if not reactions:
        raise ValueError("No supported reactions found in CRN text.")

    return reactions


parse_reactions_from_text = reactions_from_text
