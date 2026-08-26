"""Symbolic calculations for IOCRNs.

This module contains offline symbolic utilities that are intentionally kept
outside ``IOCRN`` itself. The routines here can be expensive for large CRNs,
especially Groebner-basis elimination, so they are best used for inspecting a
small number of candidate networks rather than inside a training loop.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import sympy as sp

from RL4CRN.iocrns.reactions import MassAction


def output_algebraic_equation(
    crn,
    output_label: Optional[str] = None,
    input_labels: Iterable[str] = ("u_1", "u_2", "u_3"),
    parameter_prefix: str = "k",
    use_numeric_parameters: bool = False,
    output_symbol_name: str = "z",
    simplify: bool = True,
) -> Dict[str, object]:
    """Compute an implicit polynomial equation for an IOCRN output.

    The function constructs the symbolic mass-action steady-state equations

    ``dx_i/dt = 0``

    and eliminates every non-output species using a lexicographic Groebner
    basis. The result is a polynomial

    ``P(z, u_1, u_2, u_3, k_0, k_1, ...) = 0``

    where ``z`` is the selected output species.

    Args:
        crn: Compiled or uncompiled ``IOCRN`` instance.
        output_label: Output species to keep. Defaults to the only CRN output.
            If the CRN has multiple outputs, this argument is required.
        input_labels: Input labels that should exist as symbolic variables.
            Inputs present in the CRN are also included even if omitted here.
        parameter_prefix: Prefix for symbolic reaction parameters.
        use_numeric_parameters: If true, known mass-action rate constants are
            inserted numerically. If false, each reaction receives a symbolic
            parameter ``k_0, k_1, ...``.
        output_symbol_name: Symbol name used for the output variable.
        simplify: Simplify/factor the polynomial and coefficients before
            returning.

    Returns:
        Dictionary with:
            ``polynomial``:
                SymPy expression for ``P(z, ...)``.
            ``coefficients``:
                Coefficients of ``P`` ordered from highest to lowest power of
                the output symbol.
            ``degree``:
                Degree of ``P`` in the output symbol.
            ``output_symbol``:
                SymPy symbol used for the output.
            ``input_symbols``:
                Mapping from input labels to SymPy symbols.
            ``parameter_symbols``:
                Mapping from reaction index to symbolic/numeric rate factor.
            ``species_symbols``:
                Mapping from species labels to SymPy symbols.
            ``steady_state_equations``:
                Mapping from species labels to symbolic ODE right-hand sides.
            ``groebner_basis``:
                The computed SymPy Groebner basis.

    Raises:
        NotImplementedError: If any reaction is not ``MassAction``.
        ValueError: If no nontrivial output-only polynomial is found.

    Notes:
        Groebner elimination can be very slow for large CRNs. Use this as an
        offline verifier for promising candidates, not during RL training.
    """
    if not hasattr(crn, "species_labels") or not hasattr(crn, "input_labels"):
        crn.compile()

    if output_label is None:
        if len(crn.output_labels) != 1:
            raise ValueError(
                "output_label is required when the IOCRN has multiple outputs."
            )
        output_label = crn.output_labels[0]

    if output_label not in crn.species_labels:
        raise ValueError(f"Unknown output species label: {output_label}")

    for reaction in crn.reactions:
        if not isinstance(reaction, MassAction):
            raise NotImplementedError(
                "Symbolic elimination currently supports only MassAction reactions."
            )

    output_symbol = sp.Symbol(output_symbol_name)
    species_symbols = {
        label: output_symbol if label == output_label else sp.Symbol(label)
        for label in crn.species_labels
    }
    all_input_labels = sorted(set(crn.input_labels).union(input_labels))
    input_symbols = {label: sp.Symbol(label) for label in all_input_labels}

    parameter_symbols = {}
    steady_state_equations = {
        label: sp.Integer(0) for label in crn.species_labels
    }

    for reaction_idx, reaction in enumerate(crn.reactions):
        if use_numeric_parameters and reaction.rate_constant is not None:
            rate_constant = sp.nsimplify(reaction.rate_constant)
        else:
            rate_constant = sp.Symbol(f"{parameter_prefix}_{reaction_idx}")
        parameter_symbols[reaction_idx] = rate_constant

        input_channel = reaction.input_channels[0]
        input_factor = (
            sp.Integer(1)
            if input_channel is None
            else input_symbols[input_channel]
        )

        propensity = rate_constant * input_factor
        for reactant_label in reaction.reactant_labels:
            propensity *= species_symbols[reactant_label]

        for species_label, stoich in reaction.get_stoichiometry_dict().items():
            steady_state_equations[species_label] += stoich * propensity

    equations = [
        sp.expand(steady_state_equations[label])
        for label in crn.species_labels
    ]
    hidden_symbols = [
        species_symbols[label]
        for label in crn.species_labels
        if label != output_label
    ]

    groebner_basis = sp.groebner(
        equations,
        *hidden_symbols,
        output_symbol,
        order="lex",
    )

    hidden_set = set(hidden_symbols)
    candidates = []
    for poly in groebner_basis.polys:
        expr = poly.as_expr()
        if expr == 0:
            continue
        if output_symbol not in expr.free_symbols:
            continue
        if expr.free_symbols.isdisjoint(hidden_set):
            candidates.append(sp.expand(expr))

    if not candidates:
        raise ValueError(
            "Groebner elimination did not produce a nontrivial polynomial in "
            f"the output {output_label!r}."
        )

    polynomial = min(candidates, key=lambda expr: sp.Poly(expr, output_symbol).degree())
    if simplify:
        polynomial = sp.factor(sp.simplify(polynomial))

    poly_in_output = sp.Poly(polynomial, output_symbol)
    coefficients = poly_in_output.all_coeffs()
    if simplify:
        coefficients = [sp.factor(sp.simplify(c)) for c in coefficients]

    return {
        "polynomial": polynomial,
        "coefficients": coefficients,
        "degree": poly_in_output.degree(),
        "output_symbol": output_symbol,
        "input_symbols": input_symbols,
        "parameter_symbols": parameter_symbols,
        "species_symbols": species_symbols,
        "steady_state_equations": steady_state_equations,
        "groebner_basis": groebner_basis,
    }

