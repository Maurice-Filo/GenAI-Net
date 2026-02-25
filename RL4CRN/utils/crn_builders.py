from RL4CRN.utils.input_interface import SolverCfg

def build_simple_IOCRN(
    species: list,
    production_input_map: dict,
    output_species: str,
    degradation_input_map: dict = None,
    dilution_map: dict = None,
    production_map: dict = None,
    solver: SolverCfg = SolverCfg(),
):
    """Helper to build and compile a starter IO-CRN.

    Args:
        species: List of species labels.
        input_map: Dict mapping input species indices to input channel names.
        dilution_map: Dict mapping species labels to their dilution rates.
        output_species: Label of the output species.
        production_map: Dict mapping species labels to their production rates.
        solver: Solver configuration.

    Returns:
        Tuple (crn_template, species_labels).
    """
    from RL4CRN.iocrns.iocrn import IOCRN
    from RL4CRN.iocrns.reactions import MassAction

    productions = []
    dilutions = []
    species_labels = species

    for input_species, input_name in production_input_map.items():
        productions.append(
            MassAction(
                reactant_labels=[],
                product_labels=[input_species],
                input_channels=[input_name],
                params=[1.0],
                params_controllability=[True],
            )
        )

    for disturbance_species, disturbance_name in (degradation_input_map or {}).items():
        dilutions.append(
            MassAction(
                reactant_labels=[],
                product_labels=[disturbance_species],
                input_channels=[disturbance_name],
                params=[1.0],
                params_controllability=[True],
            )
        )
    
    for species_label, dilution_rate in (dilution_map or {}).items():
        dilutions.append(
            MassAction(
                reactant_labels=[species_label],
                product_labels=[],
                input_channels=[None],
                params=[dilution_rate],
                params_controllability=[True],
            )
        )

    for species_label, production_rate in (production_map or {}).items():
        dilutions.append(
            MassAction(
                reactant_labels=[],
                product_labels=[species_label],
                input_channels=[None],
                params=[production_rate],
                params_controllability=[True],
            )
        )

    crn_template = IOCRN(
        productions + dilutions,
        output_labels=[output_species] if not isinstance(output_species, list) else output_species,
        solver=solver.algorithm,
        rtol=solver.rtol,
        atol=solver.atol,
        
    )
    crn_template.compile()
    return crn_template, species_labels


def build_logic_IOCRN(
    n_inputs: int,
    include_dilution: bool = False,
    solver: SolverCfg = SolverCfg(),
    n_support_species: int = 0,
    dilution_rate: float = 0.05,
):
    """Build and compile the template IO-CRN.

    Args:
        n_inputs: Number of inputs.
        include_dilution: Whether to include dilution reactions.
        solver: Solver configuration.
        n_support_species: Number of support species to include.
        dilution_rate: Dilution rate for species.

    Returns:
        Tuple (crn_template, species_labels).
    """
    from RL4CRN.iocrns.iocrn import IOCRN
    from RL4CRN.iocrns.reactions import MassAction

    productions = []
    dilutions = []
    species_labels = [f"X_{i+1}" for i in range(n_inputs)]

    for i, s in enumerate(species_labels):
        productions.append(
            MassAction(
                reactant_labels=[],
                product_labels=[s],
                input_channels=[f"u_{i+1}"],
                params=[1.0],
                params_controllability=[True],
            )
        )
        if include_dilution:
            dilutions.append(
                MassAction(
                    reactant_labels=[s],
                    product_labels=[],
                    input_channels=[None],
                    params=[dilution_rate],
                    params_controllability=[True],
                )
            )

    for j in range(n_support_species):
        support_label = f"S_{j+1}"
        species_labels.append(support_label)
        if include_dilution:
            dilutions.append(
                MassAction(
                    reactant_labels=[support_label],
                    product_labels=[],
                    input_channels=[None],
                    params=[dilution_rate],
                    params_controllability=[True],
                )
            )

    species_labels.append("OUT")

    crn_template = IOCRN(
        productions + dilutions,
        output_labels=["OUT"],
        solver=solver.algorithm,
        rtol=solver.rtol,
        atol=solver.atol,
    )
    crn_template.compile()
    return crn_template, species_labels