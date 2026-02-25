

from typing import Any, List


def build_MAK_library(crn_template: Any, species_labels: List[str], order: int):
    """Construct and attach a mass-action reaction library.

    Args:
        crn_template: Compiled IOCRN template.
        species_labels: Species labels used by the library.
        order: Reaction order.

    Returns:
        Tuple (library, M, K, masks).
    """
    from RL4CRN.iocrns.reaction_library import construct_mass_action_library

    library = construct_mass_action_library(species_labels=species_labels, order=order)
    crn_template.set_library_context(library)

    M = len(library.reactions)
    K = library.get_num_parameters()
    masks = {
        "continuous": library.get_parameter_mask(mode="continuous"),
        "discrete": library.get_parameter_mask(mode="discrete"),
        "logit": library.get_logit_mask(),
    }
    return library, M, K, masks