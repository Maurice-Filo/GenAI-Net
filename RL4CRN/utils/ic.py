"""
Initial-condition (IC) helpers for CRN simulations.

This module provides a lightweight `IC` container that stores one or more sets of
initial concentrations keyed by species names, and can project/reorder those
initial conditions to match the species ordering of a given CRN.

Typical use:

- Define ICs in a human-friendly order (by name).
- Call `get_ic(crn)` to obtain NumPy vectors aligned with `crn.species_labels`,
  ready to be passed to simulation / transient-response routines.
"""

import numpy as np

class IC():
    """
    Container for named initial conditions for CRN species.

    The object stores one or multiple initial-condition vectors defined over a
    fixed list of species `names`. When used with a CRN instance, it can reorder
    each vector to match the CRN's internal `species_labels` ordering.

    Args:
        names (list[str]): Species names corresponding to the entries in each IC vector.
        values (list[array-like]): One or more IC vectors. Each vector must have
            length `len(names)` and is interpreted as concentrations aligned with `names`.

    Attributes:
        names (list[str]): Species names used as the IC coordinate system.
        values (list[array-like]): Stored IC vectors aligned with `names`.
        name_to_index (dict[str,int]): Map from species name to its index in `names`.
        index_to_name (list[str]): Alias for `names` (kept for convenience).
    """

    def __init__(self, names, values):
        """
        Initialize an IC container.

        Args:
            names (list[str]): Species names.
            values (list[array-like]): One or more initial condition vectors aligned with `names`.
        """
        self.names = names
        self.values = values
        self.name_to_index = {name: idx for idx, name in enumerate(names)}
        self.index_to_name = names

    def get_ic(self, crn):
        """
        Reorder stored IC vectors to match a CRN's species ordering.

        The CRN is expected to expose `crn.species_labels` (list[str]). For each
        stored IC vector, this method produces a NumPy array whose entries are
        ordered exactly as `crn.species_labels`.

        Args:
            crn: CRN-like object with attribute `species_labels` (list[str]).

        Returns:
            list[np.ndarray]: List of IC arrays, one per entry in `self.values`,
            each of shape `(len(crn.species_labels),)`.

        Raises:
            ValueError: If the CRN contains a species label that is not present in `self.names`.
        """
        ic_list = []
        for ic in self.values:
            ic_values = []
            for species in crn.species_labels:
                if species in self.name_to_index:
                    idx = self.name_to_index[species]
                    ic_values.append(ic[idx])
                else:
                    raise ValueError(f"Initial condition for species '{species}' not found.")
            ic_list.append(np.array(ic_values))
        return ic_list
    
    def __str__(self):
        """Return a readable string representation of the IC container."""
        return f"IC(names={self.names}, values={self.values})"  