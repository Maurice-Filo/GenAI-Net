import numpy as np

class IC():
    def __init__(self, names, values):
        """ Initializes the initial conditions (IC) for species in a CRN.
        Arguments:
        - names: a list of species names.
        - values: a list of initial conditionals corresponding to the species names.
        """
        self.names = names
        self.values = values
        self.name_to_index = {name: idx for idx, name in enumerate(names)}
        self.index_to_name = names

    def get_ic(self, crn):
        """ Retrieves the initial conditions for the species in the given CRN.
        Arguments:
        - crn: a CRN instance.
        Returns:
        - A list of initial concentration values for the species in the CRN, in the order they appear in the CRN.
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