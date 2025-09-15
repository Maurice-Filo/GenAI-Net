import numpy as np
from RL4CRN.env2agent_interface.abstract_observer import AbstractObserver

class ExplicitObserver(AbstractObserver):
    def __init__(self, reaction_library):
        super().__init__()
        self.reaction_library = reaction_library
        self.iocrn = None

    def observe(self, iocrn):
        """
        Returns the explicit state representation of the IOCRN.
        The explicit state consists of:
        - A multi-hot encoding of the reactions present in the IOCRN.
        - A cross product of the reaction parameters and the reaction multi-hot encoding.
        - A multi-hot encoding of the inputs controlling the reactions.
        """
        self.iocrn = iocrn
        reaction_multihot = self.reactions_to_multihot()               # shape (M,)
        params_cross_multihot = self.params_cross_multihot()          # shape (P,)
        inputs_multihot = self.inputs_to_multihot()                   # shape (p, C)
        explicit_state = (reaction_multihot, params_cross_multihot, inputs_multihot)
        return explicit_state
        
    def reactions_to_multihot(self):
        """
        Returns a multi-hot encoding of the reactions present in the IOCRN.
        """
        # Get the indices of the reactions in the IOCRN
        idx = np.array(self.iocrn.gather_reaction_IDs(), dtype=np.long) 
        multihot = np.zeros(len(self.reaction_library)) 
        multihot[idx] = 1.
        return multihot
    
    def params_cross_multihot(self):
        multihot = np.zeros(self.reaction_library.get_num_parameters())
        for reaction in self.iocrn.reactions:
            idx = self.reaction_library.parameter_lookup_table[reaction.ID]
            for j in range(reaction.num_parameters):
                multihot[idx + j] = reaction.params[j]
        return multihot
    
    def inputs_to_multihot(self):
        multihots = []
        for i in range(self.iocrn.num_inputs):
            multihot = np.zeros(self.reaction_library.get_num_controllable_parameters())
            for reaction in self.iocrn.reactions:
                idx = self.reaction_library.controllable_parameter_lookup_table[reaction.ID]
                for j in range(reaction.get_num_controllable_parameters()):
                    multihot[idx + j] = 1 if reaction.input_channels[j] == self.iocrn.input_labels[i] else 0
            multihots.append(multihot)
        multihots = np.concatenate(multihots)
        return multihots