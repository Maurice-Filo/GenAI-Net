import numpy as np
from RL4CRN.env2agent_interface.abstract_observer import AbstractObserver

class MassActionCRNObserver(AbstractObserver):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.iocrn = None

    def observe(self, iocrn):
        reactions_indices = iocrn.reactions_indices
        rate_constants = iocrn.c
        reactions_indices_influenced_by_inputs = iocrn.list_influenced_reactions
        return reactions_indices, rate_constants, reactions_indices_influenced_by_inputs, self.M