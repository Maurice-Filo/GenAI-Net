"""
MassActionIOCRNStepper [DEPRECATED]
"""

from RL4CRN.agent2env_interface.abstract_stepper import AbstractStepper

class MassActionIOCRNStepper(AbstractStepper):

    def step(self, state, action):
        if state.num_unknown_rates > 0:
            raise NotImplementedError("Setting rate constants directly is not implemented yet.")
        else:
            state.add_reaction(action, mode='reaction index')        