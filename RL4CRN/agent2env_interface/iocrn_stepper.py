from RL4CRN.agent2env_interface.abstract_stepper import AbstractStepper

class IOCRNStepper(AbstractStepper):

    def step(self, state, action):
        if state.num_unknown_params > 0:
            raise NotImplementedError("Setting parameters directly is not implemented yet.")
        else:
            state.add_reaction(action)
            state.compile()