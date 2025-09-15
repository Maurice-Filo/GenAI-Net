from RL4CRN.agent2env_interface.abstract_actuator import AbstractActuator

class LibraryActuator(AbstractActuator):
    def __init__(self, reaction_library):
        super().__init__()
        self.reaction_library = reaction_library

    def actuate(self, policy_action):
        reaction = self.reaction_library.get_reaction(policy_action["reaction index"])
        reaction.set_parameters(policy_action["parameters"])
        return reaction