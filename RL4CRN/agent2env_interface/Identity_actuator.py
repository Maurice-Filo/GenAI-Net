from RL4CRN.agent2env_interface.abstract_actuator import AbstractActuator

class IdentityActuator(AbstractActuator):
    def __init__(self):
        super().__init__()

    def actuate(self, policy_action):
        action = policy_action
        action["rate constant"] = action["parameters"][0]
        action.pop("parameters")          
        return action