import numpy as np
from RL4CRN.Abstract.AbstractAgent import AbstractAgent

class RandomAgent(AbstractAgent):
    """Random agent for CRN environments.
    This agent randomly selects actions from the action space of the environment.
    """
    def __init__(self, env, max_rate_constant=10, allow_input_influence=False, logger=None):
        super(RandomAgent, self).__init__()
        self.env = env
        self.max_rate_constant = max_rate_constant
        self.allow_input_influence = allow_input_influence
        self.logger = logger

    def act(self):
        super(RandomAgent, self).act()
        action = {}
        action['reactants index'] = np.random.randint(self.env.action_space['reactants space'].n)
        action['products index'] = np.random.randint(self.env.action_space['products space'].n)
        if self.input_influence_flag:
            action['input influence index'] = np.random.randint(self.env.action_space['input influence space'].n)
        else:
            action['input influence index'] = 0
        action['rate constant'] = np.random.uniform(0, self.max_rate_constant)
        return action