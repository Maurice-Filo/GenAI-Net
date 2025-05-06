class AbstractAgent:
    def __init__(self):
        pass

    def act(self):
        """
        Take an action onto the environment by adding a reaction. For mass-action kinetics, the action is a dictionary of the following form:
        {
            'reactants space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'products space': gym.spaces.Discrete(self.CRN_template.get_complexes_range()),
            'input influence space': gym.spaces.Discrete(self.CRN_template.num_inputs + 1),
            'rate constant space': gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32)
        }
        NOTE: In the current implementation, the data type of the fields are numpy based meant to run on CPUs.
        """
        pass
    
    def update(self, rewards):
        """
        Update the agent with the rewards received from the environment by doing the backward pass.
        NOTE: This function does not return anything. It is meant to update the agent's internal state.
        """
        pass