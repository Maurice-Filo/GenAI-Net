class AbstractAgent:
    def __init__(self):
        pass

    def act(self):
        """
        Take an action onto the environment by adding a reaction. 
        This function should be overridden by subclasses to implement specific agent behavior.
        """
        pass
    
    def update(self, rewards):
        """
        Update the agent with the rewards received from the environment by doing the backward pass.
        NOTE: This function does not return anything. It is meant to update the agent's internal state.
        """
        pass

    def translate_state(self, state):
        """
        Translate the environment state into a format suitable for the agent.
        This function should be overridden by subclasses to implement specific translation logic.
        """
        pass