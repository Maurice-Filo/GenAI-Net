"""
Abstract agent interface.

Defines the minimal interface for RL agents used in this project. An agent is
responsible for:

- producing actions to be applied to the environment (`act`)
- updating internal parameters from observed rewards (`update`)
- translating environment states into agent-specific representations
  (`translate_state`)

Concrete agents should override these methods to implement their behavior.
"""

class AbstractAgent:
    """Base class for agents that act in and learn from a CRN environment."""

    def __init__(self):
        """Initialize the agent."""
        pass

    def act(self):
        """Produce an action to be applied to the environment.

        The returned object should be compatible with the agent-to-environment
        interface used by the environment/actuator. Concrete agents must override
        this method.

        Returns:
            A policy action representing a reaction to add or select. 
                The exact structure depends on the concrete agent and the actuator in use.
        """
        pass
    
    def update(self, rewards, hof=None):
        """Update the agent from rewards (e.g., perform a backward pass).

        This method is intended to update the agent's internal state and does not
        return anything.

        If the agent exposes a `policy` attribute and that policy defines
        `reset_template()`, this method calls it after each update. This is used
        to distinguish template reactions from reactions added during interaction.

        This method automatically resets the initial CRN template after each update.

        Args:
            rewards: Reward signal(s) returned by the environment. The expected
                type/shape is agent-specific.
            hof: Optional hall-of-fame (or similar) object carrying elite samples
                or trajectories. Concrete agents may use or ignore it.

        Returns:
            None.
        """
        # reset the initial mask after each update
        if hasattr(self, "policy"):
            if hasattr(self.policy, "reset_template"):
                self.policy.reset_template() # used to distinguish the template reactions from the added reactions
        pass

    def translate_state(self, state):
        """Translate an environment state into an agent-specific representation.

        Concrete agents should override this method to implement state encoding /
        feature extraction suitable for their policy and value function(s).

        Args:
            state: Environment state object.

        Returns:
            An agent-specific representation of `state` (e.g., tensors, feature
                vectors, graphs). The exact type depends on the implementation.
        """
        pass