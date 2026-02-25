"""
Abstract actuator interface.

Defines the minimal interface for an *actuator*, i.e., a component that converts
policy actions (produced by an RL agent) into environment actions (consumable by
a CRN environment).

Concrete actuators should implement `actuate` and specify the expected
structure of `policy_action` as well as the returned environment action.
"""

class AbstractActuator():
    """Base class for actuators mapping policy actions to environment actions."""

    def __init__(self):
        """Initialize the actuator."""
        pass

    def actuate(self, policy_action):
        """Convert a policy action into an environment action.

        Args:
            policy_action: Action produced by a policy/agent.

        Returns:
            Environment action derived from `policy_action`. Concrete actuators should document the returned type/structure.
        """
        pass