"""
Identity actuator. [DEPRECATED]

Concrete implementation of an actuator that directly maps policy actions to
environment actions.

In this actuator, each policy action is interpreted as a single reaction. The
reaction's rate constant is taken from the first entry of the policy action's
`"parameters"` field and stored under `"rate constant"`. The `"parameters"`
field is then removed. (as required for MAK reactions)

Note:
    This class is used for debugging purposes and simple scenarios.

Limitations:
    This currently assumes a single rate parameter per reaction. For reaction
    families that require multiple parameters (e.g., some higher-order kinetics),
    the mapping logic should be extended accordingly.

Expected `policy_action` structure:
    - `"parameters"`: sequence where the first element is the rate constant
    - other keys are treated as reaction metadata and passed through
"""

from RL4CRN.agent2env_interface.abstract_actuator import AbstractActuator

class IdentityActuator(AbstractActuator):
    """Actuator that maps a policy action directly to an environment reaction.

    The transformation performed by `actuate` is:
    - Set ``action["rate constant"]`` to the first element of
      ``action["parameters"]``.
    - Remove the ``"parameters"`` entry from the returned dictionary.

    Note:
        The returned action is the same dictionary object as the input
        `policy_action`, modified in-place.
    """

    def __init__(self):
        """Initialize the actuator."""
        super().__init__()

    def actuate(self, policy_action):
        """Convert a policy action dictionary into an environment action.

        Args:
            policy_action: Dictionary encoding a reaction. Must contain a
                `"parameters"` entry whose first element is the rate constant.

        Returns:
            A dictionary representing the environment action (reaction), with the
            rate constant stored under `"rate constant"` and `"parameters"`
            removed.

        Raises:
            KeyError: If `"parameters"` is missing from `policy_action`.
            IndexError: If `"parameters"` is present but empty.
        """
        action = policy_action
        action["rate constant"] = action["parameters"][0] 
        action.pop("parameters")          
        return action