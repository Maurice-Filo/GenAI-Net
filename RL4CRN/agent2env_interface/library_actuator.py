"""
Library actuator.

Concrete actuator that maps a policy action to a reaction sampled from a reaction
library.

The policy selects a reaction via an integer (or otherwise indexable) identifier,
and provides a set of parameters that are applied to the selected reaction before
returning it as the environment action.

Expected `policy_action` structure:

- `"reaction index"`: key used to select a reaction from the library
- `"parameters"`: parameter container passed to `reaction.set_parameters(...)`

Assumptions about `reaction_library`:

- provides method `get_reaction(index)` returning a reaction-like object
- the returned reaction provides `set_parameters(parameters)`
"""

from RL4CRN.agent2env_interface.abstract_actuator import AbstractActuator

class LibraryActuator(AbstractActuator):
    """Actuator that selects and parameterizes a reaction from a reaction library."""

    def __init__(self, reaction_library):
        """Initialize the actuator.

        Args:
            reaction_library: Object providing `get_reaction(index)` to retrieve a
                reaction template or instance given a policy-selected index.
        """
        super().__init__()
        self.reaction_library = reaction_library

    def actuate(self, policy_action):
        """Convert a policy action into a parameterized reaction.

        Args:
            policy_action: Dictionary specifying which reaction to select and which
                parameters to apply. Must contain `"reaction index"` and
                `"parameters"`.

        Returns:
            A reaction object obtained from the library and updated via `reaction.set_parameters(...)`.

        Raises:
            KeyError: If required keys (`"reaction index"` or `"parameters"`) are missing.
            AttributeError: If the library or reaction does not provide the required
                methods (`get_reaction` / `set_parameters`).
        """
        reaction = self.reaction_library.get_reaction(policy_action["reaction index"])
        reaction.set_parameters(policy_action["parameters"])
        return reaction