"""
Abstract observer interface.

Defines the minimal interface for an *observer*, i.e., a component that extracts
an observation from an environment state for consumption by an agent.

Observers are used in the environment-to-agent (env2agent) interface and are
typically paired with a tensorizer/encoder that converts observations into
tensors suitable for neural policies.
"""


class AbstractObserver():
    """Base class for observers that produce agent-facing observations."""

    def __init__(self):
        """Initialize the observer."""
        pass

    def observe(self):
        """Produce an observation from the environment.

        Concrete observers should override this method and define:
        
        - what inputs they require (e.g., state, environment handle, timestep),
        - the structure/type of the returned observation.

        Returns:
            An observation object (type/structure depends on the concrete observer).
        """
        pass